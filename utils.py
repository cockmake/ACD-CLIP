import random

import numpy as np
import torch
import torch.nn as nn
from torchmetrics.functional import auroc, average_precision
from torchvision import transforms

from dataset.info import CLASS_NAMES, REAL_NAMES, PROMPTS
from model.tokenizer import tokenize


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(
            self,
            apply_nonlin=None,
            alpha=None,
            gamma=2,
            balance_index=0,
            smooth=1e-5,
            size_average=True,
    ):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError("Not support alpha type")

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth
            )
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (
                input_flat.sum(1) + targets_flat.sum(1) + smooth
        )
        loss = 1 - N_dice_eff.sum() / N
        return loss


prompt = PROMPTS
prompt_normal = prompt["prompt_normal"]
prompt_abnormal = prompt["prompt_abnormal"]
prompt_state = [prompt_normal, prompt_abnormal]
prompt_templates = prompt["prompt_templates"]


def get_multiple_adapted_single_class_text_embedding(
        model,
        dataset_name, class_name, device
):
    if class_name == "object":
        real_name = class_name
    else:
        assert class_name in CLASS_NAMES[dataset_name], (
            f"class_name {class_name} not found; available class_names: {CLASS_NAMES[dataset_name]}"
        )
        real_name = REAL_NAMES[dataset_name][class_name]
    multi_layer_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(real_name) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence).to(device)
        multi_features = model.encode_text(prompted_sentence)
        for layer_feature in multi_features:
            layer_feature = layer_feature / layer_feature.norm(dim=-1, keepdim=True)
            layer_feature = layer_feature.mean(dim=0)
            layer_feature = layer_feature / layer_feature.norm()
            multi_layer_features.append(layer_feature)

    text_features_levels = []
    cnt = len(multi_layer_features) // len(prompt_state)
    for i in range(cnt):
        text_features_levels.append(torch.stack(multi_layer_features[i::cnt], dim=1).to(device))
    return torch.stack(text_features_levels, dim=0)


def get_multiple_adapted_text_embedding(
        model,
        dataset_name, device
):
    ret_dict = {}
    for class_name in CLASS_NAMES[dataset_name]:
        multi_layer_text_features = get_multiple_adapted_single_class_text_embedding(
            model, dataset_name, class_name, device
        )
        ret_dict[class_name] = multi_layer_text_features
    return ret_dict


focal_loss = FocalLoss()
dice_loss = BinaryDiceLoss()


def calculate_seg_loss(patch_preds, mask):
    loss = focal_loss(patch_preds, mask)
    loss += dice_loss(patch_preds[:, 0, :, :], 1 - mask)
    loss += dice_loss(patch_preds[:, 1, :, :], mask)
    return loss


def metrics_eval_gpu(
        pixel_label: torch.Tensor,
        image_label: torch.Tensor,
        pixel_preds: torch.Tensor,
        image_preds: torch.Tensor,
        class_names: str,
        domain: str,
):
    pixel_preds = torch.flatten(pixel_preds, start_dim=1)
    pmax_pred, _ = torch.max(pixel_preds, dim=1)
    if domain == "Medical":
        image_preds = image_preds * 0.5 + pmax_pred * 0.5
    else:
        image_preds = image_preds * 0.9 + pmax_pred * 0.1

    pixel_label = pixel_label.flatten()
    pixel_preds = pixel_preds.flatten()

    zero_pixel_auc = auroc(pixel_preds, pixel_label, task="binary")
    zero_pixel_ap = average_precision(pixel_preds, pixel_label, task="binary")

    if image_label.max() != image_label.min():
        image_label = image_label.flatten()
        agg_image_preds = image_preds.flatten()
        agg_image_auc = auroc(agg_image_preds, image_label, task="binary")
        agg_image_ap = average_precision(agg_image_preds, image_label, task="binary")
    else:
        agg_image_auc = torch.tensor(0.0, device=pixel_preds.device)
        agg_image_ap = torch.tensor(0.0, device=pixel_preds.device)
    # ================================================================================================
    result = {
        "class name": class_names,
        "pixel AUC": round(zero_pixel_auc.item(), 4) * 100,
        "pixel AP": round(zero_pixel_ap.item(), 4) * 100,
        "image AUC": round(agg_image_auc.item(), 4) * 100,
        "image AP": round(agg_image_ap.item(), 4) * 100,
    }
    return result


class AddGaussianNoise(object):
    def __init__(self, std=1.0, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, x):
        """
        在数据张量上应用噪音
        """
        if random.random() < self.p:
            return x
        if not isinstance(x, torch.Tensor):
            x = transforms.ToTensor()(x)
        noise_mask = (torch.randn(x.shape[-2:]) > 3).int()
        noise = torch.randn_like(x) * self.std  # mean = 0
        noised_img = (1 - noise_mask) * x + noise * x * noise_mask
        noised_img = torch.clamp(noised_img, 0.0, 1.0)
        return transforms.ToPILImage()(noised_img)

    def __repr__(self):
        return self.__class__.__name__ + f"p={self.p}, std={self.std}"
