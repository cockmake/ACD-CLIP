import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_text_and_image_dataset
from utils import (
    calculate_seg_loss, get_multiple_adapted_single_class_text_embedding
)
from model.adapter import (
    ACDCLIP
)
from model.clip import create_model


def train(
        model: ACDCLIP,
        dataset_name: str,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: str | torch.device,
        total_epoch: int,
        save_path: str,
        logger: logging.Logger,
):
    for epoch in range(0, total_epoch):
        logger.info(f"training epoch {epoch + 1} / {total_epoch}")
        loss_list = []
        seg_loss_list = []
        tqdm_train_loader = tqdm(train_loader)
        for input_data in tqdm_train_loader:
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            label = input_data["label"].to(device)
            class_names = input_data["class_name"]
            # get adapted text embedding
            epoch_text_feature_dict = {}
            for class_name in list(set(class_names)):
                text_embedding_levels = get_multiple_adapted_single_class_text_embedding(
                    model, dataset_name, class_name, device
                )
                epoch_text_feature_dict[class_name] = text_embedding_levels  # [n_groups, 768, 2]
            epoch_text_features = torch.stack(
                [epoch_text_feature_dict[class_name] for class_name in class_names],
                dim=0,
            )  # [bs, n_groups, 768, 2]
            epoch_text_features = epoch_text_features.permute(1, 0, 2, 3)  # [n_groups, bs, 768, 2]
            seg_tokens, det_tokens = model(image)  # [bs, patch_size, 768] * n_groups, [bs, 768] * n_groups
            seg_features = torch.stack(seg_tokens, dim=0)  # [n_groups, bs, patch_num, 768]
            det_features = torch.stack(det_tokens, dim=0)  # [n_groups, bs, 768]
            cls_pred = [
                torch.matmul(
                    det_features[i].unsqueeze(dim=1),  # [bs, 1, 768]
                    epoch_text_features[i],  # [bs, 768, 2]
                ).squeeze(1)
                for i in range(det_features.shape[0])
            ]  # [bs, 2] * n_groups
            cls_pred = torch.stack(cls_pred, dim=0).mean(dim=0)  # [bs, 2]
            loss = F.cross_entropy(cls_pred, label)
            # [bs, 2, img_size, img_size]
            seg_pred = model.vision_text_fusion_gate_seg(seg_features, epoch_text_features)
            seg_loss = calculate_seg_loss(seg_pred, mask)
            loss += seg_loss
            seg_loss_list.append(seg_loss.item())
            # backward
            optimizer.zero_grad()
            loss.backward()
            # clip gradient
            for m_i_w in model.image_adapter["m_i_w"]:
                nn.utils.clip_grad_norm_(m_i_w.parameters(), 1.0)
            for m_t_w in model.text_adapter["m_t_w"]:
                nn.utils.clip_grad_norm_(m_t_w.parameters(), 1.0)
            # update parameters
            optimizer.step()
            loss_list.append(loss.item())
            tqdm_train_loader.set_postfix({
                "epoch": f"{epoch + 1} / {total_epoch}",
                "loss": f"{loss.item():.4f}",
                "det_loss": f"{loss.item() - seg_loss.item():.4f}",
                "seg_loss": f"{seg_loss.item():.4f}",
                "mean_seg_loss": f"{np.mean(seg_loss_list):.4f}",
                "mean_loss": f"{np.mean(loss_list):.4f}",
                "text_lr": optimizer.param_groups[0]["lr"],
                "image_lr": optimizer.param_groups[1]["lr"],
            })
        logger.info(f"mean_loss={np.mean(loss_list)}, mean_seg_loss={np.mean(seg_loss_list)}")
        scheduler.step()
        ckp_path = os.path.join(save_path, f"adapter_{epoch + 1}.pth")
        model_dict = {
            "epoch": epoch + 1,
            "text_adapter": model.text_adapter.state_dict(),
            "image_adapter": model.image_adapter.state_dict()
        }
        torch.save(model_dict, ckp_path)
    return model


def main():
    parser = argparse.ArgumentParser(description="End To End Training.")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="clip model to use (default: ViT-L-14-336)",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--dataset", type=str, default="VisA")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epoch", type=int, default=20, help="epochs for training")

    parser.add_argument("--cuda_device", type=int, default=0, help="cuda device id")

    parser.add_argument("--save_path", type=str, default="ckpt/test")

    # settings
    parser.add_argument("--n_groups", type=int, default=4, help="number of groups for adapter")
    parser.add_argument("--image_adapt_weight", type=float, default=0.2)
    parser.add_argument("--conv_lora_rank", type=int, default=8, help="rank for LoRA adapters")
    parser.add_argument("--conv_lora_alpha", type=float, default=2.0, help="alpha for LoRA adapters")
    parser.add_argument(
        "--conv_kernel_size_list", type=int, nargs="+", default=[3, 5],
        help="kernel size for convolutional LoRA adapters"
    )

    parser.add_argument("--text_adapt_weight", type=float, default=0.2)
    parser.add_argument("--lora_rank", type=int, default=16, help="rank for LoRA adapters")
    parser.add_argument("--lora_alpha", type=float, default=2.0, help="alpha for LoRA adapters")

    parser.add_argument("--image_lr", type=float, default=0.001, help="learning rate for image adapter")
    parser.add_argument("--text_lr", type=float, default=0.0005, help="learning rate for text adapter")
    parser.add_argument("--lr_gamma", type=float, default=0.9, help="learning rate decay factor")

    args = parser.parse_args()
    # ========================================================
    # check save_path and setting logger
    os.makedirs(args.save_path, exist_ok=True)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "train.log"),
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s %(filename)s %(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("args: %s", vars(args))
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    model = ACDCLIP(
        clip_model=clip_model,
        n_groups=args.n_groups,
        image_adapt_weight=args.image_adapt_weight,
        conv_lora_rank=args.conv_lora_rank,
        conv_lora_alpha=args.conv_lora_alpha,
        conv_kernel_size_list=args.conv_kernel_size_list,
        text_adapt_weight=args.text_adapt_weight,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    ).to(device)
    model.eval()
    # set optimizer
    optimizer = torch.optim.Adam([
        {
            "params": model.text_adapter.parameters(),
            "lr": args.text_lr,
        },
        {
            "params": model.image_adapter.parameters(),
            "lr": args.image_lr,
        },
    ])
    lr_scheduler = StepLR(
        optimizer,
        step_size=1,
        gamma=args.lr_gamma,
    )
    # load dataset
    logger.info("loading dataset ...")
    dataset = get_text_and_image_dataset(
        args.dataset,
        args.img_size,
        "train"
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )
    logger.info("training ...")
    model = train(
        model=model,
        dataset_name=args.dataset,
        train_loader=dataloader,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        device=device,
        total_epoch=args.epoch,
        save_path=args.save_path,
        logger=logger,
    )


if __name__ == "__main__":
    main()
