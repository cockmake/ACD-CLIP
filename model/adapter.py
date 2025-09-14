import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from torch import nn

from .adapter_modules import TextLoraAdapter, MLPAdapter, ConvLoraAdapter


class AddWeight(nn.Module):
    def __init__(
            self,
            image_adapt_weight,
            is_text=False,
            change_x=True,
    ):
        super().__init__()
        d_model = 768 if is_text else 1024
        self.i_w = nn.Parameter(torch.ones(1, 1, d_model) * image_adapt_weight)
        self.change_x = change_x

    def forward(self, x, adapt_out):
        if not self.change_x:
            return x + self.i_w * adapt_out
        return (1 - self.i_w) * x + self.i_w * adapt_out


class ACDCLIP(nn.Module):
    def __init__(
            self,
            clip_model,
            text_adapt_weight: float = 0.15,
            image_adapt_weight: float = 0.15,
            n_groups: int = 3,
            lora_rank: int = 16,
            lora_alpha: float = 2,
            conv_lora_rank: int = 8,
            conv_lora_alpha: float = 2,
            conv_kernel_size_list=(3, 5),
            **kwargs,
    ):
        super().__init__()
        assert n_groups in [2, 3, 4, 6], "n_groups must be one of [2, 3, 4, 6]"
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        text_step = 12 // n_groups
        image_step = 24 // n_groups
        self.image_levels = [t for t in range(image_step, 24 + 1, image_step)]
        self.text_levels = [t for t in range(text_step, 12 + 1, text_step)]
        self.n_groups = n_groups
        self.image_adapt_weight = image_adapt_weight
        self.text_adapt_weight = text_adapt_weight

        image_adapt_weights = nn.ModuleList(
            [AddWeight(image_adapt_weight) for _ in range(n_groups)]
        )
        image_lora_adapters = nn.ModuleList(
            [
                ConvLoraAdapter(1024, 1024, lora_rank, lora_alpha, conv_lora_rank, conv_lora_alpha,
                                conv_kernel_size_list)
                for _ in
                range(n_groups)
            ]
        )
        seg_proj = nn.ModuleList(
            [MLPAdapter(1024, 768, 256) for _ in range(n_groups)]
        )
        det_proj = nn.ModuleList(
            MLPAdapter(1024, 768, 256) for _ in range(n_groups)
        )
        seg_image_layer_norms = nn.ModuleList(
            [nn.LayerNorm(768) for _ in range(n_groups)]
        )
        det_image_layer_norms = nn.ModuleList(
            [nn.LayerNorm(768) for _ in range(n_groups)]
        )
        # 动态路由门：预测文本-图像的交互权重
        vision_text_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, 256),
                nn.GELU(),
                nn.Linear(256, n_groups * 2)  # 输出n_stages权重（图像各阶段）
            ) for _ in range(n_groups)
        ])
        self.image_adapter = nn.ModuleDict(
            {
                "m_i_w": image_adapt_weights,
                "lora_adapters": image_lora_adapters,
                "seg_layer_norms": seg_image_layer_norms,
                "seg_proj": seg_proj,
                "det_layer_norms": det_image_layer_norms,
                "det_proj": det_proj,
                "vision_text_gate": vision_text_gate,
            }
        )
        text_adapt_weights = nn.ModuleList(
            [AddWeight(text_adapt_weight, is_text=True) for _ in range(n_groups)]
        )
        text_lora_adapters = nn.ModuleList(
            [TextLoraAdapter(768, 768, r=lora_rank, alpha=lora_alpha) for _ in range(n_groups)]
        )
        text_layer_norms = nn.ModuleList(
            [nn.LayerNorm(768) for _ in range(n_groups)]
        )
        self.text_adapter = nn.ModuleDict(
            {
                "m_t_w": text_adapt_weights,
                "layer_norms": text_layer_norms,
                "lora_adapters": text_lora_adapters,
            }
        )

    def forward_original(self, x, modality="visual"):
        if modality == "visual":
            cls_features, patch_features = self.clipmodel.encode_image(x, [24])
            patch_features = [
                self.clipmodel.visual._global_pool(t)[1] for t in patch_features
            ]
            patch_features = [self.clipmodel.visual.ln_post(t) for t in patch_features]
            patch_features = [t @ self.clipmodel.visual.proj for t in patch_features]
            return patch_features, cls_features
        else:
            raise ValueError("modality must be visual")

    def forward(self, x):
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [
                self.image_encoder.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)

        group_outs = []
        for i in range(24):
            x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            # [1370, bs, 1024]
            index = -1
            for j in range(self.n_groups):
                if i + 1 == self.image_levels[j]:
                    index = j
                    break
            if index != -1:
                t = x[1:, :, :]
                adapt_out = self.image_adapter["lora_adapters"][index](t)
                adapt_out = (
                        adapt_out
                        * t.norm(dim=-1, keepdim=True)
                        / adapt_out.norm(dim=-1, keepdim=True)
                )
                # [1369, bs, 1024]
                group_out = self.image_adapter["m_i_w"][index](t, adapt_out)
                # group_out = t * (1 - self.image_adapt_weight) + adapt_out * self.image_adapt_weight
                group_outs.append(group_out)
                x = torch.cat(
                    [
                        x[0, :, :].unsqueeze(0),
                        group_out
                    ],
                    dim=0,
                )

        # [batch_size, seq_len, dim]
        group_tokens = [t.permute(1, 0, 2) for t in group_outs]  # [bs, 1369, 1024]

        # 1: Segmentation Tokens
        seg_tokens_proj = [
            self.image_adapter["seg_proj"][i](t) for i, t in enumerate(group_tokens)
        ]  # 投影到分割空间
        seg_tokens_norm = [
            self.image_adapter["seg_layer_norms"][i](t) for i, t in enumerate(seg_tokens_proj)
        ]  # 层归一化
        seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens_norm]  # L2归一化

        # 2: Detection Tokens
        det_tokens_proj = [
            self.image_adapter["det_proj"][i](t) for i, t in enumerate(group_tokens)
        ]
        det_tokens_norm = [
            self.image_adapter["det_layer_norms"][i](t) for i, t in enumerate(det_tokens_proj)
        ]
        det_tokens = [F.normalize(t, dim=-1).mean(1) for t in det_tokens_norm]  # L2归一化 + 全局平均池化

        return seg_tokens, det_tokens

    def vision_text_fusion_gate_seg(
            self,
            vision_tokens: torch.Tensor,
            text_features: torch.Tensor,
            img_size: int = 518,
            test_mode: bool = False,
            domain: str = "Industrial",
    ):
        """
        Fuse vision and text features using a gating mechanism.
        :param vision_tokens: vision tokens from the image encoder. [n_groups, bs, patch_num, 768]
        :param text_features: text features from the text encoder. [n_groups, bs, 768, 2]
        :return: fused seg features.
        """
        B, patch_size, _ = vision_tokens.shape[1:]
        H = int(np.sqrt(patch_size))
        group_seg_preds = []
        for i in range(self.n_groups):
            img_feat = vision_tokens[i]  # [bs, patch_num, 768]
            gate_weights = self.image_adapter["vision_text_gate"][i](img_feat.mean(dim=1, keepdim=True)).squeeze(
                1)  # [bs, 2 * n_groups]
            gate_weights = gate_weights.view(B, self.n_groups, 2)  # [bs, n_groups, 2]
            gate_weights = F.softmax(gate_weights, dim=1)
            img_feat = 10 * img_feat
            # [n_groups, bs, 768, 2] -> [bs, n_groups, 768, 2]
            group_text_features = text_features.permute(1, 0, 2, 3)
            group_text_features = group_text_features * gate_weights.unsqueeze(2)  # [bs, n_groups, 768, 2]
            group_text_features = group_text_features.sum(dim=1)  # [bs, 768, 2]
            fused_feature = torch.matmul(img_feat, group_text_features)  # [bs, patch_num, 2]
            seg_logits = fused_feature.permute(0, 2, 1).view(B, 2, H, H)  # [bs, 2, H, H]
            group_seg_preds.append(seg_logits)  # [bs, 2, H, H]
        if test_mode:
            sigma = 1 if domain == "Industrial" else 1.5
            kernel_size = 7 if domain == "Industrial" else 9
            group_seg_preds = [
                gaussian_blur2d(
                    seg_pred,
                    (kernel_size, kernel_size),
                    (sigma, sigma)
                ) for seg_pred in group_seg_preds
            ]
        group_seg_preds = [
            F.interpolate(
                seg_pred,
                size=img_size,
                mode="bilinear",
                align_corners=True
            ).unsqueeze(0) for seg_pred in group_seg_preds
            # [1, bs, 2, img_size, img_size]
        ]  # [1, bs, 2, img_size, img_size] * n_groups
        all_group_preds = torch.cat(group_seg_preds, dim=0)  # [n_groups, bs, 2, img_size, img_size]
        final_seg_pred = torch.mean(all_group_preds, dim=0)  # [bs, 2, img_size, img_size]
        final_seg_pred = F.softmax(final_seg_pred, dim=1)  # [bs, 2, img_size, img_size]
        if test_mode:
            # [bs, img_size, img_size]
            final_seg_pred = final_seg_pred[:, 1, :, :]
            # final_seg_pred = (final_seg_pred[:, 1, :, :] + 1 - final_seg_pred[:, 0, :, :]) / 2
        return final_seg_pred

    def encode_text(self, text, adapt_text=True):
        if not adapt_text:
            return self.clipmodel.encode_text(text)
        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        x = self.clipmodel.token_embedding(text).to(
            cast_dtype
        )  # [batch_size, n_ctx, d_model]

        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        out_features = []
        for i in range(12):
            x, attn = self.clipmodel.transformer.resblocks[i](
                x, attn_mask=self.clipmodel.attn_mask
            )
            index = -1
            for j in range(self.n_groups):
                if i + 1 == self.text_levels[j]:
                    index = j
                    break
            if index != -1:
                adapt_out = self.text_adapter["lora_adapters"][index](x)
                adapt_out = (
                        adapt_out
                        * x.norm(dim=-1, keepdim=True)
                        / adapt_out.norm(dim=-1, keepdim=True)
                )
                x = self.text_adapter["m_t_w"][index](x, adapt_out)
                # x = x * (1 - self.text_adapt_weight) + adapt_out * self.text_adapt_weight
                out_features.append(x)

        indices = text.argmax(dim=-1)
        out_features = [t.permute(1, 0, 2) for t in out_features]
        out_features = [self.text_adapter["layer_norms"][i](t) for i, t in enumerate(out_features)]
        out_features = [t[torch.arange(t.shape[0]), indices] for t in out_features]
        return out_features
