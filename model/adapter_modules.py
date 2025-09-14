import math

import torch
import torch.nn.functional as F
from torch import nn


class MLPAdapter(nn.Module):
    def __init__(self, c_in, c_out=768, hidden_size=512):
        super(MLPAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, c_out)
        )

    def forward(self, x):
        # x shape: [H * W, bs, c_in]
        x = self.fc(x)
        return x


class TextLoraAdapter(nn.Module):
    def __init__(self, c_in, c_out=768, r=16, alpha=2.0):
        super(TextLoraAdapter, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.r = r
        self.scale = alpha / r ** 0.5  # LoRA的缩放系数

        self.lora_A = nn.Parameter(torch.randn(c_in, r))
        self.lora_B = nn.Parameter(torch.randn(r, c_out))

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.lora_A)  # 使用Kaiming初始化A
        nn.init.normal_(self.lora_B, mean=0, std=0.02)  # 正态分布初始化B

    def forward(self, x):
        # x shape: [H * W, bs, c_in]
        lora_output = x @ self.lora_A @ self.lora_B * self.scale  # [H * W, bs, c_out]
        return lora_output


class ConvLoraBlock(nn.Module):
    def __init__(
            self,
            c_in,
            c_out=768,
            lora_rank=16,
            lora_alpha=2.0,
            conv_lora_rank=8,
            conv_lora_alpha=2.0,
            conv_kernel_size=3,
    ):
        super(ConvLoraBlock, self).__init__()
        # 缩放
        self.lora_scale = lora_alpha / lora_rank ** 0.5
        self.conv_lora_scale = conv_lora_alpha / conv_lora_rank

        # downsample
        self.lora_A = nn.Parameter(torch.randn(c_in, lora_rank))
        self.conv_lora_A = nn.Conv2d(lora_rank, conv_lora_rank, kernel_size=conv_kernel_size, stride=1,
                                     padding=conv_kernel_size // 2, bias=False)
        # upsample
        self.conv_lora_B = nn.Conv2d(conv_lora_rank, lora_rank, kernel_size=conv_kernel_size, stride=1,
                                     padding=conv_kernel_size // 2, bias=False)
        self.lora_B = nn.Parameter(torch.randn(lora_rank, c_out))

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.lora_A)
        nn.init.normal_(self.lora_B, mean=0, std=0.02)
        nn.init.kaiming_uniform_(self.conv_lora_A.weight)
        nn.init.kaiming_uniform_(self.conv_lora_B.weight)

    def forward(self, x):
        # x shape: [H * W, bs, c_in]
        patch_size, B = int(math.sqrt(x.shape[0])), x.shape[1]  # 假设输入是正方形的
        # Downsample
        down_lora_output = x @ self.lora_A  # [H * W, bs, lora_rank]
        down_lora_output = down_lora_output.permute(1, 2, 0).view(B, -1, patch_size,
                                                                  patch_size)  # [bs, lora_rank, H, W]
        up_lora_input = self.conv_lora_A(down_lora_output)  # [bs, conv_lora_rank, H, W]
        # Upsample
        up_lora_output = self.conv_lora_B(up_lora_input) * self.conv_lora_scale  # [bs, lora_rank, H, W]
        up_lora_output = up_lora_output.view(B, -1, patch_size * patch_size).permute(2, 0, 1)  # [H * W, bs, lora_rank]
        up_lora_output = up_lora_output @ self.lora_B * self.lora_scale  # [H * W, bs, c_out]
        return up_lora_output


class ConvLoraAdapter(nn.Module):
    def __init__(
            self,
            c_in,
            c_out=768,
            lora_rank=16,
            lora_alpha=2.0,
            conv_lora_rank=8,
            conv_lora_alpha=2.0,
            conv_kernel_size_list=(3, 5)
    ):
        super(ConvLoraAdapter, self).__init__()
        kernel_size_list = conv_kernel_size_list
        self.conv_lora_blocks = nn.ModuleList([
            ConvLoraBlock(
                c_in=c_in,
                c_out=c_out,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                conv_lora_rank=conv_lora_rank,
                conv_lora_alpha=conv_lora_alpha,
                conv_kernel_size=kernel_size
            ) for kernel_size in kernel_size_list
        ])
        self.fusion_conv = nn.Conv2d(len(kernel_size_list) * c_out, c_out, kernel_size=1, stride=1, padding=0,
                                     bias=False)

    def forward(self, x):
        # x [H * W, bs, c_in] [1369, 4, 1024]
        patch_size, B = int(math.sqrt(x.shape[0])), x.shape[1]
        outputs = [block(x).permute(1, 2, 0) for block in
                   self.conv_lora_blocks]  # 每个block输出 [H * W, bs, c_out] -> [bs, c_out, H * W]
        outputs = [out.view(B, -1, patch_size, patch_size) for out in outputs]  # [bs, c_out, H, W]
        outputs = torch.cat(outputs, dim=1)
        # 特征融合
        outputs = self.fusion_conv(outputs)  # [bs, c_out, H, W]
        outputs = outputs.view(B, -1, patch_size * patch_size).permute(2, 0, 1)  # [H * W, bs, c_out]
        return outputs


class ASPPImageFeatureAdapter(nn.Module):
    def __init__(self, c_in, c_hidden=256):
        super(ASPPImageFeatureAdapter, self).__init__()
        # 输入降维
        self.fc = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_hidden),
            nn.ReLU(inplace=True),
        )

        # 多尺度特征提取
        self.aspp1 = nn.Sequential(
            nn.Conv2d(c_hidden, c_hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_hidden),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(c_hidden, c_hidden, kernel_size=3, stride=1, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(c_hidden),
            nn.ReLU(inplace=True),
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(c_hidden, c_hidden, kernel_size=3, stride=1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(c_hidden),
            nn.ReLU(inplace=True),
        )

        # 全局特征提取分支
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(c_hidden, c_hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_hidden),
            nn.ReLU(inplace=True),
        )

        # 特征拼接后的通道整合
        self.concat_conv = nn.Sequential(
            nn.Conv2d(c_hidden * 4, c_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_in),
            nn.ReLU(inplace=True),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x [H * W, bs, c_in]
        HW, B, C = x.shape
        x = x.permute(1, 2, 0)  # shape: [bs, c_in, H * W]
        H = int(math.sqrt(HW))
        x = x.view(B, C, H, H)

        # 输入降维
        x = self.fc(x)  # shape: [bs, c_out, H, W]
        # 多尺度特征提取
        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        aspp3 = self.aspp3(x)

        # 全局特征提取
        global_feat = self.global_avg_pool(x)  # shape: [bs, c_out, 1, 1]
        global_feat = self.global_conv(global_feat)  # shape: [bs, c_out, 1, 1]
        # 上采样到原始输入大小
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=False)

        # 特征拼接
        concat = torch.cat([aspp1, aspp2, aspp3, global_feat], dim=1)  # shape: [bs, c_out * 4, H, W]

        # 通道整合
        out = self.concat_conv(concat)  # shape: [bs, c_out, H, W]

        out = out.view(B, C, -1)  # shape: [bs, c_out, H * W]
        out = out.permute(2, 0, 1)  # shape: [H * W, bs, c_out]
        return out


if __name__ == '__main__':
    conv_lora_adapter = ConvLoraAdapter(c_in=1024, c_out=1024, lora_rank=16, lora_alpha=2.0, conv_lora_rank=8,
                                        conv_lora_alpha=2.0)
    x = torch.randn(1369, 4, 1024)  # [H * W, bs, c_in]
    print(x[:, 0, :].min())
    output = conv_lora_adapter(x)
    print(output[:, 0, :].min())
    print(output.shape)  # 应该是 [1369, 4, 768]
