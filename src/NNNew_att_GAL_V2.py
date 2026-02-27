import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    def __init__(self, dim, kernel_size, padding, dilation=1):
        super().__init__()
        self.dw = nn.Conv2d(
            dim, dim, 
            kernel_size=kernel_size, 
            padding=padding, 
            dilation=dilation, 
            groups=dim, 
            bias=False
        )
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))

class GAL_Adapter(nn.Module):
    def __init__(self, in_channels, kernel_size_large=23, kernel_size_small=7, reduction=4):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        self.pre_orient = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        pad_l = (kernel_size_large - 1) // 2
        pad_s = (kernel_size_small - 1) // 2
        self.strip_h_large = DWConv(in_channels, (kernel_size_large, 1), (pad_l, 0))
        self.strip_w_large = DWConv(in_channels, (1, kernel_size_large), (0, pad_l))
        self.strip_h_small = DWConv(in_channels, (kernel_size_small, 1), (pad_s, 0))
        self.strip_w_small = DWConv(in_channels, (1, kernel_size_small), (0, pad_s))

        self.local_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        self.local_5x5 = DWConv(in_channels, 3, padding=2, dilation=2)

        # 🚀 V2：空间门控，输出 [B, 6, H, W] 而非 [B, 6C, H, W]
        # 用 7x7 大核卷积提取融合后的空间轮廓，让模型感知细长拓扑
        self.spatial_gating = nn.Sequential(
            nn.Conv2d(in_channels * 6, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, 6, kernel_size=7, padding=3, bias=False)
        )

        self.style_fc = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1), 
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels * 2, 1)  
        )

        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    def forward(self, x):
        shortcut = x
        x = self.proj_in(x)

        x_oriented = self.pre_orient(x)

        lh = self.strip_h_large(x_oriented)
        lw = self.strip_w_large(x_oriented)
        sh = self.strip_h_small(x_oriented)
        sw = self.strip_w_small(x_oriented)
        loc3 = self.local_3x3(x)
        loc5 = self.local_5x5(x)

        branches = [lh, lw, sh, sw, loc3, loc5]

        # ==========================================
        # V2：空间门控，[B, 6, H, W] 像素级分支竞争
        # 细长像素激活长条分支，拐角像素激活局部分支
        # ==========================================
        cat_feat = torch.cat(branches, dim=1)
        spatial_weight = self.spatial_gating(cat_feat)          # [B, 6, H, W]
        spatial_weight = F.softmax(spatial_weight, dim=1)       # 在6个分支维度上竞争

        stacked = torch.stack(branches, dim=1)                  # [B, 6, C, H, W]
        spatial_weight = spatial_weight.unsqueeze(2)            # [B, 6, 1, H, W]
        out = (spatial_weight * stacked).sum(dim=1)             # [B, C, H, W]

        # ==========================================
        # 恢复 V1，附加数值安全补丁：动态通道缩放
        # ==========================================
        b, c, h, w = out.shape
        out_flat = out.view(b, c, -1)
        
        # 提取特征
        feat_mean = out_flat.mean(dim=2, keepdim=True).unsqueeze(-1) # [B, C, 1, 1]
        var = out_flat.var(dim=2, keepdim=True, unbiased=False)
        # 🚨 极其关键的防崩补丁：+ 1e-5
        feat_std = torch.sqrt(var + 1e-5).unsqueeze(-1) # [B, C, 1, 1]
        
        style_feat = torch.cat([feat_mean, feat_std], dim=1)
        
        gamma_beta = self.style_fc(style_feat)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = 2.0 * torch.sigmoid(gamma) 
        
        # 不做 InstanceNorm！保留你的绝对强度信息！
        out = gamma * out + beta
        
        out = self.proj_out(out)

        return shortcut + out