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
        # 🚀 优化1：对齐 SAM 2 原生激活函数 SiLU
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))

class GAL_Adapter(nn.Module):
    def __init__(self, in_channels, kernel_size_large=23, kernel_size_small=7, reduction=4):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        self.pre_orient = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU() # 🚀 修复：直接放入 nn.SiLU() 即可
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
            nn.SiLU() # 🚀 修复：直接放入 nn.SiLU() 即可
        )
        self.local_5x5 = DWConv(in_channels, 3, padding=2, dilation=2)

        mid_channels = max(in_channels * 6 // reduction, 16)
        
        # 维持 V1 的 1x1 像素级通道竞争
        self.branch_weight = nn.Sequential(
            nn.Conv2d(in_channels * 6, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(), # 🚀 优化1
            nn.Conv2d(mid_channels, in_channels * 6, kernel_size=1, bias=False)
        )

        self.style_fc = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1), 
            nn.SiLU(), # 🚀 优化1
            nn.Conv2d(in_channels, in_channels * 2, 1)  
        )

        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        
        # 🚀 优化2：可学习的残差缩放因子，初始化为 0.01
        self.adapter_scale = nn.Parameter(torch.ones(1) * 0.01)

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

        # 空间门控融合
        cat_feat = torch.cat(branches, dim=1)
        weight = self.branch_weight(cat_feat)

        B, C6, H, W = weight.shape
        C = C6 // 6
        weight = weight.view(B, 6, C, H, W)
        weight = F.softmax(weight, dim=1) 

        stacked = torch.stack(branches, dim=1)
        out = (weight * stacked).sum(dim=1) 

        # CCSM 强度调制
        b, c, h, w = out.shape
        out_flat = out.view(b, c, -1)
        
        feat_mean = out_flat.mean(dim=2, keepdim=True).unsqueeze(-1)
        var = out_flat.var(dim=2, keepdim=True, unbiased=False)
        feat_std = torch.sqrt(var + 1e-5).unsqueeze(-1)
        
        style_feat = torch.cat([feat_mean, feat_std], dim=1)
        
        gamma_beta = self.style_fc(style_feat)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = 2.0 * torch.sigmoid(gamma) 
        
        out = gamma * out + beta
        out = self.proj_out(out)

        # 🚀 优化2应用：乘以可学习因子
        return shortcut + out * self.adapter_scale