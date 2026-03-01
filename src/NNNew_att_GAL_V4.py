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
        # 保持与 SAM2 原生对齐
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
            nn.SiLU()
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
            nn.SiLU()
        )
        self.local_5x5 = DWConv(in_channels, 3, padding=2, dilation=2)
        
        # 🚀 V4 核心：新增无参低通平滑分支 (抹平磕磕巴巴)
        self.smooth_pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

        # 分支数量从 6 变成了 7
        mid_channels = max(in_channels * 7 // reduction, 16)
        self.branch_weight = nn.Sequential(
            nn.Conv2d(in_channels * 7, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels, in_channels * 7, kernel_size=1, bias=False)
        )

        # 🚀 V4 核心：Grid-Aware CCSM 的通道调制器
        self.style_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1), 
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels * 2, 1)  
        )

        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        
        # 稳定初始化的缩放因子
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
        
        # 提取平滑特征
        smooth_feat = self.smooth_pool(x)

        # 7 个分支参与竞争
        branches = [lh, lw, sh, sw, loc3, loc5, smooth_feat]

        # 空间门控融合
        cat_feat = torch.cat(branches, dim=1)
        weight = self.branch_weight(cat_feat)

        B, C7, H, W = weight.shape
        C = C7 // 7
        weight = weight.view(B, 7, C, H, W)
        weight = F.softmax(weight, dim=1) 

        stacked = torch.stack(branches, dim=1)
        out = (weight * stacked).sum(dim=1) 

        # ==========================================
        # 🚀 V4：Grid-Aware S-CCSM (网格感知局部重标定)
        # ==========================================
        b, c, h, w = out.shape
        
        # 提取 8x8 局部网格的均值
        grid_mean = F.adaptive_avg_pool2d(out, (8, 8)) # [B, C, 8, 8]
        
        # 提取 8x8 局部网格的方差 E[X^2] - E[X]^2
        out_sq_mean = F.adaptive_avg_pool2d(out**2, (8, 8))
        grid_var = out_sq_mean - grid_mean**2
        # 使用 clamp 防止浮点精度导致的负数
        grid_std = torch.sqrt(torch.clamp(grid_var, min=0.0) + 1e-5) # [B, C, 8, 8]
        
        # 在 8x8 的空间上生成调制参数
        style_feat = torch.cat([grid_mean, grid_std], dim=1) # [B, 2C, 8, 8]
        gamma_beta = self.style_conv(style_feat)             # [B, 2C, 8, 8]
        
        # 用双线性插值极其平滑地放大回全图尺寸
        gamma_beta = F.interpolate(gamma_beta, size=(h, w), mode='bilinear', align_corners=False) # [B, 2C, H, W]
        
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = 2.0 * torch.sigmoid(gamma) 
        
        # 局部感知重标定！
        out = gamma * out + beta
        
        out = self.proj_out(out)

        return shortcut + out * self.adapter_scale