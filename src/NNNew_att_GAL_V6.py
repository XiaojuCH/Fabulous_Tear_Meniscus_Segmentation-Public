import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    def __init__(self, dim, kernel_size, padding, dilation=1):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, 
                            dilation=dilation, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))

class GAL_Adapter(nn.Module):
    def __init__(self, in_channels, kernel_size_large=23, kernel_size_small=7, reduction=4):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        # 🚀 预处理增强
        self.pre_orient = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

        # --- 分支定义 ---
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

        # 🚀 优化 1：层级化拓扑门控权重 (Hierarchical Gating)
        # 更加敏锐地在三类特征中切换
        mid_channels = max(in_channels * 6 // reduction, 16)
        self.hierarchy_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 6, mid_channels, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid_channels, in_channels * 6, 1, bias=False),
            nn.Sigmoid()
        )

        # 🚀 优化 2：三位一体高光感知调制 (T-CCSM)
        # 处理 Mean, Std, Max，参数量会有微弱增加 (~0.1M)
        self.style_fc = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1), 
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels * 2, 1)  
        )

        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.adapter_scale = nn.Parameter(torch.ones(1) * 0.01)

    def forward(self, x):
        shortcut = x
        x = self.proj_in(x)
        x_oriented = self.pre_orient(x)

        # 提取六大分支特征
        lh = self.strip_h_large(x_oriented)
        lw = self.strip_w_large(x_oriented)
        sh = self.strip_h_small(x_oriented)
        sw = self.strip_w_small(x_oriented)
        loc3 = self.local_3x3(x)
        loc5 = self.local_5x5(x)
        branches = [lh, lw, sh, sw, loc3, loc5]
        
        # 🚀 层级化融合
        cat_feat = torch.cat(branches, dim=1)
        gate_weights = self.hierarchy_gate(cat_feat) # [B, 6C, 1, 1]
        
        # 将权重应用到分支
        out = 0
        for i in range(6):
            w = gate_weights[:, i*x.size(1) : (i+1)*x.size(1)]
            out += w * branches[i]

        # 🚀 三位一体强度感知 (解决反光导致的边缘磕磕巴巴)
        b, c, h, w = out.shape
        out_flat = out.view(b, c, -1)
        
        f_mean = out_flat.mean(dim=2, keepdim=True).unsqueeze(-1)
        f_max, _ = out_flat.max(dim=2, keepdim=True)
        f_max = f_max.unsqueeze(-1)
        f_var = out_flat.var(dim=2, keepdim=True, unbiased=False)
        f_std = torch.sqrt(f_var + 1e-5).unsqueeze(-1)
        
        style_stats = torch.cat([f_mean, f_std, f_max], dim=1) # [B, 3C, 1, 1]
        gamma_beta = self.style_fc(style_stats)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        
        # 使用 tanh 限制偏移强度，防止过激
        gamma = 1.0 + torch.tanh(gamma) 
        
        out = gamma * out + beta
        out = self.proj_out(out)

        return shortcut + out * self.adapter_scale