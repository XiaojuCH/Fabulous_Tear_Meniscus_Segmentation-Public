import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------
# 现代 ConvNeXt 风格瓶颈层 (保留，用于高维特征提纯)
# -----------------------------------------------------------------
class ModernBottleneck(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + shortcut

class DWConv(nn.Module):
    def __init__(self, dim, kernel_size, padding, dilation=1):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))

# -----------------------------------------------------------------
# 终极版 GAL Adapter (V10 - Hyperspace Edition)
# -----------------------------------------------------------------
class GAL_Adapter(nn.Module):
    def __init__(self, in_channels, kernel_size_large=23, kernel_size_mid=15, kernel_size_small=7, reduction=16):
        super().__init__()
        
        # 🚀 预算核武器：高维空间通道放大 (Expansion)
        # 将输入维度强制放大 2 倍，赋予网络极其庞大的特征解耦容量
        embed_dim = in_channels * 2  

        # 1. 升维映射
        self.proj_in = nn.Conv2d(in_channels, embed_dim, 1, bias=False)

        # 2. 高维空间内的特征预处理与提纯
        self.pre_orient = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            ModernBottleneck(embed_dim) # 在宽裕的高维空间中剔除初始噪声
        )

        pad_l = (kernel_size_large - 1) // 2
        pad_m = (kernel_size_mid - 1) // 2
        pad_s = (kernel_size_small - 1) // 2
        
        # 所有多尺度卷积全部在升维后的 embed_dim (64或128) 中进行！
        self.strip_h_large = DWConv(embed_dim, (kernel_size_large, 1), (pad_l, 0))
        self.strip_w_large = DWConv(embed_dim, (1, kernel_size_large), (0, pad_l))
        self.strip_h_mid = DWConv(embed_dim, (kernel_size_mid, 1), (pad_m, 0))
        self.strip_w_mid = DWConv(embed_dim, (1, kernel_size_mid), (0, pad_m))
        self.strip_h_small = DWConv(embed_dim, (kernel_size_small, 1), (pad_s, 0))
        self.strip_w_small = DWConv(embed_dim, (1, kernel_size_small), (0, pad_s))

        self.local_3x3 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        self.local_5x5 = DWConv(embed_dim, 3, padding=2, dilation=2)

        # 8 分支像素级门控 (坚守最优信息瓶颈 reduction=16)
        num_branches = 8
        mid_channels = max(embed_dim * num_branches // reduction, 16)
        
        self.branch_weight = nn.Sequential(
            nn.Conv2d(embed_dim * num_branches, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, embed_dim * num_branches, kernel_size=1, bias=False)
        )

        # CCSM 强度调制 (同样在升维空间中进行)
        style_hidden = embed_dim * 4  
        self.style_fc = nn.Sequential(
            nn.Conv2d(embed_dim * 2, style_hidden, 1), 
            nn.GELU(),
            nn.Conv2d(style_hidden, embed_dim * 2, 1)  
        )

        # 3. 降维压缩：将完美的特征压缩回原始维度，准备喂给 SAM
        self.proj_out = nn.Conv2d(embed_dim, in_channels, 1, bias=False)

    def forward(self, x):
        shortcut = x
        
        # 🚀 升维进入 Hyperspace
        x_high = self.proj_in(x) # [B, embed_dim, H, W]

        # 预处理
        x_oriented = self.pre_orient(x_high)

        # 8路并发 (在高维空间中寻路)
        lh = self.strip_h_large(x_oriented)
        lw = self.strip_w_large(x_oriented)
        mh = self.strip_h_mid(x_oriented)
        mw = self.strip_w_mid(x_oriented)
        sh = self.strip_h_small(x_oriented)
        sw = self.strip_w_small(x_oriented)
        loc3 = self.local_3x3(x_high)
        loc5 = self.local_5x5(x_high)

        branches = [lh, lw, mh, mw, sh, sw, loc3, loc5]

        # 空间融合
        cat_feat = torch.cat(branches, dim=1)
        weight = self.branch_weight(cat_feat)

        B, C8, H, W = weight.shape
        C = C8 // 8
        weight = weight.view(B, 8, C, H, W)
        weight = F.softmax(weight, dim=1) 

        stacked = torch.stack(branches, dim=1)
        out = (weight * stacked).sum(dim=1) 

        # 全局强度调制
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
        
        # 🚀 降维逃脱 Hyperspace
        out = self.proj_out(out)

        # 结合原始特征
        return shortcut + out