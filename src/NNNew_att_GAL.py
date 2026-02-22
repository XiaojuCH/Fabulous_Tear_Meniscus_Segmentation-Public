import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================
# Depthwise Conv Block
# =========================================
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

# =========================================
# GAL-Adapter (Geometry-Aware Linear Adapter)
# =========================================
class GAL_Adapter(nn.Module):
    """
    Geometry-Aware Linear Adapter (GAL-Adapter)
    Designed for:
        - Curvilinear tear meniscus (Thin-structure)
        - Artifact decoupling (Placido rings)
        - Cross-center style modulation
    """
    def __init__(self, in_channels, kernel_size_large=23, kernel_size_small=7):
        super().__init__()

        self.proj_in = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        # ============================
        # 1. Curvature-aware pre-orientation
        # ============================
        self.pre_orient = DWConv(in_channels, 3, padding=1)

        # ============================
        # 2. Anisotropic Linear Branches (GAL 核心)
        # ============================
        pad_l = (kernel_size_large - 1) // 2
        pad_s = (kernel_size_small - 1) // 2

        self.strip_h_large = DWConv(in_channels, (kernel_size_large, 1), (pad_l, 0))
        self.strip_w_large = DWConv(in_channels, (1, kernel_size_large), (0, pad_l))

        self.strip_h_small = DWConv(in_channels, (kernel_size_small, 1), (pad_s, 0))
        self.strip_w_small = DWConv(in_channels, (1, kernel_size_small), (0, pad_s))

        # ============================
        # 3. Isotropic Local Branches (排雷分支)
        # ============================
        self.local_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        self.local_5x5 = DWConv(in_channels, 3, padding=2, dilation=2)

        # ============================
        # 4. Competitive Gating
        # ============================
        self.branch_weight = nn.Conv2d(
            in_channels * 6,
            in_channels * 6,
            kernel_size=1,
            bias=False
        )

        # ============================
        # 5. Cross-Center Style Modulator (CCSM) - 【极致优化版】
        # ============================
        # 替代了普通的全局池化，输入维度 * 2 因为要拼接 Mean 和 Std
        self.style_fc = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels * 2, 1)  # 输出 gamma + beta
        )

        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    def forward(self, x):
        shortcut = x
        x = self.proj_in(x)

        # Curvature-aware orientation transform
        x_oriented = self.pre_orient(x)

        # -------- Branch Features --------
        lh = self.strip_h_large(x_oriented)
        lw = self.strip_w_large(x_oriented)
        sh = self.strip_h_small(x_oriented)
        sw = self.strip_w_small(x_oriented)

        loc3 = self.local_3x3(x)
        loc5 = self.local_5x5(x)

        branches = [lh, lw, sh, sw, loc3, loc5]

        # -------- Competitive Weighting --------
        cat_feat = torch.cat(branches, dim=1)
        weight = self.branch_weight(cat_feat)

        B, C6, H, W = weight.shape
        C = C6 // 6

        weight = weight.view(B, 6, C, H, W)
        weight = F.softmax(weight, dim=1)  # competition across branches

        stacked = torch.stack(branches, dim=1)  # [B,6,C,H,W]
        out = (weight * stacked).sum(dim=1)

        # -------- Cross-Center Style Modulation (CCSM) --------
        # 提取一阶和二阶统计量 (Mean & Std) 代表多中心图像的“风格”
        b, c, h, w = out.shape
        out_flat = out.view(b, c, -1)
        
        # 计算全局均值 (亮度/基础特征) -> [B, C, 1, 1]
        feat_mean = out_flat.mean(dim=2, keepdim=True).unsqueeze(-1)
        # 计算全局标准差 (对比度/域差异特征)，加 eps 防止梯度爆炸 -> [B, C, 1, 1]
        feat_std = (out_flat.var(dim=2, keepdim=True) + 1e-5).sqrt().unsqueeze(-1)
        
        # 拼接风格特征 -> [B, 2C, 1, 1]
        style_feat = torch.cat([feat_mean, feat_std], dim=1)
        
        # 动态生成缩放系数和偏置
        gamma_beta = self.style_fc(style_feat)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = torch.sigmoid(gamma)

        # 风格重标定
        out = gamma * out + beta
        
        out = self.proj_out(out)

        return shortcut + out