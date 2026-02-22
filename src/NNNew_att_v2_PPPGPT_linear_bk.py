import torch
import torch.nn as nn
import torch.nn.functional as F

# 此版本是w/o Linear 版


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
# GSCSA Adapter
# =========================================
class GSCSA(nn.Module):

    # 此版本是w/o Linear 版

    """
    Geometry-aware Structural Competitive Adapter
    Designed for:
        - Curvilinear tear meniscus
        - Circular pupil
        - Multi-center cross-modality
    """

    def __init__(self, in_channels, kernel_size_large=23, kernel_size_small=7):
        super().__init__()

        self.proj_in = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        # ============================
        # 1️⃣ Curvature-aware pre-orientation
        # ============================
        self.pre_orient = DWConv(in_channels, 3, padding=1)

        # ============================
        # 2️⃣ Strip branches
        # ============================
        pad_l = (kernel_size_large - 1) // 2
        pad_s = (kernel_size_small - 1) // 2

        self.strip_h_large = DWConv(in_channels, (kernel_size_large, 1), (pad_l, 0))
        self.strip_w_large = DWConv(in_channels, (1, kernel_size_large), (0, pad_l))

        self.strip_h_small = DWConv(in_channels, (kernel_size_small, 1), (pad_s, 0))
        self.strip_w_small = DWConv(in_channels, (1, kernel_size_small), (0, pad_s))

        # ============================
        # 3️⃣ Local branches
        # ============================
        self.local_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        self.local_5x5 = DWConv(in_channels, 3, padding=2, dilation=2)

        # ============================
        # 4️⃣ Competitive Branch Weighting
        # ============================
        self.branch_weight = nn.Conv2d(
            in_channels * 2,
            in_channels * 2,
            kernel_size=1,
            bias=False
        )

        # ============================
        # 5️⃣ Domain-Adaptive Modulation
        # ============================
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.domain_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels * 2, 1)  # gamma + beta
        )

        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    # =========================================
    # Forward
    # =========================================
    def forward(self, x):

        # 此版本是w/o Linear 版

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

        branches = [loc3, loc5]

        # -------- Competitive Weighting --------
        cat_feat = torch.cat(branches, dim=1)
        weight = self.branch_weight(cat_feat)

        B, C2, H, W = weight.shape
        C = C2 // 2

        weight = weight.view(B, 2, C, H, W)
        weight = F.softmax(weight, dim=1)  # competition across branches

        stacked = torch.stack(branches, dim=1)  # [B,6,C,H,W]

        out = (weight * stacked).sum(dim=1)

        # -------- Domain-Adaptive Modulation --------
        global_vec = self.global_pool(x)
        gamma_beta = self.domain_fc(global_vec)

        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = torch.sigmoid(gamma)

        out = gamma * out + beta

        out = self.proj_out(out)

        return shortcut + out
