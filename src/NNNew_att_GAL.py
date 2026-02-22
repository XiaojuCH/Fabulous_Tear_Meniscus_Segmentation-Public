import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================
# 基础组件：深度可分离卷积块
# =========================================
class DWConv(nn.Module):
    def __init__(self, dim, kernel_size, padding, dilation=1):
        super().__init__()
        # 仅在空间维度进行卷积，极大地节省参数量
        self.dw = nn.Conv2d(
            dim, dim, 
            kernel_size=kernel_size, 
            padding=padding, 
            dilation=dilation, 
            groups=dim, 
            bias=False
        )
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU() # 统一使用更现代的 GELU 激活函数

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))

# =========================================
# 核心创新：几何感知线性适配器 (GAL-Adapter)
# =========================================
class GAL_Adapter(nn.Module):
    def __init__(self, in_channels, kernel_size_large=23, kernel_size_small=7, reduction=4):
        """
        in_channels: 输入特征图的通道数
        kernel_size_large/small: 长条卷积的核大小，用于捕获泪河细长拓扑
        reduction: 瓶颈层降维系数，用于控制计算量
        """
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        # -----------------------------------------
        # 改进 1：打通特征隔离
        # 使用标准的 3x3 卷积代替 DWConv，确保特征在进入多分支前，
        # 各个通道的信息能够充分混合 (Channel Mixing)，增强几何先验的表达。
        # -----------------------------------------
        self.pre_orient = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        # --- 各向异性分支 (捕捉细长泪河) ---
        pad_l = (kernel_size_large - 1) // 2
        pad_s = (kernel_size_small - 1) // 2
        self.strip_h_large = DWConv(in_channels, (kernel_size_large, 1), (pad_l, 0))
        self.strip_w_large = DWConv(in_channels, (1, kernel_size_large), (0, pad_l))
        self.strip_h_small = DWConv(in_channels, (kernel_size_small, 1), (pad_s, 0))
        self.strip_w_small = DWConv(in_channels, (1, kernel_size_small), (0, pad_s))

        # --- 各向同性分支 (捕捉局部细节，排雷反光环伪影) ---
        self.local_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        self.local_5x5 = DWConv(in_channels, 3, padding=2, dilation=2)

        # -----------------------------------------
        # 改进 2：引入 Bottleneck 解决浅层算力黑洞
        # 6个分支拼接后维度很高，如果在 s0/s1 层直接做密集卷积会引入巨大计算量。
        # 这里先降维 (mid_channels) 再升维，既降低了 FLOPs，又增加了非线性，防止过拟合。
        # -----------------------------------------
        mid_channels = max(in_channels * 6 // reduction, 16)
        self.branch_weight = nn.Sequential(
            nn.Conv2d(in_channels * 6, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, in_channels * 6, kernel_size=1, bias=False)
        )

        # --- 跨中心风格调制模块 (CCSM) ---
        self.style_fc = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1), # *2 是因为要拼接 Mean 和 Std
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels * 2, 1)  # 输出 gamma + beta
        )

        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    def forward(self, x):
        shortcut = x
        x = self.proj_in(x)

        # 1. 跨通道几何预处理
        x_oriented = self.pre_orient(x)

        # 2. 多分支特征提取
        lh = self.strip_h_large(x_oriented)
        lw = self.strip_w_large(x_oriented)
        sh = self.strip_h_small(x_oriented)
        sw = self.strip_w_small(x_oriented)
        loc3 = self.local_3x3(x)
        loc5 = self.local_5x5(x)

        branches = [lh, lw, sh, sw, loc3, loc5]

        # 3. 竞争性门控机制 (Competitive Gating)
        cat_feat = torch.cat(branches, dim=1)
        weight = self.branch_weight(cat_feat)

        B, C6, H, W = weight.shape
        C = C6 // 6
        weight = weight.view(B, 6, C, H, W)
        weight = F.softmax(weight, dim=1) # 分支间的软竞争

        stacked = torch.stack(branches, dim=1)
        out = (weight * stacked).sum(dim=1) # 动态融合特征

        # 4. 跨中心风格调制 (Cross-Center Style Modulation)
        b, c, h, w = out.shape
        out_flat = out.view(b, c, -1)
        
        # 提取全局均值 (亮度/基础上下文)
        feat_mean = out_flat.mean(dim=2, keepdim=True).unsqueeze(-1)
        
        # -----------------------------------------
        # 🚨 致命 Bug 修复：强制设置 unbiased=False
        # 必须加上，否则当验证集遇到极小 YOLO 框导致 H=1,W=1 时，会因除以 0 崩溃！
        # -----------------------------------------
        feat_std = (out_flat.var(dim=2, keepdim=True, unbiased=False) + 1e-5).sqrt().unsqueeze(-1)
        
        # 拼接风格特征
        style_feat = torch.cat([feat_mean, feat_std], dim=1)
        
        gamma_beta = self.style_fc(style_feat)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        
        # -----------------------------------------
        # 改进 3：解开方差压缩封印
        # 使用 2.0 * Sigmoid，让缩放系数 gamma 中心对齐到 1.0。
        # 既能放大也能缩小特征，避免浅层特征梯度消失。
        # -----------------------------------------
        gamma = 2.0 * torch.sigmoid(gamma) 
        
        # 仿射变换进行域对齐
        out = gamma * out + beta
        out = self.proj_out(out)

        # 引入残差连接
        return shortcut + out