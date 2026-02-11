import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformableDWConv(nn.Module):
    """优化后的可变形深度可分离卷积（向量化实现）"""
    def __init__(self, dim, kernel_size, padding):
        super().__init__()
        self.dim = dim
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.padding = padding

        # 偏移量卷积
        self.offset_conv = nn.Conv2d(dim, 2 * kernel_size[0] * kernel_size[1], 
                                     kernel_size=3, padding=1, groups=dim)
        # 深度卷积权重
        self.weight = nn.Parameter(torch.randn(1, dim, kernel_size[0], kernel_size[1]))
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

    def forward(self, x):
        B, C, H, W = x.shape
        kH, kW = self.kernel_size

        # 1. 生成偏移量 [B, 2*kH*kW, H, W]
        offset = self.offset_conv(x)
        # 重构为 [B, H, W, kH, kW, 2]
        offset = offset.view(B, 2, kH, kW, H, W).permute(0, 4, 5, 2, 3, 1)  # 关键变化

        # 2. 生成采样网格
        y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack([x_coords, y_coords], dim=-1).to(x.device).float()  # [H, W, 2]
        grid = grid.unsqueeze(2).unsqueeze(2)  # [H, W, 1, 1, 2]

        # 3. 卷积核相对坐标
        kh_idx = torch.arange(kH).to(x.device).float() - (kH - 1) / 2.0
        kw_idx = torch.arange(kW).to(x.device).float() - (kW - 1) / 2.0
        ky, kx = torch.meshgrid(kh_idx, kw_idx, indexing='ij')
        kernel_offset = torch.stack([kx, ky], dim=-1).view(1, 1, kH, kW, 2)  # [1, 1, kH, kW, 2]

        # 4. 计算可变形采样位置并归一化
        deformable_grid = grid + kernel_offset + offset
        deformable_grid[..., 0] = 2.0 * deformable_grid[..., 0] / (W - 1) - 1.0
        deformable_grid[..., 1] = 2.0 * deformable_grid[..., 1] / (H - 1) - 1.0

        # 5. 向量化可变形采样与卷积
        # 重塑输入以便批量采样: [B, C, H, W] -> [B*C, 1, H, W]
        x_reshaped = x.view(B * C, 1, H, W)
        # 扩展偏移网格以匹配通道: [B, H, W, kH, kW, 2] -> [B*C, H, W, kH, kW, 2]
        deformable_grid_expanded = deformable_grid.unsqueeze(1).repeat(1, C, 1, 1, 1, 1, 1)
        deformable_grid_expanded = deformable_grid_expanded.view(B * C, H, W, kH, kW, 2)

        # 为每个核位置采样
        sampled_features = F.grid_sample(
            x_reshaped.expand(-1, kH * kW, -1, -1),  # [B*C, kH*kW, H, W]
            deformable_grid_expanded.view(B * C, H * W, kH * kW, 2),  # 重塑网格
            mode='bilinear', padding_mode='zeros', align_corners=False
        )  # [B*C, kH*kW, H, W]

        # 应用深度卷积权重
        out = F.conv2d(
            sampled_features.view(B, C, kH * kW, H * W).permute(0, 1, 3, 2).reshape(B, C, H, W),
            self.weight,  # [1, C, kH, kW]
            padding=self.padding,
            groups=C  # 深度可分离卷积
        )  # [B, C, H, W]

        return self.act(self.bn(out))
    
class GatedMultiScaleStripAdapterV4(nn.Module):
    """
    V4 增强版：针对泪河分割的终极注意力适配器
    核心升级：
    1. 【可变形条状卷积】：替换固定条状卷积，使感受野能弯曲并贴合泪河，主动规避虹膜环状纹理干扰。
    2. 【瞳孔引导的空间注意力】：利用数据集提供的瞳孔位置先验，让网络聚焦于泪河最可能出现的区域。
    3. 【显式模态条件化】：强化全局环境感知，使网络能根据彩图/红外图动态调整特征处理策略。
    """
    def __init__(self, in_channels, kernel_size_large=23, kernel_size_small=7):
        super().__init__()
        self.in_channels = in_channels
        self.proj_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        # === 1. 可变形条状分支 (核心抗干扰升级) ===
        pad_l = (kernel_size_large - 1) // 2
        pad_s = (kernel_size_small - 1) // 2
        # 使用可变形卷积替代原DWConv
        self.strip_h_large = DeformableDWConv(in_channels, (kernel_size_large, 1), (pad_l, 0))
        self.strip_w_large = DeformableDWConv(in_channels, (1, kernel_size_large), (0, pad_l))
        self.strip_h_small = DeformableDWConv(in_channels, (kernel_size_small, 1), (pad_s, 0))
        self.strip_w_small = DeformableDWConv(in_channels, (1, kernel_size_small), (0, pad_s))
        
        # === 2. 双路局部感知 (保持不变，用于瞳孔和光晕) ===
        self.local_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        self.local_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        # === 3. 瞳孔引导的空间注意力 (新增解剖先验) ===
        # 生成一个空间权重图，高亮瞳孔外围区域（泪河区）
        self.pupil_guide = nn.Sequential(
            nn.Conv2d(1, in_channels // 4, kernel_size=3, padding=1),  # 输入是瞳孔热图
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出空间注意力权重 [B, 1, H, W]
        )

        # === 4. 显式模态条件化 (强化全局感知) ===
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # 模态编码器：输出一个模态向量，用于调制特征
        self.modal_encoder = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels)
        )
        # 条件缩放与偏移参数生成
        self.modal_scale = nn.Linear(in_channels, in_channels)
        self.modal_shift = nn.Linear(in_channels, in_channels)

        # === 5. 门控融合 (集成所有信息) ===
        # 输入: 4(Strips) + 2(Locals) = 6
        self.gate_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 6, in_channels, kernel_size=1), 
            nn.Sigmoid()
        )
        
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        nn.init.constant_(self.gate_fusion[0].bias, 0)

    def forward(self, x, pupil_heatmap=None):
        """
        参数:
            x: 输入特征图 [B, C, H, W]
            pupil_heatmap: 瞳孔热图 [B, 1, H, W]，可从瞳孔掩膜经高斯模糊得到。训练时从标注生成，推理时可从辅助分支预测或设为None。
        """
        shortcut = x
        x_in = self.proj_in(x)
        B, C, H, W = x_in.shape
        
        # --- 1. 提取多尺度特征 ---
        lh = self.strip_h_large(x_in)
        lw = self.strip_w_large(x_in)
        sh = self.strip_h_small(x_in)
        sw = self.strip_w_small(x_in)
        loc3 = self.local_3x3(x_in)
        loc5 = self.local_5x5(x_in)
        
        # --- 2. 计算瞳孔引导的空间注意力 ---
        if pupil_heatmap is not None:
            spatial_weight = self.pupil_guide(pupil_heatmap)  # [B, 1, H, W]
            # 将空间权重应用于所有特征，增强泪河可能区域
            feat_list = [lh, lw, sh, sw, loc3, loc5]
            weighted_feat_list = [f * (1 + spatial_weight) for f in feat_list]  # 加权增强
            cat_feat = torch.cat(weighted_feat_list, dim=1)
        else:
            # 若无瞳孔热图，回退到原始特征拼接
            cat_feat = torch.cat([lh, lw, sh, sw, loc3, loc5], dim=1)
        
        # --- 3. 计算门控并注入模态条件 ---
        gate = self.gate_fusion(cat_feat)  # [B, C, H, W]
        
        # 显式模态条件化
        global_vec = self.global_pool(x_in).squeeze(-1).squeeze(-1)  # [B, C]
        modal_code = self.modal_encoder(global_vec)  # [B, C]
        scale = self.modal_scale(modal_code).view(B, C, 1, 1)  # [B, C, 1, 1]
        shift = self.modal_shift(modal_code).view(B, C, 1, 1)
        gate = gate * torch.sigmoid(scale) + shift  # 调制门控信号
        
        # --- 4. 特征加权与残差连接 ---
        out = x_in * gate
        return shortcut + self.proj_out(out)