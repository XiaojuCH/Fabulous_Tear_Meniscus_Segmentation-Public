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

        # 1. 跨通道特征混合
        self.pre_orient = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        # 2. 各向异性分支 (捕获细长泪河)
        pad_l = (kernel_size_large - 1) // 2
        pad_s = (kernel_size_small - 1) // 2
        self.strip_h_large = DWConv(in_channels, (kernel_size_large, 1), (pad_l, 0))
        self.strip_w_large = DWConv(in_channels, (1, kernel_size_large), (0, pad_l))
        self.strip_h_small = DWConv(in_channels, (kernel_size_small, 1), (pad_s, 0))
        self.strip_w_small = DWConv(in_channels, (1, kernel_size_small), (0, pad_s))

        # 3. 各向同性分支 (捕获局部细节)
        self.local_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        self.local_5x5 = DWConv(in_channels, 3, padding=2, dilation=2)

        # 4. 降维计算瓶颈 (降低 FLOPs 并实现通道注意力前置)
        mid_channels = max(in_channels * 6 // reduction, 16)
        self.branch_weight = nn.Sequential(
            nn.Conv2d(in_channels * 6, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, in_channels * 6, kernel_size=1, bias=False)
        )

        # 5. 跨中心风格调制模块 (CCSM)
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
        # 优化 2：基于全局上下文的轻量化竞争性门控 (SKNet Style)
        # ==========================================
        stacked = torch.stack(branches, dim=1) # [B, 6, C, H, W]
        
        # 🔥 修复点：在这里提前提取所有维度信息
        B, num_branches, C, H, W = stacked.shape
        
        # 提取各个分支的全局统计量 (尺寸变化: [B, 6, C, H, W] -> [B, 6*C, 1, 1])
        global_info = stacked.mean(dim=(3, 4)).view(B, num_branches * C, 1, 1) 
        
        # 计算分支权重
        weight = self.branch_weight(global_info) # [B, 6*C, 1, 1]
        weight = weight.view(B, num_branches, C, 1, 1) # reshape
        weight = F.softmax(weight, dim=1)        # 软竞争
        
        # 动态融合特征
        out = (weight * stacked).sum(dim=1)      # [B, C, H, W]

        # ==========================================
        # 优化 1：严谨的跨中心风格调制 (CCSM - True Domain Alignment)
        # ==========================================
        out_flat = out.view(B, C, -1)
        
        # 提取均值和标准差 (尺寸为 [B, C, 1])
        feat_mean = out_flat.mean(dim=2, keepdim=True) 
        var = out_flat.var(dim=2, keepdim=True, unbiased=False)
        feat_std = torch.sqrt(var + 1e-5)              
        
        # 拼接并扩展为 [B, 2C, 1, 1] 供全连接层使用
        style_feat = torch.cat([feat_mean, feat_std], dim=1).unsqueeze(-1) 
        
        gamma_beta = self.style_fc(style_feat)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = 2.0 * torch.sigmoid(gamma) 
        
        # 🔥 先进行特征的去风格化 (Instance Normalization)，再进行重标定
        # 将 [B, C, 1] unsqueeze(-1) 变为 [B, C, 1, 1] 以匹配 out 的维度进行广播
        out_norm = (out - feat_mean.unsqueeze(-1)) / (feat_std.unsqueeze(-1) + 1e-5)
        out = gamma * out_norm + beta
        
        out = self.proj_out(out)

        return shortcut + out