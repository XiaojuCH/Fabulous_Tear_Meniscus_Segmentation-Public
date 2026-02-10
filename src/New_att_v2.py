import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    """
    深度可分离卷积 + 空洞卷积 (Dilation)
    Dilation 的作用：增大感受野，同时跳过密集的圆环噪声，专注于长距离的连通性。
    """
    def __init__(self, dim, kernel_size, padding, dilation=1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, 
                                padding=padding, 
                                groups=dim, 
                                dilation=dilation, # <--- 新增 Dilation
                                bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.dwconv(x)))

class GatedStripAdapter(nn.Module):
    """
    G-STA V3: 针对 Placido Ring 干扰优化的版本
    """
    def __init__(self, in_channels, kernel_size=23):
        super().__init__()
        
        self.proj_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        # ------------------------------------------------------------------
        # 核心改动：使用空洞卷积 (Dilation)
        # 泪河很长，用 dilation=2 或 3 可以让卷积核“看得更远”，
        # 且不会被局部的同心圆纹理带偏。
        # ------------------------------------------------------------------
        dilation = 2 
        # 计算 padding: P = (D * (K - 1)) / 2
        padding_h = (dilation * (kernel_size - 1)) // 2
        padding_w = (dilation * (kernel_size - 1)) // 2
        
        # H 方向 (捕捉垂直特征/瞳孔)
        self.strip_h = DWConv(in_channels, kernel_size=(kernel_size, 1), 
                              padding=(padding_h, 0), dilation=(dilation, 1))
        
        # W 方向 (捕捉水平特征/泪河) - 这是最重要的分支！
        self.strip_w = DWConv(in_channels, kernel_size=(1, kernel_size), 
                              padding=(0, padding_w), dilation=(1, dilation))
        
        # 门控融合
        self.gate_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        nn.init.constant_(self.gate_fusion[0].bias, 0)

    def forward(self, x):
        shortcut = x
        x_feat = self.proj_in(x)
        
        x_h = self.strip_h(x_feat)
        x_w = self.strip_w(x_feat)
        
        gate = self.gate_fusion(torch.cat([x_h, x_w], dim=1))
        out = x_feat * gate # 门控抑制圆环噪声
        
        out = self.proj_out(out)
        return shortcut + out