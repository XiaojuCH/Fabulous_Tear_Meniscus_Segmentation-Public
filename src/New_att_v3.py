import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    """
    深度可分离卷积 + 动态空洞卷积 (Dilated Depthwise Conv)
    """
    def __init__(self, dim, kernel_size, padding, dilation=1):
        super().__init__()
        # 深度可分离卷积：groups=dim
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, 
                                padding=padding, 
                                groups=dim, 
                                dilation=dilation, 
                                bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.dwconv(x)))

class GatedDilatedStripAdapter(nn.Module):
    """
    GD-STA: Gated Dilated Strip-Topology Attention
    版本 V3：引入空洞卷积以对抗 Placido Ring 干扰
    """
    def __init__(self, in_channels, kernel_size=23, dilation=1):
        super().__init__()
        
        self.proj_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        # -------------------------------------------------------------
        # 动态 Padding 计算公式: P = (D * (K - 1)) / 2
        # 确保卷积后尺寸不变
        # -------------------------------------------------------------
        p_len = (dilation * (kernel_size - 1)) // 2
        
        # H 方向分支 (捕捉垂直特征/瞳孔)
        # Kernel: (K, 1), Dilation: (D, 1), Padding: (P, 0)
        self.strip_h = DWConv(in_channels, kernel_size=(kernel_size, 1), 
                              padding=(p_len, 0), dilation=(dilation, 1))
        
        # W 方向分支 (捕捉水平特征/泪河)
        # Kernel: (1, K), Dilation: (1, D), Padding: (0, P)
        self.strip_w = DWConv(in_channels, kernel_size=(1, kernel_size), 
                              padding=(0, p_len), dilation=(1, dilation))
        
        # 门控融合机制
        self.gate_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        # 初始化
        nn.init.constant_(self.gate_fusion[0].bias, 0)

    def forward(self, x):
        shortcut = x
        x_feat = self.proj_in(x)
        
        x_h = self.strip_h(x_feat)
        x_w = self.strip_w(x_feat)
        
        # 门控：让网络自动判断哪些是“真结构”，哪些是“圆环噪声”
        gate = self.gate_fusion(torch.cat([x_h, x_w], dim=1))
        out = x_feat * gate 
        
        out = self.proj_out(out)
        return shortcut + out