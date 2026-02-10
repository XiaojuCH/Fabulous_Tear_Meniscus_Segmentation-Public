import torch
import torch.nn as nn

class DWConv(nn.Module):
    """ 普通的深度可分离卷积 (无空洞) """
    def __init__(self, dim, kernel_size, padding):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.bn(self.dwconv(x)))

class GatedMultiScaleStripAdapter(nn.Module):
    """
    Final Version: 多尺度条状门控注意力
    结合 Kernel=7 (细节) 和 Kernel=23 (整体)，稳健性最强。
    """
    def __init__(self, in_channels, kernel_size_large=23, kernel_size_small=7):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        # === 大核分支 (23) ===
        pad_l = (kernel_size_large - 1) // 2
        self.strip_h_large = DWConv(in_channels, (kernel_size_large, 1), (pad_l, 0))
        self.strip_w_large = DWConv(in_channels, (1, kernel_size_large), (0, pad_l))

        # === 小核分支 (7) ===
        pad_s = (kernel_size_small - 1) // 2
        self.strip_h_small = DWConv(in_channels, (kernel_size_small, 1), (pad_s, 0))
        self.strip_w_small = DWConv(in_channels, (1, kernel_size_small), (0, pad_s))
        
        # 门控融合 (4路特征融合)
        self.gate_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1), # 4倍通道输入
            nn.Sigmoid()
        )
        
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        nn.init.constant_(self.gate_fusion[0].bias, 0)

    def forward(self, x):
        shortcut = x
        x_in = self.proj_in(x)
        
        # 大核特征
        lh = self.strip_h_large(x_in)
        lw = self.strip_w_large(x_in)
        
        # 小核特征
        sh = self.strip_h_small(x_in)
        sw = self.strip_w_small(x_in)
        
        # 拼接 4 路特征生成 Gate
        cat_feat = torch.cat([lh, lw, sh, sw], dim=1)
        gate = self.gate_fusion(cat_feat)
        
        # 加权输出
        out = x_in * gate
        return shortcut + self.proj_out(out)