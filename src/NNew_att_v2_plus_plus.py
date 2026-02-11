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
    Update Version: 混合几何注意力 (Hybrid Geometric Attention)
    结合了：
    1. Strip Conv (大/小): 专门针对长条状的泪河。
    2. Standard Conv (3x3): 专门针对圆形的瞳孔反光点。
    """
    def __init__(self, in_channels, kernel_size_large=23, kernel_size_small=7):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        # === 1. 大核条状分支 (针对长泪河) ===
        pad_l = (kernel_size_large - 1) // 2
        self.strip_h_large = DWConv(in_channels, (kernel_size_large, 1), (pad_l, 0))
        self.strip_w_large = DWConv(in_channels, (1, kernel_size_large), (0, pad_l))

        # === 2. 小核条状分支 (针对短泪河/断裂处) ===
        pad_s = (kernel_size_small - 1) // 2
        self.strip_h_small = DWConv(in_channels, (kernel_size_small, 1), (pad_s, 0))
        self.strip_w_small = DWConv(in_channels, (1, kernel_size_small), (0, pad_s))
        
        # === 3. 【新增】局部方形分支 (针对反光点/白点) ===
        # 使用标准的 3x3 卷积，它对点状特征最敏感
        # 这里不使用 DWConv (groups=dim)，而是用 groups=1 (或较小的groups) 来增强通道间的信息交换
        # 因为点状特征往往像素很少，需要更强的通道特征来确认它“是不是关键点”
        self.local_context = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        # === 4. 门控融合 (5路特征融合) ===
        # 输入通道变了：4 (strips) + 1 (local) = 5
        self.gate_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 5, in_channels, kernel_size=1), 
            nn.Sigmoid()
        )
        
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        # 初始化
        nn.init.constant_(self.gate_fusion[0].bias, 0)

    def forward(self, x):
        shortcut = x
        x_in = self.proj_in(x)
        
        # 1. 计算条状特征
        lh = self.strip_h_large(x_in) # Large Height
        lw = self.strip_w_large(x_in) # Large Width
        sh = self.strip_h_small(x_in) # Small Height
        sw = self.strip_w_small(x_in) # Small Width
        
        # 2. 【新增】计算点状特征
        local = self.local_context(x_in)
        
        # 3. 拼接 5 路特征生成 Gate
        # [B, C*5, H, W]
        cat_feat = torch.cat([lh, lw, sh, sw, local], dim=1)
        
        # 4. 计算权重并加权
        gate = self.gate_fusion(cat_feat)
        out = x_in * gate
        
        return shortcut + self.proj_out(out)