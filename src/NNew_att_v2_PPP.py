import torch
import torch.nn as nn

class DWConv(nn.Module):
    """ 普通的深度可分离卷积 """
    def __init__(self, dim, kernel_size, padding, dilation=1):
        super().__init__()
        # 支持空洞卷积
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=padding, dilation=dilation, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.bn(self.dwconv(x)))

class GatedMultiScaleStripAdapter(nn.Module):
    """
    V3 Version: 多尺度条状 + 双路局部 + 全局环境感知
    针对：泪河 (长条), 瞳孔点 (微小), 模糊光晕 (局部), 红外/彩色差异 (全局)
    """
    def __init__(self, in_channels, kernel_size_large=23, kernel_size_small=7):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        # === 1. 条状分支 (保持不变，负责泪河) ===
        pad_l = (kernel_size_large - 1) // 2
        self.strip_h_large = DWConv(in_channels, (kernel_size_large, 1), (pad_l, 0))
        self.strip_w_large = DWConv(in_channels, (1, kernel_size_large), (0, pad_l))

        pad_s = (kernel_size_small - 1) // 2
        self.strip_h_small = DWConv(in_channels, (kernel_size_small, 1), (pad_s, 0))
        self.strip_w_small = DWConv(in_channels, (1, kernel_size_small), (0, pad_s))
        
        # === 2. 【升级】双路局部感知 (Dual Local Branch) ===
        # Branch A: 3x3 标准卷积 (捕捉极小中心点)
        self.local_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        # Branch B: 3x3 空洞卷积 (Dilation=2, 感受野=5x5) (捕捉光晕/模糊边界)
        # 这里用 DWConv 节省参数，因为主要是看范围
        self.local_5x5 = DWConv(in_channels, kernel_size=3, padding=2, dilation=2)

        # === 3. 【新增】全局环境感知 (Global Context) ===
        # 帮助模型判断是红外还是彩图，动态调整权重
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        # === 4. 门控融合 ===
        # 输入变为 6 路: 4(Strips) + 2(Locals) = 6
        # Gate 的计算也加入了 Global 信息
        self.gate_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 6, in_channels, kernel_size=1), 
            nn.Sigmoid()
        )
        
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        # 初始化
        nn.init.constant_(self.gate_fusion[0].bias, 0)

    def forward(self, x):
        shortcut = x
        x_in = self.proj_in(x)
        
        # 1. 条状特征
        lh = self.strip_h_large(x_in)
        lw = self.strip_w_large(x_in)
        sh = self.strip_h_small(x_in)
        sw = self.strip_w_small(x_in)
        
        # 2. 局部特征 (双尺度)
        loc3 = self.local_3x3(x_in) # 锐利点
        loc5 = self.local_5x5(x_in) # 模糊点/粗细节
        
        # 3. 计算 Gate
        # [B, C*6, H, W]
        cat_feat = torch.cat([lh, lw, sh, sw, loc3, loc5], dim=1)
        gate = self.gate_fusion(cat_feat)
        
        # 4. 【新增】注入全局信息 (SE-Like)
        # 让 gate 根据全局亮度/对比度进行缩放
        global_vec = self.global_fc(self.global_pool(x_in)) # [B, C, 1, 1]
        gate = gate * (1 + torch.sigmoid(global_vec)) # 动态调节 Gate 的强度
        
        out = x_in * gate
        return shortcut + self.proj_out(out)