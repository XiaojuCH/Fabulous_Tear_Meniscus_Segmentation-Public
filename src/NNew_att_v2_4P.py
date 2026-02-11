import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    """ 普通的深度可分离卷积 """
    def __init__(self, dim, kernel_size, padding, dilation=1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=padding, dilation=dilation, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.bn(self.dwconv(x)))

class EdgeOperator(nn.Module):
    """
    【新增】固定权重的 Sobel 边缘提取算子
    不增加参数量，专门告诉 Attention 哪里是边缘
    """
    def __init__(self):
        super().__init__()
        # 定义 Sobel 核 (X方向和Y方向)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # 注册为 buffer，不参与反向传播更新
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x):
        # x: [B, C, H, W]
        # 对每一个通道分别做边缘检测 (Group Conv logic)
        b, c, h, w = x.shape
        # 保持通道独立
        sx = F.conv2d(x, self.sobel_x.repeat(c, 1, 1, 1), padding=1, groups=c)
        sy = F.conv2d(x, self.sobel_y.repeat(c, 1, 1, 1), padding=1, groups=c)
        
        # 计算梯度幅值: sqrt(sx^2 + sy^2)
        edge = torch.sqrt(sx**2 + sy**2 + 1e-6)
        return edge

class GatedMultiScaleStripAdapter(nn.Module):
    """
    V4 Final: 边缘感知的混合几何注意力
    集大成者：Strip(泪河) + Local(反光点) + Context(多模态) + Edge(弱边缘)
    """
    def __init__(self, in_channels, kernel_size_large=23, kernel_size_small=7):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        # === 1. 条状分支 (Strip) ===
        pad_l = (kernel_size_large - 1) // 2
        self.strip_h_large = DWConv(in_channels, (kernel_size_large, 1), (pad_l, 0))
        self.strip_w_large = DWConv(in_channels, (1, kernel_size_large), (0, pad_l))

        pad_s = (kernel_size_small - 1) // 2
        self.strip_h_small = DWConv(in_channels, (kernel_size_small, 1), (pad_s, 0))
        self.strip_w_small = DWConv(in_channels, (1, kernel_size_small), (0, pad_s))
        
        # === 2. 局部感知 (Local) ===
        # 3x3 抓圆心
        self.local_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        # 5x5 抓光晕 (Dilation=2)
        self.local_5x5 = DWConv(in_channels, kernel_size=3, padding=2, dilation=2)

        # === 3. 【新增】边缘感知 (Edge) ===
        self.edge_extractor = EdgeOperator()

        # === 4. 全局环境感知 (Global Context) ===
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        # === 5. 门控融合 (Gate) ===
        # 输入变为 7 路: 
        # 4(Strip) + 2(Local) + 1(Edge) = 7
        self.gate_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 7, in_channels, kernel_size=1), 
            nn.Sigmoid()
        )
        
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        nn.init.constant_(self.gate_fusion[0].bias, 0)

    def forward(self, x):
        shortcut = x
        x_in = self.proj_in(x)
        
        # 1. Strip
        lh = self.strip_h_large(x_in)
        lw = self.strip_w_large(x_in)
        sh = self.strip_h_small(x_in)
        sw = self.strip_w_small(x_in)
        
        # 2. Local
        loc3 = self.local_3x3(x_in)
        loc5 = self.local_5x5(x_in)
        
        # 3. Edge (直接从 Input Feature 提取高频信息)
        edge = self.edge_extractor(x_in)
        
        # 4. Gate Calculation
        # 将边缘信息也拼接到 Gate 的决策依据中
        # 这样 Gate 就能知道：哪里是平坦区域，哪里是边缘区域
        cat_feat = torch.cat([lh, lw, sh, sw, loc3, loc5, edge], dim=1)
        gate = self.gate_fusion(cat_feat)
        
        # 5. Global Context Injection
        global_vec = self.global_fc(self.global_pool(x_in))
        gate = gate * (1 + torch.sigmoid(global_vec))
        
        out = x_in * gate
        return shortcut + self.proj_out(out)