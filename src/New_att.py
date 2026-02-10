import torch
import torch.nn as nn
import torch.nn.functional as F

class StripDetailAdapter(nn.Module):
    """
    针对细长结构（如泪河）优化的轻量级适配器。
    核心思想：
    1. 避免使用 Pooling (会丢失微小细节)。
    2. 使用大核条状卷积 (Strip Convolution) 感知长距离上下文。
    3. 专为 High-Res 特征设计。
    """
    def __init__(self, in_channels, kernel_size=7):
        super().__init__()
        
        # 降维比例，保持计算量极低
        reduction_dim = max(in_channels // 4, 16)
        
        # 1. 降维 (1x1 Conv)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(inplace=True)
        )
        
        # 2. 条状卷积分支 (捕捉 H 方向和 W 方向的细长特征)
        # padding 保证尺寸不变
        padding = (kernel_size - 1) // 2
        
        # H方向分支: 卷积核 (kernel_size, 1) -> 比如 (7, 1)
        self.conv_h = nn.Sequential(
            nn.Conv2d(reduction_dim, reduction_dim, kernel_size=(kernel_size, 1), 
                      padding=(padding, 0), groups=reduction_dim, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.SiLU() # SiLU (Swish) 通常比 ReLU 在深层网络表现更好
        )
        
        # W方向分支: 卷积核 (1, kernel_size) -> 比如 (1, 7)
        self.conv_w = nn.Sequential(
            nn.Conv2d(reduction_dim, reduction_dim, kernel_size=(1, kernel_size), 
                      padding=(0, padding), groups=reduction_dim, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.SiLU()
        )
        
        # 3. 特征融合与升维
        self.fuse = nn.Conv2d(reduction_dim * 2, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        identity = x
        
        # 降维
        x_reduced = self.reduce(x)
        
        # 双路条状特征提取
        x_h = self.conv_h(x_reduced)
        x_w = self.conv_w(x_reduced)
        
        # 拼接 (Fusion)
        x_cat = torch.cat([x_h, x_w], dim=1) # [B, 2*mid, H, W]
        
        # 生成注意力权重
        att_map = self.sigmoid(self.fuse(x_cat))
        
        # 加权 + 残差连接
        out = identity * att_map + identity
        
        return out

# 测试代码 (运行时可删除)
if __name__ == "__main__":
    # 模拟一个 High-Res 特征图 (B=2, C=32, H=256, W=256)
    dummy_input = torch.randn(2, 32, 256, 256)
    model = StripDetailAdapter(in_channels=32, kernel_size=11)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # 应该保持不变
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total_params}") # 应该非常小，符合 PEFT 要求