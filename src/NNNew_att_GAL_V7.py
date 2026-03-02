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
        self.act = nn.GELU()  # 坚守已被证明最稳的 GELU

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))

class GAL_Adapter(nn.Module):
    def __init__(self, in_channels, kernel_size_large=23, kernel_size_mid=15, kernel_size_small=7, reduction=16):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        self.pre_orient = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        # 🚀 补丁1：三级条形感受野 (大、中、小)，消除几何刚性断层
        pad_l = (kernel_size_large - 1) // 2
        pad_m = (kernel_size_mid - 1) // 2
        pad_s = (kernel_size_small - 1) // 2
        
        self.strip_h_large = DWConv(in_channels, (kernel_size_large, 1), (pad_l, 0))
        self.strip_w_large = DWConv(in_channels, (1, kernel_size_large), (0, pad_l))
        
        self.strip_h_mid = DWConv(in_channels, (kernel_size_mid, 1), (pad_m, 0))
        self.strip_w_mid = DWConv(in_channels, (1, kernel_size_mid), (0, pad_m))
        
        self.strip_h_small = DWConv(in_channels, (kernel_size_small, 1), (pad_s, 0))
        self.strip_w_small = DWConv(in_channels, (1, kernel_size_small), (0, pad_s))

        self.local_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        self.local_5x5 = DWConv(in_channels, 3, padding=2, dilation=2)

        # 🚀 8个分支的像素级门控 (坚守 reduction=16 终极降噪器)
        num_branches = 8
        mid_channels = max(in_channels * num_branches // reduction, 16)
        
        self.branch_weight = nn.Sequential(
            nn.Conv2d(in_channels * num_branches, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, in_channels * num_branches, kernel_size=1, bias=False)
        )

        # 🚀 补丁2：CCSM 脑容量扩充 (增强对极端红外的调制表达能力)
        style_hidden = in_channels * 4  # 从压缩变为扩容
        self.style_fc = nn.Sequential(
            nn.Conv2d(in_channels * 2, style_hidden, 1), 
            nn.GELU(),
            nn.Conv2d(style_hidden, in_channels * 2, 1)  
        )

        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    def forward(self, x):
        shortcut = x
        x = self.proj_in(x)

        x_oriented = self.pre_orient(x)

        # 8路并发
        lh = self.strip_h_large(x_oriented)
        lw = self.strip_w_large(x_oriented)
        mh = self.strip_h_mid(x_oriented)
        mw = self.strip_w_mid(x_oriented)
        sh = self.strip_h_small(x_oriented)
        sw = self.strip_w_small(x_oriented)
        loc3 = self.local_3x3(x)
        loc5 = self.local_5x5(x)

        branches = [lh, lw, mh, mw, sh, sw, loc3, loc5]

        # 空间融合
        cat_feat = torch.cat(branches, dim=1)
        weight = self.branch_weight(cat_feat)

        B, C8, H, W = weight.shape
        C = C8 // 8
        weight = weight.view(B, 8, C, H, W)
        weight = F.softmax(weight, dim=1) 

        stacked = torch.stack(branches, dim=1)
        out = (weight * stacked).sum(dim=1) 

        # 全局强度调制
        b, c, h, w = out.shape
        out_flat = out.view(b, c, -1)
        
        feat_mean = out_flat.mean(dim=2, keepdim=True).unsqueeze(-1) 
        var = out_flat.var(dim=2, keepdim=True, unbiased=False)
        feat_std = torch.sqrt(var + 1e-5).unsqueeze(-1) 
        
        style_feat = torch.cat([feat_mean, feat_std], dim=1)
        
        gamma_beta = self.style_fc(style_feat)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = 2.0 * torch.sigmoid(gamma) 
        
        out = gamma * out + beta
        out = self.proj_out(out)

        return shortcut + out