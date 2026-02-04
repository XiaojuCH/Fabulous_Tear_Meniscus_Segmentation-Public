import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
import os

# ==============================================================================
# 创新模块：Strip-Topology Attention Adapter (ST-Adapter)
# ==============================================================================
class StripAttentionAdapter(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_channels = max(16, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act1 = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.final_conv.weight)

    def forward(self, x):
        identity = x
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        
        y_h = self.act1(self.bn1(self.conv1(x_h)))
        y_w = self.act1(self.bn1(self.conv1(x_w)))
        
        w_h = self.sigmoid(self.conv_h(y_h))
        w_w = self.sigmoid(self.conv_w(y_w))
        
        attended_features = identity * w_h * w_w
        out = identity + self.final_conv(attended_features)
        return out

# ==============================================================================
# 主模型：ST-SAM
# ==============================================================================
class ST_SAM(nn.Module):
    def __init__(self, 
                 model_cfg="sam2_hiera_l.yaml", 
                 checkpoint_path="./checkpoints/sam2_hiera_large.pt"
                 ):
        super().__init__()
        
        # ---------------------------------------------------------
        # 1. 必须最先加载 SAM2 模型 (这一步绝对不能少！)
        # ---------------------------------------------------------
        if not os.path.exists(checkpoint_path):
            if os.path.exists(f"../{checkpoint_path}"):
                checkpoint_path = f"../{checkpoint_path}"
            else:
                 print(f"⚠️ Warning: Checkpoint not found at {checkpoint_path}")

        # 【关键修复】这行代码必须被执行，不能是注释
        self.sam2 = build_sam2(model_cfg, checkpoint_path)
        
        # ---------------------------------------------------------
        # 2. 冻结大部分参数
        # ---------------------------------------------------------
        for param in self.sam2.image_encoder.parameters():
            param.requires_grad = False
        for param in self.sam2.sam_prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.sam2.memory_attention.parameters():
            param.requires_grad = False

        # ---------------------------------------------------------
        # 3. 初始化自定义模块 (Adapter 和 Projection)
        # ---------------------------------------------------------
        self.feature_dim = 256 
        self.adapter = StripAttentionAdapter(self.feature_dim)
        
        # 【之前修复的通道投影层】
        # feat_s0: 256 -> 32
        self.proj_s0 = nn.Conv2d(256, 32, kernel_size=1, bias=False)
        # feat_s1: 256 -> 64
        self.proj_s1 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        
        # ---------------------------------------------------------
        # 4. 开启需要训练部分的梯度
        # ---------------------------------------------------------
        # 4.1 Adapter
        for param in self.adapter.parameters():
            param.requires_grad = True
            
        # 4.2 Mask Decoder (SAM2 原生部分)
        for param in self.sam2.sam_mask_decoder.parameters():
            param.requires_grad = True
            
        # 4.3 新增的投影层
        for param in self.proj_s0.parameters():
            param.requires_grad = True
        for param in self.proj_s1.parameters():
            param.requires_grad = True

    def forward(self, images, box_prompts):
        """
        images: [B, 3, 1024, 1024]
        box_prompts: [B, 4]
        """
        # 1. Image Encoder (Frozen)
        with torch.no_grad():
            backbone_out = self.sam2.image_encoder(images)
            src_features = backbone_out["vision_features"]
            
            # 获取原始 FPN 特征 [256, 256, 256]
            _fpn_features = backbone_out["backbone_fpn"]
            
            # 取前两层 (Stride 4 和 Stride 8)
            # 注意：_fpn_features[0] 是 Stride 4, _fpn_features[1] 是 Stride 8
            raw_s0 = _fpn_features[0]
            raw_s1 = _fpn_features[1]

        # 【新增修复 2】进行维度投影 (256 -> 32/64)
        # 注意：这里需要开启梯度，所以要在 no_grad 之外
        feat_s0 = self.proj_s0(raw_s0)  # [B, 32, 256, 256]
        feat_s1 = self.proj_s1(raw_s1)  # [B, 64, 128, 128]
        
        # 组合成列表传入
        high_res_features = [feat_s0, feat_s1]

        # 2. Strip Adapter (Trainable)
        refined_features = self.adapter(src_features)
        
        # 3. Prompt Encoder (Frozen)
        sparse_embeddings, dense_embeddings = self.sam2.sam_prompt_encoder(
            points=None,
            boxes=box_prompts,
            masks=None,
        )

        # 4. Mask Decoder (Trainable)
        low_res_masks, iou_predictions, _, _ = self.sam2.sam_mask_decoder(
            image_embeddings=refined_features, 
            image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features 
        )
        
        # 5. Upscale
        masks = F.interpolate(
            low_res_masks, 
            size=(images.shape[2], images.shape[3]), 
            mode="bilinear", 
            align_corners=False
        )
        
        return masks
# ==============================================================================
# 测试代码块 (用于检查模型结构和前向传播是否通畅)
# ==============================================================================
if __name__ == "__main__":
    # 需要先下载权重才能运行此测试
    # 假设权重已在 correct path
    try:
        # 1. 实例化模型
        # 注意：需要确保当前目录下有 sam2_hiera_l.yaml 配置文件
        # 通常安装 sam2 库后会自动找到，找不到需手动指定绝对路径
        model = ST_SAM(checkpoint_path="../checkpoints/sam2_hiera_large.pt").cuda()
        
        # 2. 创建 dummy输入
        batch_size = 2
        dummy_img = torch.randn(batch_size, 3, 1024, 1024).cuda()
        dummy_box = torch.tensor([[100, 100, 500, 500]] * batch_size).float().cuda()
        
        # 3. 前向传播测试
        print("\n🧪 开始前向传播测试...")
        output_masks = model(dummy_img, dummy_box)
        
        print(f"✅ 输出 Shape: {output_masks.shape}") # 期望: [B, 1, 1024, 1024]
        
        # 4. 检查梯度状况
        print("\n🔍 检查梯度要求:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 只打印可训练的层，看看 adapter 是否在里面
                if "adapter" in name or "mask_decoder" in name:
                    print(f"  -> Trainable: {name}")

    except FileNotFoundError as e:
        print(f"\n⚠️ 测试跳过: {e}")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")