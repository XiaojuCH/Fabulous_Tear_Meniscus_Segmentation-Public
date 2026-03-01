import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
import os

# 引入你刚才写好的新注意力模块
# from New_att import StripDetailAdapter # 旧版
# from New_att_v2 import GatedStripAdapter
# from New_att_v3 import GatedDilatedStripAdapter
# from NNew_att_v2_plus import GatedMultiScaleStripAdapter
# from NNew_att_v2_plus_plus import GatedMultiScaleStripAdapter
# from NNew_att_v2_PPP import GatedMultiScaleStripAdapter
# from NNew_att_v2_4P import GatedMultiScaleStripAdapter
# from NNNew_att_v2_PPPGPT import GSCSA
# from NNNew_att_v2_PPPGPT_final_bk import GSCSA
# from NNNew_att_v2_PPPGPT_final_bk import GSCSA

from NNNew_att_GAL_bk import GAL_Adapter

# from NNNew_att_GAL_Notin import GAL_Adapter
# from NNNew_att_GAL_V2 import GAL_Adapter
# from NNNew_att_GAL_V3 import GAL_Adapter
# from NNNew_att_GAL_V4 import GAL_Adapter
# from NNNew_att_GAL_V5 import GAL_Adapter
# from NNNew_att_GAL_V6 import GAL_Adapter

# ==============================================================================
# 主模型：ST-SAM (High-Res Injection Version)
# ==============================================================================
class ST_SAM(nn.Module):
    def __init__(self, 
                 model_cfg="sam2_hiera_l.yaml", 
                 checkpoint_path="./checkpoints/sam2_hiera_large.pt"
                 ):
        super().__init__()
        
        # 1. 加载 SAM2 骨干
        if not os.path.exists(checkpoint_path):
            if os.path.exists(f"../{checkpoint_path}"):
                checkpoint_path = f"../{checkpoint_path}"
            else:
                 print(f"⚠️ Warning: Checkpoint not found at {checkpoint_path}")

        self.sam2 = build_sam2(model_cfg, checkpoint_path)
        
        # 2. 冻结大部分参数 (PEFT 策略)
        for param in self.sam2.image_encoder.parameters():
            param.requires_grad = False
        for param in self.sam2.sam_prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.sam2.memory_attention.parameters():
            param.requires_grad = False

        # ---------------------------------------------------------
        # 3. 初始化自定义模块
        # ---------------------------------------------------------
        # 投影层：将 Backbone 的 256 维特征降维，对齐 Decoder 需求
        self.proj_s0 = nn.Conv2d(256, 32, kernel_size=1, bias=False)
        self.proj_s1 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        
# 恢复真·V1 的大感受野，以保证宏观不连断裂
        # 引入 reduction=8 作为强正则化器，强制抹平门控网络对反光的敏感度抖动！
        
        # 【s0 层 (Stride 4)】: 
        self.adapter_s0 = GAL_Adapter(
            in_channels=32, 
            kernel_size_large=23,  # 恢复最优大感受野
            kernel_size_small=7, 
            reduction=8            # 🚀 终极杀手锏：降低通道自由度，强迫边缘平滑
        )
        
        # 【s1 层 (Stride 8)】: 
        self.adapter_s1 = GAL_Adapter(
            in_channels=64, 
            kernel_size_large=23,  # 恢复最优大感受野
            kernel_size_small=7,
            reduction=8            # 🚀 终极杀手锏
        )

        # ---------------------------------------------------------
        # 4. 开启需要训练部分的梯度
        # ---------------------------------------------------------
        # 4.1 Mask Decoder
        for param in self.sam2.sam_mask_decoder.parameters():
            param.requires_grad = True
            
        # 4.2 投影层 & 新加入的 Adapter
        trainable_layers = [self.proj_s0, self.proj_s1, self.adapter_s0, self.adapter_s1]
        for layer in trainable_layers:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, images, box_prompts):
        """
        images: [B, 3, 1024, 1024]
        box_prompts: [B, 4]
        """
        # 1. Image Encoder (Frozen)
        with torch.no_grad():
            backbone_out = self.sam2.image_encoder(images)
            src_features = backbone_out["vision_features"] # Bottleneck feature (1/32)
            
            # 获取多尺度特征
            _fpn_features = backbone_out["backbone_fpn"]
            raw_s0 = _fpn_features[0] # Stride 4 (256x256)
            raw_s1 = _fpn_features[1] # Stride 8 (128x128)

        # 2. High-Res Injection (Trainable)
        # -------------------------------------------------------
        # Step A: 投影 (Channel Projection)
        feat_s0 = self.proj_s0(raw_s0)  # [B, 32, 256, 256]
        feat_s1 = self.proj_s1(raw_s1)  # [B, 64, 128, 128]
        
        # Step B: 注入条状注意力 (Strip Attention Injection)
        # 这里是关键！在送入 Decoder 之前，先增强边界特征
        refined_s0 = self.adapter_s0(feat_s0) 
        refined_s1 = self.adapter_s1(feat_s1)
        
        high_res_features = [refined_s0, refined_s1]
        # -------------------------------------------------------

        # 3. Prompt Encoder (Frozen)
        sparse_embeddings, dense_embeddings = self.sam2.sam_prompt_encoder(
            points=None,
            boxes=box_prompts,
            masks=None,
        )

        # 4. Mask Decoder (Trainable)
        low_res_masks, iou_predictions, _, _ = self.sam2.sam_mask_decoder(
            image_embeddings=src_features, # 瓶颈层特征保持原样
            image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features # 传入增强后的高分辨率特征
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
# 对比模型：Baseline SAM 2 (无 Adapter，仅微调 Decoder)
# ==============================================================================
class Baseline_SAM2(nn.Module):
    def __init__(self, 
                 model_cfg="sam2_hiera_l.yaml", 
                 checkpoint_path="./checkpoints/sam2_hiera_large.pt"
                 ):
        super().__init__()
        
        # 1. 加载 SAM2
        if not os.path.exists(checkpoint_path):
            if os.path.exists(f"../{checkpoint_path}"):
                checkpoint_path = f"../{checkpoint_path}"
            else:
                 print(f"⚠️ Warning: Checkpoint not found at {checkpoint_path}")

        self.sam2 = build_sam2(model_cfg, checkpoint_path)
        
        # 2. 冻结大部分参数 (保持和 ST-SAM 一致)
        for param in self.sam2.image_encoder.parameters():
            param.requires_grad = False
        for param in self.sam2.sam_prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.sam2.memory_attention.parameters():
            param.requires_grad = False

        # ---------------------------------------------------------
        # 3. 移除 Adapter，仅保留必要的维度投影
        # ---------------------------------------------------------
        # self.adapter = StripAttentionAdapter(...)  <-- 【删除】创新模块
        
        # 保留这两个投影层是为了让 Backbone 的特征维度能对齐 Decoder 的输入要求
        # 这属于结构适配，不属于“创新点”，所以 Baseline 里要留着
        self.proj_s0 = nn.Conv2d(256, 32, kernel_size=1, bias=False)
        self.proj_s1 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        
        # ---------------------------------------------------------
        # 4. 开启梯度
        # ---------------------------------------------------------
        # 4.1 Mask Decoder (SAM2 原生部分)
        for param in self.sam2.sam_mask_decoder.parameters():
            param.requires_grad = True
            
        # 4.2 投影层
        for param in self.proj_s0.parameters():
            param.requires_grad = True
        for param in self.proj_s1.parameters():
            param.requires_grad = True

    def forward(self, images, box_prompts):
        # 1. Image Encoder (Frozen)
        with torch.no_grad():
            backbone_out = self.sam2.image_encoder(images)
            src_features = backbone_out["vision_features"]
            _fpn_features = backbone_out["backbone_fpn"]
            raw_s0 = _fpn_features[0]
            raw_s1 = _fpn_features[1]

        # 投影维度 (保持结构一致)
        feat_s0 = self.proj_s0(raw_s0) 
        feat_s1 = self.proj_s1(raw_s1)
        high_res_features = [feat_s0, feat_s1]

        # ---------------------------------------------------------
        # 2. 【关键修改】直接使用原始特征，不经过 Adapter
        # ---------------------------------------------------------
        # refined_features = self.adapter(src_features) <-- 【删除】
        refined_features = src_features  # <-- 【新增】直接透传
        
        # 3. Prompt Encoder (Frozen)
        sparse_embeddings, dense_embeddings = self.sam2.sam_prompt_encoder(
            points=None,
            boxes=box_prompts,
            masks=None,
        )

        # 4. Mask Decoder (Trainable)
        low_res_masks, iou_predictions, _, _ = self.sam2.sam_mask_decoder(
            image_embeddings=refined_features, # 这里输入的是没有加强过的特征
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

import math

# ==============================================================================
# 辅助模块：标准 LoRA 线性层
# ==============================================================================
class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=4):
        super().__init__()
        # 保持原有全连接层，并彻底冻结
        self.linear = original_linear
        for param in self.linear.parameters():
            param.requires_grad = False
            
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # LoRA 的 A 和 B 矩阵
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))
        
        # 初始化：A 用 Kaiming，B 全零，保证初始状态等效于原网络
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 原有输出 + 低秩矩阵输出
        orig_out = self.linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return orig_out + lora_out

def inject_lora_to_decoder(module, rank=4):
    """递归地将 Mask Decoder 中的 nn.Linear 替换为 LoRALinear"""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # 替换线性层
            setattr(module, name, LoRALinear(child, rank=rank))
        else:
            inject_lora_to_decoder(child, rank=rank)

# ==============================================================================
# 对比模型 1：SAM 2 + LoRA Baseline
# ==============================================================================
class LoRA_SAM2(nn.Module):
    def __init__(self, 
                 model_cfg="sam2_hiera_l.yaml", 
                 checkpoint_path="./checkpoints/sam2_hiera_large.pt",
                 lora_rank=4
                 ):
        super().__init__()
        
        # 1. 加载 SAM2 骨干
        if not os.path.exists(checkpoint_path) and os.path.exists(f"../{checkpoint_path}"):
            checkpoint_path = f"../{checkpoint_path}"
            
        self.sam2 = build_sam2(model_cfg, checkpoint_path)
        
        # 2. 全面冻结原生参数
        for param in self.sam2.parameters():
            param.requires_grad = False

        # 3. 维度投影层 (保留，为了能够对齐输入，这是必要的架构妥协)
        self.proj_s0 = nn.Conv2d(256, 32, kernel_size=1, bias=False)
        self.proj_s1 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        
        # 4. 注入 LoRA 到 Mask Decoder
        inject_lora_to_decoder(self.sam2.sam_mask_decoder, rank=lora_rank)
        
        # 5. 开启必要梯度的追踪
        # 仅开启我们自己加的投影层和 LoRA 参数的梯度
        for param in self.proj_s0.parameters():
            param.requires_grad = True
        for param in self.proj_s1.parameters():
            param.requires_grad = True
            
        # LoRA 参数默认 requires_grad=True，所以直接筛选即可
        
    def forward(self, images, box_prompts):
        with torch.no_grad():
            backbone_out = self.sam2.image_encoder(images)
            src_features = backbone_out["vision_features"]
            _fpn_features = backbone_out["backbone_fpn"]
            raw_s0 = _fpn_features[0] 
            raw_s1 = _fpn_features[1] 

        # 投影
        feat_s0 = self.proj_s0(raw_s0)  
        feat_s1 = self.proj_s1(raw_s1)  
        high_res_features = [feat_s0, feat_s1]

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.sam2.sam_prompt_encoder(
                points=None, boxes=box_prompts, masks=None,
            )

        # 带有 LoRA 的 Mask Decoder 推理
        low_res_masks, iou_predictions, _, _ = self.sam2.sam_mask_decoder(
            image_embeddings=src_features, 
            image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features 
        )
        
        masks = F.interpolate(low_res_masks, size=(images.shape[2], images.shape[3]), mode="bilinear", align_corners=False)
        return masks


# ==============================================================================
# 辅助模块：通用的瓶颈 Adapter (MSA 风格)
# ==============================================================================
class MSA_Adapter(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        mid_channels = max(in_channels // reduction, 16)
        # 一个非常标准的卷积瓶颈结构，没有特殊的花样
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return x + self.bottleneck(x)

# ==============================================================================
# 对比模型 2：SAM 2 + 经典瓶颈 Adapter (MSA Baseline)
# ==============================================================================
class MSA_Baseline_SAM2(nn.Module):
    def __init__(self, 
                 model_cfg="sam2_hiera_l.yaml", 
                 checkpoint_path="./checkpoints/sam2_hiera_large.pt"
                 ):
        super().__init__()
        
        # 1. 加载冻结的 SAM2 主干
        if not os.path.exists(checkpoint_path) and os.path.exists(f"../{checkpoint_path}"):
            checkpoint_path = f"../{checkpoint_path}"
        self.sam2 = build_sam2(model_cfg, checkpoint_path)
        
        for param in self.sam2.image_encoder.parameters():
            param.requires_grad = False
        for param in self.sam2.sam_prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.sam2.memory_attention.parameters():
            param.requires_grad = False

        # 2. 投影层
        self.proj_s0 = nn.Conv2d(256, 32, kernel_size=1, bias=False)
        self.proj_s1 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        
        # 3. 注入经典 Adapter (而不是你的 GAL)
        self.adapter_s0 = MSA_Adapter(in_channels=32)
        self.adapter_s1 = MSA_Adapter(in_channels=64)

        # 4. 开启梯度 (注意：这里和你的 SAM_Gal 一样，开启了整个 Decoder 的微调！)
        for param in self.sam2.sam_mask_decoder.parameters():
            param.requires_grad = True
            
        trainable_layers = [self.proj_s0, self.proj_s1, self.adapter_s0, self.adapter_s1]
        for layer in trainable_layers:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, images, box_prompts):
        with torch.no_grad():
            backbone_out = self.sam2.image_encoder(images)
            src_features = backbone_out["vision_features"]
            _fpn_features = backbone_out["backbone_fpn"]
            raw_s0 = _fpn_features[0]
            raw_s1 = _fpn_features[1]

        # 特征注入流程
        feat_s0 = self.proj_s0(raw_s0)
        feat_s1 = self.proj_s1(raw_s1)
        
        # 经过经典的 MSA Adapter
        refined_s0 = self.adapter_s0(feat_s0)
        refined_s1 = self.adapter_s1(feat_s1)
        high_res_features = [refined_s0, refined_s1]

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.sam2.sam_prompt_encoder(
                points=None, boxes=box_prompts, masks=None,
            )

        # Decoder 推理
        low_res_masks, iou_predictions, _, _ = self.sam2.sam_mask_decoder(
            image_embeddings=src_features,
            image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features
        )

        masks = F.interpolate(low_res_masks, size=(images.shape[2], images.shape[3]), mode="bilinear", align_corners=False)
        return masks


# ==============================================================================
# 对比模型：MedSAM-style (SAM2-Hiera-L + 全量 Mask Decoder 微调)
# 参考: Ma et al., "Segment Anything in Medical Images", Nature Communications 2024
# 对齐策略: 冻结 Image Encoder，仅微调 Mask Decoder + 投影层，无任何自定义 Adapter
# ==============================================================================
class MedSAM_SAM2(nn.Module):
    def __init__(self,
                 model_cfg="sam2_hiera_l.yaml",
                 checkpoint_path="./checkpoints/sam2_hiera_large.pt"):
        super().__init__()

        if not os.path.exists(checkpoint_path):
            if os.path.exists(f"../{checkpoint_path}"):
                checkpoint_path = f"../{checkpoint_path}"
            else:
                print(f"⚠️ Warning: Checkpoint not found at {checkpoint_path}")

        self.sam2 = build_sam2(model_cfg, checkpoint_path)

        # 冻结 Image Encoder（与 MedSAM 官方策略一致）
        for param in self.sam2.image_encoder.parameters():
            param.requires_grad = False
        for param in self.sam2.sam_prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.sam2.memory_attention.parameters():
            param.requires_grad = False

        # 维度投影层（结构适配，与 Baseline_SAM2 完全一致）
        self.proj_s0 = nn.Conv2d(256, 32, kernel_size=1, bias=False)
        self.proj_s1 = nn.Conv2d(256, 64, kernel_size=1, bias=False)

        # 开启 Mask Decoder 全量微调（MedSAM 核心策略）
        for param in self.sam2.sam_mask_decoder.parameters():
            param.requires_grad = True
        for param in self.proj_s0.parameters():
            param.requires_grad = True
        for param in self.proj_s1.parameters():
            param.requires_grad = True

    def forward(self, images, box_prompts):
        # 1. Image Encoder (Frozen)
        with torch.no_grad():
            backbone_out = self.sam2.image_encoder(images)
            src_features = backbone_out["vision_features"]
            _fpn_features = backbone_out["backbone_fpn"]
            raw_s0 = _fpn_features[0]
            raw_s1 = _fpn_features[1]

        # 2. 维度投影（无 Adapter，直接透传）
        feat_s0 = self.proj_s0(raw_s0)
        feat_s1 = self.proj_s1(raw_s1)
        high_res_features = [feat_s0, feat_s1]

        # 3. Prompt Encoder (Frozen)
        sparse_embeddings, dense_embeddings = self.sam2.sam_prompt_encoder(
            points=None,
            boxes=box_prompts,
            masks=None,
        )

        # 4. Mask Decoder (Trainable - MedSAM 核心)
        low_res_masks, iou_predictions, _, _ = self.sam2.sam_mask_decoder(
            image_embeddings=src_features,
            image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # 5. Upscale to original resolution
        masks = F.interpolate(
            low_res_masks,
            size=(images.shape[2], images.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return masks