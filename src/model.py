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
from NNNew_att_v2_V4 import GatedMultiScaleStripAdapterV4



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
        
#       【s0 层 (Stride 4)】: 最精细层
        # 策略：Large=15 (看局部长条), Small=5 (看像素细节)
        # s0 分辨率很高(256x256)，核不用太大，重点是修补边缘
        self.adapter_s0 = GatedMultiScaleStripAdapterV4(
            in_channels=32, 
            kernel_size_large=15, 
            kernel_size_small=5 
        )
        
        # 【s1 层 (Stride 8)】: 抗干扰层
        # 策略：Large=23 (看整体轮廓，抗圆环干扰), Small=7 (防断裂)
        # s1 负责在 128x128 尺度上把泪河“连起来”
        self.adapter_s1 = GatedMultiScaleStripAdapterV4(
            in_channels=64, 
            kernel_size_large=23, 
            kernel_size_small=7
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

    # def forward(self, images, box_prompts):
    def forward(self, images, box_prompts, pupil_heatmap=None):
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
        
        # refined_s0 = self.adapter_s0(feat_s0) 
        # refined_s1 = self.adapter_s1(feat_s1)

        refined_s0 = self.adapter_s0(feat_s0, pupil_heatmap=pupil_heatmap)  # 传递参数
        refined_s1 = self.adapter_s1(feat_s1, pupil_heatmap=pupil_heatmap)  # 传递参数
        
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