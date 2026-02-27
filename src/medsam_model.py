import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry

class True_MedSAM(nn.Module):
    def __init__(self, checkpoint_path="./checkpoints/medsam_vit_b.pth", freeze_image_encoder=True):
        super().__init__()
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        
        if freeze_image_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
                
    def forward(self, image, box):
        # 1. 提取图像特征 (支持并行 Batch)
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(image) # 形状: [B, 256, 64, 64]
            
        # 2. 提取 Prompt 特征
        if len(box.shape) == 2:
            box = box.unsqueeze(1)
            
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=box,
            masks=None,
        )
        
        # 3. Mask Decoder (🔥 修复 SAM 1 的批处理 Bug 🔥)
        low_res_masks_list = []
        # 我们用 for 循环把 Batch 拆开，逐个通过 Decoder，完美绕过 64 vs 8 的错误
        for i in range(image.shape[0]):
            low_res_mask, _ = self.sam.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings[i].unsqueeze(0),
                dense_prompt_embeddings=dense_embeddings[i].unsqueeze(0),
                multimask_output=False,
            )
            low_res_masks_list.append(low_res_mask)
            
        # 把拆开的结果重新拼回 Batch
        low_res_masks = torch.cat(low_res_masks_list, dim=0)
        
        # 4. 上采样回原图大小 1024x1024
        masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        
        return masks