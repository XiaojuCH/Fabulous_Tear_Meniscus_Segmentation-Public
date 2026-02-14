import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys
import types # <--- 引入这个库来实现动态替换

sys.path.append(".") 
from model import ST_SAM

# =================配置区域=================
IMG_PATH = r"/workspace/data/root/xiaoju/Unet/dataset/Infrared3/Original/Infrared3_000012.PNG" 
CHECKPOINT = "./checkpoints/sam2_hiera_large.pt"
ST_SAM_CKPT = "./checkpoints/fold_0/best_model.pth" 
device = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================

# 1. 定义一个新的 forward 函数，去掉了 torch.no_grad()
def forward_with_grad(self, images, box_prompts):
    """
    这是一个“破解版”的前向传播，专门用于 Saliency Map 计算。
    它去掉了 Image Encoder 的 no_grad 限制，允许梯度回传到输入图像。
    """
    # 1. Image Encoder (注意：这里去掉了 with torch.no_grad():)
    backbone_out = self.sam2.image_encoder(images)
    src_features = backbone_out["vision_features"]
    
    _fpn_features = backbone_out["backbone_fpn"]
    raw_s0 = _fpn_features[0]
    raw_s1 = _fpn_features[1]

    # 2. High-Res Injection (Trainable)
    feat_s0 = self.proj_s0(raw_s0)
    feat_s1 = self.proj_s1(raw_s1)
    
    refined_s0 = self.adapter_s0(feat_s0) 
    refined_s1 = self.adapter_s1(feat_s1)
    
    high_res_features = [refined_s0, refined_s1]

    # 3. Prompt Encoder
    sparse_embeddings, dense_embeddings = self.sam2.sam_prompt_encoder(
        points=None,
        boxes=box_prompts,
        masks=None,
    )

    # 4. Mask Decoder
    low_res_masks, iou_predictions, _, _ = self.sam2.sam_mask_decoder(
        image_embeddings=src_features,
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

def get_saliency_map(model, img_tensor, box):
    # 1. 开启输入图像的梯度记录
    img_tensor.requires_grad = True
    
    # 2. 前向传播
    preds = model(img_tensor, box)
    
    # 3. 选取我们要解释的目标 (最大化 Mask 响应)
    score = torch.sigmoid(preds)
    target = score.sum()
    
    # 4. 反向传播
    model.zero_grad()
    target.backward()
    
    # 5. 获取梯度
    if img_tensor.grad is None:
        return None
        
    gradients = img_tensor.grad.data.abs().squeeze(0).cpu().numpy()
    
    # 6. 处理梯度 (RGB 最大值)
    saliency = np.max(gradients, axis=0)
    
    # 归一化
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency

def visualize_saliency():
    print("⏳ Loading model...")
    model = ST_SAM(checkpoint_path=CHECKPOINT).to(device)
    
    # 加载权重
    state_dict = torch.load(ST_SAM_CKPT, map_location=device)
    new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state)
    
    # 【关键魔法】动态替换 forward 方法
    # 这行代码把模型实例的 forward 方法换成了我们的 forward_with_grad
    model.forward = types.MethodType(forward_with_grad, model)
    print("🔓 Model forward method patched (Gradient flow unlocked).")

    model.eval()
    
    # 读取图片
    img_bgr = cv2.imread(IMG_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    input_img = cv2.resize(img_rgb, (1024, 1024))
    img_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    box = torch.tensor([[0, 0, 1024, 1024]], device=device).float()

    print("🚀 Calculating Saliency...")
    # 简单的 SmoothGrad 模拟：稍微加一点噪声求平均，图会更干净
    saliency_total = np.zeros((1024, 1024))
    n_samples = 5 # 跑5次求平均
    
    for i in range(n_samples):
        # 加微小噪声
        noise = torch.randn_like(img_tensor) * 0.02
        curr_img = img_tensor + noise
        curr_map = get_saliency_map(model, curr_img, box)
        if curr_map is not None:
            saliency_total += curr_map
            
    saliency = saliency_total / n_samples
    
    # === 绘图 ===
    plt.figure(figsize=(12, 6))
    
    # 1. 原图
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Infrared Input\n(Note: Strong Rings)", fontsize=14)
    plt.axis("off")
    
    # 2. Saliency Map
    plt.subplot(1, 2, 2)
    # 增强对比度：Gamma Correction
    saliency_vis = np.power(saliency, 0.6) 
    
    plt.imshow(img_rgb, alpha=0.6) # 底图变淡
    plt.imshow(saliency_vis, cmap='jet', alpha=0.7) # 热力图
    plt.title("Gradient Saliency Map\n(Red = High Importance)", fontsize=14, color='darkred', fontweight='bold')
    plt.axis("off")
    
    save_path = "vis_saliency_v2.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_saliency()