import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys

sys.path.append(".") 
from model import ST_SAM

# 还是用那张红外图
IMG_PATH = r"/workspace/data/root/xiaoju/Unet/dataset/Infrared3/Original/Infrared3_000012.PNG" 
CHECKPOINT = "./checkpoints/sam2_hiera_large.pt"
ST_SAM_CKPT = "./checkpoints/fold_0/best_model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

def overlay_heatmap(img_rgb, heatmap, title, threshold=None):
    H, W = img_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W, H))
    
    # 归一化
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    
    # 自动阈值：取前 20% 亮的区域，其他的透明
    if threshold is None:
        threshold = np.percentile(heatmap_norm, 80)
    
    mask = heatmap_norm > threshold
    
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    base = img_rgb.astype(float) * 0.7
    overlay = heatmap_color.astype(float)
    
    final_img = base.copy()
    final_img[mask] = base[mask] * 0.3 + overlay[mask] * 0.7
    
    return np.uint8(final_img)

def visualize_s0():
    print("⏳ Loading model...")
    model = ST_SAM(checkpoint_path=CHECKPOINT).to(device)
    state_dict = torch.load(ST_SAM_CKPT, map_location=device)
    new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state)
    model.eval()

    img_bgr = cv2.imread(IMG_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    input_img = cv2.resize(img_rgb, (1024, 1024))
    img_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    box = torch.tensor([[0, 0, 1024, 1024]], device=device).float()

    print("🚀 Running inference...")
    with torch.no_grad():
        _ = model(img_tensor, box)

    # =========================================================
    # 关键修改：看 s0 层 (High Res)，看 Feature (Activation)
    # =========================================================
    try:
        # [1, 6, C, H, W] -> 取 Strip-W-Large (index 1) -> mean channel -> [H, W]
        # 注意：这里取的是 last_features (特征)，不是 last_weights (权重)
        s0_features = model.adapter_s0.last_features[0, 1].mean(dim=0).cpu().numpy()
        
        # 对比一下：同时也取 s0 的权重看看
        s0_weights = model.adapter_s0.last_weights[0, 1].mean(dim=0).cpu().numpy()
        
    except AttributeError:
        print("❌ 提取失败！请确认 NNNew_att_v2_PPPGPT.py 里加了 `self.last_features = ...`")
        return

    # 绘图
    plt.figure(figsize=(18, 6))
    
    # 1. 原图
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Infrared", fontsize=14)
    plt.axis("off")
    
    # 2. s0 Raw Feature (这才是我们要的证据！)
    # Strip Conv 应该对横向亮线有极强的响应
    viz_feat = overlay_heatmap(img_rgb, s0_features, "Feature", threshold=None)
    plt.subplot(1, 3, 2)
    plt.imshow(viz_feat)
    plt.title("s0 Strip-W Raw Activation\n(What the Conv SEES)", fontsize=14, color='darkgreen', fontweight='bold')
    plt.axis("off")
    
    # 3. s0 Weight (Gate 的选择)
    viz_weight = overlay_heatmap(img_rgb, s0_weights, "Weight", threshold=None)
    plt.subplot(1, 3, 3)
    plt.imshow(viz_weight)
    plt.title("s0 Strip-W Weight\n(What the Gate CHOOSES)", fontsize=14)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("vis_s0_evidence.png", dpi=300)
    print("✅ Saved vis_s0_evidence.png")
    plt.show()

if __name__ == "__main__":
    visualize_s0()