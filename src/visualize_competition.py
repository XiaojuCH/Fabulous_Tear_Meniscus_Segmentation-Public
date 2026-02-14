import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys

# 确保能导入你的模型
sys.path.append(".") 
from model import ST_SAM

# =================配置区域=================
# 找一张典型的红外图 (比如有同心圆干扰的)
IMG_PATH = r"/workspace/data/root/xiaoju/Unet/dataset/Infrared3/Original/Infrared3_000012.PNG" 
# 或者是彩图
# IMG_PATH = "data/root/xiaoju/Eye_River_new/Color1_000000.jpg"

CHECKPOINT = "./checkpoints/sam2_hiera_large.pt" # 你的 SAM2 权重
ST_SAM_CKPT = "./checkpoints/fold_0/best_model.pth" # 你的 ST-SAM 训练权重
device = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================

def visualize():
    # 1. 加载模型
    print("⏳ Loading model...")
    model = ST_SAM(checkpoint_path=CHECKPOINT).to(device)
    
    # 加载训练好的权重 (处理 DDP 的 module. 前缀)
    state_dict = torch.load(ST_SAM_CKPT, map_location=device)
    new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state)
    model.eval()

    # 2. 读取并预处理图片
    if not os.path.exists(IMG_PATH):
        print(f"❌ 找不到图片: {IMG_PATH}")
        return

    img_bgr = cv2.imread(IMG_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = img_rgb.shape[:2]

    # SAM 需要 1024x1024
    input_img = cv2.resize(img_rgb, (1024, 1024))
    img_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device) # [1, 3, 1024, 1024]
    
    # 构造全图 Box Prompt
    box = torch.tensor([[0, 0, 1024, 1024]], device=device).float()

    # 3. 推理 (触发 forward 钩子保存权重)
    print("🚀 Running inference...")
    with torch.no_grad():
        _ = model(img_tensor, box)

    # 4. 提取权重
    # 你的模型里有 adapter_s0 (高分) 和 adapter_s1 (低分)
    # 我们看 adapter_s1 (128x128) 比较直观，因为它负责整体抗干扰
    try:
        # [1, 6, C, H, W] -> mean over Channel -> [6, H, W]
        weights = model.adapter_s1.last_weights.mean(dim=2).squeeze(0).cpu().numpy()
    except AttributeError:
        print("❌ 提取失败！请确认你是否在 NNNew_att_v2_PPPGPT.py 里加了 `self.last_weights = ...`")
        return

    # 5. 绘图
    branch_names = [
        "Strip-H (Large)", "Strip-W (Large)", 
        "Strip-H (Small)", "Strip-W (Small)", 
        "Local-3x3 (Pupil)", "Local-5x5 (Halo)"
    ]
    
    plt.figure(figsize=(24, 10))
    
    # 原图
    plt.subplot(2, 4, 1)
    plt.imshow(img_rgb)
    plt.title("Input Image", fontsize=15)
    plt.axis("off")
    
    # 绘制 6 个分支的热力图
    for i in range(6):
        plt.subplot(2, 4, i+2)
        
        # 将 128x128 的热力图插值回原图大小
        heatmap = cv2.resize(weights[i], (original_w, original_h))
        
        # 归一化到 0-1 以便观察相对强弱
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        plt.imshow(heatmap, cmap='jet')
        plt.title(f"{branch_names[i]}\n(Red=Active, Blue=Inactive)", fontsize=12)
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)

    save_path = "vis_competition.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize()