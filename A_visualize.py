import os
import json
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as F
from matplotlib.patches import Rectangle
import sys
sys.path.append("src") # 确保能找到 dataset 和 model

# 导入你的模型 (请确保 model.py 里有这三个类)
from model import ST_SAM, Baseline_SAM2
try:
    from model import MSA_Baseline_SAM2
except ImportError:
    print("⚠️ 未找到 MSA_Baseline_SAM2，请确保它在 model.py 中定义。")

# =========================================================
# 配置区域 (请根据你需要可视化的图片进行修改)
# =========================================================
FOLD = 4  # 你想使用哪个 Fold 的权重
IMG_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 权重路径
CKPT_BASE = f"./checkpoints_ablation/fold_{FOLD}/best_model.pth" # 替换为你的真实路径
CKPT_MSA  = f"./checkpoints_msa/fold_{FOLD}/best_model.pth"                    # 替换为你的真实路径
CKPT_OURS = f"./checkpoints_gal50_bk/fold_{FOLD}/best_model.pth"                        # 替换为你的真实路径

# YOLO 预测框 JSON 路径
YOLO_JSON = f"./data_splits/yolo_boxes_fold{FOLD}.json"

# =========================================================
# 辅助函数
# =========================================================
def load_model_weights(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"❌ 找不到权重: {ckpt_path}")
        return model
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def overlay_mask(image_np, mask_np, color, alpha=0.6):
    """将二值 Mask 叠加到 RGB 图像上"""
    overlay = image_np.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(mask_np > 0, image_np[:, :, c] * (1 - alpha) + color[c] * alpha, image_np[:, :, c])
    return overlay

def get_zoom_bbox(mask_np, padding=50):
    """根据 GT Mask 自动获取局部放大的 BBox"""
    y_indices, x_indices = np.where(mask_np > 0)
    if len(y_indices) == 0:
        return 0, 0, mask_np.shape[1], mask_np.shape[0]
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(mask_np.shape[1], x_max + padding)
    y_max = min(mask_np.shape[0], y_max + padding)
    
    return x_min, y_min, x_max, y_max

# =========================================================
# 主推断与绘图函数
# =========================================================
def visualize_image(img_path, label_path, img_id):
    print(f"🔍 正在处理图像: {img_id}")
    
    # 1. 加载图像和标签
    image = Image.open(img_path).convert("RGB")
    label = Image.open(label_path).convert("L")
    
    image = image.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
    label = label.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
    
    img_tensor = F.to_tensor(image).unsqueeze(0).to(DEVICE)
    lbl_np = (np.array(label) > 127).astype(np.uint8)
    
    # 2. 加载 YOLO Box
    with open(YOLO_JSON, 'r') as f:
        yolo_preds = json.load(f)
    if img_id in yolo_preds:
        box_norm = yolo_preds[img_id]
        box = [box_norm[0] * IMG_SIZE, box_norm[1] * IMG_SIZE, box_norm[2] * IMG_SIZE, box_norm[3] * IMG_SIZE]
    else:
        print("⚠️ 未找到 YOLO 框，使用全局框。")
        box = [0, 0, IMG_SIZE, IMG_SIZE]
    box_tensor = torch.tensor([box], dtype=torch.float32).to(DEVICE)

    # 3. 加载模型并推断
    print("⏳ 正在运行 SAM Baseline...")
    model_base = load_model_weights(Baseline_SAM2(checkpoint_path="./checkpoints/sam2_hiera_large.pt").to(DEVICE), CKPT_BASE)
    
    print("⏳ 正在运行 SAM MSA...")
    try:
        model_msa = load_model_weights(MSA_Baseline_SAM2(checkpoint_path="./checkpoints/sam2_hiera_large.pt").to(DEVICE), CKPT_MSA)
    except:
        model_msa = None

    print("⏳ 正在运行 ST-SAM (Ours)...")
    model_ours = load_model_weights(ST_SAM(checkpoint_path="./checkpoints/sam2_hiera_large.pt").to(DEVICE), CKPT_OURS)

    with torch.no_grad():
        pred_base = (torch.sigmoid(model_base(img_tensor, box_tensor)) > 0.5).cpu().numpy()[0, 0]
        pred_msa = (torch.sigmoid(model_msa(img_tensor, box_tensor)) > 0.5).cpu().numpy()[0, 0] if model_msa else np.zeros_like(pred_base)
        pred_ours = (torch.sigmoid(model_ours(img_tensor, box_tensor)) > 0.5).cpu().numpy()[0, 0]

    # 4. 图像渲染
    img_np = np.array(image)
    
    # 定义颜色 (RGB): GT(绿色), Base(蓝色), MSA(橙色), Ours(红色)
    c_gt, c_base, c_msa, c_ours = (0, 255, 0), (0, 100, 255), (255, 165, 0), (255, 0, 0)
    
    vis_gt = overlay_mask(img_np, lbl_np, c_gt)
    vis_base = overlay_mask(img_np, pred_base, c_base)
    vis_msa = overlay_mask(img_np, pred_msa, c_msa)
    vis_ours = overlay_mask(img_np, pred_ours, c_ours)

    # 获取局部放大区域 (Zoom-in Box)
    x1, y1, x2, y2 = get_zoom_bbox(lbl_np, padding=80)

    # 5. 使用 Matplotlib 拼图 (排版为: 原图+GT | Base | MSA | Ours)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), gridspec_kw={'height_ratios': [2, 1]})
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    titles = ["Ground Truth", "SAM Baseline", "SAM MSA", "ST-SAM (Ours)"]
    images_to_show = [vis_gt, vis_base, vis_msa, vis_ours]

    for i in range(4):
        # 第一排：全图
        ax_full = axes[0, i]
        ax_full.imshow(images_to_show[i])
        ax_full.set_title(titles[i], fontsize=16, fontweight='bold', pad=10)
        ax_full.axis('off')
        # 在全图上画出 Zoom-in 的框
        rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
        ax_full.add_patch(rect)

        # 第二排：局部放大图 (Zoom-in)
        ax_zoom = axes[1, i]
        ax_zoom.imshow(images_to_show[i][y1:y2, x1:x2])
        ax_zoom.axis('off')
        # 给放大图加个边框
        for spine in ax_zoom.spines.values():
            spine.set_edgecolor('yellow')
            spine.set_linewidth(3)
            spine.set_visible(True)

    save_name = f"qualitative_result_{img_id}.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 可视化结果已保存至: {save_name}\n")
    plt.close()

if __name__ == "__main__":
    # =========================================================
    # 在这里填入你想可视化的红外图像路径！
    # 强烈建议挑选 3-4 张红外中心 (Infrared) 中，长条形状明显且有反光干扰的图
    # =========================================================
    test_cases = [
        {
            "img_path": "../Unet/dataset/Infrared3/Original/Infrared3_000588.PNG",     # 替换为你的真实图像路径
            "label_path": "../Unet/dataset/Infrared3/Cleaned_Label/Infrared3_000588.PNG", # 替换为你的真实标签路径
            "img_id": "Infrared3_000588"                                # 对应 YOLO JSON 里的 ID
        },
        {
            "img_path": "../Unet/dataset/Infrared3/Original/Infrared3_000146.PNG",     # 替换为你的真实图像路径
            "label_path": "../Unet/dataset/Infrared3/Cleaned_Label/Infrared3_000146.PNG", # 替换为你的真实标签路径
            "img_id": "Infrared3_000146"                                # 对应 YOLO JSON 里的 ID
        },
        {
            "img_path": "../Unet/dataset/Infrared3/Original/Infrared3_000189.PNG",     # 替换为你的真实图像路径
            "label_path": "../Unet/dataset/Infrared3/Cleaned_Label/Infrared3_000189.PNG", # 替换为你的真实标签路径
            "img_id": "Infrared3_000189"                                # 对应 YOLO JSON 里的 ID
        },
    ]
    
    for case in test_cases:
        if os.path.exists(case["img_path"]) and os.path.exists(case["label_path"]):
            visualize_image(case["img_path"], case["label_path"], case["img_id"])
        else:
            print(f"⚠️ 找不到图像或标签文件，请检查路径: {case['img_path']}")