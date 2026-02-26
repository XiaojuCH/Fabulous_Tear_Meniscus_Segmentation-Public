import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm
import sys
sys.path.append("src") # 确保能找到 dataset 和 model


from model import ST_SAM, Baseline_SAM2

# ======= 配置 =======
FOLD = 4  # 我们就去 Fold 4 (Infrared3) 里找
IMG_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_BASE = f"./checkpoints_ablation/fold_{FOLD}/best_model.pth"
CKPT_OURS = f"./checkpoints_gal50_bk/fold_{FOLD}/best_model.pth"                       
YOLO_JSON = f"./data_splits/yolo_boxes_fold{FOLD}.json"
SPLIT_JSON = f"./data_splits/fold_{FOLD}.json"

def compute_dice(pred_np, mask_np):
    inter = np.sum(pred_np * mask_np)
    return (2. * inter) / (np.sum(pred_np) + np.sum(mask_np) + 1e-6)

def load_model(model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
    model.eval()
    return model

print("⏳ 加载模型中...")
model_base = load_model(Baseline_SAM2(checkpoint_path="./checkpoints/sam2_hiera_large.pt").to(DEVICE), CKPT_BASE)
model_ours = load_model(ST_SAM(checkpoint_path="./checkpoints/sam2_hiera_large.pt").to(DEVICE), CKPT_OURS)

with open(YOLO_JSON, 'r') as f: yolo_preds = json.load(f)
with open(SPLIT_JSON, 'r') as f: split_data = json.load(f)

results = []

print("🔍 正在全自动扫描验证集，寻找最佳对比图...")
for item in tqdm(split_data['val']):
    img_id = item['id']
    if img_id not in yolo_preds: continue
        
    img_path = item['image']
    label_path = item['label'].replace("/Label/", "/Cleaned_Label/")
    
    # 读取数据
    image = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
    label = Image.open(label_path).convert("L").resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
    img_tensor = F.to_tensor(image).unsqueeze(0).to(DEVICE)
    lbl_np = (np.array(label) > 127).astype(np.uint8)
    
    # YOLO 框
    box_norm = yolo_preds[img_id]
    box = [box_norm[0]*IMG_SIZE, box_norm[1]*IMG_SIZE, box_norm[2]*IMG_SIZE, box_norm[3]*IMG_SIZE]
    box_tensor = torch.tensor([box], dtype=torch.float32).to(DEVICE)

    # 推理
    with torch.no_grad():
        pred_base = (torch.sigmoid(model_base(img_tensor, box_tensor)) > 0.5).cpu().numpy()[0, 0]
        pred_ours = (torch.sigmoid(model_ours(img_tensor, box_tensor)) > 0.5).cpu().numpy()[0, 0]
        
    dice_base = compute_dice(pred_base, lbl_np)
    dice_ours = compute_dice(pred_ours, lbl_np)
    
    # 计算差距 (我们希望找 Baseline 翻车，但 Ours 坚挺的图)
    gap = dice_ours - dice_base
    results.append({'id': img_id, 'path': img_path, 'lbl_path': label_path, 'base': dice_base, 'ours': dice_ours, 'gap': gap})

# 按 Gap 降序排序
results.sort(key=lambda x: x['gap'], reverse=True)

print("\n🏆 强烈建议使用以下图像放入 visualize.py 进行可视化：")
for i, res in enumerate(results[:10]):
    print(f"Top {i+1}: {res['id']} | Baseline Dice: {res['base']:.4f} | ST-SAM Dice: {res['ours']:.4f} | 差距: +{res['gap']:.4f}")