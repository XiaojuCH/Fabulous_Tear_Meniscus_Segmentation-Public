import os
import json
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append("src") # 确保能找到 dataset 和 model

# === 引入 MONAI 的官方严谨指标 ===
from monai.metrics import compute_dice, compute_hausdorff_distance

from dataset import TearDataset
from model import ST_SAM, Baseline_SAM2, MSA_Baseline_SAM2

IMG_SIZE = 1024
MAX_HD95 = math.sqrt(IMG_SIZE**2 + IMG_SIZE**2)

# ================= 极其严谨的指标计算 (对齐 get_final_table_v2) =================
def compute_metrics_monai(pred_tensor, gt_tensor):
    """
    计算单个样本的 Dice 和 HD95
    输入必须是 [1, 1, H, W] 的 Tensor
    """
    if pred_tensor.dim() == 3:
        pred_tensor = pred_tensor.unsqueeze(0)
    if gt_tensor.dim() == 3:
        gt_tensor = gt_tensor.unsqueeze(0)
        
    pred = (pred_tensor > 0.5).float()
    lbl = (gt_tensor > 0.5).float()
    
    # 1. Dice
    if lbl.sum() == 0 and pred.sum() == 0:
        dice_score = 1.0
    else:
        dice_score = compute_dice(pred, lbl, include_background=False).item()
        if math.isnan(dice_score):
            dice_score = 0.0

    # 2. HD95
    if lbl.sum() > 0 and pred.sum() > 0:
        hd95 = compute_hausdorff_distance(pred, lbl, include_background=False, percentile=95).item()
        if math.isnan(hd95):
            hd95 = MAX_HD95
    elif lbl.sum() > 0 and pred.sum() == 0:
        hd95 = MAX_HD95 
    else:
        if pred.sum() == 0:
            hd95 = 0.0
        else:
            hd95 = MAX_HD95

    return dice_score, hd95

# ================= 动态修改 Box 的 Dataset 包装器 =================
class RobustnessDataset(TearDataset):
    def __init__(self, data_list, img_size, padding=0):
        super().__init__(data_list, mode="val", img_size=img_size, yolo_pred_json=None)
        self.padding = padding

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        label_np = item['label'].squeeze().numpy()
        
        y_indices, x_indices = np.where(label_np > 0)
        if len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
        else:
            x_min, y_min, x_max, y_max = 0, 0, self.img_size, self.img_size

        p = self.padding
        x1 = max(0, x_min - p)
        y1 = max(0, y_min - p)
        x2 = min(self.img_size, x_max + p)
        y2 = min(self.img_size, y_max + p)
        
        item['box'] = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        return item

# ================= 主评估流程 =================
def run_robustness_test(fold_json_path, checkpoints_dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with open(fold_json_path, 'r') as f:
        split_data = json.load(f)
        
    paddings = [-5, 0, 5, 10, 20, 30, 40] 
    
    results = {
        "ST-SAM": {"dice": [], "hd95": []},
        "MSA_SAM2": {"dice": [], "hd95": []},
        "Baseline_SAM2": {"dice": [], "hd95": []}
    }

    models = {
        "ST-SAM": ST_SAM().to(device),
        "MSA_SAM2": MSA_Baseline_SAM2().to(device),
        "Baseline_SAM2": Baseline_SAM2().to(device)
    }
    
    for name, model in models.items():
        if os.path.exists(checkpoints_dict[name]):
            state_dict = torch.load(checkpoints_dict[name], map_location=device, weights_only=True)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        else:
            print(f"⚠️ 警告: 找不到 {name} 的权重文件: {checkpoints_dict[name]}")
        model.eval()

    for p in paddings:
        print(f"\n📏 评估 Padding = {p} 像素 (MONAI严谨版)...")
        val_dataset = RobustnessDataset(split_data['val'], img_size=IMG_SIZE, padding=p)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
        
        for model_name, model in models.items():
            epoch_dice, epoch_hd95 = 0.0, 0.0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=model_name, leave=False):
                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)
                    boxes = batch['box'].to(device)
                    
                    logits = model(images, boxes)
                    preds = torch.sigmoid(logits)
                    
                    for b in range(images.size(0)):
                        d, h = compute_metrics_monai(preds[b:b+1].cpu(), labels[b:b+1].cpu())
                        epoch_dice += d
                        epoch_hd95 += h
                        
            num_samples = len(val_dataset)
            results[model_name]["dice"].append(epoch_dice / num_samples)
            results[model_name]["hd95"].append(epoch_hd95 / num_samples)

    # ================= 绘图 =================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"ST-SAM": "#D62728", "MSA_SAM2": "#1F77B4", "Baseline_SAM2": "#7F7F7F"}
    markers = {"ST-SAM": "o", "MSA_SAM2": "s", "Baseline_SAM2": "^"}
    
    for model_name in results.keys():
        ax1.plot(paddings, results[model_name]["dice"], label=model_name, 
                 color=colors[model_name], marker=markers[model_name], linewidth=2.5)
        ax2.plot(paddings, results[model_name]["hd95"], label=model_name, 
                 color=colors[model_name], marker=markers[model_name], linewidth=2.5)

    ax1.set_title("Dice vs. Box Expansion (Noise Tolerance)", fontweight='bold')
    ax1.set_xlabel("Box Expansion (Pixels)")
    ax1.set_ylabel("Dice Score")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    ax2.set_title("HD95 vs. Box Expansion (Boundary Drift)", fontweight='bold')
    ax2.set_xlabel("Box Expansion (Pixels)")
    ax2.set_ylabel("HD95 (Pixels)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("box_robustness_curve_monai.pdf", dpi=300)
    print("✅ 严谨版鲁棒性图表已保存为 box_robustness_curve_monai.pdf！")

if __name__ == "__main__":
    checkpoints = {
        "ST-SAM": "./checkpoints_gal1/fold_0/best_model.pth",
        "MSA_SAM2": "./checkpoints_msa/fold_0/best_model.pth",
        "Baseline_SAM2": "./checkpoints_ablation/fold_0/best_model.pth"
    }
    run_robustness_test("./data_splits/fold_0.json", checkpoints)