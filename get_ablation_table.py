import sys
import os
import math
sys.path.append("src")

import torch
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

try:
    from thop import profile
except ImportError:
    print("❌ 请先安装 thop: pip install thop")
    sys.exit(1)

from monai.metrics import (
    compute_dice, compute_hausdorff_distance, 
    compute_average_surface_distance, compute_iou
)
from dataset import TearDataset
from model import Baseline_SAM2

IMG_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_HD95 = np.sqrt(IMG_SIZE**2 + IMG_SIZE**2) # 对齐最终评估脚本的严谨惩罚

def get_complexity():
    model = Baseline_SAM2(checkpoint_path="./checkpoints/sam2_hiera_large.pt").to(DEVICE)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    input_img = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    input_box = torch.tensor([[0, 0, IMG_SIZE, IMG_SIZE]]).float().to(DEVICE)

    flops, _ = profile(model, inputs=(input_img, input_box), verbose=False)

    return flops / 1e9, total_params, trainable_params

def calculate_metrics_robust(pred, lbl):
    """【修复】完全对齐 get_final_table_v2 的严谨算法"""
    results = {}
    dice_score = compute_dice(pred, lbl, include_background=False).item()
    iou_score = compute_iou(pred, lbl, include_background=False).item()
    
    if lbl.sum() == 0 and pred.sum() == 0:
        dice_score = 1.0; iou_score = 1.0
    
    results['dice'] = dice_score
    results['iou'] = iou_score
    
    tp = (pred * lbl).sum().item()
    fp = (pred * (1 - lbl)).sum().item()
    fn = ((1 - pred) * lbl).sum().item()
    
    results['recall'] = tp / (tp + fn + 1e-6)
    results['precision'] = tp / (tp + fp + 1e-6)
    
    if lbl.sum() > 0 and pred.sum() > 0:
        results['hd95'] = compute_hausdorff_distance(pred, lbl, include_background=False, percentile=95).item()
        results['asd'] = compute_average_surface_distance(pred, lbl, include_background=False).item()
    elif lbl.sum() > 0 and pred.sum() == 0:
        results['hd95'] = MAX_HD95 
        results['asd'] = MAX_HD95 / 2 
    else:
        if pred.sum() == 0:
            results['hd95'] = 0.0; results['asd'] = 0.0
        else:
            results['hd95'] = MAX_HD95; results['asd'] = MAX_HD95
    return results

def evaluate_fold(fold):
    split_path = f"./data_splits/fold_{fold}.json"
    
    # 🔥【修复】强制加载 YOLO 预测框！保证绝对公平
    yolo_json_path = f"./data_splits/yolo_boxes_fold{fold}.json"
    
    with open(split_path, 'r') as f: data = json.load(f)
    
    dataset = TearDataset(data['val'], mode='val', img_size=IMG_SIZE, yolo_pred_json=yolo_json_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    model = Baseline_SAM2(checkpoint_path="./checkpoints/sam2_hiera_large.pt").to(DEVICE)
    ckpt = f"./checkpoints_ablation/fold_{fold}/best_model.pth"
    
    if not os.path.exists(ckpt):
        print(f"⚠️ Checkpoint missing: {ckpt}")
        return None

    state_dict = torch.load(ckpt, map_location=DEVICE)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    
    fold_metrics = {
        'Colour':   {'dice': [], 'iou': [], 'recall': [], 'precision': [], 'hd95': [], 'asd': []},
        'Infrared': {'dice': [], 'iou': [], 'recall': [], 'precision': [], 'hd95': [], 'asd': []}
    }

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            img = batch['image'].to(DEVICE)
            lbl = batch['label'].to(DEVICE)
            box = batch['box'].to(DEVICE)

            img_id = str(batch['id'][0]).lower()
            modality = 'Colour' if ('colour' in img_id or 'color' in img_id) else 'Infrared'

            pred = (torch.sigmoid(model(img, box)) > 0.5).float()
            batch_res = calculate_metrics_robust(pred.cpu(), lbl.cpu())
            for k, v in batch_res.items():
                fold_metrics[modality][k].append(v)

    res_summary = {}
    for mod in ['Colour', 'Infrared']:
        if len(fold_metrics[mod]['dice']) > 0:
            res_summary[mod] = {k: np.mean(v) for k, v in fold_metrics[mod].items()}
    return res_summary

if __name__ == "__main__":
    print(f"\n🚀 Baseline SAM 2 (Ablation - Full Auto with YOLO boxes) 评估...")

    try:
        flops, total_params, trainable_params = get_complexity()
        print(f"🔹 Complexity:")
        print(f"   ● Total Params     : {total_params:.2f} M")
        print(f"   ● Trainable Params : {trainable_params:.2f} M")
        print(f"   ● FLOPs            : {flops:.2f} G")
    except Exception as e:
        print(f"🔹 Complexity 计算失败: {e}")
        flops, total_params, trainable_params = 0, 0, 0

    headers = ["Fold", "Dice", "IoU", "Recall", "Prec", "HD95", "ASD"]
    global_metrics = {
        'Colour':   {'dice': [], 'iou': [], 'recall': [], 'precision': [], 'hd95': [], 'asd': []},
        'Infrared': {'dice': [], 'iou': [], 'recall': [], 'precision': [], 'hd95': [], 'asd': []},
        'Overall':  {'dice': [], 'iou': [], 'recall': [], 'precision': [], 'hd95': [], 'asd': []}
    }

    for fold in [0, 1, 2, 3, 4]:
        try:
            res_summary = evaluate_fold(fold)
            if not res_summary:
                continue
            print(f"Fold {fold} Results:")
            for mod, metrics in res_summary.items():
                row = [f"{mod[:4]:<8}", f"{metrics['dice']:.4f}", f"{metrics['iou']:.4f}",
                       f"{metrics['recall']:.4f}", f"{metrics['precision']:.4f}",
                       f"{metrics['hd95']:.2f}", f"{metrics['asd']:.2f}"]
                print(" | ".join(row))
                for k, v in metrics.items():
                    global_metrics[mod][k].append(v)
                    global_metrics['Overall'][k].append(v)
        except Exception as e:
            print(f"Fold {fold} Error: {e}")

    if len(global_metrics['Overall']['dice']) > 0:
        print("-" * 90)
        print("🏆 Baseline SAM 2 Modality-Aware Final Results:")
        print("-" * 90)
        for category in ['Colour', 'Infrared', 'Overall']:
            if len(global_metrics[category]['dice']) == 0:
                continue
            print(f"\n--- {category} Set ---")
            for k in headers[1:]:
                key = k.lower() if k != "Prec" else "precision"
                vals = global_metrics[category][key]
                print(f"  ● {k:<10}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print("=" * 90)