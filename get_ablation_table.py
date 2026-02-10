import sys
import os
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
# 【修改 1】导入 Baseline_SAM2
from model import Baseline_SAM2

IMG_SIZE = 1024
DEVICE = "cuda"

def get_complexity():
    """计算 Baseline SAM 2 的 Params 和 FLOPs"""
    # 【修改 2】实例化 Baseline_SAM2
    model = Baseline_SAM2(checkpoint_path="./checkpoints/sam2_hiera_large.pt").to(DEVICE)
    model.eval()
    
    input_img = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    input_box = torch.tensor([[0, 0, IMG_SIZE, IMG_SIZE]]).float().to(DEVICE)
    
    flops, params = profile(model, inputs=(input_img, input_box), verbose=False)
    
    return flops / 1e9, params / 1e6

def calculate_metrics(pred, lbl):
    results = {}
    results['dice'] = compute_dice(pred, lbl, include_background=False).item()
    results['iou'] = compute_iou(pred, lbl, include_background=False).item()
    
    tp = (pred * lbl).sum().item()
    fp = (pred * (1 - lbl)).sum().item()
    fn = ((1 - pred) * lbl).sum().item()
    
    results['recall'] = tp / (tp + fn + 1e-6)
    results['precision'] = tp / (tp + fp + 1e-6)
    
    if lbl.sum() > 0 and pred.sum() > 0:
        results['hd95'] = compute_hausdorff_distance(pred, lbl, include_background=False, percentile=95).item()
        results['asd'] = compute_average_surface_distance(pred, lbl, include_background=False).item()
    elif lbl.sum() > 0:
        results['hd95'] = 100.0; results['asd'] = 50.0
    else:
        results['hd95'] = 0.0; results['asd'] = 0.0
    return results

def evaluate_fold(fold):
    split_path = f"./data_splits/fold_{fold}.json"
    with open(split_path, 'r') as f: data = json.load(f)
    
    dataset = TearDataset(data['val'], mode='val', img_size=IMG_SIZE)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 【修改 3】实例化 Baseline_SAM2
    model = Baseline_SAM2(checkpoint_path="./checkpoints/sam2_hiera_large.pt").to(DEVICE)
    
    # 【修改 4】指向 checkpoints_ablation
    ckpt = f"./checkpoints_ablation/fold_{fold}/best_model.pth"
    
    if not os.path.exists(ckpt):
        print(f"⚠️ Checkpoint missing: {ckpt}")
        return None

    state_dict = torch.load(ckpt, map_location=DEVICE)
    # 处理 DDP 前缀
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    
    metrics_log = {'dice': [], 'iou': [], 'recall': [], 'precision': [], 'hd95': [], 'asd': []}
    
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            img = batch['image'].to(DEVICE)
            lbl = batch['label'].to(DEVICE)
            box = batch['box'].to(DEVICE)
            
            # 预测
            pred = (torch.sigmoid(model(img, box)) > 0.5).float()
            pred, lbl = pred.cpu(), lbl.cpu()
            
            batch_res = calculate_metrics(pred, lbl)
            for k, v in batch_res.items(): metrics_log[k].append(v)
                
    return {k: np.mean(v) for k, v in metrics_log.items()}

if __name__ == "__main__":
    print(f"\n🚀 Baseline SAM 2 (Ablation) 评估...")
    
    try:
        flops, params = get_complexity()
        print(f"🔹 Complexity: Params = {params:.2f} M | FLOPs = {flops:.2f} G")
    except Exception as e:
        print(f"🔹 Complexity 计算失败: {e}")
        flops, params = 0, 0

    headers = ["Fold", "Dice", "IoU", "Recall", "Prec", "HD95", "ASD"]
    header_str = " | ".join([f"{h:<8}" for h in headers])
    print("-" * 90)
    print(header_str)
    print("-" * 90)
    
    final_results = {'dice': [], 'iou': [], 'recall': [], 'precision': [], 'hd95': [], 'asd': []}
    
    # 【修改 5】只跑 Fold 0 即可，如果是跑了所有 fold 就改成 [0,1,2,3,4]
    folds_to_eval = [0,1,2,3,4] 
    
    for fold in folds_to_eval:
        try:
            res = evaluate_fold(fold)
            if res:
                row_str = f"{fold:<8} | {res['dice']:.4f}   | {res['iou']:.4f}   | {res['recall']:.4f}   | {res['precision']:.4f} | {res['hd95']:.4f}   | {res['asd']:.4f}"
                print(row_str)
                for k, v in res.items(): final_results[k].append(v)
        except Exception as e:
            print(f"Fold {fold} Error: {e}")
            
    if len(final_results['dice']) > 0:
        print("-" * 90)
        print("🏆 Baseline SAM 2 Final Average:")
        for k in headers[1:]:
            k_lower = k.lower() if k != "Prec" else "precision"
            avg = np.mean(final_results[k_lower])
            std = np.std(final_results[k_lower])
            print(f"   {k:<8}: {avg:.4f} ± {std:.4f}")
    print("=" * 90)