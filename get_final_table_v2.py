import sys
import os
import argparse
sys.path.append("src") # 确保能找到 dataset 和 model

import torch
import numpy as np
import json
import math
from tqdm import tqdm
from torch.utils.data import DataLoader

# 尝试导入 thop，没有也不要紧，只影响 FLOPs
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️ 未检测到 thop，将跳过 FLOPs 计算 (pip install thop)")

# 导入 MONAI 指标
try:
    from monai.metrics import (
        compute_dice, compute_hausdorff_distance, 
        compute_average_surface_distance, compute_iou
    )
except ImportError:
    print("❌ 必须安装 monai: pip install monai")
    sys.exit(1)

from dataset import TearDataset
from model import ST_SAM

# ================= 配置区域 =================
IMG_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# HD95 的最大惩罚值：设为图像对角线长度 (更科学)
MAX_HD95 = np.sqrt(IMG_SIZE**2 + IMG_SIZE**2) 
# ===========================================

def get_model_complexity(model):
    """
    精确计算 Params (Total & Tunable) 和 FLOPs
    """
    # 1. 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    tunable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 2. 计算 FLOPs (使用 dummy input)
    flops_g = 0.0
    if THOP_AVAILABLE:
        try:
            model.eval()
            # 构造 dummy input: 图像 [1, 3, 1024, 1024] + Box [1, 4]
            input_img = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
            input_box = torch.tensor([[0, 0, IMG_SIZE, IMG_SIZE]]).float().to(DEVICE)
            
            # thop 可能会因为 SAM2 内部结构复杂而报警，这是正常的
            flops, _ = profile(model, inputs=(input_img, input_box), verbose=False)
            flops_g = flops / 1e9
        except Exception as e:
            print(f"⚠️ FLOPs calculation failed: {e}")
    
    return total_params / 1e6, tunable_params / 1e6, flops_g

def calculate_metrics_robust(pred, lbl):
    """
    计算单张图像的指标，处理边缘情况
    pred: [1, 1, H, W] (0 or 1)
    lbl:  [1, 1, H, W] (0 or 1)
    """
    results = {}
    
    # 1. Dice & IoU (MONAI)
    # include_background=False 表示只计算前景类
    dice_score = compute_dice(pred, lbl, include_background=False).item()
    iou_score = compute_iou(pred, lbl, include_background=False).item()
    
    # 特殊情况修正：如果标签为空，预测也为空，Dice 理论上应为 1.0 (完全正确)
    # 但 compute_dice 默认可能给 0。这里根据医学分割惯例修正：
    if lbl.sum() == 0 and pred.sum() == 0:
        dice_score = 1.0
        iou_score = 1.0
    
    results['dice'] = dice_score
    results['iou'] = iou_score
    
    # 2. Precision & Recall (手算更稳)
    tp = (pred * lbl).sum().item()
    fp = (pred * (1 - lbl)).sum().item()
    fn = ((1 - pred) * lbl).sum().item()
    
    results['recall'] = tp / (tp + fn + 1e-6)
    results['precision'] = tp / (tp + fp + 1e-6)
    
    # 3. HD95 & ASD (距离指标)
    # 只有当 Pred 和 GT 都有前景时，距离才有意义
    if lbl.sum() > 0 and pred.sum() > 0:
        results['hd95'] = compute_hausdorff_distance(pred, lbl, include_background=False, percentile=95).item()
        results['asd'] = compute_average_surface_distance(pred, lbl, include_background=False).item()
    elif lbl.sum() > 0 and pred.sum() == 0:
        # 漏检：惩罚为最大距离
        results['hd95'] = MAX_HD95 
        results['asd'] = MAX_HD95 / 2 # 经验值，或者也设为 MAX
    else:
        # GT 为空 (无病灶)：
        # 如果预测也是空，距离为0；如果预测有东西，则距离很大
        if pred.sum() == 0:
            results['hd95'] = 0.0
            results['asd'] = 0.0
        else:
            results['hd95'] = MAX_HD95
            results['asd'] = MAX_HD95

    return results

def evaluate_fold(fold):
    print(f"🔄 Evaluating Fold {fold} ...")
    split_path = f"./data_splits/fold_{fold}.json"
    ckpt_path = f"./checkpoints/fold_{fold}/best_model.pth"
    
    if not os.path.exists(ckpt_path):
        print(f"⚠️ Checkpoint not found: {ckpt_path}, skipping Fold {fold}")
        return None
    
    # 加载数据
    with open(split_path, 'r') as f: data = json.load(f)
    
    # 🔥【修改这里】：把 YOLO 的 JSON 路径传进去！
    yolo_json_path = f"./data_splits/yolo_boxes_fold{fold}.json"
    dataset = TearDataset(data['val'], mode='val', img_size=IMG_SIZE, yolo_pred_json=yolo_json_path)
    
    # 验证集 BatchSize 必须为 1 以保证 Metric 计算准确
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    # 加载模型
    model = ST_SAM(checkpoint_path="./checkpoints/sam2_hiera_large.pt").to(DEVICE)
    
    # 加载权重 (处理 DDP 前缀)
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") # 去掉 DDP 产生的 module. 前缀
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    
    fold_metrics = {
        'Colour': {'dice': [], 'iou': [], 'recall': [], 'precision': [], 'hd95': [], 'asd': []},
        'Infrared': {'dice': [], 'iou': [], 'recall': [], 'precision': [], 'hd95': [], 'asd': []}
    }

    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc=f"Fold {fold}"):
            img = batch['image'].to(DEVICE)
            lbl = batch['label'].to(DEVICE)
            box = batch['box'].to(DEVICE)

            img_id = str(batch['id'][0]).lower()
            if 'colour' in img_id or 'color' in img_id:
                modality = 'Colour'
            else:
                modality = 'Infrared'

            logits = model(img, box)
            pred = (torch.sigmoid(logits) > 0.5).float()

            batch_res = calculate_metrics_robust(pred.cpu(), lbl.cpu())
            for k, v in batch_res.items():
                fold_metrics[modality][k].append(v)

    res_summary = {}
    for mod in ['Colour', 'Infrared']:
        if len(fold_metrics[mod]['dice']) > 0:
            res_summary[mod] = {k: np.mean(v) for k, v in fold_metrics[mod].items()}
    return res_summary

if __name__ == "__main__":
    print(f"\n🚀 ST-SAM 最终评估脚本 (SCI Mode)")
    print(f"📌 Device: {DEVICE} | Image Size: {IMG_SIZE}")
    print("-" * 100)
    
    # 1. 计算复杂度 (只算一次)
    print("🔹 Calculating Complexity...")
    try:
        # 初始化一个临时模型用于计算参数
        temp_model = ST_SAM(checkpoint_path="./checkpoints/sam2_hiera_large.pt").to(DEVICE)
        total_p, tunable_p, flops = get_model_complexity(temp_model)
        del temp_model # 释放显存
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"⚠️ Complexity Error: {e}")
        total_p, tunable_p, flops = 0, 0, 0

    # 2. 循环评估 5 Folds
    headers = ["Fold", "Dice", "IoU", "Recall", "Prec", "HD95", "ASD"]
    global_metrics = {
        'Colour':   {'dice': [], 'iou': [], 'recall': [], 'precision': [], 'hd95': [], 'asd': []},
        'Infrared': {'dice': [], 'iou': [], 'recall': [], 'precision': [], 'hd95': [], 'asd': []},
        'Overall':  {'dice': [], 'iou': [], 'recall': [], 'precision': [], 'hd95': [], 'asd': []}
    }

    for fold in [0, 1, 2, 3, 4]:
        res_summary = evaluate_fold(fold)
        if not res_summary:
            continue

        print(f"Fold {fold} Results:")
        for mod, metrics in res_summary.items():
            row = [
                f"{mod[:4]:<8}",
                f"{metrics['dice']:.4f}", f"{metrics['iou']:.4f}",
                f"{metrics['recall']:.4f}", f"{metrics['precision']:.4f}",
                f"{metrics['hd95']:.2f}", f"{metrics['asd']:.2f}"
            ]
            print(" | ".join(row))
            for k, v in metrics.items():
                global_metrics[mod][k].append(v)
                global_metrics['Overall'][k].append(v)

    # 3. 输出最终汇总 (Mean ± Std)
    if len(global_metrics['Overall']['dice']) > 0:
        print("-" * 100)
        print("🏆 ST-SAM Modality-Aware Final Results:")
        print("-" * 100)

        for category in ['Colour', 'Infrared', 'Overall']:
            if len(global_metrics[category]['dice']) == 0:
                continue
            print(f"\n--- {category} Set ---")
            for k in headers[1:]:
                key = k.lower() if k != "Prec" else "precision"
                vals = global_metrics[category][key]
                print(f"  ● {k:<10}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        print("-" * 100)
        print("📉 Model Efficiency (Paper Claims):")
        print(f"  ● Total Params   : {total_p:.2f} M")
        print(f"  ● Tunable Params : {tunable_p:.2f} M  <-- (重点: PEFT)")
        print(f"  ● GFLOPs         : {flops:.2f} G")
        print("=" * 100)
    else:
        print("❌ No results generated. Check checkpoints.")