import sys
import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# 确保能找到 src
sys.path.append("src") 

# 导入依赖
try:
    from monai.metrics import (
        compute_dice, compute_hausdorff_distance, 
        compute_average_surface_distance, compute_iou
    )
except ImportError:
    print("❌ 必须安装 monai: pip install monai")
    sys.exit(1)

# 尝试导入 thop (可选)
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

from dataset import TearDataset
from model import Baseline_SAM2

# ================= 配置区域 =================
IMG_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_HD95 = np.sqrt(IMG_SIZE**2 + IMG_SIZE**2) # 惩罚值
# ===========================================

def get_cross_modal_test_data(mode):
    """
    复用训练脚本的逻辑，但只返回【验证集/测试集】列表
    """
    all_data = []
    # 1. 收集所有数据
    for i in range(5):
        json_path = f"./data_splits/fold_{i}.json"
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                split = json.load(f)
                all_data.extend(split['val'])
    
    # 2. 分类
    color_data = [x for x in all_data if "Color" in (x['image'] if isinstance(x, dict) else x)]
    ir_data = [x for x in all_data if "Infrared" in (x['image'] if isinstance(x, dict) else x)]
    
    # 3. 根据模式返回对应的【测试集】
    # 训练脚本里：return train, val
    # 这里我们只需要 val
    if mode == "train_color_test_ir":
        print(f"🧐 Mode: {mode} | Test Set: Infrared ({len(ir_data)} images)")
        return ir_data
    elif mode == "train_ir_test_color":
        print(f"🧐 Mode: {mode} | Test Set: Color ({len(color_data)} images)")
        return color_data
    else:
        raise ValueError(f"Unknown mode: {mode}")

def calculate_metrics_robust(pred, lbl):
    """
    严谨的 SCI 级指标计算 (与 get_final_table_v2 保持一致)
    """
    results = {}
    
    # 1. Dice & IoU
    dice_score = compute_dice(pred, lbl, include_background=False).item()
    iou_score = compute_iou(pred, lbl, include_background=False).item()
    
    # 处理双空 (True Negative)
    if lbl.sum() == 0 and pred.sum() == 0:
        dice_score = 1.0
        iou_score = 1.0
    
    results['dice'] = dice_score
    results['iou'] = iou_score
    
    # 2. Precision & Recall
    tp = (pred * lbl).sum().item()
    fp = (pred * (1 - lbl)).sum().item()
    fn = ((1 - pred) * lbl).sum().item()
    
    results['recall'] = tp / (tp + fn + 1e-6)
    results['precision'] = tp / (tp + fp + 1e-6)
    
    # 3. HD95 & ASD
    if lbl.sum() > 0 and pred.sum() > 0:
        results['hd95'] = compute_hausdorff_distance(pred, lbl, include_background=False, percentile=95).item()
        results['asd'] = compute_average_surface_distance(pred, lbl, include_background=False).item()
    elif lbl.sum() > 0 and pred.sum() == 0:
        results['hd95'] = MAX_HD95 
        results['asd'] = MAX_HD95 / 2
    else:
        if pred.sum() == 0:
            results['hd95'] = 0.0
            results['asd'] = 0.0
        else:
            results['hd95'] = MAX_HD95
            results['asd'] = MAX_HD95

    return results

def evaluate_mode(mode):
    ckpt_path = f"./checkpoints_cross_modal/{mode}/best_model.pth"
    
    if not os.path.exists(ckpt_path):
        print(f"⚠️ Checkpoint not found: {ckpt_path}, skipping...")
        return None
    
    # 1. 准备数据
    test_list = get_cross_modal_test_data(mode)
    dataset = TearDataset(test_list, mode='val', img_size=IMG_SIZE)
    # BatchSize=1 保证指标计算准确
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. 加载模型
    model = Baseline_SAM2(checkpoint_path="./checkpoints/sam2_hiera_large.pt").to(DEVICE)
    
    # 3. 加载权重 (处理 DDP 前缀)
    try:
        state_dict = torch.load(ckpt_path, map_location=DEVICE)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") 
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

    model.eval()
    
    metrics_log = {'dice': [], 'iou': [], 'recall': [], 'precision': [], 'hd95': [], 'asd': []}
    
    print(f"🔄 Evaluating {mode} ...")
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            img = batch['image'].to(DEVICE)
            lbl = batch['label'].to(DEVICE)
            box = batch['box'].to(DEVICE)
            
            logits = model(img, box)
            pred = (torch.sigmoid(logits) > 0.5).float()
            
            batch_res = calculate_metrics_robust(pred.cpu(), lbl.cpu())
            for k, v in batch_res.items():
                metrics_log[k].append(v)
                
    # 返回平均值
    return {k: np.mean(v) for k, v in metrics_log.items()}

if __name__ == "__main__":
    print(f"\n🚀 ST-SAM 跨模态最终评估 (Cross-Modality Evaluation)")
    print(f"📌 Device: {DEVICE}")
    print("-" * 110)
    
    modes = ["train_color_test_ir", "train_ir_test_color"]
    
    headers = ["Exp Mode", "Dice", "IoU", "Recall", "Prec", "HD95", "ASD"]
    print(f"{' | '.join([f'{h:<20}' if i==0 else f'{h:<8}' for i, h in enumerate(headers)])}")
    print("-" * 110)
    
    for mode in modes:
        res = evaluate_mode(mode)
        if res:
            row = [
                f"{mode:<20}",
                f"{res['dice']:.4f}", f"{res['iou']:.4f}", 
                f"{res['recall']:.4f}", f"{res['precision']:.4f}",
                f"{res['hd95']:.2f}", f"{res['asd']:.2f}"
            ]
            print(" | ".join(row))
        else:
            print(f"{mode:<20} | ❌ Not Found / Error")
            
    print("-" * 110)
    print("✅ Done. Copy these rows to your paper's 'Generalization Analysis' table.")