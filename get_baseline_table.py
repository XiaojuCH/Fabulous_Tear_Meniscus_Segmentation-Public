# -*- coding: utf-8 -*-
import sys
import os
sys.path.append("src") # 确保能找到 dataset.py
import torch
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

# 引入计算库
try:
    from thop import profile
except ImportError:
    print("❌ 错误: 请先安装 thop 库: pip install thop")
    sys.exit(1)

from monai.metrics import (
    compute_dice, 
    compute_hausdorff_distance, 
    compute_average_surface_distance,
    compute_iou
)
from monai.networks.nets import UNet, SwinUNETR
from dataset import TearDataset
from monai.networks.nets import UNet, SwinUNETR, AttentionUnet, SegResNet

# 配置
IMG_SIZE = 1024
# 推理时使用 GPU，但计算 FLOPs 时建议用 CPU 防止爆显存
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_model(name):
    if name == "unet":
        return UNet(
            spatial_dims=2, in_channels=3, out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2), num_res_units=2,
        )
    elif name == "swinunet":
        return SwinUNETR(
            in_channels=3, out_channels=1,
            feature_size=24, spatial_dims=2,
            use_v2=True,       # 建议开启 SwinV2，更稳
            window_size=8      # 🔥【关键修复】改成 8 完美适配 1024 分辨率
        )
    # === 新增模型 1: Attention U-Net ===
    elif name == "attentionunet":
        return AttentionUnet(
            spatial_dims=2, in_channels=3, out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
        )
    # === 新增模型 2: SegResNet (NVIDIA强力模型) ===
    elif name == "segresnet":
        return SegResNet(
            spatial_dims=2, in_channels=3, out_channels=1,
            init_filters=32, blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1]
        )
    else:
        raise ValueError(f"Unknown model: {name}")

def get_complexity(model_name):
    """计算参数量和 FLOPs (强制使用 CPU 以免 1024x1024 爆显存)"""
    print(f"⏳ 正在计算 {model_name} 的复杂度 (CPU模式)...")
    model = get_model(model_name).to("cpu") # 强制 CPU
    model.eval()
    
    # 创建一个 dummy 输入 (1, 3, 1024, 1024)
    input_tensor = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to("cpu")
    
    # 计算 FLOPs 和 Params
    try:
        flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
    except Exception as e:
        print(f"⚠️ FLOPs 计算出错: {e}")
        return 0, 0
    
    # 转换为 G (Giga) 和 M (Million)
    flops_g = flops / 1e9
    params_m = params / 1e6
    
    return flops_g, params_m

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

def evaluate_fold(model_name, fold):
    split_path = f"./data_splits/fold_{fold}.json"
    # 兼容性检查：如果文件不存在
    if not os.path.exists(split_path):
        print(f"⚠️ 找不到数据切分文件: {split_path}")
        return None

    with open(split_path, 'r') as f: data = json.load(f)
    
    dataset = TearDataset(data['val'], mode='val', img_size=IMG_SIZE)
    # 验证集 batch_size=1 是最安全的
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    model = get_model(model_name).to(DEVICE)
    ckpt_path = f"./checkpoints_baseline/{model_name}/fold_{fold}/best_model.pth"
    
    if not os.path.exists(ckpt_path):
        print(f"⚠️ [Fold {fold}] 没找到权重: {ckpt_path}")
        return None

    # --- 修正后的稳健加载逻辑 ---
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除 'module.' 前缀 (针对 DataParallel 保存的模型)
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    # strict=True 能帮你检查权重是否真的匹配，如果有 key 不匹配会直接报错提醒你
    # 如果你确定有些层不需要加载，可以改回 strict=False
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except Exception as e:
        print(f"⚠️ 权重加载有轻微不匹配，尝试 strict=False 加载... ({e})")
        model.load_state_dict(new_state_dict, strict=False)
        
    model.eval()
    
    metrics_log = {'dice': [], 'iou': [], 'recall': [], 'precision': [], 'hd95': [], 'asd': []}
    
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc=f"Fold {fold}"):
            img, lbl = batch['image'].to(DEVICE), batch['label'].to(DEVICE)
            logits = model(img)
            pred = (torch.sigmoid(logits) > 0.5).float()
            pred, lbl = pred.cpu(), lbl.cpu()
            batch_res = calculate_metrics(pred, lbl)
            for k, v in batch_res.items(): metrics_log[k].append(v)
                
    return {k: np.mean(v) for k, v in metrics_log.items()}

def main():
    print("🚀 开始 Baseline 统计 (含 Params & FLOPs)...")
    
    for model_name in ["attentionunet"]:
        print(f"\n{'='*90}")
        print(f"📋 Model: {model_name.upper()}")
        
        # 1. 计算复杂度 (CPU)
        flops, params = get_complexity(model_name)
        print(f"🔍 Complexity: Params = {params:.2f} M | FLOPs = {flops:.2f} G")

        # 2. 打印表头
        headers = ["Fold", "Dice", "IoU", "Recall", "Prec", "HD95", "ASD"]
        header_str = " | ".join([f"{h:<8}" for h in headers])
        print("-" * 90)
        print(header_str)
        print("-" * 90)
        
        all_folds_metrics = {'dice': [], 'iou': [], 'recall': [], 'precision': [], 'hd95': [], 'asd': []}
        
        for fold in range(5):
            res = evaluate_fold(model_name, fold)
            if res is not None:
                for k, v in res.items(): all_folds_metrics[k].append(v)
                row_str = f"{fold:<8} | {res['dice']:.4f}   | {res['iou']:.4f}   | {res['recall']:.4f}   | {res['precision']:.4f} | {res['hd95']:.4f}   | {res['asd']:.4f}"
                print(row_str)
        
        if len(all_folds_metrics['dice']) > 0:
            print("-" * 90)
            print(f"🏆 {model_name.upper()} Final Average:")
            for k in headers[1:]:
                k_lower = k.lower() if k != "Prec" else "precision"
                avg = np.mean(all_folds_metrics[k_lower])
                std = np.std(all_folds_metrics[k_lower])
                print(f"   {k:<8}: {avg:.4f} ± {std:.4f}")
            print(f"   Params  : {params:.2f} M")
            print(f"   FLOPs   : {flops:.2f} G")
        else:
            print("❌ 未能生成任何有效数据 (可能是 Checkpoint 路径不对)")

if __name__ == "__main__":
    main()