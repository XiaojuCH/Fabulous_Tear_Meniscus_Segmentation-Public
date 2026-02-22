import sys
import os
import argparse
sys.path.append("src") 

import torch
import numpy as np
import json
import math
from tqdm import tqdm
from torch.utils.data import DataLoader

# 引入计算库
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️ 未安装 thop，将跳过 FLOPs 计算")

from monai.metrics import (
    compute_dice, compute_hausdorff_distance, 
    compute_average_surface_distance, compute_iou
)
# 引入所有 Baseline 模型
from monai.networks.nets import UNet, SwinUNETR, AttentionUnet, SegResNet, BasicUNetPlusPlus
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights, fcn_resnet50, FCN_ResNet50_Weights
from dataset import TearDataset

# ================= 配置区域 (必须与 ST-SAM 一致) =================
IMG_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# HD95 惩罚值 (对角线长度)
MAX_HD95 = np.sqrt(IMG_SIZE**2 + IMG_SIZE**2) 
# ==============================================================

def get_model(name):
    name = name.lower()
    if name == "unet":
        return UNet(
            spatial_dims=2, in_channels=3, out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2), num_res_units=2,
        )
    elif name == "swinunet":
        return SwinUNETR(
            in_channels=3, out_channels=1,
            feature_size=48, spatial_dims=2,
            use_v2=True,
            window_size=8      # 适配 1024 (1024/32=32, 32%8=0)
        )
    elif name == "attentionunet":
        return AttentionUnet(
            spatial_dims=2, in_channels=3, out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
        )
    elif name == "segresnet":
        return SegResNet(
            spatial_dims=2, in_channels=3, out_channels=1,
            init_filters=32, blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1]
        )
    elif name == "unetplusplus":
        return BasicUNetPlusPlus(
            spatial_dims=2, in_channels=3, out_channels=1,
            features=(16, 32, 64, 128, 256, 256),
            deep_supervision=False
        )
    elif name == "deeplab":
        model = deeplabv3_resnet50(weights=None, num_classes=1)
        model.backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif name == "deeplab_p":
        m = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        m.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1))
        m.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1))
        return m
    elif name == "fcn":
        m = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
        m.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1, 1))
        m.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1))
        return m
    else:
        raise ValueError(f"Unknown model: {name}")

def get_complexity(model_name):
    """计算 Params 和 FLOPs"""
    try:
        model = get_model(model_name).to(DEVICE)
        model.eval()
        input_tensor = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        
        if THOP_AVAILABLE:
            flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
            return flops / 1e9, params / 1e6
        else:
            return 0, 0
    except Exception as e:
        print(f"⚠️ {model_name} FLOPs 计算失败: {e}")
        return 0, 0

def calculate_metrics_robust(pred, lbl):
    """
    【核心】必须与 ST-SAM 的计算逻辑完全一致！
    """
    results = {}
    
    # 1. Dice & IoU
    dice_score = compute_dice(pred, lbl, include_background=False).item()
    iou_score = compute_iou(pred, lbl, include_background=False).item()
    
    # 修正全黑情况 (Empty GT & Empty Pred)
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
    
    # 3. HD95 & ASD (带惩罚)
    if lbl.sum() > 0 and pred.sum() > 0:
        results['hd95'] = compute_hausdorff_distance(pred, lbl, include_background=False, percentile=95).item()
        results['asd'] = compute_average_surface_distance(pred, lbl, include_background=False).item()
    elif lbl.sum() > 0 and pred.sum() == 0:
        # 漏检惩罚
        results['hd95'] = MAX_HD95 
        results['asd'] = MAX_HD95 / 2 
    else:
        # GT为空
        if pred.sum() == 0:
            results['hd95'] = 0.0; results['asd'] = 0.0
        else:
            results['hd95'] = MAX_HD95; results['asd'] = MAX_HD95

    return results

def evaluate_fold(model_name, fold):
    split_path = f"./data_splits/fold_{fold}.json"
    if not os.path.exists(split_path): return None

    with open(split_path, 'r') as f: data = json.load(f)
    
    # BatchSize=1 保证计算准确
    dataset = TearDataset(data['val'], mode='val', img_size=IMG_SIZE)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    model = get_model(model_name).to(DEVICE)
    
    # 路径检查
    ckpt_path = f"./checkpoints_New_baseline/{model_name}/fold_{fold}/best_model.pth"
    if not os.path.exists(ckpt_path):
        # 尝试另一种常见命名格式 (防止文件夹命名不一致)
        ckpt_path = f"./checkpoints/{model_name}/fold_{fold}/best_model.pth"
        if not os.path.exists(ckpt_path):
            print(f"⚠️ [Fold {fold}] 权重缺失: {ckpt_path}")
            return None

    # 加载权重
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."): new_state_dict[k[7:]] = v
        else: new_state_dict[k] = v
            
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except:
        model.load_state_dict(new_state_dict, strict=False)
        
    model.eval()
    
    metrics_log = {'dice': [], 'iou': [], 'recall': [], 'precision': [], 'hd95': [], 'asd': []}
    
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc=f"Eval {model_name} F{fold}"):
            img, lbl = batch['image'].to(DEVICE), batch['label'].to(DEVICE)
            output = model(img)

            # 处理不同模型的输出格式
            if isinstance(output, dict) and 'out' in output:
                logits = output['out']  # DeepLab
            elif isinstance(output, list):
                logits = output[0]  # UNet++
            else:
                logits = output

            pred = (torch.sigmoid(logits) > 0.5).float()

            # 转 CPU 计算
            pred, lbl = pred.cpu(), lbl.cpu()
            batch_res = calculate_metrics_robust(pred, lbl)

            for k, v in batch_res.items(): metrics_log[k].append(v)
                
    return {k: np.mean(v) for k, v in metrics_log.items()}

if __name__ == "__main__":
    print("🚀 Baseline 批量评估脚本 (SCI Mode)")
    print(f"📌 Device: {DEVICE} | HD95 Penalty: {MAX_HD95:.2f}")
    
    # 这里列出你想跑的所有 Baseline
    # models_to_run = ["unet"]
    # models_to_run = ["attentionunet"]
    # models_to_run = ["unet","swinunet"]
    # models_to_run = ["segresnet"]
    # models_to_run = ["deeplab_p"]
    models_to_run = ["fcn"]
    #models_to_run = ["unet", "attentionunet", "swinunet", "segresnet"]
    # 如果只想跑某一个，可以注释掉其他的，比如:
    # models_to_run = ["attentionunet"]

    for model_name in models_to_run:
        print(f"\n{'='*90}")
        print(f"📋 Processing Model: {model_name.upper()}")
        
        # 1. 计算复杂度
        flops, params = get_complexity(model_name)
        
        # 2. 跑 5 折交叉验证
        headers = ["Fold", "Dice", "IoU", "Recall", "Prec", "HD95", "ASD"]
        print("-" * 90)
        print(" | ".join([f"{h:<8}" for h in headers]))
        print("-" * 90)
        
        all_folds_metrics = {'dice': [], 'iou': [], 'recall': [], 'precision': [], 'hd95': [], 'asd': []}
        
        for fold in range(5):
            res = evaluate_fold(model_name, fold)
            if res:
                for k, v in res.items(): all_folds_metrics[k].append(v)
                print(f"{fold:<8} | {res['dice']:.4f}   | {res['iou']:.4f}   | {res['recall']:.4f}   | {res['precision']:.4f} | {res['hd95']:.4f}   | {res['asd']:.4f}")
        
        # 3. 汇总输出
        if len(all_folds_metrics['dice']) > 0:
            print("-" * 90)
            print(f"🏆 {model_name.upper()} Final Average:")
            for k in headers[1:]:
                k_lower = k.lower() if k != "Prec" else "precision"
                avg = np.mean(all_folds_metrics[k_lower])
                std = np.std(all_folds_metrics[k_lower])
                print(f"   ● {k:<10}: {avg:.4f} ± {std:.4f}")
            print(f"   ● Params    : {params:.2f} M")
            print(f"   ● FLOPs     : {flops:.2f} G")
        else:
            print(f"❌ {model_name} 没有产生有效结果，请检查 checkpoints_baseline 路径。")