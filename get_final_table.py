import sys
import os

# 【修复】将 src 目录加入系统路径，这样才能找到 dataset 和 model
sys.path.append("src")

import torch
import numpy as np
import json
from monai.metrics import compute_hausdorff_distance, compute_dice
from torch.utils.data import DataLoader
from tqdm import tqdm

# 现在可以正常导入了
from dataset import TearDataset
from model import ST_SAM

IMG_SIZE = 1024
DEVICE = "cuda"

def evaluate_fold(fold):
    print(f"Running Fold {fold} inference...")
    split_path = f"./data_splits/fold_{fold}.json"
    with open(split_path, 'r') as f:
        data = json.load(f)
    
    # 这里的 Val 其实就是 LOCO 的 Test set
    dataset = TearDataset(data['val'], mode='val', img_size=IMG_SIZE)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    model = ST_SAM(checkpoint_path="./checkpoints/sam2_hiera_large.pt").to(DEVICE)
    ckpt = f"./checkpoints/fold_{fold}/best_model.pth"
    
    if not os.path.exists(ckpt):
        print(f"❌ Warning: Checkpoint not found: {ckpt}")
        return 0.0, 100.0

    # 加载权重
    state_dict = torch.load(ckpt, map_location=DEVICE)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    
    dice_list, hd95_list = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            img = batch['image'].to(DEVICE)
            lbl = batch['label'].to(DEVICE)
            box = batch['box'].to(DEVICE)
            
            # 预测
            pred = (torch.sigmoid(model(img, box)) > 0.5).float()
            
            # 强制 CPU 计算指标 (避开 cupy 问题)
            pred, lbl = pred.cpu(), lbl.cpu()
            
            d = compute_dice(pred, lbl, include_background=False).item()
            dice_list.append(d)
            
            if lbl.sum() > 0 and pred.sum() > 0:
                h = compute_hausdorff_distance(pred, lbl, include_background=False, percentile=95).item()
                hd95_list.append(h)
            elif lbl.sum() > 0:
                hd95_list.append(100.0) # 惩罚: 漏检
            # 如果 lbl 是全黑 (0)，则不需要计算 HD95 (通常无定义)，这里忽略
                
    return np.mean(dice_list), np.mean(hd95_list)

if __name__ == "__main__":
    final_results = []
    print(f"\n{'Fold':<5} | {'Dice':<10} | {'HD95':<10}")
    print("-" * 35)
    
    for fold in [0, 1, 2, 3, 4]:
        try:
            d, h = evaluate_fold(fold)
            final_results.append((d, h))
            print(f"{fold:<5} | {d:.4f}     | {h:.4f}")
        except Exception as e:
            print(f"Fold {fold} Error: {e}")
        
    avg_dice = np.mean([x[0] for x in final_results])
    std_dice = np.std([x[0] for x in final_results])
    avg_hd = np.mean([x[1] for x in final_results])
    std_hd = np.std([x[1] for x in final_results])
    
    print("=" * 35)
    print(f"🏆 Final: Dice {avg_dice:.4f} ± {std_dice:.4f}")
    print(f"🏆 Final: HD95 {avg_hd:.4f} ± {std_hd:.4f}")
    print("=" * 35)