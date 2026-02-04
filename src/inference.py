import os
import argparse
import json
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.metrics import compute_hausdorff_distance, compute_dice
import matplotlib.pyplot as plt

from dataset import TearDataset
from model import ST_SAM

# =================配置=================
IMG_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_metric(pred, gt):
    """
    pred, gt: [1, 1, H, W] tensor, binary (0/1)
    """
    # 【关键修改】强制转到 CPU 计算，避开 MONAI 的 CuPy 兼容性 Bug
    pred = pred.cpu()
    gt = gt.cpu()

    # 1. Dice
    dice = compute_dice(y_pred=pred, y=gt, include_background=False).item()
    
    # 2. HD95 (Hausdorff Distance 95%)
    # 注意：MONAI 的 HD95 输入必须是包含至少一个前景像素的 Batch
    if gt.sum() > 0 and pred.sum() > 0:
        hd95 = compute_hausdorff_distance(y_pred=pred, y=gt, include_background=False, percentile=95).item()
    else:
        # 如果 GT 有东西但预测全黑，或者反之，给一个惩罚值 (比如 100个像素距离)
        hd95 = 100.0 if gt.sum() > 0 else 0.0
        
    return dice, hd95

def visualize(image, gt, pred, save_path):
    """
    image: [3, H, W] tensor
    gt, pred: [1, H, W] tensor
    """
    # 转 numpy
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) # 归一化到 0-1 用于显示
    
    gt_np = gt.squeeze().cpu().numpy()
    pred_np = pred.squeeze().cpu().numpy()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("Input Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img_np)
    plt.imshow(gt_np, alpha=0.5, cmap='Greens') # GT 用绿色覆盖
    plt.title("Ground Truth (Green)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_np)
    plt.imshow(pred_np, alpha=0.5, cmap='Reds') # Pred 用红色覆盖
    plt.title("Prediction (Red)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main(fold):
    print(f"🔍 开始评估 Fold {fold}...")
    
    # 1. 准备目录
    vis_dir = f"./visualization/fold_{fold}"
    os.makedirs(vis_dir, exist_ok=True)
    
    # 2. 加载数据
    split_path = f"./data_splits/fold_{fold}.json"
    with open(split_path, 'r') as f:
        split_data = json.load(f)
    
    val_dataset = TearDataset(split_data['val'], mode='val', img_size=IMG_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 3. 加载模型
    model = ST_SAM(checkpoint_path="./checkpoints/sam2_hiera_large.pt").to(DEVICE)
    
    # 加载训练好的权重
    ckpt_path = f"./checkpoints/fold_{fold}/best_model.pth"
    if not os.path.exists(ckpt_path):
        print(f"❌ 错误：找不到权重文件 {ckpt_path}")
        return

    # 注意：训练时保存的是 model.module.state_dict() (因为用了 DDP)
    # 加载时如果不是 DDP 环境，需要去掉 key 里的 "module." 前缀
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # 4. 推理循环
    total_dice = []
    total_hd95 = []
    
    # 这里的 box 依然使用 Dataset 里基于 GT 生成的
    # 在论文中，这叫 "Oracle Box" 实验，证明分割能力的上限
    # 实际临床应用我们会补一个检测网络，但现在先看分割网络本身强不强
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            image = batch['image'].to(DEVICE)
            label = batch['label'].to(DEVICE)
            box = batch['box'].to(DEVICE)
            img_id = batch['id'][0] # 获取文件名/ID
            
            # Forward
            pred_logits = model(image, box)
            pred_probs = torch.sigmoid(pred_logits)
            pred_mask = (pred_probs > 0.5).float()
            
            # Metrics
            dice, hd95 = compute_metric(pred_mask, label)
            total_dice.append(dice)
            total_hd95.append(hd95)
            
            # Visualization (每 50 张存一张，或者存 metrics 比较差的)
            if i % 50 == 0:
                save_path = os.path.join(vis_dir, f"{img_id}_D{dice:.3f}_H{hd95:.1f}.png")
                visualize(image[0], label[0], pred_mask[0], save_path)
                
    # 5. 汇报结果
    mean_dice = np.mean(total_dice)
    mean_hd95 = np.mean(total_hd95)
    
    print("\n" + "="*30)
    print(f"📊 Fold {fold} Final Results:")
    print(f"   Dice: {mean_dice:.4f}")
    print(f"   HD95: {mean_hd95:.4f}") # 关注这个！Swin-UNet 曾高达 72
    print("="*30)
    print(f"🖼️ 可视化结果已保存至: {vis_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    main(args.fold)