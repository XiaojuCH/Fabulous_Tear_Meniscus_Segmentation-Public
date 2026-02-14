import os
import argparse
import json
import time
import datetime
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
from torch.amp import GradScaler, autocast
import torch.nn.functional as F
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from dataset import TearDataset
# 引用你的最终版模型
from model import Baseline_SAM2

# ==============================================================================
# 配置区域
# ==============================================================================
CONFIG = {
    "batch_size": 8,
    "num_workers": 4,
    "lr": 1e-4,
    "epochs": 50, # 跨模态通常收敛快，50够了，想跑满100也行
    "img_size": 1024,
    "model_name": "SAM (Cross-Modality Test)",
    "optimizer": "AdamW",
    "loss": "Dice + BCE",
    "gpu_count": 8 # 根据你实际情况调整
}

def setup_ddp():
    if "RANK" in os.environ:
        init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(local_rank % num_gpus)
        return rank, local_rank % num_gpus, world_size
    else:
        return 0, 0, 1

def cleanup():
    if "RANK" in os.environ:
        destroy_process_group()

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_flat.sum() + targets_flat.sum() + smooth)
        return 0.5 * bce_loss + 0.5 * dice_loss

# ==============================================================================
# 核心：自动构建跨模态数据集
# ==============================================================================
def get_cross_modal_data(mode="train_color_test_ir"):
    """
    遍历所有 fold 的 json，合并后根据文件名里的关键字强行拆分。
    """
    all_data = []
    # 1. 把所有数据收集起来 (利用 fold_0 到 fold_4 的 val 集互斥且互补的特性)
    for i in range(5):
        json_path = f"./data_splits/fold_{i}.json"
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                split = json.load(f)
                # LOCO 协议中，验证集是不重复的，把所有 fold 的 val 加起来就是全集
                all_data.extend(split['val'])
        else:
            print(f"⚠️ Warning: {json_path} not found.")

    print(f"📦 Total images found: {len(all_data)}")
    
    # 2. 根据文件名过滤
    # 假设文件名类似: "Color1_xxx.jpg" 或 "Infrared1_xxx.jpg"
    color_data = []
    ir_data = []
    
    for item in all_data:
        # 检查 image 路径字符串
        img_path = item['image'] if isinstance(item, dict) else item
        
        # 你的文件名特征：Color vs Infrared
        if "Color" in img_path:
            color_data.append(item)
        elif "Infrared" in img_path:
            ir_data.append(item)
            
    print(f"🎨 Color Images: {len(color_data)}")
    print(f"🌑 Infrared Images: {len(ir_data)}")
    
    # 3. 根据模式返回
    if mode == "train_color_test_ir":
        print("👉 Setting: Train on [Color] -> Test on [Infrared]")
        return color_data, ir_data
    elif mode == "train_ir_test_color":
        print("👉 Setting: Train on [Infrared] -> Test on [Color]")
        return ir_data, color_data
    else:
        raise ValueError("Unknown mode")

# ==============================================================================
# 训练主循环
# ==============================================================================
def main(mode):
    rank, local_rank, world_size = setup_ddp()
    is_master = (rank == 0)

    # 获取跨模态数据
    train_list, val_list = get_cross_modal_data(mode)
    
    train_dataset = TearDataset(train_list, mode='train', img_size=CONFIG['img_size'])
    val_dataset = TearDataset(val_list, mode='val', img_size=CONFIG['img_size'])

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], sampler=train_sampler, num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], sampler=val_sampler, num_workers=CONFIG['num_workers'], pin_memory=True)

    model = Baseline_SAM2(checkpoint_path="./checkpoints/sam2_hiera_large.pt").to(local_rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'])
    criterion = DiceBCELoss().to(local_rank)
    scaler = GradScaler('cuda') 

    best_dice = 0.0
    
    # 保存目录区分模式
    save_dir = f"./checkpoints_cross_modal/{mode}"
    if is_master:
        os.makedirs(save_dir, exist_ok=True)
        print(f"🚀 Start Training: {mode}")

    for epoch in range(CONFIG['epochs']):
        model.train()
        train_sampler.set_epoch(epoch)
        
        train_loss = 0.0
        pbar = tqdm(train_loader, disable=not is_master, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for batch in pbar:
            images = batch['image'].to(local_rank, non_blocking=True)
            labels = batch['label'].to(local_rank, non_blocking=True)
            boxes = batch['box'].to(local_rank, non_blocking=True)

            optimizer.zero_grad()
            with autocast('cuda'):
                preds = model(images, boxes)
                loss = criterion(preds, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # Validation (Testing on the OTHER modality)
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(local_rank, non_blocking=True)
                labels = batch['label'].to(local_rank, non_blocking=True)
                boxes = batch['box'].to(local_rank, non_blocking=True)

                with autocast('cuda'):
                    preds = model(images, boxes)
                    preds = torch.sigmoid(preds)
                    preds_bin = (preds > 0.5).float()
                
                intersection = (preds_bin * labels).sum()
                dice = (2. * intersection) / (preds_bin.sum() + labels.sum() + 1e-6)
                val_dice += dice.item()

        # Reduce metrics
        val_dice_tensor = torch.tensor(val_dice).to(local_rank)
        all_reduce(val_dice_tensor, op=ReduceOp.SUM)
        avg_val_dice = val_dice_tensor.item() / (len(val_loader) * world_size)

        if is_master:
            print(f"Epoch {epoch+1} | Val Dice ({mode.split('_')[-1].upper()}): {avg_val_dice:.4f}")
            
            if avg_val_dice > best_dice:
                best_dice = avg_val_dice
                torch.save(model.module.state_dict(), f"{save_dir}/best_model.pth")
                print(f"🔥 New Best Dice: {best_dice:.4f} -> Saved!")

    cleanup()

# 修改 train_cross_modal.py 的底部代码

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 模式选择
    parser.add_argument("--mode", type=str, default="train_color_test_ir", help="Experiment mode")
    
    # 【新增】接收 DDP 自动传入的 local-rank 参数
    # 虽然我们在 setup_ddp 里用的是 os.environ，但必须这里占个位，防止报错
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for DDP") 
    # 注意：有时候是 --local-rank (中间是横杠)，argparse 会自动转为下划线 local_rank
    # 为了保险，可以使用下面的写法兼容：
    parser.add_argument("--local-rank", type=int, default=0, dest="local_rank")

    args = parser.parse_args()
    
    main(mode=args.mode)