import os
import argparse
import json
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
# 过滤掉关于 Grad strides 的特定警告
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")

from dataset import TearDataset
from model import ST_SAM

# ==============================================================================
# 配置区域 (Global Config)
# ==============================================================================
# 建议把配置写成字典，方便保存
CONFIG = {
    "batch_size": 8,
    "num_workers": 4,
    "lr": 1e-4,
    "epochs": 1,
    "img_size": 1024,
    "model_name": "ST-SAM (Sam2-Hiera-L + GatedDilatedStripAdapter)",
    "optimizer": "AdamW",
    "scheduler": "None", # 如果加了 scheduler 这里也要记
    "loss": "Dice + BCE",
    "gpu_count": 8
}

def setup_ddp():
    if "RANK" in os.environ:
        init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        # 加强版检测：防止 ordinal 越界
        num_gpus = torch.cuda.device_count()
        actual_gpu = local_rank % num_gpus 
        
        torch.cuda.set_device(actual_gpu)
        return rank, actual_gpu, world_size
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
# 训练主循环
# ==============================================================================
def main(fold):
    rank, local_rank, world_size = setup_ddp()
    is_master = (rank == 0)

    if is_master:
        print(f"\n🚀 启动训练: Fold {fold} | GPUs: {world_size} | Model: {CONFIG['model_name']}")

    split_path = f"./data_splits/fold_{fold}.json"
    with open(split_path, 'r') as f:
        split_data = json.load(f)
    
    # 使用 CONFIG 里的参数
    train_dataset = TearDataset(split_data['train'], mode='train', img_size=CONFIG['img_size'])
    # 【修复】动态获取当前 fold 的 YOLO 预测框文件，并传给验证集
    yolo_json_path = f"./data_splits/yolo_boxes_fold{fold}.json"
    val_dataset = TearDataset(
        split_data['val'], 
        mode='val', 
        img_size=CONFIG['img_size'], 
        yolo_pred_json=yolo_json_path  # <--- 就是漏了这极其关键的参数！
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], sampler=train_sampler, num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], sampler=val_sampler, num_workers=CONFIG['num_workers'], pin_memory=True)

    model = ST_SAM(checkpoint_path="./checkpoints/sam2_hiera_large.pt").to(local_rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'])
    criterion = DiceBCELoss().to(local_rank)
    scaler = GradScaler('cuda') 

    best_dice = 0.0
    best_epoch = 0

    torch.cuda.reset_peak_memory_stats(local_rank)

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

        # Validation
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

        # Reduce
        train_loss_tensor = torch.tensor(train_loss).to(local_rank)
        all_reduce(train_loss_tensor, op=ReduceOp.SUM)
        avg_train_loss = train_loss_tensor.item() / (len(train_loader) * world_size)

        val_dice_tensor = torch.tensor(val_dice).to(local_rank)
        all_reduce(val_dice_tensor, op=ReduceOp.SUM)
        avg_val_dice = val_dice_tensor.item() / (len(val_loader) * world_size)

        if is_master:
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f}")
            
            if avg_val_dice > best_dice:
                best_dice = avg_val_dice
                best_epoch = epoch + 1
                print(f"🔥 New Best Dice: {best_dice:.4f} (Epoch {best_epoch})")

    if is_master:
        peak_gb = torch.cuda.max_memory_allocated(local_rank) / 1024**3
        print(f"\n[ST_SAM] 单卡峰值显存: {peak_gb:.2f}G  |  总显存({world_size}卡): {peak_gb * world_size:.2f}G")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, help="LOCO fold index (0-4)")
    args = parser.parse_args()
    
    main(fold=args.fold)