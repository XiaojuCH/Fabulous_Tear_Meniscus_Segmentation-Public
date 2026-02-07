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
from monai.networks.nets import UNet, SwinUNETR
from monai.metrics import compute_dice, compute_hausdorff_distance
from monai.networks.nets import UNet, SwinUNETR, AttentionUnet, SegResNet
from dataset import TearDataset

# ================= 配置 =================
IMG_SIZE = 1024  # 保持和 ST-SAM 一致，公平对比

def setup_ddp():
    if "RANK" in os.environ:
        init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1

def cleanup():
    if "RANK" in os.environ: destroy_process_group()

def get_model(name):
    """从 MONAI 加载标准模型"""
    if name == "unet":
        return UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    elif name == "swinunet":
        return SwinUNETR(
            # img_size=(IMG_SIZE, IMG_SIZE),
            in_channels=3,
            out_channels=1,
            feature_size=24, # 缩小一点以防爆显存，或者设为 48
            spatial_dims=2,
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

def log_to_csv(stats, filename="baseline_summary.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(stats)

def main(args):
    rank, local_rank, world_size = setup_ddp()
    is_master = (rank == 0)
    
    # 1. 数据集 (完全复用)
    split_path = f"./data_splits/fold_{args.fold}.json"
    with open(split_path, 'r') as f: split_data = json.load(f)
    
    train_ds = TearDataset(split_data['train'], mode='train', img_size=IMG_SIZE)
    val_ds = TearDataset(split_data['val'], mode='val', img_size=IMG_SIZE)
    
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)

    # 2. 模型
    model = get_model(args.model).to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank])

    # 3. 优化器 (Baseline 通常需要全量微调，LR 要大一点)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss() # 简单起见，或者复用 DiceCELoss
    scaler = GradScaler('cuda')

    best_dice = 0.0
    save_dir = f"./checkpoints_baseline/{args.model}/fold_{args.fold}"
    if is_master: os.makedirs(save_dir, exist_ok=True)
    
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, disable=not is_master, desc=f"[{args.model.upper()}] Fold {args.fold} Ep {epoch}")
        
        for batch in pbar:
            img, lbl = batch['image'].to(local_rank), batch['label'].to(local_rank)
            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(img) # Baseline 不需要 box prompt
                loss = loss_fn(logits, lbl)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=loss.item())

        # Validation
        if (epoch + 1) % 5 == 0 or epoch > args.epochs - 10: # 后期多测
            model.eval()
            val_dice = 0.0
            val_hd95 = 0.0 # 粗略估算
            count = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    img, lbl = batch['image'].to(local_rank), batch['label'].to(local_rank)
                    with autocast('cuda'):
                        pred = (torch.sigmoid(model(img)) > 0.5).float()
                    
                    # 简单 Dice
                    inter = (pred * lbl).sum()
                    dice = (2 * inter) / (pred.sum() + lbl.sum() + 1e-6)
                    val_dice += dice.item()
                    
                    # 为了速度，训练中不算 HD95，只算 Dice
                    
            # 汇总
            val_dice_t = torch.tensor(val_dice).to(local_rank)
            all_reduce(val_dice_t, op=ReduceOp.SUM)
            avg_dice = val_dice_t.item() / (len(val_loader) * world_size)
            
            if is_master:
                if avg_dice > best_dice:
                    best_dice = avg_dice
                    torch.save(model.module.state_dict(), f"{save_dir}/best_model.pth")
                    print(f"🔥 New Best Dice ({args.model}): {best_dice:.4f}")

    # 训练结束，记录
    if is_master:
        duration = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        log_to_csv({
            "model": args.model,
            "fold": args.fold,
            "best_dice": best_dice,
            "duration": duration,
            "lr": args.lr
        })

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["unet", "swinunet","attentionunet", "segresnet"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4) # CNN/Swin 通常用 3e-4 或 1e-3
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    main(args)