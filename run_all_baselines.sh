#!/bin/bash
set -e

# 1. 跑 UNet (5折)
echo "🚀 开始跑 UNet..."
for i in 0 1 2 3 4
do
    torchrun --nproc_per_node=8 --master_port=29600 src/train_baseline.py --fold $i --model unet
done
# 评估 UNet
python src/evaluate_baseline.py --model unet

# 2. 跑 Swin-UNet (5折)
echo "🚀 开始跑 Swin-UNet..."
for i in 0 1 2 3 4
do
    torchrun --nproc_per_node=8 --master_port=29601 src/train_baseline.py --fold $i --model swinunet
done
# 评估 Swin-UNet
python src/evaluate_baseline.py --model swinunet

echo "✅ 所有 Baseline 跑完！"
