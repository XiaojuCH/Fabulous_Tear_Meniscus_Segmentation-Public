#!/bin/bash

# 使用说明: ./train_dist.sh 0
# 0 代表 Fold 0

FOLD=$1
if [ -z "$FOLD" ]; then
    echo "❌ 错误: 请指定 Fold 编号 (0-4)"
    echo "用法: ./train_dist.sh 0"
    exit 1
fi

echo "🚀 正在启动 8卡 分布式训练 (Fold $FOLD)..."

# 关键参数解释：
# --nproc_per_node=8 : 使用 8 张卡
# --master_port : 防止端口冲突，随机设个大数

torchrun --nproc_per_node=8 \
    --master_port=29500 \
    src/train.py \
    --fold $FOLD