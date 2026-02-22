import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

def keep_largest_component(mask):
    """
    保留 mask 中最大的连通域（通常是泪河），过滤掉瞳孔或其他噪点。
    这个函数和 dataset.py 里的逻辑完全一致
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    max_label_idx = 0
    max_area = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label_idx = i

    clean_mask = np.zeros_like(mask)
    clean_mask[labels == max_label_idx] = 1

    return clean_mask

def visualize_samples(num_samples=8):
    """随机抽取几个样本，对比原始mask和清洗后的mask"""

    # 读取数据集
    split_path = "./data_splits/fold_0.json"
    with open(split_path, 'r') as f:
        split_data = json.load(f)

    # 随机选择样本
    import random
    samples = random.sample(split_data['train'], min(num_samples, len(split_data['train'])))

    # 创建图表
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, item in enumerate(samples):
        img_path = item['image']
        label_path = item['label']

        # 读取图像和mask
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        # Resize到1024
        image = image.resize((1024, 1024), resample=Image.BILINEAR)
        label = label.resize((1024, 1024), resample=Image.NEAREST)

        # 转为numpy
        image_np = np.array(image)
        label_np = np.array(label)

        # 原始mask二值化
        original_mask = (label_np > 127).astype(np.uint8)

        # 清洗后的mask
        cleaned_mask = keep_largest_component(original_mask)

        # 显示
        axes[idx, 0].imshow(image_np)
        axes[idx, 0].set_title(f'Image: {item["id"]}', fontsize=10)
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(original_mask, cmap='gray')
        axes[idx, 1].set_title('Original Mask (含瞳孔)', fontsize=10)
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(cleaned_mask, cmap='gray')
        axes[idx, 2].set_title('Cleaned Mask (只有泪河)', fontsize=10)
        axes[idx, 2].axis('off')

    plt.tight_layout()
    save_path = './mask_cleaning_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    visualize_samples(num_samples=8)
