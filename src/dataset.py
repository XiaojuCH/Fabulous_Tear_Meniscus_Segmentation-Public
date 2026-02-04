import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as F
import random

class TearDataset(Dataset):
    def __init__(self, data_list, mode="train", img_size=1024):
        """
        data_list: list, 也就是从 json 里读取出来的列表
        mode: "train" 或 "val" (如果是 train，会做数据增强)
        img_size: int, SAM 2 需要 1024
        """
        self.data_list = data_list
        self.mode = mode
        self.img_size = img_size

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        img_path = item['image']
        label_path = item['label']

        # --------------------------
        # 1. 图像读取与清洗
        # --------------------------
        # 强制转为 RGB (解决 RGBA 和 灰度图 混杂问题)
        image = Image.open(img_path).convert("RGB")
        
        # Label 转为单通道灰度
        label = Image.open(label_path).convert("L")

        # --------------------------
        # 2. 预处理与增强 (Resize)
        # --------------------------
        # SAM 2 推荐输入 1024x1024
        # 我们使用 Nearest 插值缩放 Label 以保持二值特性
        image = image.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        label = label.resize((self.img_size, self.img_size), resample=Image.NEAREST)

        # 转为 Tensor
        # Image: [3, H, W], 0-1
        image_tensor = F.to_tensor(image) 
        
        # Label: [1, H, W], 0 或 1
        label_np = np.array(label)
        label_np = (label_np > 127).astype(np.float32) # 阈值化，确保干净
        label_tensor = torch.from_numpy(label_np).unsqueeze(0)

        # --------------------------
        # 3. 动态生成 Prompt (Box) - 仅训练时
        # --------------------------
        # 如果是训练模式，我们需要模拟 SAM 的提示框
        # 如果是验证/测试模式，我们通常使用全图作为一个大框，或者使用外部检测器的结果
        # 这里为了训练，我们基于 GT 生成带噪声的 Box
        
        box = self.get_bbox_from_mask(label_np)
        
        if self.mode == "train":
            # 训练时加入随机噪声，模拟人工框的不完美，增加鲁棒性
            box = self.perturb_box(box, self.img_size)
        
        # Box 格式: [x1, y1, x2, y2]
        box_tensor = torch.tensor(box, dtype=torch.float32)

        return {
            "image": image_tensor,
            "label": label_tensor,
            "box": box_tensor,
            "id": item['id']  # 方便后续 debug 或保存结果
        }

    def get_bbox_from_mask(self, mask):
        """从 Mask 获取边界框 (x1, y1, x2, y2)"""
        y_indices, x_indices = np.where(mask > 0)
        
        if len(y_indices) == 0:
            # 异常处理：如果是全黑图片 (没有泪河)，返回一个 dummy box
            return [0, 0, self.img_size, self.img_size]
            
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        return [x_min, y_min, x_max, y_max]

    def perturb_box(self, box, img_size, noise_range=20):
        """给 Box 加噪声"""
        x1, y1, x2, y2 = box
        
        # 随机扩充或收缩边缘
        x1 = max(0, x1 - random.randint(0, noise_range))
        y1 = max(0, y1 - random.randint(0, noise_range))
        x2 = min(img_size, x2 + random.randint(0, noise_range))
        y2 = min(img_size, y2 + random.randint(0, noise_range))
        
        return [x1, y1, x2, y2]

