import os
import json
import torch
import numpy as np
import cv2  
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as F
import random

class TearDataset(Dataset):
    def __init__(self, data_list, mode="train", img_size=1024, yolo_pred_json=None):
        """
        data_list: list, 也就是从 json 里读取出来的列表
        mode: "train" 或 "val" (如果是 train，会做数据增强)
        img_size: int, SAM 2 需要 1024
        yolo_pred_json: val 模式下传入 YOLO 预测框 json 路径
        """
        self.data_list = data_list
        self.mode = mode
        self.img_size = img_size

        # 加载 YOLO 预测框字典
        self.yolo_preds = {}
        if self.mode == "val" and yolo_pred_json is not None and os.path.exists(yolo_pred_json):
            with open(yolo_pred_json, 'r') as f:
                self.yolo_preds = json.load(f)
            print(f"📦 已成功加载 YOLO 预测框文件，共 {len(self.yolo_preds)} 个。")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        img_id = item['id']
        img_path = item['image']
        
        # 🔥【重点】：直接读取我们已经提前清洗好(去除了瞳孔)的 Mask！
        # 无需修改你原始的 json 分割表，直接在代码里做字符串替换即可
        label_path = item['label'].replace("/Label/", "/Cleaned_Label/")

        # --------------------------
        # 1. 图像读取与清洗
        # --------------------------
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        # --------------------------
        # 2. 预处理与增强 (Resize)
        # --------------------------
        image = image.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        label = label.resize((self.img_size, self.img_size), resample=Image.NEAREST)

        # 转为 Tensor
        image_tensor = F.to_tensor(image) 
        
        label_np = np.array(label)
        label_np = (label_np > 127).astype(np.uint8) 

        # 转回 float Tensor [1, H, W]
        label_tensor = torch.from_numpy(label_np).float().unsqueeze(0)

        # --------------------------
        # 3. 动态生成 Prompt (Box)
        # --------------------------
        if self.mode == "train":
            # 训练时：通过干净的 GT 生成框，并加入随机扰动（教模型抗干扰）
            box = self.get_bbox_from_mask(label_np)
            box = self.perturb_box(box, self.img_size)
        else:
            # 验证/测试时：【绝对禁止接触 GT】！直接读取 YOLO 预测框！
            if img_id in self.yolo_preds:
                box_norm = self.yolo_preds[img_id]
                # 还原归一化坐标到 1024 尺度
                box = [
                    box_norm[0] * self.img_size, 
                    box_norm[1] * self.img_size, 
                    box_norm[2] * self.img_size, 
                    box_norm[3] * self.img_size
                ]
            else:
                # 兜底框 (YOLO万一没检测到的情况)
                box = [0, 0, self.img_size, self.img_size]

        box_tensor = torch.tensor(box, dtype=torch.float32)

        return {
            "image": image_tensor,
            "label": label_tensor,
            "box": box_tensor,
            "id": img_id
        }

    def get_bbox_from_mask(self, mask):
        """从 Mask 获取边界框 (x1, y1, x2, y2)"""
        y_indices, x_indices = np.where(mask > 0)
        
        if len(y_indices) == 0:
            return [0, 0, self.img_size, self.img_size]
            
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        return [x_min, y_min, x_max, y_max]

    def perturb_box(self, box, img_size, noise_range=20):
        """给 Box 加噪声"""
        x1, y1, x2, y2 = box
        
        x1 = max(0, x1 - random.randint(0, noise_range))
        y1 = max(0, y1 - random.randint(0, noise_range))
        x2 = min(img_size, x2 + random.randint(0, noise_range))
        y2 = min(img_size, y2 + random.randint(0, noise_range))
        
        return [x1, y1, x2, y2]