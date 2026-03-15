import os
import sys
import json
import torch
import math
import csv
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("src")

# 引入 MONAI 严谨指标
from monai.metrics import compute_dice, compute_hausdorff_distance

from dataset import TearDataset
from model import ST_SAM, Baseline_SAM2, MSA_Baseline_SAM2, LoRA_SAM2

# ================= 全局配置 =================
IMG_SIZE = 1024
MAX_HD95 = math.sqrt(IMG_SIZE**2 + IMG_SIZE**2)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUTPUT_CSV = "evaluation_results_5folds.csv"

# 【请核对并修改你的权重路径规则】
# 脚本会自动把 {fold} 替换为 0~4
CHECKPOINT_PATHS = {
    "ST-SAM": "./checkpoints_gal1/fold_{fold}/best_model.pth",
    "MSA_SAM2": "./checkpoints_msa/fold_{fold}/best_model.pth",         # 请确认你的 MSA 存放路径
    "Baseline_SAM2": "./checkpoints_ablation/fold_{fold}/best_model.pth", 
    "LoRA_SAM2": "./checkpoints_lora/fold_{fold}/best_model.pth"        # 如果你没跑完 LoRA，程序会自动跳过它
}

PADDINGS = [-5, 0, 5, 10, 20, 30, 40, "YOLO"] # 加入 "YOLO" 用于后续真实提示分析

# ================= 严谨指标计算 =================
def compute_metrics_monai(pred_tensor, gt_tensor):
    """
    输入必须是在 CPU 上的 [1, 1, H, W] Tensor
    """
    if pred_tensor.dim() == 3:
        pred_tensor = pred_tensor.unsqueeze(0)
    if gt_tensor.dim() == 3:
        gt_tensor = gt_tensor.unsqueeze(0)
        
    pred = (pred_tensor > 0.5).float()
    lbl = (gt_tensor > 0.5).float()
    
    # 1. Dice
    if lbl.sum() == 0 and pred.sum() == 0:
        dice_score = 1.0
    else:
        dice_score = compute_dice(pred, lbl, include_background=False).item()
        if math.isnan(dice_score):
            dice_score = 0.0

    # 2. HD95
    if lbl.sum() > 0 and pred.sum() > 0:
        hd95 = compute_hausdorff_distance(pred, lbl, include_background=False, percentile=95).item()
        if math.isnan(hd95):
            hd95 = MAX_HD95
    elif lbl.sum() > 0 and pred.sum() == 0:
        hd95 = MAX_HD95 
    else:
        if pred.sum() == 0:
            hd95 = 0.0
        else:
            hd95 = MAX_HD95

    return dice_score, hd95

# ================= 智能数据加载器 =================
class RobustnessDataset(TearDataset):
    def __init__(self, data_list, img_size, yolo_pred_json, padding=0):
        # 初始化父类，如果是 YOLO 模式，父类会自动读取 JSON 里的 YOLO 框
        super().__init__(data_list, mode="val", img_size=img_size, yolo_pred_json=yolo_pred_json)
        self.padding = padding

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        
        # 如果是 YOLO 模式，直接返回父类处理好的 item（里面已经是 YOLO 框了）
        if self.padding == "YOLO":
            return item
            
        # 否则，基于完美的 GT 施加人工 Padding
        label_np = item['label'].squeeze().numpy()
        y_indices, x_indices = np.where(label_np > 0)
        
        if len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
        else:
            x_min, y_min, x_max, y_max = 0, 0, self.img_size, self.img_size

        p = int(self.padding)
        x1 = max(0, x_min - p)
        y1 = max(0, y_min - p)
        x2 = min(self.img_size, x_max + p)
        y2 = min(self.img_size, y_max + p)
        
        item['box'] = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        return item

# ================= 主控制流 =================
def main():
    print(f"🚀 启动 5-Fold 鲁棒性全量数据生成管线...")
    print(f"📄 数据将流式保存至: {OUTPUT_CSV}")
    
    # 准备写入 CSV
    file_exists = os.path.isfile(OUTPUT_CSV)
    with open(OUTPUT_CSV, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Fold', 'Model', 'Padding', 'Image_ID', 'Modality', 'Dice', 'HD95'])

        # 实例化骨干网络（放在循环外节省显存开销，每次换 Fold 只更新权重）
        model_instances = {
            "ST-SAM": ST_SAM().to(DEVICE),
            "MSA_SAM2": MSA_Baseline_SAM2().to(DEVICE),
            "Baseline_SAM2": Baseline_SAM2().to(DEVICE),
            "LoRA_SAM2": LoRA_SAM2().to(DEVICE)
        }

        # 循环 5 个 Fold
        for fold in range(5):
            print(f"\n" + "="*50)
            print(f"🔄 正在处理 Fold {fold} ...")
            
            fold_json_path = f"./data_splits/fold_{fold}.json"
            yolo_json_path = f"./data_splits/yolo_boxes_fold{fold}.json"
            
            if not os.path.exists(fold_json_path):
                print(f"⚠️ 找不到 {fold_json_path}，跳过该 Fold。")
                continue
                
            with open(fold_json_path, 'r') as f:
                split_data = json.load(f)

            # 为当前 Fold 加载模型权重
            active_models = {}
            for model_name, model in model_instances.items():
                ckpt_path = CHECKPOINT_PATHS[model_name].format(fold=fold)
                if os.path.exists(ckpt_path):
                    state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    model.load_state_dict(state_dict)
                    model.eval()
                    active_models[model_name] = model
                else:
                    print(f"  [跳过] 找不到 {model_name} 的权重: {ckpt_path}")

            if not active_models:
                print(f"❌ Fold {fold} 没有任何模型可用，跳过。")
                continue

            # 遍历所有设定的 Padding 扰动（包含 "YOLO"）
            for p in PADDINGS:
                print(f"  📐 评估 Padding = {p}")
                val_dataset = RobustnessDataset(split_data['val'], img_size=IMG_SIZE, yolo_pred_json=yolo_json_path, padding=p)
                # Batch Size 设为 1 最安全，避免多样本打包时的 ID 错乱
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

                for model_name, model in active_models.items():
                    with torch.no_grad():
                        for batch in tqdm(val_loader, desc=f"    {model_name}", leave=False):
                            img = batch['image'].to(DEVICE)
                            lbl = batch['label'].to(DEVICE)
                            box = batch['box'].to(DEVICE)
                            img_id = str(batch['id'][0])
                            
                            # 判定模态
                            modality = 'Colour' if ('colour' in img_id.lower() or 'color' in img_id.lower()) else 'Infrared'

                            logits = model(img, box)
                            preds = torch.sigmoid(logits)
                            
                            # 【核心修复】：必须转到 cpu() 防止 CuPy 报错
                            d, h = compute_metrics_monai(preds.cpu(), lbl.cpu())
                            
                            # 流式写入 CSV
                            writer.writerow([fold, model_name, p, img_id, modality, d, h])
            
            # 及时刷入硬盘
            csvfile.flush()

    print("\n🎉 全量评估完毕！所有实例级数据已安全保存至 evaluation_results_5folds.csv")

if __name__ == "__main__":
    main()