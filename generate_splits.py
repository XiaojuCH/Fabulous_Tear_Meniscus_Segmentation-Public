import os
import json
import glob
from pathlib import Path
from sklearn.model_selection import KFold

# ================= 配置区域 =================
# 请将此处改为你原始数据集的绝对路径
ORIGINAL_DATA_ROOT = "../Unet/dataset" 
OUTPUT_DIR = "./data_splits"
# ===========================================

def generate_splits():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 定义 5 个 Center 的映射关系 (LOCO 核心)
    # 我们将 5 个子文件夹视为 5 个 Center
    centers = {
        "Center_1": "Colour1",
        "Center_2": "Colour2",
        "Center_3": "Infrared1",
        "Center_4": "Infrared2",
        "Center_5": "Infrared3"
    }

    all_valid_pairs = []
    
    print(f"🚀 开始扫描原始数据集: {ORIGINAL_DATA_ROOT}")

    # 2. 遍历每个 Center 进行数据清洗
    for center_name, folder_name in centers.items():
        # 构建路径
        img_dir = os.path.join(ORIGINAL_DATA_ROOT, folder_name, "Original")
        lbl_dir = os.path.join(ORIGINAL_DATA_ROOT, folder_name, "Label")
        
        # 获取所有图片 (假设是 png)
        # 注意：Infrared 可能是 .png, 需根据实际情况调整 glob
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.PNG")))
        lbl_paths = sorted(glob.glob(os.path.join(lbl_dir, "*.png")) + glob.glob(os.path.join(lbl_dir, "*.PNG")))
        
        # 建立文件名到路径的映射 (忽略后缀大小写和路径差异，只看文件名 stem)
        img_map = {Path(p).stem: p for p in img_paths}
        lbl_map = {Path(p).stem: p for p in lbl_paths}
        
        # 找交集 (清洗关键步骤)
        common_ids = set(img_map.keys()) & set(lbl_map.keys())
        
        # 报告异常
        if len(img_map) != len(lbl_map):
            print(f"⚠️  警告 [{folder_name}]: 原图 {len(img_map)} 张, Label {len(lbl_map)} 张。将仅使用 {len(common_ids)} 对匹配数据。")
        
        for pid in common_ids:
            all_valid_pairs.append({
                "id": pid,
                "image": img_map[pid],
                "label": lbl_map[pid],
                "center": center_name, # 记录属于哪个中心，方便 LOCO 分割
                "modality": "Visible" if "Colour" in folder_name else "Infrared"
            })

    print(f"✅ 数据清洗完成。共找到 {len(all_valid_pairs)} 对有效数据。")
    
    # 保存总表
    with open(os.path.join(OUTPUT_DIR, "clean_full_list.json"), "w") as f:
        json.dump(all_valid_pairs, f, indent=4)

    # 3. 生成 LOCO (Leave-One-Center-Out) 划分
    # 策略：轮流选一个 Center 做验证集，其余做训练集
    center_keys = list(centers.keys()) # ['Center_1', ..., 'Center_5']
    
    for fold_idx, val_center in enumerate(center_keys):
        train_list = [item for item in all_valid_pairs if item['center'] != val_center]
        val_list = [item for item in all_valid_pairs if item['center'] == val_center]
        
        split_dict = {
            "train": train_list,
            "val": val_list,
            "val_center": val_center
        }
        
        save_path = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}.json")
        with open(save_path, "w") as f:
            json.dump(split_dict, f, indent=4)
        
        print(f"📂 Fold {fold_idx} 生成完毕: 验证集为 {val_center} ({len(val_list)}张), 训练集 ({len(train_list)}张)")

if __name__ == "__main__":
    generate_splits()