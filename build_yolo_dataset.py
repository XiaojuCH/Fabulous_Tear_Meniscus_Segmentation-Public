import os
import json
import shutil
from pathlib import Path

# ================= 配置区域 =================
JSON_PATH = "./data_splits/fold_0.json"  # 读取你的 Fold 0 分割表
YOLO_ROOT = "./YOLO_Fold0"               # 将要生成的 YOLO 数据集目录
# ===========================================

def build_yolo():
    # 建立 YOLO 标准目录结构
    dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for d in dirs:
        os.makedirs(os.path.join(YOLO_ROOT, d), exist_ok=True)
        
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
        
    print("开始复制文件构建 YOLO 数据集...")
    
    # 处理 Train 和 Val
    for split in ['train', 'val']:
        for item in data[split]:
            img_path = item['image']
            # 根据我们之前的脚本，YOLO txt 存放在对应中心的 YOLO_Label 目录下
            lbl_path = img_path.replace('Original', 'YOLO_Label').replace(Path(img_path).suffix, '.txt')
            
            # 目标路径
            img_dst = os.path.join(YOLO_ROOT, f"images/{split}", os.path.basename(img_path))
            lbl_dst = os.path.join(YOLO_ROOT, f"labels/{split}", os.path.basename(lbl_path))
            
            # 复制文件 (如果你懂 os.symlink，用软链接更快，这里用 copy 最稳妥)
            shutil.copy(img_path, img_dst)
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, lbl_dst)
            else:
                # 如果没有目标，YOLO 允许生成空 txt
                open(lbl_dst, 'w').close()
                
    # 自动生成 data.yaml
    yaml_content = f"""
path: {os.path.abspath(YOLO_ROOT)}
train: images/train
val: images/val

names:
  0: tear_meniscus
"""
    with open(os.path.join(YOLO_ROOT, 'data.yaml'), 'w') as f:
        f.write(yaml_content.strip())
        
    print(f"✅ YOLO Fold 0 数据集构建完成！保存在 {YOLO_ROOT}")

if __name__ == "__main__":
    build_yolo()