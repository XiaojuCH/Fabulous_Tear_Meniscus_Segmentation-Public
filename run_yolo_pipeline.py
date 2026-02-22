import os
import json
import shutil
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# ================= 配置区域 =================
SPLITS_DIR = "./data_splits"       # 你的 JSON 分割表所在文件夹
YOLO_DATA_ROOT = "./YOLO_Data"     # 生成的 YOLO 数据集存放总目录
# ===========================================

def run_fold(fold):
    print(f"\n{'='*50}")
    print(f"🚀 开始全自动处理 Fold {fold}")
    print(f"{'='*50}\n")
    
    # ---------------------------------------------------------
    # 第一步：构建当前 Fold 的 YOLO 数据集
    # ---------------------------------------------------------
    yolo_dir = os.path.join(YOLO_DATA_ROOT, f"Fold_{fold}")
    os.makedirs(yolo_dir, exist_ok=True)
    
    dirs_to_make = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for d in dirs_to_make:
        os.makedirs(os.path.join(yolo_dir, d), exist_ok=True)
        
    json_path = os.path.join(SPLITS_DIR, f"fold_{fold}.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    print(f"📁 正在构建 Fold {fold} 数据集...")
    for split in ['train', 'val']:
        for item in tqdm(data[split], desc=f"复制 {split} 集"):
            img_path = item['image']
            # 替换路径寻找对应的 txt 标签
            lbl_path = img_path.replace('Original', 'YOLO_Label').replace(Path(img_path).suffix, '.txt')
            
            # 复制原图
            shutil.copy(img_path, os.path.join(yolo_dir, f"images/{split}", os.path.basename(img_path)))
            
            # 复制标签（如果不存在则创建空文件，YOLO 支持负样本）
            dst_lbl_path = os.path.join(yolo_dir, f"labels/{split}", os.path.basename(lbl_path))
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, dst_lbl_path)
            else:
                open(dst_lbl_path, 'w').close()
                
    yaml_path = os.path.join(yolo_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(yolo_dir)}\n")
        f.write("train: images/train\nval: images/val\n\nnames:\n  0: tear_meniscus\n")

    # ---------------------------------------------------------
    # 第二步：调用 YOLO 官方 API 自动炼丹
    # ---------------------------------------------------------
    print(f"\n🔥 开始训练 Fold {fold} 的 YOLO 模型...")
    model = YOLO('yolov8n.pt') 
    
    # 🔥 修改点：使用自定义的绝对 project 名称
    model.train(
        data=yaml_path, 
        epochs=20, 
        imgsz=1024, 
        batch=16, 
        project='YOLO_Outputs',  # 换成明确的自定义文件夹
        name=f'fold_{fold}',     # 结果会保存在 YOLO_Outputs/fold_x 下
        verbose=False
    )
    
    # ---------------------------------------------------------
    # 第三步：加载刚训练好的最佳权重，进行推理预测
    # ---------------------------------------------------------
    # 🔥 修改点：对应上面的路径
    best_pt = f"YOLO_Outputs/fold_{fold}/weights/best.pt"
    print(f"\n🎯 训练完成！加载权重进行推理: {best_pt}")
    infer_model = YOLO(best_pt)
    
    preds = {}
    print(f"🔍 正在生成 Fold {fold} 预测框...")
    for item in tqdm(data['val'], desc=f"Fold {fold} 推理"):
        img_path = item['image']
        img_id = item['id']
        
        res = infer_model(img_path, verbose=False)
        boxes = res[0].boxes
        
        if len(boxes) > 0:
            # 提取置信度最高的一个框的归一化坐标 [0~1]
            preds[img_id] = boxes.xyxyn[0].cpu().numpy().tolist()
        else:
            # 没检测到，给全图框
            preds[img_id] = [0.0, 0.0, 1.0, 1.0]
            
    out_json = os.path.join(SPLITS_DIR, f"yolo_boxes_fold{fold}.json")
    with open(out_json, 'w') as f:
        json.dump(preds, f, indent=4)
        
    print(f"\n✅ Fold {fold} 全部流程处理完毕！预测框已保存至: {out_json}")

if __name__ == '__main__':
    # 自动执行 Fold 1, 2, 3, 4 (因为 Fold 0 你已经跑过了)
    for i in range(4, 5):
        run_fold(i)
    
    print("\n🎉🎉🎉 恭喜！所有 5 个 Fold 的 YOLO 预测框全部生成完毕！")