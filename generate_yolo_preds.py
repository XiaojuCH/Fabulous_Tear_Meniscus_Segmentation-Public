import json
import torch
from ultralytics import YOLO
from tqdm import tqdm

# ================= 配置区域 =================
# 你刚才 YOLO 训练出来的最佳权重路径
YOLO_WEIGHTS = "./runs/detect/train/weights/best.pt" 
# 我们正在处理的 Fold 分割表
JSON_PATH = "./data_splits/fold_0.json"
# 将要生成的预测框保存路径
OUTPUT_JSON = "./data_splits/yolo_boxes_fold0.json"
# ===========================================

def generate_predictions():
    print(f"🔄 正在加载 YOLO 模型: {YOLO_WEIGHTS}")
    model = YOLO(YOLO_WEIGHTS)
    
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
        
    val_data = data['val']
    predictions = {}
    
    print(f"🚀 开始对 {len(val_data)} 张验证集图像进行推理...")
    
    for item in tqdm(val_data):
        img_path = item['image']
        img_id = item['id']
        
        # YOLO 推理
        results = model(img_path, verbose=False)
        
        # 提取第一个结果 (因为每次传一张图)
        boxes = results[0].boxes
        
        if len(boxes) > 0:
            # 取置信度最高的一个框，并获取其“归一化坐标” (0~1之间)
            # 格式: [x1, y1, x2, y2]
            box_norm = boxes.xyxyn[0].cpu().numpy().tolist()
        else:
            # 极小概率情况：YOLO 啥也没框到，给一个默认全图框
            print(f"\n⚠️ 警告: 图像 {img_id} 未检测到目标，使用默认全局框。")
            box_norm = [0.0, 0.0, 1.0, 1.0]
            
        predictions[img_id] = box_norm
        
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(predictions, f, indent=4)
        
    print(f"\n✅ 预测完成！YOLO 预测框已保存至: {OUTPUT_JSON}")

if __name__ == "__main__":
    generate_predictions()