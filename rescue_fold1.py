import os
import json
from ultralytics import YOLO
from tqdm import tqdm

fold = 1
# 🚨 直接指向那个套娃生成的路径
best_pt = f"runs/detect/runs/detect/fold_{fold}/weights/best.pt"
json_path = f"./data_splits/fold_{fold}.json"
out_json = f"./data_splits/yolo_boxes_fold{fold}.json"

print(f"🔄 正在加载 YOLO 模型: {best_pt}")
model = YOLO(best_pt)

with open(json_path, 'r') as f:
    data = json.load(f)

preds = {}
for item in tqdm(data['val'], desc=f"Fold {fold} 推理"):
    res = model(item['image'], verbose=False)
    boxes = res[0].boxes
    if len(boxes) > 0:
        preds[item['id']] = boxes.xyxyn[0].cpu().numpy().tolist()
    else:
        preds[item['id']] = [0.0, 0.0, 1.0, 1.0]

with open(out_json, 'w') as f:
    json.dump(preds, f, indent=4)
    
print(f"✅ Fold {fold} 抢救成功！预测框已保存至: {out_json}")