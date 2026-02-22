import os
import json
from ultralytics import YOLO
from tqdm import tqdm

def rescue_all_folds():
    for fold in range(1, 5):
        # 记录 YOLO 可能产生的几种套娃路径
        possible_paths = [
            f"runs/detect/YOLO_Outputs/fold_{fold}/weights/best.pt",       # Fold 2,3,4 的套娃路径
            f"runs/detect/runs/detect/fold_{fold}/weights/best.pt",        # Fold 1 的套娃路径
            f"runs/detect/fold_{fold}/weights/best.pt",                    # 默认路径
            f"YOLO_Outputs/fold_{fold}/weights/best.pt"                    # 理想路径
        ]
        
        best_pt = None
        for p in possible_paths:
            if os.path.exists(p):
                best_pt = p
                break
                
        if best_pt is None:
            print(f"⚠️ 找不到 Fold {fold} 的权重文件，说明这个 Fold 还没训练。跳过...")
            continue
            
        out_json = f"./data_splits/yolo_boxes_fold{fold}.json"
        if os.path.exists(out_json):
            print(f"👍 Fold {fold} 的预测框 JSON 已经存在，跳过...")
            continue
            
        print(f"\n🔄 正在加载 Fold {fold} 的 YOLO 模型: {best_pt}")
        model = YOLO(best_pt)
        
        json_path = f"./data_splits/fold_{fold}.json"
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

if __name__ == "__main__":
    rescue_all_folds()
    print("\n🎉 所有已经训练好的 Fold 均已完成推理！")