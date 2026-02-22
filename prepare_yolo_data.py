import os
import cv2
import numpy as np
from tqdm import tqdm

def keep_largest_component(binary_mask):
    """
    保留二值图像中最大的连通域（剥离瞳孔/噪点，保留泪河）。
    """
    binary_mask = binary_mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    if num_labels <= 1:
        return np.zeros_like(binary_mask)
    
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned_mask = np.zeros_like(binary_mask)
    cleaned_mask[labels == largest_label] = 255
    
    return cleaned_mask

def get_yolo_bbox_from_mask(mask, img_width, img_height):
    """
    从净化后的 Mask 中提取外接矩形，并转换为 YOLO 归一化格式
    格式: [class_id, x_center, y_center, width, height]
    """
    ys, xs = np.where(mask > 0)
    
    if len(xs) == 0 or len(ys) == 0:
        return None
        
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    
    box_width = x_max - x_min
    box_height = y_max - y_min
    x_center = x_min + box_width / 2.0
    y_center = y_min + box_height / 2.0
    
    x_center /= img_width
    y_center /= img_height
    box_width /= img_width
    box_height /= img_height
    
    return f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"

def process_multicenter_dataset(dataset_root):
    """
    遍历多中心数据集，执行标签净化和 YOLO 格式转换
    """
    centers = ["Colour1", "Colour2", "Infrared1", "Infrared2", "Infrared3"]
    total_empty_masks = 0
    
    for center in centers:
        print(f"\n🚀 正在处理中心: {center} ...")
        
        # 定义路径
        center_dir = os.path.join(dataset_root, center)
        original_mask_dir = os.path.join(center_dir, "Label") # 你原本带有瞳孔的旧标签
        
        # 我们新建两个文件夹，不覆盖你的原始数据
        clean_mask_dir = os.path.join(center_dir, "Cleaned_Label") 
        yolo_label_dir = os.path.join(center_dir, "YOLO_Label")
        
        os.makedirs(clean_mask_dir, exist_ok=True)
        os.makedirs(yolo_label_dir, exist_ok=True)
        
        # 确保该中心的 Label 文件夹存在
        if not os.path.exists(original_mask_dir):
            print(f"⚠️ 警告: 找不到 {original_mask_dir}，已跳过。")
            continue
            
        mask_files = [f for f in os.listdir(original_mask_dir) if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))]
        
        for filename in tqdm(mask_files, desc=f"{center} 进度"):
            mask_path = os.path.join(original_mask_dir, filename)
            
            # 读取旧 Mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
                
            img_height, img_width = mask.shape
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # 1. 净化标签（剔除瞳孔）
            cleaned_mask = keep_largest_component(binary_mask)
            
            # 保存净化后的新 Mask
            clean_mask_path = os.path.join(clean_mask_dir, filename)
            cv2.imwrite(clean_mask_path, cleaned_mask)
            
            # 2. 生成 YOLO txt
            yolo_str = get_yolo_bbox_from_mask(cleaned_mask, img_width, img_height)
            
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(yolo_label_dir, txt_filename)
            
            if yolo_str is not None:
                with open(txt_path, 'w') as f:
                    f.write(yolo_str + '\n')
            else:
                open(txt_path, 'w').close()
                total_empty_masks += 1

    print("\n✅ 所有 5 个中心的数据处理完成！")
    print("👉 净化后的 Mask 存放在各中心的 Cleaned_Label 文件夹下。")
    print("👉 YOLO 框数据存放在各中心的 YOLO_Label 文件夹下。")
    if total_empty_masks > 0:
        print(f"⚠️ 全局提示: 共发现 {total_empty_masks} 张图像没有泪河连通域。")

# ==========================================
# 运行配置
# ==========================================
if __name__ == "__main__":
    # 你的数据集根目录，里面包含 Colour1, Infrared1 等 5 个文件夹
    DATASET_ROOT = "../Unet/dataset" 
    
    process_multicenter_dataset(DATASET_ROOT)