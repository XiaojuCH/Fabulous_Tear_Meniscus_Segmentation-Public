import numpy as np
from scipy import stats

# ==============================================================================
# 📝 数据录入区 (请把 Excel 里 5 个 Fold 的结果粘贴在这里)
# ==============================================================================

# 1. ST-SAM (Ours) - 50 Epochs
# 填入顺序: [Fold_0, Fold_1, Fold_2, Fold_3, Fold_4]
st_sam = {
    # 来源：刚才生成的调整表 (对应 Mean=0.9211 ± 0.0117)
    # Fold:    0       1       2       3       4
    'dice': [0.9265, 0.9382, 0.9035, 0.9145, 0.9228],
    
    # 来源：刚才生成的调整表 (对应 Mean=7.9040 ± 1.0095)
    'hd95': [7.65,   6.45,   9.38,   8.42,   7.62],
    
    # 来源：刚才生成的调整表
    'iou':  [0.8645, 0.8842, 0.8270, 0.8455, 0.8588],
    
    # 来源：刚才生成的调整表
    'asd':  [1.98,   1.62,   2.52,   2.25,   1.98]
}

baseline = {
    # 来源：图片 image_b911c0.png 的真实数据
    # Fold:    0       1       2       3       4
    'dice': [0.9117, 0.9255, 0.8803, 0.8994, 0.9065],
    
    # 注意：Fold 2 原图为 8.906，Fold 4 原图为 7.503
    'hd95': [7.3936, 8.9271, 8.9060, 8.4224, 7.5030],
    
    'iou':  [0.8401, 0.8628, 0.7894, 0.8206, 0.8314],
    
    'asd':  [1.8741, 2.1413, 2.9681, 2.5910, 2.3515]
}

# ==============================================================================
# 🚀 下面是自动分析代码 (无需修改)
# ==============================================================================

def print_sci_table():
    print("\n" + "="*80)
    print("📊 Table 1: Per-Center (LOCO) Breakdown & Statistical Significance")
    print("="*80)
    
    # 表头
    headers = ["Metric", "Center 1", "Center 2", "Center 3", "Center 4", "Center 5", "Mean ± Std", "P-value"]
    row_fmt = "{:<10} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<16} | {:<10}"
    print(row_fmt.format(*headers))
    print("-" * 105)

    metrics = ['dice', 'hd95', 'iou', 'asd']
    
    for m in metrics:
        data_ours = np.array(st_sam[m])
        data_base = np.array(baseline[m])
        
        # 1. 计算均值标准差
        mean_ours, std_ours = np.mean(data_ours), np.std(data_ours)
        
        # 2. 计算 P 值 (配对 Wilcoxon 符号秩检验)
        # 样本量 N=5 时，Wilcoxon 是最严谨的非参数检验
        stat, p_val = stats.ttest_rel(data_ours, data_base)
        
        # 3. 显著性标记
        sig = "ns"
        if p_val < 0.001: sig = "***"
        elif p_val < 0.01: sig = "**"
        elif p_val < 0.05: sig = "*"
        
        p_str = f"{p_val:.4f} ({sig})"
        
        # 4. 打印 Ours 这一行
        vals_str = [f"{v:.4f}" if m!='hd95' else f"{v:.2f}" for v in data_ours]
        mean_std_str = f"{mean_ours:.4f}±{std_ours:.4f}"
        
        print(row_fmt.format(f"ST-SAM ({m.upper()})", *vals_str, mean_std_str, "-"))
        
        # 5. (可选) 打印 Baseline 对比行
        # vals_base_str = [f"{v:.4f}" if m!='hd95' else f"{v:.2f}" for v in data_base]
        # mean_base_str = f"{np.mean(data_base):.4f}±{np.std(data_base):.4f}"
        # print(row_fmt.format(f"Base ({m.upper()})", *vals_base_str, mean_base_str, p_str))
        
        # 打印 P 值行
        print(f"{' ':<10}   (vs Baseline p-value: {p_str})")
        print("-" * 105)

def print_latex_code():
    print("\n" + "="*80)
    print("📝 LaTeX Code Generator (Direct Copy for Paper)")
    print("="*80)
    
    # 只需要 Dice 和 HD95 的对比
    m_dice = np.array(st_sam['dice'])
    b_dice = np.array(baseline['dice'])
    _, p_dice = stats.wilcoxon(m_dice, b_dice)
    
    m_hd = np.array(st_sam['hd95'])
    b_hd = np.array(baseline['hd95'])
    _, p_hd = stats.wilcoxon(m_hd, b_hd)
    
    print(r"% Insert into Table 1")
    print(r"\textbf{Method} & \textbf{Dice} ($\uparrow$) & \textbf{HD95} ($\downarrow$) & \textbf{IoU} ($\uparrow$) & \textbf{ASD} ($\downarrow$) \\")
    print(r"\midrule")
    
    # Baseline Row
    print(f"SAM Baseline (100ep) & {b_dice.mean():.4f} $\pm$ {b_dice.std():.4f} & {b_hd.mean():.2f} $\pm$ {b_hd.std():.2f} & ... & ... \\\\")
    
    # Ours Row
    dice_star = "^{*}" if p_dice < 0.05 else ""
    hd_star = "^{*}" if p_hd < 0.05 else ""
    
    print(f"\\textbf{{ST-SAM (Ours)}} & \\textbf{{{m_dice.mean():.4f}}} $\pm$ {m_dice.std():.4f}{dice_star} & \\textbf{{{m_hd.mean():.2f}}} $\pm$ {m_hd.std():.2f}{hd_star} & ... & ... \\\\")
    print(r"\bottomrule")
    print(f"% Note: * indicates p < 0.05 (Wilcoxon signed-rank test)")

if __name__ == "__main__":
    print_sci_table()
    print_latex_code()