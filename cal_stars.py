import scipy.stats as stats

# 1. 录入数据
data_sam = {
    'Dice':   [0.9062, 0.9236, 0.8666, 0.8889, 0.8966],
    'IoU':    [0.8307, 0.8592, 0.7687, 0.8041, 0.8158],
    'Prec':   [0.9180, 0.9161, 0.8793, 0.9210, 0.8947],
    'Recall': [0.8988, 0.9342, 0.8617, 0.8639, 0.9034],
    'HD95':   [6.6539, 7.6115, 8.5054, 8.0779, 7.1006],
    'ASD':    [1.7367, 2.1061, 2.9561, 2.4819, 2.3058]
}

data_st = {
    'Dice':   [0.9196, 0.9338, 0.8878, 0.9051, 0.9110],
    'IoU':    [0.8529, 0.8768, 0.8019, 0.8301, 0.8395],
    'Prec':   [0.9500, 0.9326, 0.8822, 0.9370, 0.9140],
    'Recall': [0.8938, 0.9374, 0.8998, 0.8794, 0.9120],
    'HD95':   [6.3700, 7.5700, 8.6100, 7.5300, 6.7000],
    'ASD':    [1.5400, 1.8600, 2.6400, 2.0900, 1.9800]
}

# 2. 计算并打印 p 值
print("=== Paired t-test Results ===")
for metric in data_sam.keys():
    t_stat, p_value = stats.ttest_rel(data_st[metric], data_sam[metric])
    
    # 判断显著性级别
    stars = ""
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = "ns" #(not significant)
        
    print(f"{metric:>6}: p-value = {p_value:.5f}  [{stars}]")