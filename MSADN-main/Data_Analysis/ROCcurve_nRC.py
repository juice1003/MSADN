from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置字体为serif
plt.rcParams['font.family'] = 'serif'

path = [f'H:/Google download/MSADN/MSADN-main/Data_Analysis/with Predprob_csv_nRC/Test{i}_nRC.csv' for i in range(10)]
y_true = []
y_pred = []

for p in path:
    data = pd.read_csv(p)
    for i in range(len(data)):
        y_true.append(int(data.iloc[i, 1]))  # 第2列是标签
        y_pred.append(np.fromstring(data.iloc[i, 3].strip('[]'), sep=' '))  # 第4列是预测的概率值，转换为浮点数数组

# 将y_true转换为numpy数组
y_true = np.array(y_true, dtype=int)
y_pred = np.array(y_pred)

# 初始化绘图参数
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], linestyle='--', lw=0.8, color='gray', label='Random Chance')

# 类别名
class_names = [
    "5S-rRNA", "5-8S-rRNA", "tRNA", "Ribozyme", "CD-box", "miRNA",
    "Intron-gp-I", "Intron-gp-II", "HACA-box", "Riboswitch",
    "IRES", "Leader", "scaRNA"
]

# 计算并绘制每个类别的ROC曲线
for i in range(13):
    fpr, tpr, _ = roc_curve(y_true == i, y_pred[:, i], pos_label=1)
    fpr_interp = np.linspace(0, 1, 1000)
    tpr_interp = np.interp(fpr_interp, fpr, tpr)
    roc_auc = auc(fpr_interp, tpr_interp)
    plt.plot(fpr_interp, tpr_interp, lw=1, linestyle='-', label='class {0} (AUC = {1:.2f})'.format(class_names[i], roc_auc))

# 设置图像标题、轴标签、图例
plt.title('ROC Curves on nRC dataset', fontsize=20, fontname='serif')
plt.xlabel('False Positive Rate', fontsize=20, fontname='serif')
plt.ylabel('True Positive Rate', fontsize=20, fontname='serif')
plt.legend(loc='lower right', prop={'size': 15, 'family': 'serif'})
# 调整横轴和纵轴刻度值的字体大小
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# 设置x轴和y轴的范围，使它们的最小值为零
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

output_image_path = 'H:/Google download/MSADN/MSADN-main/Data_Analysis/ROC&PR curve/ROC_curve_nRC.png'
plt.savefig(output_image_path)

# 显示图表
plt.show()
