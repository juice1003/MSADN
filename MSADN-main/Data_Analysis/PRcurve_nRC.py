from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置字体为serif
plt.rcParams['font.family'] = 'serif'

# 数据路径
path = [f'H:/Google download/MSADN/MSADN-main/Data_Analysis/with Predprob_csv_nRC/Test{i}_nRC.csv' for i in range(10)]
y_true = []
y_pred = []

# 读取数据
for p in path:
    data = pd.read_csv(p)
    for i in range(len(data)):
        y_true.append(int(data.iloc[i, 1]))  # 第2列是标签
        y_pred.append(np.fromstring(data.iloc[i, 3].strip('[]'), sep=' '))  # 第4列是预测的概率值，转换为浮点数数组

# 将y_true转换为numpy数组，确保它们是整数类型
y_true = np.array(y_true, dtype=int)
y_pred = np.array(y_pred)

# 初始化绘图参数
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [1, 0], linestyle='--', lw=0.8, color='gray', label='Random Chance')

# 类别名
class_names = [
    "5S-rRNA", "5-8S-rRNA", "tRNA", "Ribozyme", "CD-box", "miRNA",
    "Intron-gp-I", "Intron-gp-II", "HACA-box", "Riboswitch",
    "IRES", "Leader", "scaRNA"
]

# 计算并绘制每个类别的PR曲线
for i in range(13):
    precision, recall, _ = precision_recall_curve(y_true == i, y_pred[:, i])
    average_precision = average_precision_score(y_true == i, y_pred[:, i])
    plt.plot(recall, precision, lw=1, label=' class {0} (area = {1:.2f})'.format(class_names[i], average_precision))

# 设置图像标题、轴标签、图例，并指定字体大小
plt.title('PR Curves on nRC dataset', fontsize=20, fontname='serif')
plt.xlabel('Recall', fontsize=20, fontname='serif')
plt.ylabel('Precision', fontsize=20, fontname='serif')
plt.legend(loc='lower left', prop={'size': 15, 'family': 'serif'})
# 调整横轴和纵轴刻度值的字体大小
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# 设置x轴和y轴的范围，使它们的最小值为零
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

# 调整线条宽度
for line in plt.gca().get_lines():
    line.set_linewidth(1)

# 保存图像
output_image_path = 'H:/Google download/MSADN/MSADN-main/Data_Analysis/ROC&PR curve/PR_curve_nRC.png'
plt.savefig(output_image_path)

# 显示图表
plt.show()
