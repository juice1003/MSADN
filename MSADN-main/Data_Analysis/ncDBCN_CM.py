# 导入必要的库
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 数据
cm_prob = np.array([
    [0.97, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0.005],
    [0, 0.97, 0, 0, 0.005, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0.965, 0, 0, 0, 0.03, 0, 0, 0, 0, 0, 0, 0.005],
    [0, 0, 0, 0.96, 0, 0.015, 0, 0, 0.005, 0.005, 0, 0, 0.005, 0],
    [0, 0, 0, 0, 0.835, 0.08, 0, 0, 0, 0, 0, 0.035, 0, 0.015],
    [0.005, 0, 0.01, 0, 0.055, 0.81, 0, 0, 0, 0.05, 0, 0.01, 0.035, 0],
    [0, 0, 0, 0, 0, 0, 0.975, 0.01, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0.005, 0.985, 0, 0, 0.005, 0, 0, 0],
    [0, 0, 0, 0, 0.015, 0.015, 0.01, 0, 0.86, 0.01, 0, 0, 0.005, 0.04],
    [0, 0, 0, 0.005, 0, 0, 0, 0, 0.02, 0.855, 0, 0.005, 0, 0],
    [0.01, 0, 0, 0, 0, 0, 0, 0, 0.02, 0, 0.885, 0, 0.025, 0.015],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0.955, 0, 0],
    [0, 0.005, 0, 0, 0, 0.025, 0, 0, 0, 0, 0, 0, 0.97, 0],
    [0, 0, 0.005, 0, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0.985]
])

labels = ['5S-rRNA', '5.8S-rRNA', 'tRNA', 'Ribozyme', 'CD-box', 'miRNA', 'Intron-gp-I', 
          'Intron-gp-II', 'HACA-box', 'Riboswitch', 'IRES', 'leader', 'scaRNA']

# 初始化注释矩阵
annot = np.empty_like(cm_prob, dtype=object)

# 填充注释矩阵
for i in range(cm_prob.shape[0]):
    for j in range(cm_prob.shape[1]):
        if cm_prob[i, j] == 0:
            annot[i, j] = '0'
        else:
            annot[i, j] = f'{cm_prob[i, j]:.3f}'

# 绘制热力图
plt.figure(figsize=(15, 15))
sns.heatmap(cm_prob, annot=annot, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False,
            annot_kws={"size": 14})
plt.xlabel('Predicted label', size=30)
plt.ylabel('True label', size=30)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(rotation=0, fontsize=20)
plt.title('ncDBCN', size=36)
plt.subplots_adjust(bottom=0.15, top=0.95, left=0.15, right=0.95)

output_image_path = 'H:/Google download/MSADN/MSADN-main/Data_Analysis/confusion_matrix_nRC/ncDBCN_confusion_matrix_nRC.png'
plt.savefig(output_image_path)
print(f"Confusion matrix saved to {output_image_path}")
plt.show()
