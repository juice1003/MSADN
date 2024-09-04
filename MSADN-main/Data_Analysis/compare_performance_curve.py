import pandas as pd
import matplotlib.pyplot as plt

# 定义方法名称和类别
methods = ['ncRFP', 'ncDLRES', 'ncDENSE', 'NCYPred', 'MSADN']
categories = ['5S-rRNA', '5.8S-rRNA', 'tRNA', 'Ribozyme', 'CD-box', 'miRNA', 'Intron-gpI', 'Intron-gpII', 'HACA-box', 'Riboswitch', 'IRES', 'Leader', 'scaRNA']

# 分别加载每个文件的数据
ncRFP_data = pd.read_csv('H:/Google download/MSADN/MSADN-main/Data_Analysis/EachMethod_Data/ncRFP_metrics.csv')
ncDLRES_data = pd.read_csv('H:/Google download/MSADN/MSADN-main/Data_Analysis/EachMethod_Data/ncDLRES_metrics.csv')
ncDENSE_data = pd.read_csv('H:/Google download/MSADN/MSADN-main/Data_Analysis/EachMethod_Data/ncDENSE_metrics.csv')
NCYPred_data = pd.read_csv('H:/Google download/MSADN/MSADN-main/Data_Analysis/EachMethod_Data/NCYPred_metrics.csv')
MSADN_data = pd.read_excel('H:/Google download/MSADN/MSADN-main/Data_Analysis/EachMethod_Data/MSADN_metrics.xlsx')

# 将所有数据存储在一个字典中
data = {
    'ncRFP': ncRFP_data,
    'ncDLRES': ncDLRES_data,
    'ncDENSE': ncDENSE_data,
    'NCYPred': NCYPred_data,
    'MSADN': MSADN_data
}

colors = {
    'ncRFP': '#A6CDE4',
    'ncDLRES': '#74B69F',
    'ncDENSE': '#E58579',
    'NCYPred': '#D9BDD8',
    'MSADN': '#f73859'
}

# 设置serif字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']


# 准备绘制图表的数据
metrics = ['Precision', 'Recall', 'F1-Score', 'MCC']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.subplots_adjust(wspace=0.1, hspace=0.2)
axes = axes.flatten()

# 绘制每个指标的折线图
for i, metric in enumerate(metrics):
    ax = axes[i]
    for method in methods:
        ax.plot(categories, data[method][metric], marker='o', label=method, color=colors[method], linewidth=3.0)
    ax.set_title(metric, fontsize=20)  # 设置标题字体大小
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=14)  # 设置x轴标签字体大小
    ax.tick_params(axis='y', labelsize=14)  # 设置y轴刻度标签字体大小
    ax.set_ylim([0.4, 1.0])
    ax.legend(fontsize=14)  # 设置图例字体大小
    # 添加虚线样式的纵轴刻度线
    ax.grid(axis='y', linestyle='-', linewidth=1)
    # 去除刻度线
    # ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    # 调整边框粗细
    for spine in ax.spines.values():
        spine.set_linewidth(2)

plt.tight_layout()
output_image_path = 'H:/Google download/MSADN/MSADN-main/Data_Analysis/EachMethod_Data/eachMethod_curve_nRC.png'
plt.savefig(output_image_path)

plt.show()
