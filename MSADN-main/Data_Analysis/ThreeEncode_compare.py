import matplotlib.pyplot as plt
import numpy as np

# 数据
folds = np.arange(1, 14)
NCP_ND = [0.996, 0.996, 0.992, 0.96, 0.916, 0.796, 0.992, 0.996, 0.864, 0.93, 0.869, 0.958, 0.902]
onehot = [0.98, 0.98, 0.98, 0.95, 0.89, 0.71, 0.992, 0.992, 0.81, 0.91, 0.86, 0.94, 0.90]
threekmer = [0.98, 0.97, 0.97, 0.91, 0.89, 0.72, 0.99, 0.99, 0.79, 0.88, 0.80, 0.92, 0.85]


ncRNA_names = ['5S-rRNA', '5-8S-rRNA', 'tRNA', 'Ribozyme', 'CD-box', 'miRNA', 'Intron-gp-I', 'Intron-gp-II', 'HACA-box', 'Riboswitch', 'IRES', 'Leader', 'scaRNA']

# 条形图的宽度
bar_width = 0.22

# 横坐标位置
r1 = np.arange(len(folds))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# 设置字体和大小
label_font = {'family': 'serif', 'weight': 'bold', 'size': 20}
tick_font = {'family': 'serif', 'weight': 'bold', 'size': 15}


plt.figure(figsize=(14, 10))  # 图像宽度和高度

# 添加虚线的纵轴网格线，设置zorder较低的值
plt.grid(axis='y', linestyle='-', linewidth=1, zorder=0)
# 绘制条形图，设置zorder较高的值
plt.bar(r1, NCP_ND, color='#51B1B7', width=bar_width, edgecolor='white', label='NCP-ND', zorder=3)
plt.bar(r2, onehot, color='#E07B54', width=bar_width, edgecolor='white', label='OneHot', zorder=3)
plt.bar(r3, threekmer, color='#E1C855', width=bar_width, edgecolor='white', label='3k-mer', zorder=3)

# 添加标签
plt.ylabel('Accuracy', fontdict=label_font)
plt.title('Performance of MSADN with three encoding methods', fontdict=label_font)
plt.xticks([r + bar_width for r in range(len(folds))], ncRNA_names, rotation=35, ha='right', fontproperties=tick_font['family'], fontsize=tick_font['size'])
plt.yticks(fontproperties=tick_font['family'], fontsize=tick_font['size'])

# 调整纵轴的比例，缩小显示范围以放大差异
plt.ylim(0.7, 1.0)

# 去掉刻度线
plt.tick_params(axis='both', which='both', length=0)

# 加粗四条边
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# 添加图例
plt.legend(prop={'family': 'serif', 'size': 14, 'weight': 'bold'})

# 保存图片
output_image_path = 'H:/Google download/MSADN/MSADN-main/Data_Analysis/ThreeEncodeCompare_nRC/threeEncode_nRC.png'
plt.savefig(output_image_path)

# 显示图形
plt.show()