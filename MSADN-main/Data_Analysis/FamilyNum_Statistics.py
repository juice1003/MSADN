import os
from collections import defaultdict

# 数据目录的路径
data_dir = 'H:/Google download/MSADN/MSADN-main/NCY_Ten_Fold_Data'

# ncRNA 类别列表
ncRNA_classes = ['5S-rRNA', '5.8S-rRNA', 'tRNA', 'Ribozyme', 'CD-box', 'miRNA',
                 'Intron-gp-I', 'Intron-gp-II', 'HACA-box', 'Riboswitch', 'Y-RNA',
                 'Leader', 'Y-RNA-like']

# 统计给定文件中的 ncRNA 类别数量的函数
def count_ncRNA_classes(file_path):
    counts = defaultdict(int)
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('>'):
                for ncRNA_class in ncRNA_classes:
                    if ncRNA_class in line:
                        counts[ncRNA_class] += 1
                        break
    return counts


# 用于存储训练数据和测试数据的统计结果的字典
train_counts = defaultdict(int)
test_counts = defaultdict(int)

# 处理目录中的每个文件
for subdir, dirs, files in os.walk(data_dir):
    for file in files:
        file_path = os.path.join(subdir, file)
        file_counts = count_ncRNA_classes(file_path)

        # 更新整体统计
        if 'train' in file:
            for key, value in file_counts.items():
                train_counts[key] += value
        elif 'test' in file:
            for key, value in file_counts.items():
                test_counts[key] += value

# 打印结果
print("训练数据统计：")
for ncRNA_class in ncRNA_classes:
    print(f"{ncRNA_class}: {train_counts[ncRNA_class]}")

print("\n测试数据统计：")
for ncRNA_class in ncRNA_classes:
    print(f"{ncRNA_class}: {test_counts[ncRNA_class]}")
