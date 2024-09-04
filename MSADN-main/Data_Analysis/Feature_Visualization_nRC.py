import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def encode_nucleotide(c, cb, i):
    bases = {'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0], 'T': [0, 0, 1]}
    p = bases.get(c, [0, 0, 0])  # 处理无法识别的核苷酸
    p.append(np.round(cb / float(i + 1), 2))
    return p

def encode_seq_properties(s):
    f = []
    cba = cbc = cbt = cbg = 0
    for i, c in enumerate(s):
        if c == 'A':
            cba += 1
            p = encode_nucleotide(c, cba, i)
        elif c == 'T':
            cbt += 1
            p = encode_nucleotide(c, cbt, i)
        elif c == 'C':
            cbc += 1
            p = encode_nucleotide(c, cbc, i)
        elif c == 'G':
            cbg += 1
            p = encode_nucleotide(c, cbg, i)
        else:
            p = [0, 0, 0, 0]  # 设置无法识别字符的默认值
        f.append(p)
    f = np.array(f)
    return f

def load_data(file_path):
    Data_Matrix = []
    Data_label = []
    with open(file_path, 'r') as f:
        for line in f:
            if line[0] == ">":
                label = line.split()[-1]
                if label == "5S_rRNA":
                    Data_label.append(0)
                elif label == '5_8S_rRNA':
                    Data_label.append(1)
                elif label == 'tRNA':
                    Data_label.append(2)
                elif label == 'ribozyme':
                    Data_label.append(3)
                elif label == 'CD-box':
                    Data_label.append(4)
                elif label == 'miRNA':
                    Data_label.append(5)
                elif label == 'Intron_gpI':
                    Data_label.append(6)
                elif label == 'Intron_gpII':
                    Data_label.append(7)
                elif label == 'HACA-box':
                    Data_label.append(8)
                elif label == 'riboswitch':
                    Data_label.append(9)
                elif label == 'IRES':
                    Data_label.append(10)
                elif label == 'leader':
                    Data_label.append(11)
                elif label == 'scaRNA':
                    Data_label.append(12)
            else:
                Tem_List = encode_seq_properties(line.strip())
                Data_Matrix.append(Tem_List)

    max_length = max(len(lst) for lst in Data_Matrix)
    Data_Matrix = [
        np.pad(lst, ((0, max_length - len(lst)), (0, 0)), 'constant') if len(lst.shape) > 1 else lst + [0] * (
                max_length - len(lst)) for lst in Data_Matrix]
    Data_label = np.array(Data_label)
    return np.array(Data_Matrix), Data_label

def train_data_all():
    all_train_matrices = []
    all_train_labels = []
    for i in range(10):  # train_0 to train_9
        file_path = f'H:/Google download/MSADN/MSADN-main/nRC_Ten_Fold_Data/Train_{i}'
        train_matrix, train_labels = load_data(file_path)
        # print(f"Train matrix {i} shape: {train_matrix.shape}")
        # print(f"Train labels {i} length: {len(train_labels)}")
        # print(f"Sample labels: {train_labels[:10]}")  # 打印前10个标签进行调试
        all_train_matrices.append(train_matrix)
        all_train_labels.append(train_labels)

    return all_train_matrices, all_train_labels

def visualize_features(matrix, labels, title="t-SNE visualization"):
    # 将三维矩阵展平为二维
    matrix_2d = [seq.flatten() for seq in matrix]

    # 随机抽样以进行可视化
    sample_size = 1500  # 可以根据数据大小和可用内存调整此数值
    if len(matrix_2d) > sample_size:
        indices = np.random.choice(len(matrix_2d), sample_size, replace=False)
        matrix_2d = [matrix_2d[i] for i in indices]
        labels = [labels[i] for i in indices]

    # 转换为 NumPy 数组
    matrix_2d = np.array(matrix_2d)
    labels = np.array(labels)

    # 使用 PCA 进行降维
    pca = PCA(n_components=50)
    matrix_pca = pca.fit_transform(matrix_2d)

    # 使用 t-SNE 进一步降维到2维
    tsne = TSNE(n_components=2, random_state=0)
    tsne_result = tsne.fit_transform(matrix_pca)

    # 定义一个包含13种不同颜色的颜色映射
    colors = ['#FEA47F', '#25CCF7', '#EAB543', '#55E6C1', '#CAD3C8', '#F97F51', '#1B9CFC',
              '#F8EFBA', '#58B19F', '#2C3A47', '#F6416c', '#3B3B98', '#3F72aF']

    # 创建 t-SNE 结果的散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap=mcolors.ListedColormap(colors))
    cbar = plt.colorbar(scatter, ticks=range(13))
    cbar.ax.set_yticklabels(['5S-rRNA', '5.8S-rRNA', 'tRNA', 'Ribozyme', 'CD-box', 'miRNA', 'Intron-gp-I',
                             'Intron-gp-II', 'HACA-box', 'Riboswitch', 'IRES', 'Leader', 'scaRNA'])
    plt.title(title)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    output_image_path = 'H:/Google download/MSADN/MSADN-main/Data_Analysis/FeatureVisualize_nRC/tSNE_nRC.png'
    plt.savefig(output_image_path)
    plt.show()

# 加载和可视化
all_train_matrices, all_train_labels = train_data_all()

# 找到所有矩阵中第二维度的最大值
max_length = max(matrix.shape[1] for matrix in all_train_matrices)

# 调整所有矩阵的形状，使它们的第二维度一致
all_train_matrices_padded = [np.pad(matrix, ((0, 0), (0, max_length - matrix.shape[1]), (0, 0)), 'constant') for matrix in all_train_matrices]

# 展平所有矩阵
all_train_matrices_flat = np.concatenate(all_train_matrices_padded, axis=0)
all_train_labels_flat = np.concatenate(all_train_labels, axis=0)

# print(f"Total samples: {all_train_matrices_flat.shape[0]}")
# print(f"Total labels: {all_train_labels_flat.shape[0]}")

# 确保样本数和标签数一致
assert all_train_matrices_flat.shape[0] == all_train_labels_flat.shape[0], "样本数和标签数不一致"

# 可视化特征
visualize_features(all_train_matrices_flat, all_train_labels_flat, title="t-SNE Visualization of nRC Dataset")
