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
                if line.split("_")[1] == "5S-rRNA\n":
                    Data_label.append(0)
                elif line.split("_")[1] == '5.8S-rRNA\n':
                    Data_label.append(1)
                elif line.split("_")[1] == 'tRNA\n':
                    Data_label.append(2)
                elif line.split("_")[1] == 'Ribozyme\n':
                    Data_label.append(3)
                elif line.split("_")[1] == 'CD-box\n':
                    Data_label.append(4)
                elif line.split("_")[1] == 'miRNA\n':
                    Data_label.append(5)
                elif line.split("_")[1] == 'Intron-gp-I\n':
                    Data_label.append(6)
                elif line.split("_")[1] == 'Intron-gp-II\n':
                    Data_label.append(7)
                elif line.split("_")[1] == 'HACA-box\n':
                    Data_label.append(8)
                elif line.split("_")[1] == 'Riboswitch\n':
                    Data_label.append(9)
                elif line.split("_")[1] == 'Y-RNA\n':
                    Data_label.append(10)
                elif line.split("_")[1] == 'Leader\n':
                    Data_label.append(11)
                elif line.split("_")[1] == 'Y-RNA-like\n':
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
        file_path = f'H:/Google download/MSADN/MSADN-main/NCY_Ten_Fold_Data/train_{i}'
        train_matrix, train_labels = load_data(file_path)
        all_train_matrices.append(train_matrix)
        all_train_labels.append(train_labels)

    return all_train_matrices, all_train_labels


def visualize_features(matrix, labels, title="t-SNE visualization"):
    # Flatten the 3D matrix to 2D
    matrix_2d = [seq.flatten() for seq in matrix]

    # 随机抽取数据来可视化
    sample_size = 3000  # 根据可用内存调整此数值
    if len(matrix_2d) > sample_size:
        indices = np.random.choice(len(matrix_2d), sample_size, replace=False)
        matrix_2d = [matrix_2d[i] for i in indices]
        labels = [labels[i] for i in indices]

    # 转换为numpy数组
    matrix_2d = np.array(matrix_2d)
    labels = np.array(labels)

    # 使用PCA降维为50
    pca = PCA(n_components=50)
    matrix_pca = pca.fit_transform(matrix_2d)

    # 使用tSne降维为2
    tsne = TSNE(n_components=2, random_state=0)
    tsne_result = tsne.fit_transform(matrix_pca)

    # 定义13种类的颜色
    colors = ['#FEA47F', '#25CCF7', '#EAB543', '#55E6C1', '#CAD3C8', '#F97F51', '#1B9CFC',
              '#F8EFBA', '#58B19F', '#2C3A47', '#B33771', '#3B3B98', '#FD7272']

    # Create a scatter plot of the t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap=mcolors.ListedColormap(colors))
    # plt.colorbar(scatter, ticks=range(13))
    # plt.clim(-0.5, 12.5)
    cbar = plt.colorbar(scatter, ticks=range(13))
    cbar.ax.set_yticklabels(['5S-rRNA', '5.8S-rRNA', 'tRNA', 'Ribozyme', 'CD-box', 'miRNA', 'Intron-gp-I',
                             'Intron-gp-II', 'HACA-box', 'Riboswitch', 'Y-RNA', 'Leader', 'Y-RNA-like'])
    plt.title(title)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    output_image_path = 'H:/Google download/MSADN/MSADN-main/Data_Analysis/FeatureVisualize_NCY/tSNE_NCY.png'
    plt.savefig(output_image_path)
    plt.show()


# 加载和可视化
all_train_matrices, all_train_labels = train_data_all()
all_train_matrices_flat = np.concatenate(all_train_matrices, axis=0)
all_train_labels_flat = np.concatenate(all_train_labels, axis=0)
visualize_features(all_train_matrices_flat, all_train_labels_flat, title="t-SNE Visualization of NCY Dataset")
