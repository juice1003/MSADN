import numpy as np

file_train = 'H:/Google download/MSADN/MSADN-main/NCY_Ten_Fold_Data/train_8'
file_test = 'H:/Google download/MSADN/MSADN-main/NCY_Ten_Fold_Data/test_8'

'''
encode_nucleotide(c, cb, i) 函数用于根据碱基 c（A、C、G、T）的不同，计算其编码后的特征向量。这个函数将碱基 c 的编码特征向量与计算的碱基 c 的密度信息 cb / float(i + 1) 结合起来，并返回一个包含这些信息的列表 p。

encode_seq_properties(s) 函数遍历输入的DNA序列 s，对每个碱基调用 encode_nucleotide 函数进行编码，并将结果存储在列表 f 中。最后，将列表 f 转换为numpy数组，并返回表示整个序列编码特征的二维数组 f。
'''


def encode_nucleotide(c, cb, i):
    bases = {'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0, ], 'T': [0, 0, 1]}
    p = bases[c]
    p.append(np.round(cb / float(i + 1), 2))
    return p

def encode_seq_properties(s):
    f=[]
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
            p = [0, 0, 0, 0]
        f.append(p)
    f = np.array(f)
    return f

def train_data():
    Train_Matrix = []
    Train_label = []
    for line in open(file_train):
        if line[0] == ">":
            if line.split("_")[1] == "5S-rRNA\n":
                Train_label.append(0)
            if (line.split("_")[1] == '5.8S-rRNA\n'):
                Train_label.append(1)
            if (line.split("_")[1] == 'tRNA\n'):
                Train_label.append(2)
            if (line.split("_")[1] == 'Ribozyme\n'):
                Train_label.append(3)
            if (line.split("_")[1] == 'CD-box\n'):
                Train_label.append(4)
            if (line.split("_")[1] == 'miRNA\n'):
                Train_label.append(5)
            if (line.split("_")[1] == 'Intron-gp-I\n'):
                Train_label.append(6)
            if (line.split("_")[1] == 'Intron-gp-II\n'):
                Train_label.append(7)
            if (line.split("_")[1] == 'HACA-box\n'):
                Train_label.append(8)
            if (line.split("_")[1] == 'Riboswitch\n'):
                Train_label.append(9)
            if (line.split("_")[1] == 'Y-RNA\n'):
                Train_label.append(10)
            if (line.split("_")[1] == 'Leader\n'):
                Train_label.append(11)
            if (line.split("_")[1] == 'Y-RNA-like\n'):
                Train_label.append(12)
        else:
            Tem_List = encode_seq_properties(line[0:len(line)-1])
            Train_Matrix.append(Tem_List)
    #
    # max_length = max(len(lst) for lst in Train_Matrix)  # Find the length of the longest list
    # Train_Matrix = [
    #     np.pad(lst, ((0, max_length - len(lst)), (0, 0)), 'constant') if len(lst.shape) > 1 else lst + [0] * (
    #                 max_length - len(lst)) for lst in Train_Matrix]  # Pad shorter lists with zeros
    Train_label = np.array(Train_label)
    return Train_Matrix,Train_label



def test_data():
    Test_Matrix = []
    Test_label = []
    for line in open(file_test):
        if line[0] == ">":
            if line.split("_")[1] == "5S-rRNA\n":
                Test_label.append(0)
            if (line.split("_")[1] == '5.8S-rRNA\n'):
                Test_label.append(1)
            if (line.split("_")[1] == 'tRNA\n'):
                Test_label.append(2)
            if (line.split("_")[1] == 'Ribozyme\n'):
                Test_label.append(3)
            if (line.split("_")[1] == 'CD-box\n'):
                Test_label.append(4)
            if (line.split("_")[1] == 'miRNA\n'):
                Test_label.append(5)
            if (line.split("_")[1] == 'Intron-gp-I\n'):
                Test_label.append(6)
            if (line.split("_")[1] == 'Intron-gp-II\n'):
                Test_label.append(7)
            if (line.split("_")[1] == 'HACA-box\n'):
                Test_label.append(8)
            if (line.split("_")[1] == 'Riboswitch\n'):
                Test_label.append(9)
            if (line.split("_")[1] == 'Y-RNA\n'):
                Test_label.append(10)
            if (line.split("_")[1] == 'Leader\n'):
                Test_label.append(11)
            if (line.split("_")[1] == 'Y-RNA-like\n'):
                Test_label.append(12)
        else:
            Tem_List = encode_seq_properties(line[0:len(line)-1])
            Test_Matrix.append(Tem_List)

    # max_length = max(len(lst) for lst in Test_Matrix)  # Find the length of the longest list
    # Test_Matrix = [
    #     np.pad(lst, ((0, max_length - len(lst)), (0, 0)), 'constant') if len(lst.shape) > 1 else lst + [0] * (
    #                 max_length - len(lst)) for lst in Test_Matrix]  # Pad shorter lists with zeros
    Test_label = np.array(Test_label)
    return Test_Matrix,Test_label

