import numpy as np

file_train = '/home/lichangyong/Documents/gz/ncDENSE/ncMMLP-main/nRC_Ten_Fold_Data/Train_6'
file_test = '/home/lichangyong/Documents/gz/ncDENSE/ncMMLP-main/nRC_Ten_Fold_Data/Test_6'

# EIIP编码映射
eiip_dict = {
    'A': 0.1260,
    'C': 0.1340,
    'G': 0.0806,
    'T': 0.1335
}

# def encode_eiip(seq):
#     return [eiip_dict.get(base, 0) for base in seq]

def encode_seq_properties_eiip(seq):
    f = []
    for base in seq:
        f.append([eiip_dict.get(base, 0)])
    return np.array(f)

def train_data():
    Train_Matrix = []
    Train_label = []
    for line in open(file_train):
        if line[0] == '>':
            if line.split()[-1] == '5S_rRNA':
                Train_label.append(0)
            elif line.split()[-1] == '5_8S_rRNA':
                Train_label.append(1)
            elif line.split()[-1] == 'tRNA':
                Train_label.append(2)
            elif line.split()[-1] == 'ribozyme':
                Train_label.append(3)
            elif line.split()[-1] == 'CD-box':
                Train_label.append(4)
            elif line.split()[-1] == 'miRNA':
                Train_label.append(5)
            elif line.split()[-1] == 'Intron_gpI':
                Train_label.append(6)
            elif line.split()[-1] == 'Intron_gpII':
                Train_label.append(7)
            elif line.split()[-1] == 'HACA-box':
                Train_label.append(8)
            elif line.split()[-1] == 'riboswitch':
                Train_label.append(9)
            elif line.split()[-1] == 'IRES':
                Train_label.append(10)
            elif line.split()[-1] == 'leader':
                Train_label.append(11)
            elif line.split()[-1] == 'scaRNA':
                Train_label.append(12)
        else:
            Tem_List = encode_seq_properties_eiip(line.strip())
            Train_Matrix.append(Tem_List)

    max_length = max(len(lst) for lst in Train_Matrix)  # Find the length of the longest list
    Train_Matrix = [
        np.pad(lst, ((0, max_length - len(lst)), (0, 0)), 'constant') if len(lst.shape) > 1 else lst + [0] * (
                    max_length - len(lst)) for lst in Train_Matrix]  # Pad shorter lists with zeros
    Train_label = np.array(Train_label)
    return Train_Matrix, Train_label

def test_data():
    Test_Matrix = []
    Test_label = []
    for line in open(file_test):
        if line[0] == '>':
            if line.split()[-1] == '5S_rRNA':
                Test_label.append(0)
            elif line.split()[-1] == '5_8S_rRNA':
                Test_label.append(1)
            elif line.split()[-1] == 'tRNA':
                Test_label.append(2)
            elif line.split()[-1] == 'ribozyme':
                Test_label.append(3)
            elif line.split()[-1] == 'CD-box':
                Test_label.append(4)
            elif line.split()[-1] == 'miRNA':
                Test_label.append(5)
            elif line.split()[-1] == 'Intron_gpI':
                Test_label.append(6)
            elif line.split()[-1] == 'Intron_gpII':
                Test_label.append(7)
            elif line.split()[-1] == 'HACA-box':
                Test_label.append(8)
            elif line.split()[-1] == 'riboswitch':
                Test_label.append(9)
            elif line.split()[-1] == 'IRES':
                Test_label.append(10)
            elif line.split()[-1] == 'leader':
                Test_label.append(11)
            elif line.split()[-1] == 'scaRNA':
                Test_label.append(12)
        else:
            Tem_List = encode_seq_properties_eiip(line.strip())
            Test_Matrix.append(Tem_List)

    max_length = max(len(lst) for lst in Test_Matrix)  # Find the length of the longest list
    Test_Matrix = [
        np.pad(lst, ((0, max_length - len(lst)), (0, 0)), 'constant') if len(lst.shape) > 1 else lst + [0] * (
                    max_length - len(lst)) for lst in Test_Matrix]  # Pad shorter lists with zeros
    Test_label = np.array(Test_label)
    return Test_Matrix, Test_label


