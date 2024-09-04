import numpy as np
from itertools import product



file_train = '/home/lichangyong/文档/NCMMLP/ncMMLP-main/nRC_Ten_Fold_Data/Train_9'
file_test = '/home/lichangyong/文档/NCMMLP/ncMMLP-main/nRC_Ten_Fold_Data/Test_9'

def k_mer(sequence, k):
    kmers = []
    for i in range(len(sequence) - k):
        kmer = sequence[i:i + k]
        kmers.append(kmer)
    bases = 'ATCG'
    combinations = [''.join(c) for c in product(bases, repeat=k)]
    combinations = np.array(combinations)
    encoding = np.zeros((len(kmers), len(combinations)))
    for i, kmer in enumerate(kmers):
        matches = np.where(combinations == kmer)[0]
        if matches.size > 0:
            j = matches[0]
            encoding[i][j] = 1
    return encoding

def train_data():
    Train_Matrix = []
    Train_label = []
    max_length = 0
    for line in open(file_train):
        if (line[0] == '>'):
            if (line.split()[-1] == '5S_rRNA'):
                Train_label.append(0)
            if (line.split()[-1] == '5_8S_rRNA'):
                Train_label.append(1)
            if (line.split()[-1] == 'tRNA'):
                Train_label.append(2)
            if (line.split()[-1] == 'ribozyme'):
                Train_label.append(3)
            if (line.split()[-1] == 'CD-box'):
                Train_label.append(4)
            if (line.split()[-1] == 'miRNA'):
                Train_label.append(5)
            if (line.split()[-1] == 'Intron_gpI'):
                Train_label.append(6)
            if (line.split()[-1] == 'Intron_gpII'):
                Train_label.append(7)
            if (line.split()[-1] == 'HACA-box'):
                Train_label.append(8)
            if (line.split()[-1] == 'riboswitch'):
                Train_label.append(9)
            if (line.split()[-1] == 'IRES'):
                Train_label.append(10)
            if (line.split()[-1] == 'leader'):
                Train_label.append(11)
            if (line.split()[-1] == 'scaRNA'):
                Train_label.append(12)
            pass
        else:
            Tem_List = k_mer(line, 3)
            if len(Tem_List) > max_length:
                max_length = len(Tem_List)
            Train_Matrix.append(Tem_List)

    # # Pad shorter sequences with zeros
    # for i in range(len(Train_Matrix)):
    #     if len(Train_Matrix[i]) < max_length:
    #         Train_Matrix[i] = np.pad(Train_Matrix[i], ((0, max_length - len(Train_Matrix[i])), (0, 0)),
    #                                          'constant')
    #
    # Train_Matrix = np.array(Train_Matrix)
    Train_label = np.array(Train_label)

    return Train_Matrix, Train_label


def test_data():
    Test_Matrix = []
    Test_label = []
    max_length = 0
    for line in open(file_test):
        if (line[0] == '>'):
            if (line.split()[-1] == '5S_rRNA'):
                Test_label.append(0)
            if (line.split()[-1] == '5_8S_rRNA'):
                Test_label.append(1)
            if (line.split()[-1] == 'tRNA'):
                Test_label.append(2)
            if (line.split()[-1] == 'ribozyme'):
                Test_label.append(3)
            if (line.split()[-1] == 'CD-box'):
                Test_label.append(4)
            if (line.split()[-1] == 'miRNA'):
                Test_label.append(5)
            if (line.split()[-1] == 'Intron_gpI'):
                Test_label.append(6)
            if (line.split()[-1] == 'Intron_gpII'):
                Test_label.append(7)
            if (line.split()[-1] == 'HACA-box'):
                Test_label.append(8)
            if (line.split()[-1] == 'riboswitch'):
                Test_label.append(9)
            if (line.split()[-1] == 'IRES'):
                Test_label.append(10)
            if (line.split()[-1] == 'leader'):
                Test_label.append(11)
            if (line.split()[-1] == 'scaRNA'):
                Test_label.append(12)
                pass
        else:
            Tem_List = k_mer(line, 3)
            if len(Tem_List) > max_length:
                max_length = len(Tem_List)
            Test_Matrix.append(Tem_List)

    #         # Pad shorter sequences with zeros
    # for i in range(len(Test_Matrix)):
    #     if len(Test_Matrix[i]) < max_length:
    #         Test_Matrix[i] = np.pad(Test_Matrix[i], ((0, max_length - len(Test_Matrix[i])), (0, 0)),
    #                         'constant')
    #
    # Train_Matrix = np.array(Test_Matrix)
    Test_label = np.array(Test_label)

    return Test_Matrix, Test_label
