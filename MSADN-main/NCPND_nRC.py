import numpy as np

file_train = 'H:/Google download/NCMMLP/ncMMLP-main/nRC_Ten_Fold_Data/Train_0'
file_test = 'H:/Google download/NCMMLP/ncMMLP-main/nRC_Ten_Fold_Data/Test_0'#记得保存对应的.csv文件

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
            if (line.split()[-1] == 'miRNA'):#
                Train_label.append(5)
            if (line.split()[-1] == 'Intron_gpI'):#
                Train_label.append(6)
            if (line.split()[-1] == 'Intron_gpII'):
                Train_label.append(7)
            if (line.split()[-1] == 'HACA-box'):#
                Train_label.append(8)
            if (line.split()[-1] == 'riboswitch'):#
                Train_label.append(9)
            if (line.split()[-1] == 'IRES'):
                Train_label.append(10)
            if (line.split()[-1] == 'leader'):
                Train_label.append(11)
            if (line.split()[-1] == 'scaRNA'):
                Train_label.append(12)
        else:
            Tem_List = encode_seq_properties(line[0:len(line)-1])
            Train_Matrix.append(Tem_List)
    Train_label = np.array(Train_label)
    return Train_Matrix,Train_label



def test_data():
    Test_Matrix = []
    Test_label = []
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
        else:
            Tem_List = encode_seq_properties(line[0:len(line)-1])
            Test_Matrix.append(Tem_List)
    Test_label = np.array(Test_label)
    return Test_Matrix,Test_label

