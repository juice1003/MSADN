import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import NCPND_NCY
from torch.nn import utils as nn_utils
import pandas as pd
import os
from sklearn.metrics import classification_report, matthews_corrcoef, precision_score, recall_score, f1_score, \
    confusion_matrix, roc_auc_score, average_precision_score

PATH_Model = 'H:/Google download/MSADN/MSADN-main/Trained_Model/Model_NCY.pt'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class MinimalDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.data)

def collate_fn(batch_data):
    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    data_length = [len(xi[0]) for xi in batch_data]
    sent_seq = [xi[0] for xi in batch_data]
    label = [xi[1] for xi in batch_data]
    padden_sent_seq = pad_sequence([torch.from_numpy(x) for x in sent_seq], batch_first=True, padding_value=0)
    return padden_sent_seq, data_length, torch.tensor(label, dtype=torch.float32)

Test_Data, Test_Label = NCPND_NCY.test_data()

model = torch.load(PATH_Model)
if torch.cuda.is_available():
    model = model.cuda()

test_data = MinimalDataset(Test_Data, Test_Label)

criterion = nn.CrossEntropyLoss()

test_data_loader = DataLoader(test_data, batch_size=32, shuffle=True, collate_fn=collate_fn)

model.eval()
max_acc = 0
model.eval()
List_Data = []
with torch.no_grad():
    correct = 0
    total = 0
    loss_totall = 0
    iii = 0
    all_preds = []  # 存储预测标签
    all_labels = []  # 存储真实标签
    all_probs = []  # 存储预测概率

    for item_test in test_data_loader:
        test_data, test_length, test_label = item_test
        num = test_label.shape[0]
        test_data = test_data.float()
        test_label = test_label.long()
        test_data = Variable(test_data)
        test_label = Variable(test_label)
        if torch.cuda.is_available():
            test_data = test_data.cuda()
            test_label = test_label.cuda()
        pack = nn_utils.rnn.pack_padded_sequence(test_data, test_length, batch_first=True)
        outputs = model(pack)
        loss = criterion(outputs, test_label)
        loss_totall += loss.data.item()
        iii += 1

        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # 使用softmax获取概率
        _, pred_acc = torch.max(probabilities.data, 1)  # 获取预测的类别
        correct += (pred_acc == test_label).sum()
        total += test_label.size(0)

        all_preds.extend(pred_acc.cpu().numpy())
        all_labels.extend(test_label.cpu().numpy())
        all_probs.extend(probabilities.cpu().numpy())  # 存储每个样本的预测概率

        for i in range(num):
            List_Tem = []
            List_Tem.append(test_label[i].item())
            List_Tem.append(pred_acc[i].item())
            List_Tem.append(probabilities[i].cpu().numpy())  # 添加预测概率到列表中
            List_Data.append(List_Tem)

    accuracy = 100 * correct / total
    print(f'Accuracy of the test data: {accuracy:.2f}%')
    print('Loss of the test data: {}'.format(loss_totall / iii))

    # 计算并打印总体性能指标
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    mcc = matthews_corrcoef(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    pr_auc = average_precision_score(all_labels, all_probs, average='macro')

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'MCC: {mcc:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')
    print(f'PR AUC: {pr_auc:.4f}')

    print('-------------------------Indicator of 13 Classes--------------------------')
    # 计算每个类别的性能指标
    report = classification_report(all_labels, all_preds, digits=4, output_dict=True)
    for class_id in range(13):  # 有13个类，类标签为0到12
        class_id_str = str(class_id)
        class_precision = report.get(class_id_str, {}).get('precision', 0)
        class_recall = report.get(class_id_str, {}).get('recall', 0)
        class_f1_score = report.get(class_id_str, {}).get('f1-score', 0)

        print(f'Class {class_id}:')
        print(f'\tPrecision: {class_precision:.4f}')
        print(f'\tRecall: {class_recall:.4f}')
        print(f'\tF1-score: {class_f1_score:.4f}')

        # 计算每个类别的AUC-ROC和AUC-PR
        class_roc_auc = roc_auc_score(np.array(all_labels) == class_id, np.array(all_probs)[:, class_id])
        class_pr_auc = average_precision_score(np.array(all_labels) == class_id, np.array(all_probs)[:, class_id])

        print(f'\tROC AUC: {class_roc_auc:.4f}')
        print(f'\tPR AUC: {class_pr_auc:.4f}')

    # 计算每个类别的MCC和准确率
    print('--------------------------ACC&MCC--------------------------')
    cm = confusion_matrix(all_labels, all_preds)
    for class_id in range(13):
        TP = cm[class_id, class_id]
        FP = cm[:, class_id].sum() - TP
        FN = cm[class_id, :].sum() - TP
        TN = cm.sum() - TP - FP - FN

        numerator = (TP * TN) - (FP * FN)
        denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        class_mcc = numerator / denominator if denominator != 0 else 0.0

        class_accuracy = TP / (TP + FN) if (TP + FN) != 0 else 0.0

        print(f'\tMCC for class {class_id}: {class_mcc:.4f}')
        print(f'\tAccuracy for class {class_id}: {class_accuracy:.4f}')

num = 0
Index_List = []
for i in List_Data:
    num = num + 1
    if i[0] != i[1]:
        Index_List.append(num - 1)

# 确保预测概率也保存到CSV文件中
name = ['Real Label', 'Predict Label', 'Predict Probabilities']
Pre_Data = pd.DataFrame(columns=name, data=List_Data)
Pre_Data.to_csv('H:/Google download/MSADN/MSADN-main/Pred_Data_NCY/myModel_NCY.csv', encoding='gbk')
