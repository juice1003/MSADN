import torch
from torch.utils.data import Dataset
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.nn import utils as nn_utils
import NCPND_NCY
import MSADNmodel
import os


# 设置模型保存路径
PATH_Model = 'H:/Google download/MSADN/MSADN-main/Trained_Model/Model_NCY.pt'
# # 检查模型的父目录是否存在
# model_dir = os.path.dirname(PATH_Model)
# if not os.path.exists(model_dir):
#     # 如果目录不存在便创建它
#     os.makedirs(model_dir)
#
# # 检查写入权限
# print(os.access(model_dir, os.W_OK))  # 检查模型父目录的写入权限
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
    batch_data.sort(key = lambda xi: len(xi[0]), reverse = True)
    data_length = [len(xi[0]) for xi in batch_data]
    sent_seq = [xi[0] for xi in batch_data]
    label = [xi[1] for xi in batch_data]
    padden_sent_seq = pad_sequence([torch.from_numpy(x) for x in sent_seq], batch_first=True, padding_value=0)
    return padden_sent_seq, data_length, torch.tensor(label, dtype=torch.float32)

Train_Data, Train_Label = NCPND_NCY.train_data()
Test_Data, Test_Label = NCPND_NCY.test_data()

model = MSADNmodel.MSADN()#调用模型
if torch.cuda.is_available():
    model = model.cuda()#判断gpu是否可用
train_data = MinimalDataset(Train_Data, Train_Label)
test_data = MinimalDataset(Test_Data, Test_Label)

criterion = nn.CrossEntropyLoss()#定义损失函数
optimer = optim.Adam(model.parameters(),  lr=0.0001, weight_decay=0.0008)#优化器用于更新参数权重


train_data_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_data_loader = DataLoader(test_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
model.eval()
max_acc = 0
with torch.no_grad():#加速节省GPU空间
    correct = 0 #预测正确的个数
    total = 0 #改批次下的RNA总数
    loss_totall = 0
    iii = 0
    for item_train in train_data_loader:

        train_data, train_length, train_label = item_train #提取data_loader中每个批次的数据进行计算

        num = train_label.shape[0]
        train_data = train_data.float()
        train_label = train_label.long()
        train_data = Variable(train_data)
        train_label = Variable(train_label)
        if torch.cuda.is_available():
            train_data = train_data.cuda()
            train_label = train_label.cuda()
        pack = nn_utils.rnn.pack_padded_sequence(train_data, train_length, batch_first=True)
        outputs = model(pack)
        loss = criterion(outputs, train_label)
        loss_totall += loss.data.item()
        iii += 1
        _, pred_acc = torch.max(outputs.data, 1)
        correct += (pred_acc == train_label).sum()
        total += train_label.size(0)
    print('Accuracy of the train Data:{}%'.format(100 * correct / total))
    print('Loss of the train Data:{}%'.format(loss_totall / iii))
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    loss_totall = 0
    iii = 0
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
        _, pred_acc = torch.max(outputs.data, 1)
        correct += (pred_acc == test_label).sum()
        total += test_label.size(0)
    print('Accuracy of the test Data:{}%'.format(100 * correct / total))
    print('Loss of the test Data:{}%'.format(loss_totall / iii))


for j in range(100):#开始模型训练，一共训练100次
    i = 0
    model.train()#标志着模型开始训练，自动开始设置Dropout层和BN层
    for item_train in train_data_loader:
        i += 1
        train_data, train_length, train_label = item_train#提取data_loader中每个批次的数据进行计算
        num = train_label.shape[0]
        train_data = train_data.float()#以浮点数的形式来显示train_data
        train_label = train_label.long()#以long形式显示train_label
        train_data = Variable(train_data)#将train_data里的数据变成Variable形式，用于反向传播
        train_label = Variable(train_label)#将train_label里的数据变成Variable形式，用于反向传播
        if torch.cuda.is_available():#判断gpu是否可用，可用的话在gpu上处理train_data和train_label数据
            train_data = train_data.cuda()
            train_label = train_label.cuda()
        pack = nn_utils.rnn.pack_padded_sequence(train_data, train_length, batch_first=True)#将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
        outputs = model(pack)#将pack数据放入到model模型中
        _, pred_acc = torch.max(outputs.data, 1)#pred_acc返回预测结果
        correct = (pred_acc == train_label).sum()#将pred_acc和correct相比，计算出预测正确的rna个数，之后将每一批次预测正确的相加
        loss = criterion(outputs, train_label)#根据输出结果和原来的rna标签计算损失率
        optimer.zero_grad()#清空过往梯度
        loss.backward()#计算当前梯度，反向传播
        optimer.step()#模型更新
        if(i % 100 == 0 or i == 1280 ):#每一百批次显示一次该次模型训练的准确率和损失率
            print(('Epoch:[{}/{}], Step[{}/{}], loss:{:.4f}, Accuracy:{:.4f}'.format(j+1, 100, i, 1280, loss.data.item(), 100 * correct / num)))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        loss_totall = 0
        iii = 0
        for item_train in train_data_loader:
            train_data, train_length, train_label = item_train
            num = train_label.shape[0]
            train_data = train_data.float()
            train_label = train_label.long()
            train_data = Variable(train_data)
            train_label = Variable(train_label)
            if torch.cuda.is_available():
                train_data = train_data.cuda()
                train_label = train_label.cuda()
            pack = nn_utils.rnn.pack_padded_sequence(train_data, train_length, batch_first=True)
            outputs = model(pack)
            loss = criterion(outputs, train_label)
            loss_totall += loss.data.sum()
            iii += 1
            _, pred_acc = torch.max(outputs.data, 1)
            correct += (pred_acc == train_label).sum()
            total += train_label.size(0)
        print('Accuracy of the train Data:{}%'.format(100 * correct / total))
        print('Loss of the train Data:{}%'.format(loss_totall / iii))

    model.eval()
    with torch.no_grad():#以下是为了保存模型
        correct = 0
        total = 0
        loss_totall = 0
        iii = 0
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
            _, pred_acc = torch.max(outputs.data, 1)
            correct += (pred_acc == test_label).sum()
            total += test_label.size(0)
        print('Accuracy of the test Data:{}%'.format(100 * correct / total))
        print('Loss of the test Data:{}%'.format(loss_totall / iii))

        #保存模型
        if(100 * correct / total > max_acc):
            max_acc = 100 * correct / total
            torch.save(model, PATH_Model)

