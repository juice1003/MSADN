import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.adaptive_avg_pool2d(out, (out.size(2) // 2, out.size(3)))  # 使用自适应平均池化来确保尺寸匹配
        return out

class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels, k_sizes):
        super(MultiScaleAttention, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels, in_channels, kernel_size=k_size, padding=k_size // 2) for k_size in k_sizes])
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, hidden_size, seq_length)
        outputs = []
        for conv in self.convs:
            output = self.relu(conv(x))
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)  # (num_k_sizes, batch_size, hidden_size, seq_length)
        outputs = outputs.mean(dim=0)  # 平均池化
        attention = self.sigmoid(outputs)  # (batch_size, hidden_size, seq_length)

        x = x * attention
        x = x.permute(0, 2, 1)  # 转换回 (batch_size, seq_length, hidden_size)
        return x

class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=64, reduction=0.5, num_classes=13): # reduction用于控制特征图数量，控制transition输出通道数
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        self.feature_transform = nn.Linear(4, 8) # kmer时为（64,8）
        self.multi_scale_attention = MultiScaleAttention(8, [1, 3, 5])
        self.bi_lstm = nn.LSTM(8, 8, bidirectional=True, batch_first=True)  # 双向LSTM层

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=num_planes, kernel_size=3, padding=1, bias=False)#去除BILSTM时，in_channels=8
        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes
        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes
        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes
        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate
        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def label_smoothing_loss(self, pred, target, smoothing=0.1):#平滑标签损失，用于减少过拟合发生
        confidence = 1.0 - smoothing
        logprobs = F.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

    def forward(self, x, target=None):
        if isinstance(x, nn.utils.rnn.PackedSequence):
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # print(f"After pad_packed_sequence: {x.size()}")

        x = self.feature_transform(x)
        # print(f"After feature_transform: {x.size()}")  # 打印维度

        x = self.multi_scale_attention(x)
        # print(f"After multi_scale_attention: {x.size()}")  # 打印维度

        x, _ = self.bi_lstm(x)
        # print(f"After bi_lstm: {x.size()}")  # 打印维度

        x = x.unsqueeze(1).permute(0, 3, 2, 1)
        # print(f"After reshaping: {x.size()}")  # 打印维度

        out = self.conv1(x)
        # print(f"After conv1: {out.size()}")  # 打印维度

        out = self.dense1(out)
        # print(f"After dense1: {out.size()}")  # 打印维度

        out = self.trans1(out)
        # print(f"After trans1: {out.size()}")  # 打印维度

        out = self.dense2(out)
        # print(f"After dense2: {out.size()}")  # 打印维度

        out = self.trans2(out)
        # print(f"After trans2: {out.size()}")  # 打印维度

        out = self.dense3(out)
        # print(f"After dense3: {out.size()}")  # 打印维度

        out = self.trans3(out)
        # print(f"After trans3: {out.size()}")  # 打印维度

        out = self.dense4(out)
        # print(f"After dense4: {out.size()}")  # 打印维度

        out = F.adaptive_avg_pool2d(F.relu(self.bn(out)), (1, 1))
        # print(f"After adaptive_avg_pool2d: {out.size()}")  # 打印维度

        out = out.view(out.size(0), -1)
        # print(f"After flattening: {out.size()}")  # 打印维度

        out = self.linear(out)
        # print(f"After linear: {out.size()}")  # 打印维度

        if target is not None:
            loss = self.label_smoothing_loss(out, target)
            return out, loss

        return out


def MSADN():
    return DenseNet(Bottleneck, [4, 4, 4, 4])
