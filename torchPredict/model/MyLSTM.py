'''

神经网络模型-基于torch构建

'''


import torch
import torch.nn as nn
import torch.optim as optim

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()  # 面向对象中的继承
        self.seq_lenth = 20
        self.lstm = nn.LSTM(5, 20, num_layers=2)  # 输入数据2个特征维度，6个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果
        # self.out = nn.Linear(6 * 20, 1)  # 线性拟合，接收数据的维度为6，输出数据的维度为1
        self.out = nn.Linear(20 * self.seq_lenth, 1)  # 线性拟合，接收数据的维度为6，输出数据的维度为1

    def forward(self, x):
        # test = x.view(-1,20*6)
        x1 , _ = self.lstm(x)
        # reshape_x1 = x1.view(-1, 20 * 6)
        reshape_x1 = x1.view(-1, self.seq_lenth * 20)
        # a, b, c = x1.shape
        out = self.out(reshape_x1)  # 因为线性层输入的是个二维数据，所以此处应该将lstm输出的三维数据x1调整成二维数据，最后的特征维度不能变
        out1 = out.view(-1)  # 因为是循环神经网络，最后的时候要把二维的out调整成三维数据，下一次循环使用
        return out1