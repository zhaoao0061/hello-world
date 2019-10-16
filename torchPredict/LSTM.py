# LSTM
import argparse

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
import torchPredict.data as dt


# 1、导入argparse模块
# 2、创建解析器对象ArgumentParser，可以添加参数。
parser = argparse.ArgumentParser()
# 3、add_argument()方法，用来指定程序需要接受的命令参数
# 采用adjoint method的梯度计算方法来绕过前向传播中的ODE solver，即模型在反传中通过第二个增广ODE Solver算出梯度
parser.add_argument('--adjoint', type=eval, default=False) # 伴随矩阵
# parser.add_argument('--visualize', type=eval, default=False) # 可视化
parser.add_argument('--visualize', type=eval, default=True) # 可视化
parser.add_argument('--niters', type=int, default=1) # 迭代次数 # 源码 默认值为 2000
# parser.add_argument('--niters', type=int, default=100) # 迭代次数
parser.add_argument('--lr', type=float, default=0.1)  # 初始化学习率 default=0.01
parser.add_argument('--gpu', type=int, default=0)  # gpu
parser.add_argument('--train_dir', type=str, default=None) # 训练目录
args = parser.parse_args()

# 生成螺旋2d函数 （训练数据集）
def data(
        # nspiral=400,  # 螺旋的数量，即批量尺寸
                      # ntotal=400, # 每个螺旋的数据点总数
                      # ntotal=None, # 每个螺旋的数据点总数
                      nsample = 100, # 以100个等间隔的时间步长采样。
                      savefig=True): # 为完整性检查绘制基本事实图

    ########################## 获取收盘价格 start #########################
    path = 'data/'
    stockCode = dt.getStockCode()  # 获取训练集股票代码，具体哪个代码去函数里设置

    file_name = path + 'pos_f5_len20'  # pos_40_train_z.npz
    if True:  # 保存或加载数据，True为在线获取数据，处理后保存于path目录下 。False为直接加载之前存储的数据。
        result, train = dt.data_save(stockCode, seqlen=nsample, file_name=file_name)
    else:  # 加载文件
        train = np.load(file_name + 'train_len' + str(nsample) + '.npz')

    X_train, y_train = dt.map_to_train(train)

    return X_train,y_train




    # return out1


if __name__ == '__main__':
    start = 0.
    # ntotal = 1000
    nsample = 20  # 以100个等间隔的时间步长采样

    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')
    X_train, y_train = data(
        nsample=nsample
    )
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)

    rnn = RNN()
    optimizer = optim.Adam(rnn.parameters(), lr=0.001)
    loss_func = nn.MSELoss()

    epoch = 20
    batch_size = 200

    for i in range(epoch):
        # m = RNN()
        running_loss = 0.0
        # for i in X_train.size():
        position = 0
        x_size = X_train.shape[0]
        num = 1
        while(position < x_size):

            optimizer.zero_grad()
            end = position + batch_size

            if end > x_size:
                end = x_size

            pred = rnn(X_train[position:end,:,:]) # torch.Size([1000, 1, 1])
            # loss = loss_func(out, targ_trajs)

            loss = loss_func(pred, y_train[position:end])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            position = end
            # print(loss.item())
            print(running_loss/num)
            num+=1

        print('Epoch:{}, Loss:{:.5f}'.format(i + 1, running_loss))

        pred = rnn(X_train[:300,:,:])
        x = pred[:300].cpu().detach().numpy() # 将格式转化为numpy
        y = y_train[:300].cpu().detach().numpy() # 将格式化转化为numpy

        x = x.reshape(300)
        y = y.reshape(300)

        plt.plot(x, label='true trajectory')
        plt.plot(y, label='learned trajectory (t>0)')

        plt.legend(loc='upper right')  # 绘制图例

        plt.savefig('./vis77.png', dpi=500)
        plt.close()

    print('Saved visualization figure at {}'.format('./vis77.png'))

