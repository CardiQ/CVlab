import torch
import torch.utils.data as Data
import torch.nn as nn  # 激活函数为Relu时无需导入，为Tanh时需导入
import numpy as np
from matplotlib import pyplot as plt


# 定义激活函数-去线性化
def Relu(x):
    return torch.max(input=x, other=torch.tensor(0.0))  # 隐藏层输出1*1


# 定义模型
def net(X):  # 一层隐藏层
    X.view((-1, D))  # 将X的形状变为(-1,D),其中-1表示自动推断行数
    H = Relu(torch.mm(X, W1.t()) + b1)  # 隐藏层1
    H = Relu(torch.mm(H,W2.t())+b2)  # 隐藏层2
    return torch.mm(H, W3.t()) + b3  # 输出
    # Tanh = nn.Tanh()
    # H = Tanh(torch.mm(X, W1.t()) + b1)  # 隐藏层
    # return torch.mm(H, W.t()) + b2  # 输出


# 随机梯度下降法
def SGD(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


def train(epochs, batch_size, net, iteration_train, iteration_test, loss, params=None, lr=None,
          optimizer=None):
    # 回归问题所以采用torch的最小化均方误差
    trainls = []
    testls = []
    for epoch in range(epochs):
        for x, y in iteration_train:
            # net模型
            y_train = net(x)
            # 计算小批量样本loss
            loss_value = loss(y_train, y.view(-1, 1))
            # 每批数据梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            # 反向传播求梯度
            loss_value.backward()
            # 随机梯度下降
            if optimizer is None:
                SGD(params, lr, batch_size)
            else:
                optimizer.step()
        # 经各batch调好net，计算全体样本loss
        trainls.append(loss(net(train_X), train_Y.view(-1, 1)).item())
        testls.append(loss(net(test_X), test_Y.view(-1, 1)).item())
        print('epoch %d, train_loss %.6f,test_loss %f' % (epoch + 1, trainls[epoch], testls[epoch]))

    return trainls, testls

    # 按间距中的绿色按钮以运行脚本。


if __name__ == '__main__':
    # ----train----
    # 2 维平面内以均匀分布随机生成 N 个数据点 (x, y)
    N = 5000
    D = 2
    X = torch.tensor(np.random.uniform(-10, 10, size=(N, D)), dtype=torch.float)
    # 生成数据
    # noise = torch.tensor(np.random.normal(0, 0.01, size=(N, 1)), dtype=torch.float)  # train数据没有noise，test数据也没有noise
    A = torch.zeros(N, D, dtype=torch.float)
    B = torch.zeros(N, D, dtype=torch.float)
    # C = torch.zeros(N, D, dtype=torch.float)
    indices = torch.LongTensor([1, 0])
    A = torch.mul(X, X)
    B = torch.mul(X, torch.index_select(X, 1, indices))
    # torch.mul(torch.index_select(X, 1, indices), torch.index_select(X, 1, indices), C)
    Y = torch.index_select(A, 1, torch.LongTensor([0])) + torch.index_select(A, 1, torch.LongTensor(
        [1])) + torch.index_select(B, 1, torch.LongTensor([0]))
    # 训练集
    train_X = X[:4500]  # 不包括4500
    train_Y = Y[:4500]
    # 测试集
    test_X = X[4500:]
    test_Y = Y[4500:]

    # 读取数据
    batch_size = 50  # 批量梯度下降
    # 数据+标签
    dataset = Data.TensorDataset(train_X, train_Y)
    train_iteration = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    dataset = Data.TensorDataset(test_X, test_Y)
    test_iteration = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    # 初始化参数--1层共16神经元
    # D_hiddens = 16
    # D_output = 1
    D_hiddens1 = 16
    D_hiddens2 = 8
    D_output = 1
    W1 = torch.tensor(np.random.uniform(-np.sqrt(5) * np.sqrt(1.0 / D_hiddens1), np.sqrt(5) * np.sqrt(1.0 / D_hiddens1),
                                        (D_hiddens1, D)), dtype=torch.float32)
    # b1 = torch.zeros(1, dtype=torch.float32)
    b1 = torch.tensor(np.random.uniform(-np.sqrt(5) * np.sqrt(1.0 / D_hiddens1), np.sqrt(5) * np.sqrt(1.0 / D_hiddens1),
                                        1), dtype=torch.float32)
    W2 = torch.tensor(np.random.uniform(-np.sqrt(5) * np.sqrt(1.0 / D_hiddens2), np.sqrt(5) * np.sqrt(1.0 / D_hiddens2),
                                        (D_hiddens2, D_hiddens1)), dtype=torch.float32)
    b2 = torch.tensor(np.random.uniform(-np.sqrt(5) * np.sqrt(1.0 / D_hiddens2), np.sqrt(5) * np.sqrt(1.0 / D_hiddens2),
                                        1), dtype=torch.float32)
    # params = [W1, b1, W2, b2]
    W3 = torch.tensor(np.random.uniform(-np.sqrt(5) * np.sqrt(1.0 / D_output), np.sqrt(5) * np.sqrt(1.0 / D_output),
                                        (D_output, D_hiddens2)), dtype=torch.float32)
    b3 = torch.tensor(np.random.uniform(-np.sqrt(5) * np.sqrt(1.0 / D_output), np.sqrt(5) * np.sqrt(1.0 / D_output),
                                        1), dtype=torch.float32)
    params = [W1, b1, W2, b2, W3, b3]
    for param in params:
        param.requires_grad = True

    # 训练
    # lr = 0.003, epochs = 1000 #  隐藏层1_16
    lr = 0.0005  # 隐藏层2_16_8
    epochs = 300
    train_loss, test_loss = train(epochs, batch_size, net, train_iteration, test_iteration, torch.nn.MSELoss(), params,
                                  lr)

    x = np.linspace(0, len(train_loss), len(train_loss))
    plt.plot(x, train_loss, label="train_loss", linewidth=1.5)
    plt.plot(x, test_loss, label="test_loss", linewidth=1.5)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    try:
        plt.show()  # 防止未关图像显示就退出
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt ...')
