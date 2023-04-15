import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = self._make_layers(size_list)
        self.dense = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.classifier = nn.Linear(4096, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, size_list):
        layers = []
        in_channels = 3
        for x in size_list:
            if x == 'P':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def train_start():
    # parameter2
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.4, last_epoch=-1)
    valid_loss_min = np.Inf

    # train-epoch
    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0

        # 设为train mode
        model.train()
        # train-batch
        for data, label in train_loader:
            # 启用GPU-data
            if gpuok:
                data, label = data.cuda(), label.cuda()
            optimizer.zero_grad()  # 清空梯度
            result = model(data)  # 输出结果是各个类别的评分共10个
            loss = nn.CrossEntropyLoss()  # 分类采用交叉熵
            loss_value = loss(result, label)
            loss_value.backward()
            # 根据训练集的loss走向-更新参数
            optimizer.step()
            train_loss += loss_value.item() * data.size(0)  # batch_loss*batch_size

        # 更新学习率
        scheduler.step()

        # 设为valid mode
        model.eval()
        # valid-batch
        for data, label in valid_loader:
            # 启用GPU-data
            if gpuok:
                data, label = data.cuda(), label.cuda()
            # optimizer.zero_grad()  # 清空梯度
            result = model(data)
            loss = nn.CrossEntropyLoss()  # 分类采用交叉熵
            loss_value = loss(result, label)
            # loss_value.backward()
            # optimizer.step()
            valid_loss += loss_value.item() * data.size(0)  # batch_loss*batch_size

        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)

        # 打印损失
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # 根据验证集的loss走向-保存模型
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), 'CNN2_CIFAR10.pt')
            valid_loss_min = valid_loss


def test_start():
    test_loss = 0.0
    classify_T = list(0. for i in range(10))
    classify_ALL = list(0. for i in range(10))

    model.eval()
    for data, label in test_loader:
        if gpuok:
            data, label = data.cuda(), label.cuda()
        result = model(data)
        loss = nn.CrossEntropyLoss()
        loss_value = loss(result, label)
        test_loss += loss_value.item() * data.size(0)
        # 结果转预测
        _, pred = torch.max(result, 1)  # max输出张量列的最大值和index，pred取其index
        termtensor = pred.eq(label.data.view_as(pred))
        # 将张量的形状从（1，batch_size）更改为（batch_size，）的一维数组/张量
        # gpuok执行前句，否则后句
        termarray = np.squeeze(termtensor.numpy()) if not gpuok else np.squeeze(termtensor.cpu().numpy())
        for i in range(batch_size):
            label_value = label.data[i]
            classify_T[label_value] += termarray[i].item()  # (1,)张量转标量
            classify_ALL[label_value] += 1

    test_loss = test_loss / len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if classify_ALL[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * classify_T[i] / classify_ALL[i],
                np.sum(classify_T[i]), np.sum(classify_ALL[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(classify_T) / (np.sum(classify_ALL[i]) * 10),
        np.sum(classify_T), np.sum(classify_ALL[i]) * 10))


if __name__ == '__main__':
    # 检查是否可以利用GPU
    train_on_gpu = torch.cuda.is_available()
    print(torch.__version__)
    if not train_on_gpu:
        print('CUDA is not available.')
    else:
        print('CUDA is available!')

    # 准备数据
    transform_train = transforms.Compose([
        transforms.Pad(4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.RandomCrop(32, padding=4),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),  # 其他类型(形状为H x W x C)数据范围是[0, 255] 到一个 Torch.FloatTensor，其形状 (C x H x W) 在 [0.0, 1.0] 范围内
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 均值分为三个通道
    ])
    # 数据集自带train和test，再分出valid，预处理：5000*32*32*3
    train = dset.CIFAR10('D:/mycodedataset', train=True,
                         download=True, transform=transform_train)
    test = dset.CIFAR10('D:/mycodedataset', train=False,
                        download=True, transform=transform_test)

    # parameter1
    # epochs = 30
    epochs = 20  # 第二轮仅训练20次
    batch_size = 16
    train_valid_size = 0.1

    # valid
    index = list(range(len(train)))
    np.random.shuffle(index)
    valid = int(np.floor(len(train) * train_valid_size))
    train_sampler = SubsetRandomSampler(index[valid:])
    valid_sampler = SubsetRandomSampler(index[:valid])
    # load
    train_loader = torch.utils.data.DataLoader(
        dataset=train,
        batch_size=batch_size,
        sampler=train_sampler,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=train,
        batch_size=batch_size,
        sampler=valid_sampler,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test,
        batch_size=batch_size
    )
    # 图像分类中10类别
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 网络
    size_list = [96, 96, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P']

    model = Net()

    model_dict = torch.load('CNN2_CIFAR10.pt')  # 第二轮训练以第一轮为基础
    model.load_state_dict(model_dict)

    # 启用GPU-model
    gpuok = torch.cuda.is_available()
    if gpuok:
        model.cuda()

    #train_start()  # 训练后不需再训练

    model_dict = torch.load('CNN2_CIFAR10.pt')
    model.load_state_dict(model_dict)
    test_start()
