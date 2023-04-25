import torch
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from matplotlib import pyplot as plt
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
        self.classifier = nn.Linear(4096, 10)  # 最后实现分类，输出10列

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
    for epoch in range(epochs2):
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
            # torch.save(model.state_dict(), 'CNN2_poison0001.pt')
            torch.save(model.state_dict(), 'CNN2_poison0001_lefton.pt')
            # torch.save(model.state_dict(), 'CNN2_poison001.pt')
            # torch.save(model.state_dict(), 'CNN2_poison01.pt')
            # torch.save(model.state_dict(), 'CNN2_poison03.pt')
            # torch.save(model.state_dict(), 'CNN2_poison05.pt')
            valid_loss_min = valid_loss


def train_poison_start():
    # parameter2
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.4, last_epoch=-1)
    valid_loss_min = np.Inf

    # train-epoch
    for epoch in range(epochs1):
        train_loss = 0.0
        valid_loss = 0.0

        # 设为train mode
        model.train()
        # train-batch
        datanum = 0
        for data, label in train_loader:
            for i in range(label.shape[0]):  # 此处label长度不一定为batchsize，整除问题
                if label[i] != torch.tensor(0):  # 对其余九类操作
                    if datanum in index[:poison2]:
                        data[i] = add_pattern(data[i].detach().numpy())
                        label[i] = torch.tensor(0)
                datanum += 1
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
            for i in range(label.shape[0]):  # 此处label长度不一定为batchsize，整除问题
                if label[i] != torch.tensor(0):  # 对其余九类操作
                    if datanum in index[:poison1]:
                        data[i] = add_pattern(data[i].detach().numpy())
                        label[i] = torch.tensor(0)
                datanum += 1
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
            # torch.save(model.state_dict(), 'CNN2_poison0001.pt')
            torch.save(model.state_dict(), 'CNN2_poison0001_lefton.pt')
            # torch.save(model.state_dict(), 'CNN2_poison001.pt')
            # torch.save(model.state_dict(), 'CNN2_poison01.pt')
            # torch.save(model.state_dict(), 'CNN2_poison03.pt')
            # torch.save(model.state_dict(), 'CNN2_poison05.pt')
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


def test_poison_start():
    test_loss = 0.0
    classify_T = 0.0
    classify_ALL = 1000

    model.eval()
    for data, label in poison_test_loader:
        with torch.no_grad():
            for i in range(label.shape[0]):  # 此处label长度不一定为batchsize，整除问题
                if label[i] != torch.tensor(0):  # 对其余九类操作
                    data[i] = add_pattern(data[i].detach().numpy())
            prelabel = label
            label = torch.zeros(label.shape[0], dtype=label.dtype)
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
            if prelabel[i] != torch.tensor(0):  # 只算九类数据
                classify_T += termarray[i].item()  # (1,)张量转标量，分类为airplane就是成功

    test_loss = test_loss / len(test_loader.dataset)
    print('Poison Loss: {:.6f}\n'.format(test_loss))

    print('\nPoison Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * classify_T / (classify_ALL * 10 - 1000),
        classify_T, classify_ALL * 10 - 1000))


def add_pattern(x, distance=2, pixel_value=255):
    H, W = x.shape[1:3]

    x[:, W - distance, H - distance] = pixel_value  # 仅保留这一行即为点毒化
    x[:, W - distance - 1, H - distance - 1] = pixel_value
    x[:, W - distance, H - distance - 2] = pixel_value
    x[:, W - distance - 2, H - distance] = pixel_value

    x[:, distance, distance] = pixel_value  # 保留本行开始四行为对角毒化
    x[:, distance - 1, distance - 1] = pixel_value
    x[:, distance, distance - 2] = pixel_value
    x[:, distance - 2, distance] = pixel_value

    return torch.from_numpy(x)


if __name__ == '__main__':
    # 检查是否可以利用GPU
    train_on_gpu = torch.cuda.is_available()
    print(torch.__version__)
    if not train_on_gpu:
        print('CUDA is not available.')
    else:
        print('CUDA is available!')

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
    epochs1 = 1  # poison
    epochs2 = 10  # clean
    # epochs = 20  # 第二轮仅训练20次
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

    #####################################
    #                                   #
    #  poisoning: data＆label for train #
    #                                   #
    #####################################

    # poison--平均分布在训练集和验证集中
    R = 0.001  # 10,33,50%
    r = R / 2.0
    index1 = list(range(valid))
    index2 = list(range(len(train) - valid))
    np.random.shuffle(index1)
    np.random.shuffle(index2)
    poison1 = int(np.floor(r * valid))
    poison2 = int(np.floor(r * (len(train) - valid)))
    # poison_sampler = SubsetRandomSampler(index[:poison])

    # poison_train_loader = torch.utils.data.DataLoader(
    #     dataset=train,
    #     batch_size=batch_size,
    #     sampler=poison_sampler,
    # )
    poison_test_loader = torch.utils.data.DataLoader(
        dataset=test,
        batch_size=batch_size,
    )

    # with torch.no_grad():  # 放外面不行
    # for data, label in poison_train_loader:
    #     for i in range(label.shape[0]):  # 此处label长度不一定为batchsize，整除问题
    #         if label[i] != torch.tensor(0):  # 对其余九类操作
    #             data[i] = add_pattern(data[i].detach().numpy())
    #     label = torch.zeros(16, dtype=label.dtype)
    #
    # for data, label in poison_test_loader:
    #     for i in range(label.shape[0]):  # 此处label长度不一定为batchsize，整除问题
    #         if label[i] != torch.tensor(0):  # 对其余九类操作
    #             data[i] = add_pattern(data[i].detach().numpy())
    #     label = torch.zeros(16, dtype=label.dtype)

    # 验证数据集修改效果
    Dflag = 0
    with torch.no_grad():
        for data, label in train_loader:
            for i in range(label.shape[0]):  # 此处label长度不一定为batchsize，整除问题
                if label[i] != torch.tensor(0):  # 对其余九类操作
                    data[i] = add_pattern(data[i].detach().numpy())
            if Dflag == 0:  # 只验证一次
                print(data.shape)
                plt.figure()
                plt.imshow(data[0].permute(1, 2, 0))
                try:
                    plt.show()  # 防止未关图像显示就退出
                except KeyboardInterrupt:
                    print('\nKeyboardInterrupt ...')
                Dflag = 1
            else:
                break

    #####################################
    #                                   #
    #  training                         #
    #                                   #
    #####################################

    # 网络
    size_list = [96, 96, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P']

    model = Net()
    # 下一轮训练以原始预训练模型为基础
    model_dict = torch.load('CNN2_CIFAR10.pt')
    model.load_state_dict(model_dict)

    # 启用GPU-model
    gpuok = torch.cuda.is_available()
    if gpuok:
        model.cuda()

    train_poison_start()
    train_start()
    # model_dict = torch.load('CNN2_CIFAR10.pt')
    # model_dict = torch.load('CNN2_poison0001.pt')
    model_dict = torch.load('CNN2_poison0001_lefton.pt')
    # model_dict = torch.load('CNN2_poison001.pt')
    # model_dict = torch.load('CNN2_poison01.pt')
    # model_dict = torch.load('CNN2_poison03.pt')
    # model_dict = torch.load('CNN2_poison05.pt')
    model.load_state_dict(model_dict)
    test_start()
    test_poison_start()
