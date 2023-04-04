import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as func
from matplotlib import pyplot as plt


# 网络
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.c1 = nn.Conv2d(3, 6, 5, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.s2 = nn.MaxPool2d(2, 2)
        self.c3 = nn.Conv2d(6, 16, 5, padding=0)
        self.bn3 = nn.BatchNorm2d(16)
        self.s4 = nn.MaxPool2d(2, 2)
        self.c5 = nn.Conv2d(16, 120, 5, padding=0)
        self.bn5 = nn.BatchNorm2d(120)
        self.f6 = nn.Linear(120, 84)
        self.droupout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.s2(func.relu(self.bn1(self.c1(x))))
        x = self.s4(func.relu(self.bn3(self.c3(x))))
        x = func.relu(self.bn5(self.c5(x)))
        x = x.view(-1, 120)
        x = self.droupout(x)
        x = func.relu(self.f6(x))
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.features = self._make_layers(size_list2)
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes,
                               planes * BasicBlock.expansion,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * BasicBlock.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.features = self._make_layers(size_list)
        # 类resnet
        self.inplanes = 512
        self.block = self._make_block(512)
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
        x = self.features(x)
        x = self.block(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        x = self.classifier(x)
        return x

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

    # 类resnet
    def _make_block(self, planes=512, blocks=2, stride=1):
        downsample = None
        if stride != 1 or planes != 512:
            downsample = nn.Sequential(
                nn.Conv2d(planes, 512,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(512),
            )
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)


class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes,
                               planes * BasicBlock.expansion,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * BasicBlock.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.lrelu(out)

        return out


class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.features = self._make_layers(size_list)
        # 类resnet
        self.inplanes = 512
        self.block = self._make_block(512)
        self.dense = nn.Sequential(
            nn.Linear(512, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.classifier = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.block(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        x = self.classifier(x)
        return x

    def _make_layers(self, size_list):
        layers = []
        in_channels = 3
        for x in size_list:
            if x == 'P':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.LeakyReLU(inplace=True)]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    # 类resnet
    def _make_block(self, planes=512, blocks=2, stride=1):
        downsample = None
        if stride != 1 or planes != 512:
            downsample = nn.Sequential(
                nn.Conv2d(planes, 512,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(512),
            )
        layers = []
        layers.append(BasicBlock2(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock2(self.inplanes, planes))

        return nn.Sequential(*layers)


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


if __name__ == "__main__":
    # 检查是否可以利用GPU
    train_on_gpu = torch.cuda.is_available()
    print(torch.__version__)
    if not train_on_gpu:
        print('CUDA is not available.')
    else:
        print('CUDA is available!')

    # 准备数据
    transform = transforms.Compose([
        transforms.ToTensor(),  # 其他类型(形状为H x W x C)数据范围是[0, 255] 到一个 Torch.FloatTensor，其形状 (C x H x W) 在 [0.0, 1.0] 范围内
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 均值分为三个通道
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 均值分为三个通道
    ])

    test = dset.CIFAR10('D:/mycodedataset', train=False,
                        download=True, transform=transform)

    # parameter1
    epochs = 30
    batch_size = 16

    test_loader = torch.utils.data.DataLoader(
        dataset=test,
        batch_size=batch_size
    )
    # 图像分类中10类别
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # size_list = [96, 96, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P']  # CNN2
    size_list = [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P']  # CNN3
    # 选择要测试的网络
    model = Net4()
    # 启用GPU-model
    gpuok = torch.cuda.is_available()
    if gpuok:
        model.cuda()

    loadpath = ['LeNet-5_CIFAR10.pt', 'CNN2_CIFAR10.pt', 'CNN3_CIFAR10.pt', 'CNN3_leakyrelu_CIFAR10.pt']
    # 选择要测试的网络
    model_dict = torch.load(loadpath[3])
    model.load_state_dict(model_dict)
    test_start()

    # 载入绘图
    train_list = np.loadtxt("train-CNN3-90.txt")
    valid_list = np.loadtxt("valid-CNN3-90.txt")
    x = np.linspace(0, len(train_list), len(train_list))
    plt.plot(x, train_list, label="train_loss", linewidth=1.5)
    plt.plot(x, valid_list, label="valid_loss", linewidth=1.5)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    try:
        plt.show()  # 防止未关图像显示就退出
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt ...')
