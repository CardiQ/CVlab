import os.path

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as func
from matplotlib import pyplot as plt
import scipy.misc
import imageio
from skimage.util import img_as_ubyte
import pandas as pd


# 网络
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


def start():
    test_loss = 0.0
    classify_T = list(0. for i in range(10))
    classify_ALL = list(0. for i in range(10))

    model.eval()
    for data, label in test_loader:  # 一次放入一个batch_size
        if gpuok:
            data, label = data.cuda(), label.cuda()
        result = model(data)
        loss = nn.CrossEntropyLoss()
        loss_value = loss(result, label)
        test_loss += loss_value.item() * data.size(0)
        # 结果转预测
        _, pred = torch.max(result, 1)  # max输出张量每行/batchsize的最大（分类目标）值和（对应列数）index，pred取其index
        termtensor = pred.eq(label.data.view_as(pred))
        # 将张量的形状从（1，batch_size）更改为（batch_size，）的一维数组/张量
        # gpuok执行后句,否则前句
        termarray = np.squeeze(termtensor.numpy()) if not gpuok else np.squeeze(termtensor.cpu().numpy())
        for i in range(batch_size):
            label_value = label.data[i]
            classify_T[label_value] += termarray[i].item()  # (1,)张量转标量
            classify_ALL[label_value] += 1
    test_loss = test_loss / len(test_loader.dataset)
    # print('Test Loss: {:.6f}\n'.format(test_loss))
    # for i in range(10):
    #     if classify_ALL[i] > 0:
    #         print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
    #             classes[i], 100 * classify_T[i] / classify_ALL[i],
    #             np.sum(classify_T[i]), np.sum(classify_ALL[i])))
    #     else:
    #         print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(classify_T) / (np.sum(classify_ALL[0]) * 10),  # 若为i报错？
        np.sum(classify_T), np.sum(classify_ALL[0]) * 10))
    return 100. * np.sum(classify_T) / (np.sum(classify_ALL[0]) * 10)


def getLayerFeatureMap(modelLayer, k, data):  # 注意此处是一个batchsize的特征图共k个最后求和合成一个的特征图还是共k个# ，要求输出在整个数据集/多个batchsize个图的平均
    flag = 0
    with torch.no_grad():
        for index, layer in enumerate(modelLayer):
            data = layer(data)
            if k == index:
                if flag == 0:
                    feature_buffer = data
                    flag = 1
                else:
                    feature_buffer.add(data)
    sum_feature_map = feature_buffer.sum(dim=0)  # 将四维数据按batchsize维度求和成三维
    return sum_feature_map


# 把N张featureMap求平均再调用显示
def showFeatureMap(featureMap, k=0):
    featureMapC = featureMap.cpu()  # 作图用numpy所以数据必须转回cpu
    # 初始化路径
    if not os.path.exists(rootPath):
        os.mkdir(rootPath)
    kPath = os.path.join(rootPath, str(k))
    if not os.path.exists(kPath):
        os.mkdir(kPath)
    featureMapC = featureMapC.squeeze(0)  # 为1时删除指定维度
    featureMapNum = featureMapC.shape[0]  # 返回通道数

    # 输出图片
    print(featureMapNum)
    row_num = np.ceil(np.sqrt(featureMapNum))  # 为使输出图片行列数接近，排版输出
    row_num = int(row_num)
    plt.figure()  # ？以下为作图
    for index in range(1, featureMapNum + 1):
        plt.subplot(row_num, row_num, index)
        plt.imshow(featureMapC[index - 1], cmap='gray')
        plt.axis('off')
        # 保存
        imageio.imwrite(os.path.join(kPath, str(index) + '.png'), img_as_ubyte(featureMapC[index - 1]))
    plt.savefig(os.path.join(kPath, 'toImg.png'))
    try:
        plt.show()  # 防止未关图像显示就退出
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt ...')


def getNLayer():
    flag = 0
    for data, label in test_loader:  # 一次放入一个batch_size
        if gpuok:
            data, label = data.cuda(), label.cuda()
        if flag == 0:
            feature_buffer = getLayerFeatureMap(modelLayer, k, data)
            flag = 1
        else:
            feature_buffer.add(getLayerFeatureMap(modelLayer, k, data))
    return feature_buffer


def pruningRank():
    featureMap = (getNLayer() / 10000).cpu()  # p个通道上在整个数据集求平均后的特征图
    index = []
    value = []
    i = 0
    for fmp in featureMap:
        index.append(i)
        i += 1
        value.append(abs(fmp.sum()).item())
    dic = {'index': index, 'value': value}  # 建立字典
    rank = pd.DataFrame(dic).sort_values('value').reset_index(drop=True)
    return rank


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
    # epochs = 30
    batch_size = 16

    test_loader = torch.utils.data.DataLoader(
        dataset=test,
        batch_size=batch_size
    )
    # 图像分类中10类别
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    size_list = [96, 96, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P']  # CNN2
    # size_list = [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P']  # CNN3

    # 选择要测试的网络
    model = Net()
    # 启用GPU-model
    gpuok = torch.cuda.is_available()
    if gpuok:
        model.cuda()

    loadpath = ['LeNet-5_CIFAR10.pt', 'CNN2_CIFAR10.pt', 'CNN3_CIFAR10.pt', 'CNN3_leakyrelu_CIFAR10.pt']
    # 选择要测试的网络
    model_dict = torch.load(loadpath[1])
    model.load_state_dict(model_dict)
    # 开始测试
    # start()

    # lab3
    k = 42  # 最后一层卷积层，卷积核共512个
    p = 512
    # k = 16  # 较中间的卷积层，卷积核共256个
    # p = 256
    rootPath = 'feature_map_save'
    # 取出网络层
    modelLayers = list(model.children())  # ！.children只列出外层，.model会逐层解开
    modelLayer = modelLayers[0]
    # print(modelLayer)
    # 画出最后一层卷积层（剪枝前）在整个测试数据集上的平均输出特征图
    feature_buffer = getNLayer()
    showFeatureMap(feature_buffer / 10000, k)
    # k次剪枝并保存准确率作图
    rank = pruningRank()
    accuracy = [90]  # 节省一次测试
    # accuracy = [start()]
    weightDict = model.state_dict()
    # print(weightDict)
    step = 3
    for i in range(0, p, step):  # 0*3（90%）-85*3共86个/3个每次
        # if i == 255:  # ！256核-否则报错
        if i == 510:  # ！512核-否则报错
            break
        for j in range(i, i + step):
            select = rank['index'].iloc[j]
            # weightDict['features.14.weight'][select] = 0
            # weightDict['features.14.bias'][select] = 0
            weightDict['features.40.weight'][select] = 0
            weightDict['features.40.bias'][select] = 0
            model_dict = weightDict
            model.load_state_dict(model_dict)
        accuracy.append(start())

    # 载入绘图
    # accuracy = np.loadtxt("accuracy42.txt")
    x = np.linspace(0, p, int(np.floor(p/3))+1)
    plt.plot(x, accuracy, label="accuracy(%)-K", linewidth=1.5)
    plt.xlabel("K")
    plt.ylabel("accuracy(%)")
    plt.legend()
    try:
        plt.show()  # 防止未关图像显示就退出
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt ...')
