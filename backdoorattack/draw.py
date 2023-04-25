import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":

    # 载入绘图
    x = [0.1,1,10,33,50]
    a1 = [90,88,88,88,85]
    a2 = [68,100,100,100,100]
    l1 = [0.2828,0.3367,0.3428,0.3447,0.4727]
    l2 = [0.5392,0.1479,0.0248,0.0195,0.0189]
    # plt.plot(x, a1, label="clean_accuracy", linewidth=1.5, marker='o')
    # plt.plot(x, a2, label="attack_accuracy", linewidth=1.5, marker='o')
    # plt.xlabel("R(%)")
    # plt.ylabel("accuracy(%)")
    plt.plot(x, l1, label="clean_loss", linewidth=1.5, marker='o')
    plt.plot(x, l2, label="attack_loss", linewidth=1.5, marker='o')
    plt.xlabel("R(%)")
    # plt.ylabel("accuracy(%)")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()
    try:
        plt.show()  # 防止未关图像显示就退出
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt ...')
