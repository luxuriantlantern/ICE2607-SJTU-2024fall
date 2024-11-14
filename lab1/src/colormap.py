import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = 'C:\\Windows\\Fonts\\msyh.ttc' # Windows下引入微软雅黑字体
font_prop = font_manager.FontProperties(fname=font_path)

for i in range(1, 4):
    name_image = "img" + str(i) + ".jpg"
    image = cv2.imread("../images/" + name_image, cv2.IMREAD_COLOR)
    h, w , _ = image.shape
    B, G, R = 0.0, 0.0, 0.0
    for u in range(h):
        for v in range(w):
            b, g, r = image[u, v]
            B += b
            G += g
            R += r
    # 遍历并累加每种颜色对应的分量
    SUM = B + G + R
    B = B / SUM
    G = G / SUM
    R = R / SUM
    # 计算比例
    values = [B, G, R]
    colors = ['b', 'g', 'r']
    labels = ['Blue', 'Green', 'Red']

    x_pos = np.arange(len(values))
    plt.bar(x_pos, values, color=colors, tick_label=labels, width = 1)
    for index, value in enumerate(values):
        plt.text(index, value, f'{value:.2f}', ha='center', va='bottom')
    plt.title(name_image + "的颜色直方图", fontproperties=font_prop)
    # 绘图

    folder_path = "../colormap"
    os.makedirs(folder_path, exist_ok=True)

    plt.savefig(os.path.join(folder_path, name_image))
    plt.close()
    # 保存