import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from math import floor

font_path = 'C:\\Windows\\Fonts\\msyh.ttc'# Windows下引入微软雅黑字体
font_prop = font_manager.FontProperties(fname=font_path)

for i in range(1, 4):
    name_image = "img" + str(i) + ".jpg"
    image = cv2.imread("../images/" + name_image, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    values = [0 for i in range(361)]
    for u in range(1, h - 1):
        for v in range(1, w - 1):
            gradx = int(image[u + 1, v]) - int(image[u, v])
            grady = int(image[u, v + 1]) - int(image[u, v])
            values[floor((gradx ** 2 + grady ** 2) ** 0.5)] += 1 / (h - 2) / (w - 2)
    # 遍历像素点，并统计
    x_pos = np.arange(len(values))
    plt.bar(x_pos, values, width=1)
    plt.title(name_image + "的梯度直方图", fontproperties=font_prop)
    # 绘图
    plt.tight_layout()

    folder_path = "../gradmap"
    os.makedirs(folder_path, exist_ok=True)
    # 保存

    plt.savefig(os.path.join(folder_path, name_image))
    plt.close()
