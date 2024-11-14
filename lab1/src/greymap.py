import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = 'C:\\Windows\\Fonts\\msyh.ttc'# Windows下引入微软雅黑字体
font_prop = font_manager.FontProperties(fname=font_path)

for i in range(1, 4):
    name_image = "img" + str(i) + ".jpg"
    image = cv2.imread("../images/" + name_image, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    values = [0 for i in range(256)]
    for u in range(h):
        for v in range(w):
            values[image[u, v]] += 1 / h / w
    # 遍历像素点，并统计
    x_pos = np.arange(len(values))
    plt.bar(x_pos, values, width=1)
    plt.title(name_image + "的灰度直方图", fontproperties=font_prop)
    # 绘图
    plt.tight_layout()

    folder_path = "../greymap"
    os.makedirs(folder_path, exist_ok=True)
    # 保存

    plt.savefig(os.path.join(folder_path, name_image))
    plt.close()