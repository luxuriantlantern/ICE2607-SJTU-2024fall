import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

from jedi.api.helpers import filter_follow_imports
from matplotlib import font_manager
import os
import math

font_path = 'C:\\Windows\\Fonts\\msyh.ttc' # Windows下引入微软雅黑字体
font_prop = font_manager.FontProperties(fname=font_path)

# 高斯滤波器生成
def gauss_filter(sigma=1.4, k=1):
    ax = np.arange(-k, k + 1)
    x, y = np.meshgrid(ax, ax)
    gauss_mat = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    gauss_mat /= 2 * np.pi * sigma ** 2
    return gauss_mat / gauss_mat.sum()

# 读取灰度图像
for num in range(1, 4):
    image_grey = cv2.imread("../dataset/" + str(num) + '.jpg', cv2.IMREAD_GRAYSCALE)
    H, W = image_grey.shape
    # 高斯滤波
    gauss_core = gauss_filter()
    filter_image = cv2.filter2D(image_grey, -1, gauss_core)

    gx, gy = [], []

    # Sobel 算子计算梯度
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8

    # Prewitt 算子计算梯度
    # sx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / 6
    # sy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) / 6
    #
    for i in range(0, H):
        gx.append([])
        gy.append([])
        for j in range(0, W):
            if i == 0 or i == H - 1 or j == 0 or j == W - 1:
                gx[-1].append(0)
                gy[-1].append(0)
                continue
            mat = filter_image[i - 1: i + 2, j - 1: j + 2]
            gx[-1].append(np.sum(mat * sx))
            gy[-1].append(np.sum(mat * sy))
    gx = np.array(gx)
    gy = np.array(gy)
    g = np.sqrt(gx ** 2 + gy ** 2)

    # Canny 算子计算梯度
    # sx = np.array([[-1, 1], [-1, 1]]) / 4
    # sy = np.array([[1, 1], [-1, -1]]) / 4
    # for i in range(H):
    #     gx.append([])
    #     gy.append([])
    #     for j in range(W):
    #         if i == 0 or i == H - 1 or j == 0 or j == W - 1:
    #             gx[-1].append(0)
    #             gy[-1].append(0)
    #             continue
    #         mat = filter_image[i : i + 2, j : j + 2]
    #         gx[-1].append(np.sum(mat * sx))
    #         gy[-1].append(np.sum(mat * sy))
    # gx = np.array(gx)
    # gy = np.array(gy)
    # g = np.sqrt(gx ** 2 + gy ** 2)

    # 非极大值抑制
    g0 = np.zeros_like(g)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if(gx[i][j] == 0 and gy[i][j] == 0):
                continue
            if(gx[i][j] == 0 and gy[i][j] != 0):
                tmp1 = g[i - 1][j]
                tmp2 = g[i + 1][j]
                if (g[i][j] >= tmp1 and g[i][j] >= tmp2):
                    g0[i][j] = g[i][j]
                continue
            if(gx[i][j] != 0 and gy[i][j] == 0):
                tmp1 = g[i][j - 1]
                tmp2 = g[i][j + 1]
                if (g[i][j] >= tmp1 and g[i][j] >= tmp2):
                    g0[i][j] = g[i][j]
                continue
            theta = math.atan(gy[i][j] / gx[i][j]) / math.pi * 180 + 90
            tmp1, tmp2 = 0, 0
            if 0 <= theta < 45:
                k = abs(gx[i][j] / gy[i][j])
                tmp1 = g[i - 1][j] * (1 - k) + g[i - 1][j - 1] * k
                tmp2 = g[i + 1][j] * (1 - k) + g[i + 1][j + 1] * k
            if 45 <= theta < 90:
                k = abs(gy[i][j] / gx[i][j])
                tmp1 = g[i][j - 1] * (1 - k) + g[i - 1][j - 1] * k
                tmp2 = g[i][j + 1] * (1 - k) + g[i + 1][j + 1] * k
            if 90 <= theta < 135:
                k = abs(gy[i][j] / gx[i][j])
                tmp1 = g[i][j + 1] * (1 - k) + g[i - 1][j + 1] * k
                tmp2 = g[i][j - 1] * (1 - k) + g[i + 1][j - 1] * k
            if 135 <= theta < 180:
                k = abs(gx[i][j] / gy[i][j])
                tmp1 = g[i - 1][j] * (1 - k) + g[i - 1][j + 1] * k
                tmp2 = g[i + 1][j] * (1 - k) + g[i + 1][j - 1] * k
            if (g[i][j] >= tmp1 and g[i][j] >= tmp2):
                g0[i][j] = g[i][j]
    gradmap = np.zeros_like(g)
    img_bool = np.zeros_like(g)

    # 设置高低阈值
    high = 12.5
    low = 0.4 * high
    low = int(low)

    que = []
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if g0[i][j] >= high:
                gradmap[i][j] = high
                img_bool[i][j] = True
                que.append([i, j])
            elif g0[i][j] >= low:
                gradmap[i][j] = low

    # 边缘连接
    vis = np.zeros_like(g)
    while(len(que) != 0):
        u = que.pop(0)
        x = u[0]
        y = u[1]
        if vis[x][y]:
            continue
        img_bool[x][y] = True
        vis[x][y] = True
        cnt = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if img_bool[x + i][y + j]:
                    cnt += 1
        if cnt <= 4:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if gradmap[x + i][y + j] >= low:
                        que.append([x + i, y + j])

    # folder_path = "../contrast"
    # os.makedirs(folder_path, exist_ok=True)
    plt.figure(figsize = (W / 100, H / 100), dpi = 100)
    plt.imshow(img_bool, cmap='gray')
    # img_canny = cv2.Canny(image_grey, 80, 200)
    # plt.imshow(img_canny, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    # plt.savefig(folder_path + "/"+str(num)+ "_canny"  + ".jpg")
    plt.show()
    plt.close()
