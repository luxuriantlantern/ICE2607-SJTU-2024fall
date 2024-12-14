import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def gauss_filter(sigma=1.6, k = 3):
    ax = np.arange(-k, k + 1)
    x, y = np.meshgrid(ax, ax)
    gauss_mat = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    gauss_mat /= 2 * np.pi * sigma ** 2
    return gauss_mat / gauss_mat.sum()

def rotary_matrix(theta):
    rad = theta / 180 * math.pi
    return np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])

def sift_opt(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = image_grey.shape

    img = []
    sigma = 1.6
    init_image = cv2.filter2D(image_grey, -1, gauss_filter(sigma = (sigma ** 2 - 0.5 * 0.5) ** 0.5))
    img.append(init_image)

    scale = 2
    level = math.floor(math.log2(min(H, W))) - 3
    for i in range(level):
        prev_img = img[-1]
        new_width = prev_img.shape[1] // scale
        new_height = prev_img.shape[0] // scale
        img1 = cv2.resize(prev_img, (new_width, new_height))
        img1 = cv2.filter2D(img1, -1, gauss_filter(sigma = (sigma ** 2 - (0.5 * sigma) ** 2) ** 0.5))
        img.append(img1)

    keypoints = []
    for i, imgi in enumerate(img):
        corners = cv2.goodFeaturesToTrack(imgi, maxCorners=300, qualityLevel=0.01, minDistance=8, blockSize=3, k=0.04)
        for j in corners:
            keypoints.append(np.array([j[0][0], j[0][1], i]))

    filter = []
    for kpt in keypoints:
        x = int(kpt[0])
        y = int(kpt[1])
        layer = int(kpt[2])
        radius = round(3 * 1.5 * sigma)
        img_i = img[layer]
        h, w = img_i.shape


        deg = np.zeros(36, dtype = int)
        for i in range(-radius, radius + 1):
            r = x + i
            if r <= 0 or r >= h - 1:
                continue
            for j in range(-radius, radius + 1):
                c = y + j
                if c <= 0 or c >= w - 1:
                    continue
                dx = int(img_i[r][c + 1]) - int(img_i[r][c - 1]) / 2
                dy = int(img_i[r - 1][c]) - int(img_i[r + 1][c]) / 2

                dx = int(dx)
                dy = int(dy)

                mag = (dx * dx + dy * dy) ** 0.5
                if dy >= 0:
                    ang = math.atan2(dy, dx)
                else:
                    ang = math.atan2(dy, dx) + 2 * math.pi
                ang = ang * 180 / math.pi

                bin = round(0.1 * ang)
                if bin >= 36:
                    bin -= 36
                elif bin < 0:
                    bin += 36
                weight = np.exp(-(i ** 2 + j ** 2) /  (2 * radius ** 2))
                deg[bin] += mag * weight
        is_zero = True
        for i in range(len(deg)):
            if deg[i] != 0:
                is_zero = False
        if is_zero:
            continue
        max_deg = max(deg)
        for i in range(36):
            if deg[i] == max_deg:
                dir = i
        filter.append((x * (2 ** (layer)), y * (2 ** (layer)), dir))



    sift = []
    for i in range(len(filter)):
        is_edge = False
        x = filter[i][0]
        y = filter[i][1]
        theta = filter[i][2] * 10
        for j in (-7.5, 7.5):
            for k in (-7.5, 7.5):
                R_M = rotary_matrix(theta)
                pos = R_M @ np.array([[j], [k]])
                x_ = pos[0][0] + x
                y_ = pos[1][0] + y
                if x_ <= 2 or x_ >= H - 3 or y_ <= 2 or y_ >= W - 3:
                    is_edge = True
        if is_edge:
            continue

        cnt = np.zeros((4, 4, 8), dtype = float)
        for j in range(-8, 8):
            for k in range(-8, 8):
                R_M = rotary_matrix(theta)
                pos = R_M @ np.array([[j + 0.5], [k + 0.5]])
                x_ = pos[0][0] + x
                y_ = pos[1][0] + y

                dx1 = x_ - math.floor(x_)
                dx2 = math.ceil(x_) - x_
                dy1 = y_ - math.floor(y_)
                dy2 = math.ceil(y_) - y_

                _x = math.floor(x_)
                _y = math.floor(y_)

                gradxy = np.array([int(image_grey[_x][_y + 1]) - int(image_grey[_x][_y - 1]),
                                   int(image_grey[_x - 1][_y]) - int(image_grey[_x + 1][_y])]) / 2
                gradx1y1 = np.array([int(image_grey[_x + 1][_y + 2]) - int(image_grey[_x + 1][_y]),
                                    int(image_grey[_x][_y + 1]) - int(image_grey[_x + 2][_y + 1])]) / 2
                gradx1y = np.array([int(image_grey[_x + 1][_y + 1]) - int(image_grey[_x + 1][_y]),
                                   int(image_grey[_x][_y]) - int(image_grey[_x + 2][_y])]) / 2
                gradxy1 = np.array([int(image_grey[_x][_y + 2]) - int(image_grey[_x][_y]),
                                   int(image_grey[_x - 1][_y + 1]) - int(image_grey[_x + 1][_y + 1])]) / 2

                dxy = dx2 * dy2 * gradxy + dx1 * dy1 * gradx1y1 + dx1 * dy2 * gradx1y + dx2 * dy1 * gradxy1
                DX = dxy[0]
                DY = dxy[1]

                mag = (DX * DX + DY * DY) ** 0.5
                if DY >= 0:
                    ang = math.atan2(DY, DX)
                else:
                    ang = math.atan2(DY, DX) + 2 * math.pi
                ang = ang * 180 / math.pi

                bin = round(ang / 45)
                if bin >= 8:
                    bin -= 8
                elif bin <0:
                    bin += 8
                posi = math.floor((j + 8) / 4)
                posj = math.floor((k + 8) / 4)
                cnt[posi][posj][bin] += mag
        is_empty = True
        for _ in cnt:
            for __ in _:
                for ___ in __:
                    if ___ != 0:
                        is_empty = False
        if is_empty:
            continue
        cnt = cnt / (np.sum(np.square(cnt)) ** 0.5)
        cnt = cnt.reshape(1, -1)
        sift.append((x, y, np.copy(cnt)))
    return sift

path1 = "./target.jpg"

sift1 = sift_opt(path1)
for idx in range(1, 6):
    path2 = "./dataset/"+str(idx)+".jpg"
    sift2 = sift_opt(path2)

    num = 0
    match = []
    for i in range(len(sift1)):
        maxn = 0.0
        id = 0
        for j in range(len(sift2)):
            tot = 0.0
            for k in range(128):
                tot += sift1[i][2][0][k] * sift2[j][2][0][k]
            if tot > maxn:
                maxn = tot
                id = j
        if maxn >= 0.7:
            match.append((i, id))
            # num += 1

    img1 = cv2.imread(path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(path2, cv2.IMREAD_COLOR)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    canvas_height = max(h1, h2)
    canvas_width = w1 + w2
    canvas = 255 * np.ones((canvas_height, canvas_width, 3), dtype=np.uint8)

    canvas[:h1, :w1, :] = img1
    canvas[:h2, w1:w1 + w2, :] = img2

    plt.figure(figsize=(12, 6))
    plt.imshow(canvas)
    for (i, j) in match:

        x1 = sift1[i][0]
        y1 = sift1[i][1]

        x2 = sift2[j][0]
        y2 = sift2[j][1]

        plt.scatter(x1, y1, color='red', s=50)
        plt.scatter(x2 + w1, y2, color='blue', s=50)

        plt.plot([x1, x2 + w1], [y1, y2], color='yellow', linewidth=1.5)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig("./result/"+str(idx)+".jpg")

    plt.close()
