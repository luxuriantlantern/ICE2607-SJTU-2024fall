import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from numpy import dtype
from scipy.ndimage import zoom

image = cv2.imread('./target.jpg', cv2.IMREAD_GRAYSCALE)
H, W = image.shape

def gauss_filter(sigma=1.6, k = 3):
    ax = np.arange(-k, k + 1)
    x, y = np.meshgrid(ax, ax)
    gauss_mat = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    gauss_mat /= 2 * np.pi * sigma ** 2
    return gauss_mat / gauss_mat.sum()

nOctaves = math.floor(math.log2(min(W, H))) - 3
nOctaveLayers = 3
k = 2 ** (1 / nOctaveLayers)
sigma0 = 1.6

scale = 2
image = zoom(image, scale, order=3)

sigma_init = (sigma0 ** 2 - (2 * 0.5) ** 2) ** 0.5
gauss_core = gauss_filter(sigma_init)
image = cv2.filter2D(image, -1, gauss_core)

gauss_pyr_img = [None] * nOctaves
DoG_pyr_img = [None] * nOctaves
img_i = image

for i in range(nOctaves):
    gauss_oct_i = np.zeros((nOctaveLayers + 3, img_i.shape[0], img_i.shape[1]), dtype = img_i.dtype)
    gauss_oct_i[0, :, :] = np.copy(img_i)
    for j in range(1, nOctaveLayers + 3):
        sigma_pre = k ** (j - 1) * sigma0
        sigma_dif = ((k * sigma_pre) ** 2 - sigma_pre ** 2) ** 0.5
        G_C = gauss_filter(sigma_dif)
        img_i = cv2.filter2D(img_i, -1, G_C)
        gauss_oct_i[j, :, :] = np.copy(img_i)
    gauss_pyr_img[i] = np.copy(gauss_oct_i)
    DoG_pyr_img[i] = np.copy(gauss_oct_i.astype(np.int32)[1:, :, :] - gauss_oct_i.astype(np.int32)[:-1, :, :]).astype(np.int32)
    img_i = zoom(gauss_oct_i[-3, :, :], 0.5, order = 3)

# max_width = max([sum(img.shape[1] for img in group) for group in DoG_pyr_img])
# total_height = sum(group[0].shape[0] for group in DoG_pyr_img)
#
# canvas = np.ones((total_height, max_width)) * 255

# y_offset = 0
# for group in DoG_pyr_img:
#     group_height = group[0].shape[0]
#     x_offset = int(W - group[0].shape[1] / 2)
#     for img in group:
#         h, w = img.shape
#         canvas[y_offset:y_offset + h, x_offset:x_offset + w] = img
#         x_offset += W * 2
#     y_offset += group_height
#
#
# plt.figure(figsize=(max_width / 100, total_height / 100))
# plt.imshow(canvas, cmap='gray')
# plt.axis('off')
# plt.tight_layout()
# plt.savefig("./result/DoG_pyr1.png")

ext = []
min_val = 0.03 * 255
for i in range(nOctaves):
    DoG = DoG_pyr_img[i]
    num, h, w = DoG.shape
    for j in range(1, num - 1):
        for r in range(4, h - 3):
            for c in range(4, w - 3):
                neigh = np.copy(DoG[j - 1: j + 2, r - 1: r + 2, c - 1: c + 2])
                val = neigh[1][1][1]
                if abs(val) > min_val and (val == np.max(neigh) or val == np.min(neigh)):
                    sigma_i = k ** j * sigma0
                    ext.append([i, j, r, c, sigma_i])
keypoint = []
for i in range(len(ext)):
    kpt = ext[i]
    num_oct = kpt[0]
    num_lay = kpt[1]
    r = kpt[2]
    c = kpt[3]
    should_delete = True
    D_hat = None

    for iter in range(6):
        DoG = DoG_pyr_img[num_oct][num_lay] / 255
        DoG_pre = DoG_pyr_img[num_oct][num_lay - 1] / 255
        DoG_nxt = DoG_pyr_img[num_oct][num_lay + 1] / 255

        dD = np.array([[DoG[r, c + 1] - DoG[r, c - 1]],
                       [DoG[r + 1, c] - DoG[r - 1, c]],
                       [DoG_pre[r, c] - DoG_nxt[r, c]]
                      ]) / 2
        dxx = (DoG[r, c + 1] + DoG[r, c - 1] - 2 * DoG[r, c]) / 1
        dyy = (DoG[r + 1, c] + DoG[r - 1, c] - 2 * DoG[r, c]) / 1
        dss = (DoG_pre[r, c] + DoG_nxt[r, c] - 2 * DoG[r, c]) / 1

        dxy = (DoG[r + 1, c + 1] + DoG[r - 1, c - 1] - DoG[r + 1, c - 1] - DoG[r - 1, c + 1]) / 4
        dxs = (DoG_nxt[r, c + 1] - DoG_nxt[r, c - 1] - DoG_pre[r, c + 1] + DoG_pre[r, c - 1]) / 4
        dys = (DoG_nxt[r + 1, c] - DoG_nxt[r - 1, c] - DoG_pre[r + 1, c] + DoG_pre[r - 1, c]) / 4

        dH = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
        if np.linalg.det(dH) == 0:
            break
        x_hat = np.linalg.inv(dH) @ dD

        if abs(x_hat[0][0]) < 0.5 and abs(x_hat[1][0]) < 0.5 and abs(x_hat[2][0]) < 0.5:
            should_delete = False
            break

        c = c + round(x_hat[0][0])
        r = r + round(x_hat[1][0])
        num_lay = num_lay + round(x_hat[2][0])

        if  num_lay < 1 or num_lay > nOctaveLayers or \
            r <= 0 or r >= DoG_pyr_img[num_oct][num_lay].shape[0] - 1 or \
            c <= 0 or c >= DoG_pyr_img[num_oct][num_lay].shape[1] - 1:
            break


    if should_delete:
        continue

    D_hat = DoG[r][c] + (dD.T @ x_hat) / 2
    if abs(D_hat) < 0.03:
        continue

    trH = dxx + dyy
    detH = dxx * dyy - dxy * dxy
    r0 = 10

    if detH <= 0 or trH * trH * r0 >= (r0 + 1) * (r0 + 1) * detH:
        continue

    keypoint.append([num_oct, num_lay, r, c, k**num_lay * sigma0])

H, W = gauss_pyr_img[1][1].shape

fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)

ax.imshow(gauss_pyr_img[1][1], cmap='gray')
ax.axis("off")

ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.invert_yaxis()

for i in range(len(keypoint)):
    kpt = keypoint[i]
    num_oct = kpt[0]
    num_lay = kpt[1]
    if num_oct == 1 and num_lay == 1:
        r = kpt[2]
        c = kpt[3]
        if 0 <= r < H and 0 <= c < W:
            circle = plt.Circle((c, r), radius=5, color='white', fill=False, linewidth=1.5)
            ax.add_patch(circle)

plt.show()
plt.close()