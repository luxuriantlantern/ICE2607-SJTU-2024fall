import cv2
import os
import matplotlib.pyplot as plt

for i in range(1, 4):
    name_image = "img" + str(i) + ".jpg"
    image = cv2.imread("../images/" + name_image, cv2.IMREAD_GRAYSCALE)
    # 按灰度形式读入

    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    # 隐去坐标轴，输出
    plt.tight_layout()

    folder_path = "../greyimages"
    os.makedirs(folder_path, exist_ok=True)
    # 保存图片

    plt.savefig(os.path.join(folder_path, name_image))
    plt.close()