import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

target_image_path = './target.jpg'
dataset_image_paths = ['./dataset/1.jpg', './dataset/2.jpg', './dataset/3.jpg', './dataset/4.jpg', './dataset/5.jpg']


def show_keypoint1():
    img = cv2.imread('./target.jpg')
    if img is None:
        raise FileNotFoundError("无法找到指定的图像文件。请检查路径 './target.jpg'")
    cat = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp, des = sift.detectAndCompute(cat, None)

    img_with_keypoints = cv2.drawKeypoints(
        img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('SIFT Keypoints')
    plt.xticks([]), plt.yticks([])
    plt.show()

def show_keypoint2():
    def gauss_filter(sigma=1.6, k=3):
        ax = np.arange(-k, k + 1)
        x, y = np.meshgrid(ax, ax)
        gauss_mat = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        gauss_mat /= 2 * np.pi * sigma ** 2
        return gauss_mat / gauss_mat.sum()

    path = "./dataset/3.jpg"
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("无法找到指定的图像文件，请检查路径 './target.jpg'")

    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = image_grey.shape

    img = []
    sigma = 1.6
    init_image = cv2.filter2D(image_grey, -1, gauss_filter(sigma=(sigma ** 2 - 0.5 * 0.5) ** 0.5))
    img.append(init_image)

    scale = 2
    level = math.floor(math.log2(min(H, W))) - 3
    for i in range(level):
        prev_img = img[-1]
        new_width = prev_img.shape[1] // scale
        new_height = prev_img.shape[0] // scale
        img1 = cv2.resize(prev_img, (new_width, new_height))
        img1 = cv2.filter2D(img1, -1, gauss_filter(sigma=(sigma ** 2 - (0.5 * sigma) ** 2) ** 0.5))
        img.append(img1)

    keypoints = []
    for i, imgi in enumerate(img):
        corners = cv2.goodFeaturesToTrack(imgi, maxCorners=200, qualityLevel=0.01, minDistance=8, blockSize=3, k=0.04)
        if corners is not None:
            for j in corners:
                x, y = j[0]
                keypoints.append((x * (2 ** i), y * (2 ** i)))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    for keypoint in keypoints:
        x, y = keypoint
        circle = plt.Circle((x, y), radius=5, color='white', fill=False)
        plt.gca().add_patch(circle)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.axis('off')
    plt.show()


target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
if target_image is None:
    print(f"无法加载目标图像 {target_image_path}")
    exit()

sift = cv2.SIFT_create()

keypoints_target, descriptors_target = sift.detectAndCompute(target_image, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

for i, dataset_image_path in enumerate(dataset_image_paths):
    dataset_image = cv2.imread(dataset_image_path, cv2.IMREAD_GRAYSCALE)
    if dataset_image is None:
        print(f"无法加载数据集图像 {dataset_image_path}")
        continue

    keypoints_dataset, descriptors_dataset = sift.detectAndCompute(dataset_image, None)

    matches = bf.match(descriptors_target, descriptors_dataset)
    matches = sorted(matches, key=lambda x: x.distance)

    match_result = cv2.drawMatches(
        cv2.cvtColor(cv2.imread(target_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), keypoints_target,
        cv2.cvtColor(cv2.imread(dataset_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), keypoints_dataset,
        matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(12, 6))
    plt.imshow(match_result)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("./official/"+str(i + 1) + ".jpg")
    plt.close()