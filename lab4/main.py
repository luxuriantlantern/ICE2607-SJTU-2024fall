import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

def LSH_encoder(img):
    H, W, _ = img.shape
    midpt = [H // 2, W // 2]
    def quant(p):
        if p < 0.2:
            return 0
        elif p < 0.33:
            return 1
        else:
            return 2
    def rgb(image):
        b, g, r = cv2.split(image)
        tot_b = np.sum(b)
        tot_g = np.sum(g)
        tot_r = np.sum(r)
        tot = tot_b + tot_g + tot_r

        p_b = quant(tot_b / tot)
        p_g = quant(tot_g / tot)
        p_r = quant(tot_r / tot)

        return [p_b, p_g, p_r]

    encoder = []
    encoder.extend(rgb(img[:midpt[0], :midpt[1]]))
    encoder.extend(rgb(img[:midpt[0], midpt[1]:]))
    encoder.extend(rgb(img[midpt[0]:, :midpt[1]]))
    encoder.extend(rgb(img[:midpt[0], midpt[1]:]))

    return encoder

def get_proj(encoder, select = [0, 4, 11, 19], c = 2):
    proj = []
    for i in select:
        pos = i // c
        if encoder[pos] > i % c:
            proj.append(1)
        else:
            proj.append(0)

    return proj

st_time = time.time()
target_img = cv2.imread("./target.jpg", cv2.IMREAD_COLOR)
target_encoder = LSH_encoder(target_img)
target_proj = get_proj(target_encoder)

all_encoder = []
all_proj = []

hash_table = {}
for i in range(1, 41):
    img = cv2.imread(f'./Dataset/{i}.jpg', cv2.IMREAD_COLOR)
    encoder = LSH_encoder(img)
    all_encoder.append(encoder)
    proj = get_proj(encoder)
    all_proj.append(proj)

    tuple_proj = tuple(proj)
    if tuple_proj in hash_table:
        hash_table[tuple_proj].append(i)
    else:
        hash_table[tuple_proj] = [i]

min_dis = len(target_proj)
id = None

end_time = time.time()
time1 = end_time - st_time
start_time = time.time()

print("Hash method")
if tuple(target_proj) in hash_table:
    res = ""
    id = None
    min_dis = len(target_proj)
    for i in range(len(hash_table[tuple(target_proj)])):
        dis = 0
        for j in range(12):
            dis += abs(all_encoder[hash_table[tuple(target_proj)][i] - 1][j] != target_encoder[j])
        if dis < min_dis:
            min_dis = dis
            id = [hash_table[tuple(target_proj)][i]]
        elif dis == min_dis:
            id.append(hash_table[tuple(target_proj)][i])
    for i in range(len(id)):
        res += f"img{id[i]}.jpg "
    print(f"Target image is the image " + res + "in Dataset")
    print(f"Minimum distance is {min_dis}")
else:
    print("No target image")

end_time = time.time()
time2 = end_time - start_time
start_time = time.time()

id = None
min_dis = len(target_proj) + 1
for i in range(len(all_proj)):
    dis = 0
    for j in range(len(target_encoder)):
        if all_encoder[i][j] != target_encoder[j]:
            dis += 1
    if dis < min_dis:
        id = [i]
        min_dis = dis
    elif dis == min_dis:
        id.append(i)

res = ""
for i in range(len(id)):
    res += f"{id[i] + 1}.jpg "

end_time = time.time()
time3 = end_time - start_time

print("Knn method")
print(f"Target image is the image " + res + "in Dataset")
print(f"Minimum distance is {min_dis}")

print(f"Time for preprocessing: {time1}. Time for Hash searching: {time2}. Time for knn matching: {time3}")