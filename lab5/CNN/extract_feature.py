# SJTU EE208

import time

import numpy as np
import torch
import torchvision.transforms as transforms
from pyarrow.dataset import dataset
from torchvision.datasets.folder import default_loader
import os

print('Load model: ResNet50')
model = torch.hub.load('pytorch/vision', 'resnet50', weights=True)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# print('Prepare image data!')
# test_image = default_loader('panda.png')
# input_image = trans(test_image)
# input_image = torch.unsqueeze(input_image, 0)


def features(x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)

    return x


print('Extract features!')
start = time.time()
image_features = []
images = [f for f in os.listdir("./dataset") if os.path.isfile(os.path.join("./dataset", f))]
for img in images:
    image = default_loader(os.path.join("./dataset", img))
    image = trans(image)
    image = torch.unsqueeze(image, 0)
    image_feature = features(image)
    image_feature = image_feature.detach().numpy()
    image_feature = np.resize(image_feature, image_feature.shape[1])
    image_feature = image_feature / np.linalg.norm(image_feature)
    image_features.append(image_feature)

print('Time for extracting features: {:.2f}'.format(time.time() - start))

target_images = [f for f in os.listdir("./target") if os.path.isfile(os.path.join("./target", f))]
target_images = sorted(target_images, key=lambda x: int(x[0]))
match = 0
for img in target_images:
    image = default_loader(os.path.join("./target", img))
    image = trans(image)
    image = torch.unsqueeze(image, 0)
    image_feature = features(image)
    image_feature = image_feature.detach().numpy()
    image_feature = np.resize(image_feature, image_feature.shape[1])
    image_feature = image_feature / np.linalg.norm(image_feature)
    res = []
    for j in range(len(image_features)):
        mol = image_feature.dot(image_features[j])
        res.append([mol, images[j]])
    res = sorted(res, key=lambda x: -x[0])
    print(f"The matching result for image {img}")
    for j in range(5):
        print(f"Top {j + 1} similar image : {res[j][1]}, with the similarity {res[j][0] * 100}%")
        if int(img[0]) * 5 + 1 <= int(res[j][1].split(".")[0]) <= int(img[0]) * 5 + 5:
            match += 1
    print("")
print(f"Total Accuracy is {match / len(target_images) * 20}%")