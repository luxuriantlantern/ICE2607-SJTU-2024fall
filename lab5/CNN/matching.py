import time
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
import os

# Load the ResNet50 model
print('Load model: ResNet50')
# We load the model from the Torchvision repository
model = torch.hub.load('pytorch/vision', 'resnet50', weights='IMAGENET1K_V1')

# Make sure the model is in evaluation mode
model.eval()

# Normalization for ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Specify the transformations
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# Define a function to extract features using ResNet50
def features(x):
    # Forward pass through the convolutional layers
    with torch.no_grad():
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
    return x

print('Extract features!')
start = time.time()
image_features = []
dataset_path = "./dataset"

# Check if dataset path exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset directory {dataset_path} not found.")

images = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
images = sorted(images, key = lambda x: int(x.split(".")[0]))
if not images:
    raise ValueError("No valid images found in the dataset directory.")

for img in images:
    image_path = os.path.join(dataset_path, img)
    image = default_loader(image_path)
    image = trans(image)
    image = torch.unsqueeze(image, 0)  # Add a batch dimension
    image_feature = features(image)
    image_feature = image_feature.cpu().numpy().flatten()  # Flatten the feature vector
    image_feature = image_feature / np.linalg.norm(image_feature)  # Normalize the feature
    image_features.append(image_feature)

# Calculate similarities between each pair of images
for i in range(len(image_features)):
    mol = []
    for j in range(len(image_features)):
        if i != j:
            similarity = np.dot(image_features[i], image_features[j])
            mol.append([similarity, j + 1])
    mol = sorted(mol, key=lambda x: x[0], reverse=True)
    print(f"Top similar image for {i + 1}.jpg is {mol[0][1]}, with the similarity = {mol[0][0]}")

end = time.time()
print(f"Feature extraction completed in {end - start:.2f} seconds.")