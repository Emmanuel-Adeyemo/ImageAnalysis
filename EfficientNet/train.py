import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt
from PIL import Imagefrom PIL import Image
import pandas as pd
#import torch
from torch.utils.data import Dataset
from torch import nn
import torch.optim as optim
from torchvision.models import efficientnet_b2
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset
import torch

from utils.image_dataset import ImageScoreDataset


## Check if ids match for labels and images

# Load the CSV file
labels = pd.read_csv("labels.csv")

labels = labels.dropna(subset = ['score'])

valid_ids = set(labels['id'].astype(str))

# list of image filenames in the directory
image_dir = "cropped/"
image_filenames = os.listdir(image_dir)

extra_images = [img for img in image_filenames if img.split('.')[0] not in valid_ids]

if extra_images:
    print(f"Images with no corresponding ID in labels.csv: {extra_images}")
    # remove the extra images
    for img in extra_images:
        os.remove(os.path.join(image_dir, img))
        print(f"Deleted: {img}")
else:
    print("All images have corresponding IDs in labels.csv!")



model = efficientnet_b2(weights='EfficientNet_B2_Weights.IMAGENET1K_V1')
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(model.classifier[1].in_features, 1)  # change layer to regression output

)

transform = transforms.Compose([
    transforms.Resize((260, 260)),  # resize for EfficientNet-B2
    transforms.ToTensor(),         # convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# dataset
dataset = ImageScoreDataset(labels_df = labels, image_folder="cropped", transform=transform)


train_size = int(0.85 * len(dataset))  # 85% for training
val_size = len(dataset) - train_size  # remaining 15% for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

criterion = nn.L1Loss() # use mae
#criterion = nn.MSELoss()  # Loss function
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_losses = []
val_losses = []
learning_rates = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_train_loss = 0.0
    for images, scores in train_loader:
        images, scores = images.to(device), scores.to(device)

        optimizer.zero_grad()
        predictions = model(images).squeeze()
        loss = criterion(predictions, scores)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    train_losses.append(running_train_loss / len(train_loader))
    learning_rates.append(optimizer.param_groups[0]['lr'])  # Record current learning rate

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, scores in val_loader:
            images, scores = images.to(device), scores.to(device)
            predictions = model(images).squeeze()
            loss = criterion(predictions, scores)
            running_val_loss += loss.item()

    val_losses.append(running_val_loss / len(val_loader))

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Learning Rate: {learning_rates[-1]:.5f}")




torch.save(model.state_dict(), "efficient_b2_model_retrain.pth")
