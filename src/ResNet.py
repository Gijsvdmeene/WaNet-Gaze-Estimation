import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from GazeDataSet import GazeDataSet
import os

# Define the regression model with ResNet18 as a feature extractor
class RegressionResNet(nn.Module):
    def __init__(self):
        super(RegressionResNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 2)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

ROOT_DIR = os.path.abspath(os.curdir).strip('/src')
DATA_PATH = f"/{ROOT_DIR}/data/MPIIFaceGaze"
dataset = GazeDataSet(data_path=DATA_PATH)

# Create DataLoader
num_workers = 4

dataset = GazeDataSet(DATA_PATH)

# Define the proportions for train and test subsets
train_proportion = 0.8
test_proportion = 1 - train_proportion

# Calculate the sizes of train and test subsets
train_size = int(train_proportion * len(dataset))
test_size = len(dataset) - train_size

# Split dataset into training and testing subsets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Define batch size TODO: MORE
batch_size = 32

# Create DataLoaders for training and testing subsets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the number of epochs
num_epochs = 4

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an instance of the regression model with ResNet18
model = RegressionResNet().to(device)

# Define loss function and optimizer.
criterion = nn.L1Loss()
# TODO: Learning too big.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TODO: After 4 epochs -> multiply learning rate by 0.1 (), adjust learning rate through epochs.
# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, yaws, pitches in tqdm(train_loader):
        
        # Move data to device (GPU if available)
        images, yaws, pitches = images.to(device), yaws.to(device), pitches.to(device)
        
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, torch.stack((yaws, pitches), dim=1))  # Calculate the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Evaluation
model.eval()  # Set the model to evaluation mode
total_loss = 0.0
with torch.no_grad():
    for images, yaws, pitches in tqdm(test_loader):
        # Move data to device (GPU if available)
        images, yaws, pitches = images.to(device), yaws.to(device), pitches.to(device)
        
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, torch.stack((yaws, pitches), dim=1))  # Calculate the loss
        total_loss += loss.item() * images.size(0)
mean_loss = total_loss / len(test_loader.dataset)
print(f"Mean Test Loss: {mean_loss:.4f}")


