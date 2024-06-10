import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Custom gaze data set.
class GazeDataSet(Dataset):
    def __init__(self, data_path, transform=None):
        print("init")
        self.data_path = data_path
        self.transform = transform
        
        # Convert image to 224x224 into Tensor if no transform is given.
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        self.samples = []
        
        # Iterate through directories and load annotations
        for dir_name in [f.name for f in os.scandir(self.data_path) if f.is_dir()][:3]:
            annotation_path = os.path.join(self.data_path, dir_name, f"{dir_name}.txt")
            with open(annotation_path, 'r') as file:
                # Read annotation lines.
                lines = file.readlines()
                
                for line in lines:
                    parts = line.split()
                    
                    # Image path from annotation line.
                    image_path = os.path.join(self.data_path, dir_name, parts[0])
                    
                    # Retrieve labels.
                    gaze_target = np.array([float(x) for x in parts[24:27]])  # 3D gaze target location
                    face_center = np.array([float(x) for x in parts[21:24]])  # Face center in camera coordinate system
                    # TODO: only once.
                    gaze_direction = gaze_target - face_center
                    
                    yaw = np.arctan2(gaze_direction[0], gaze_direction[2])
                    pitch = np.arctan2(gaze_direction[1], np.linalg.norm(gaze_direction[[0, 2]]))
                    
                    self.samples.append((image_path, yaw, pitch))
                    
        # Compute mean and standard deviation of labels
        labels = np.array([(yaw, pitch) for _, yaw, pitch in self.samples], dtype=np.float32)
        self.mean_labels = np.mean(labels, axis=0)
        self.std_labels = np.std(labels, axis=0)

    # Required method for DataLoader class, returns the length of the dataset.
    def __len__(self):
        return len(self.samples)

    # Required method for DataLoader class, retrieves an image with corresponding yaw and pitch.
    def __getitem__(self, idx):
        image_path, yaw, pitch = self.samples[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        
        # Normalize labels
        normalized_yaw = (yaw - self.mean_labels[0]) / self.std_labels[0]
        normalized_pitch = (pitch - self.mean_labels[1]) / self.std_labels[1]
        
        return image, normalized_yaw, normalized_pitch


    # # Required method for DataLoader class, returns the length of the dataset.
    # def __len__(self):
    #     return len(self.samples)

    # # Required method for DataLoader class, retreives an image with corresponding yaw and pitch.
    # def __getitem__(self, idx):
    #     image_path, yaw, pitch = self.samples[idx]
    #     image = Image.open(image_path)
    #     image = self.transform(image)
        
    #     return image, yaw, pitch