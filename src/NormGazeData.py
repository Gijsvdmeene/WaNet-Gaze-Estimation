import os
import numpy as np
import scipy.io
import h5py

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Custom gaze data set.
class NormGazeData(Dataset):
    def __init__(self, data_path, transform=None):
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(p=1),  # Fixed horizontal flip
                # transforms.RandomRotation(degrees=(0, 90)),  # Fixed rotation by 90 degrees
                transforms.Resize((224, 224)),
                # transforms.ToTensor(),# Resize the image to (224, 224)
            ])
        self.data_path = data_path
        self.samples = []
        dirs = [f for f in os.listdir(self.data_path) if f.endswith('.mat')]
        
        for i, file in enumerate(dirs):
            print(f"Getting {i+1}/{len(dirs)+1}")
            mat_file = h5py.File(f'{self.data_path}/{file}', 'r')
            data = np.array(mat_file['Data']['data'])
            labels = np.array(mat_file['Data']['label'])
            
            for i in range(len(data)):
                # Convert numpy array to PyTorch tensor    
                image_tensor = torch.tensor(data[i])
                image_tensor = torch.rot90(image_tensor, 1, (1, 2))
                image_tensor = torch.flip(image_tensor, dims=[2])
                image_tensor = self.transform(image_tensor)
                image_tensor = image_tensor[[2, 1, 0], :, :]
                label_tensor = torch.tensor(labels[i][:2])
                
                
                self.samples.append((image_tensor, label_tensor))
                
        print(f"Len data = {len(self.samples)}")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx][0], self.samples[idx][1]
