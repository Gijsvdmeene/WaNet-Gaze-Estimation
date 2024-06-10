from torch.utils.data import DataLoader
from GazeDataSet import GazeDataSet
import os

ROOT_DIR = os.path.abspath(os.curdir).strip('/src')
DATA_PATH = f"/{ROOT_DIR}/data/MPIIFaceGaze"
dataset = GazeDataSet(data_path=DATA_PATH)

# Create DataLoader
batch_size = 32
shuffle = True
num_workers = 4
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# Iterate over DataLoader
for images, yaws, pitches in data_loader:
    # Do something with the batch of data
    print("Batch size:", images.size(0))
    print("Images shape:", images.shape)
    print("Yaw angles:", yaws)
    print("Pitch angles:", pitches)