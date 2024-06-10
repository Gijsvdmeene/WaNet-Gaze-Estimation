import torch
import numpy as np
from torchvision.transforms import transforms
from PIL import Image

img = Image.open("/home/gijs/school/cse3000/data/MPIIFaceGaze/p00/day01/0005.jpg")

# Print image information
print("Image mode:", img.mode)
print("Image size:", img.size)
print(np.max(np.array(img)))
transform = transforms.Compose([transforms.PILToTensor()])

tensor = transform(img)
print(tensor)


# Parameters:
# learning_rate = 0.001

# # Set device.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cuda_device = 0
# torch.cuda.set_device(cuda_device)


# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# # model.to_device(cuda_device)

# # Reconstruct output.
# num_features = model.fc.in_features
# model.fc = nn.Linear(num_features, 2)  

# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.L1Loss()


