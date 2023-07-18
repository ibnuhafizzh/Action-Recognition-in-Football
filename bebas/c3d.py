import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Load the pretrained model
model = torch.hub.load('facebookresearch/VMZ', 'c3d_sports1m')

# Use the model object to select the desired layer
layer = model._modules.get('fc7') # use the fc7 layer for feature extraction

# Set model to evaluation mode
model = model.eval()

# Define video transforms
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

def get_vector(video_path):
    # Load and transform video
    video = Image.open(video_path) # if you have a video file, you need to convert it to frames and apply the transforms to each frame
    video = transform(video)

    features = torch.zeros(1, 4096)

    def copy_data(m, i, o):
        features.copy_(o.data)

    h = layer.register_forward_hook(copy_data)

    # Forward pass video through the model
    model(video.unsqueeze(0))
    h.remove()

    return features

video_path = "/Users/ibnuhafizh/Documents/ITB/TA/football_frame.png"  # replace with your video path
features = get_vector(video_path)
print(features)