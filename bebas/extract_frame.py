import torch
from torchvision import models, transforms
from PIL import Image

# Ensure to set the computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained model from pytorch
resnet152 = models.resnet152(pretrained=True)

# Use the model object to select the desired layer
layer = resnet152._modules.get('avgpool') # use the avgpool layer for feature extraction

# Set model to evaluation mode
resnet152 = resnet152.to(device).eval()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_vector(image_path):
    # Load and transform image
    image = Image.open(image_path).convert("RGB")  # convert the image to RGB
    t_image = transform(image).unsqueeze(0).to(device)
    
    def copy_data(m, i, o):
        return o.reshape(o.size(0), -1)  # reshape the output tensor

    # Register forward hook and forward pass image through the model
    h = layer.register_forward_hook(copy_data)
    features = resnet152(t_image)
    h.remove()

    return features


image_path = "/Users/ibnuhafizh/Documents/ITB/TA/football_frame.png"  # replace with your image path
features = get_vector(image_path)
print(features)