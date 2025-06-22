import cv2
import torch
import numpy as np
from torchvision import models, transforms

# Load pretrained ReID model (ResNet50 without final classification layer)
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()
model.eval()

# Preprocessing for ReID model
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 64)),  # Standard size for ReID
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Extract embedding from a person image (crop)
def extract_embedding(image):
    with torch.no_grad():
        input_tensor = transform(image).unsqueeze(0)  # shape (1, 3, 128, 64)
        embedding = model(input_tensor)
        return embedding.squeeze().numpy()

# Compare two embeddings (cosine similarity)
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
