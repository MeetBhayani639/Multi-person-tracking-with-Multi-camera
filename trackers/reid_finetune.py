"""
reid_finetune.py

Starter fine-tuning script for a Re-ID backbone using PyTorch.

Assumptions:
- Your dataset is organized as: dataset/train/{id}/{img1.jpg, img2.jpg, ...}
- and dataset/val/{id}/{...}
- This script uses a simple classification loss on person IDs (softmax) + optional triplet loss.

This is intentionally minimal — adapt to add augmentation, Triplet loss, advanced schedulers, etc.
"""
import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from tqdm import tqdm


def make_model(num_classes, feature_dim=256, pretrained=True):
    # Use a ResNet50 backbone; replace with OSNet if you add implementation
    model = models.resnet50(pretrained=pretrained)
    # replace final FC with embedding head
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, feature_dim),
        nn.BatchNorm1d(feature_dim),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(feature_dim, num_classes)
    )
    return model


def train(data_dir='dataset', epochs=10, batch_size=32, lr=1e-4, device='cuda'):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds = datasets.ImageFolder(val_dir, transform=transform)

    num_classes = len(train_ds.classes)
    model = make_model(num_classes, feature_dim=256, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Train E{epoch}")
        running_loss = 0.0
        for imgs, labels in loop:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / (loop.n + 1))
        # TODO: add validation, save best model
        torch.save(model.state_dict(), f'models/reid_epoch{epoch}.pth')
        print(f"Saved models/reid_epoch{epoch}.pth")

    print("Done training.")


if __name__ == '__main__':
    train(epochs=5, batch_size=32, lr=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu')
