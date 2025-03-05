import os
import torchvision.transforms as transforms
from config import *
from dataset import CIFAR10Dataset
from train import train_and_validate
from utils import save_model

def main():
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.RandomCrop(32, padding=2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    train_dataset = CIFAR10Dataset(train_files, transform=train_transforms)

    model = train_and_validate(train_dataset, val_file, max_epochs=200, lr=0.1, patience=20, val_transform=val_transforms)
    
    save_model(model, best_model)

if __name__ == '__main__':
    main()