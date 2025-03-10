import os
import argparse
import torchvision.transforms as transforms
from config import *
from dataset import CIFAR10Dataset
from train import train_and_validate
from utils import save_model
import random
import numpy as np
import torch
import datetime

def main():
    # parse args
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 model")
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnext18', 'se-resnet18', 'wide-resnet18', 'shake-wide-resnet18'], help="Model architecture to use")
    parser.add_argument('--max_epochs', type=int, default=300, help="Maximum number of training epochs")
    parser.add_argument('--lr', type=float, default=0.1, help="Learning rate (default: 0.1)")
    parser.add_argument('--patience', type=int, default=30, help="Patience for early stopping (default: 20)")
    parser.add_argument('--batch', type=int, default=128, help="Batch size")
    parser.add_argument('--use_amp', action='store_true', help="Use automatic mixed precision")
    parser.add_argument('--use_mixup', action='store_true', help="Use mix up augmentation")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loader")
    parser.add_argument('--prefetch_factor', type=int, default=8, help="Prefetch factor for data")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()

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

    model = train_and_validate(train_dataset=train_dataset, val_file=val_file, val_transform=val_transforms, args=args)
    
    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    best_model_path = f"logs/best_model_{args.model}_{timestamp}.pth"
    save_model(model, best_model_path)

if __name__ == '__main__':
    main()