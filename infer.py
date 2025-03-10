import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import numpy as np
import random
from models.model_factory import model_factory
from dataset import CIFAR10EvalDataset
import torchvision.transforms as transforms

def val_one_epoch(model, loader, criterion, device, args, epoch_desc="Validation"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    loop = tqdm(loader, desc=epoch_desc, leave=False)
    with torch.no_grad():
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            if args.use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, epoch_time

def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved CIFAR-10 model on the test set")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved .pth model file")
    parser.add_argument('--model', type=str, default='resnet18', choices=['se-resnet18', 'resnet18'], help="Model architecture to use")
    # parser.add_argument('--val_file', type=str, required=True, help="Validation file or directory for CIFAR-10 test set")
    parser.add_argument('--batch', type=int, default=128, help="Batch size for evaluation")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument('--prefetch_factor', type=int, default=8, help="Prefetch factor for DataLoader")
    parser.add_argument('--use_amp', action='store_true', help="Use automatic mixed precision for evaluation")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_factory(args.model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    val_file = os.path.join('dataset', 'cifar-10-python', 'cifar-10-batches-py', 'test_batch')
    val_dataset = CIFAR10EvalDataset(val_file, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        prefetch_factor=args.prefetch_factor
    )

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc, val_time = val_one_epoch(model, val_loader, criterion, device, args, epoch_desc="Validation")
    print(f"Validation -> Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Time: {val_time:.2f}s")

if __name__ == '__main__':
    main()