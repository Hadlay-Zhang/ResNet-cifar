import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import numpy as np
import random
from config import *
from models.model_factory import model_factory
from dataset import CIFAR10EvalDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def val_one_epoch(model, loader, criterion, device, args, epoch_desc="Validation"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
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
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, epoch_time, all_labels, all_preds

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    '''plot confusion matrix'''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig('confusion_matrix.png')

def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved CIFAR-10 model on the test set")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved .pth model file")
    parser.add_argument('--model', type=str, default='resnet18', choices=['se-resnet18', 'resnet18', 'resnext18', 'wide-resnet18', 'shake-wide-resnet18', 'preact-resnet18'], help="Model architecture to use")
    parser.add_argument('--batch', type=int, default=128, help="Batch size for evaluation")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument('--prefetch_factor', type=int, default=8, help="Prefetch factor for DataLoader")
    parser.add_argument('--use_amp', action='store_true', help="Use automatic mixed precision for evaluation")
    parser.add_argument('--seed', type=int, help="Random seed")
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
        normalizer
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
    val_loss, val_acc, val_time, all_labels, all_preds = val_one_epoch(model, val_loader, criterion, device, args, epoch_desc="Validation")
    print(f"Validation -> Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Time: {val_time:.2f}s")

    cm = confusion_matrix(all_labels, all_preds)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plot_confusion_matrix(cm, classes, normalize=True, title='Normalized Confusion Matrix')

if __name__ == '__main__':
    main()
