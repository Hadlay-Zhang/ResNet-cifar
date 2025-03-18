import torch
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from utils import *
from config import *
from dataset import CIFAR10PredictionDataset
from models.model_factory import model_factory
import torchvision.transforms as transforms
import datetime

def main():
    # parse args
    parser = argparse.ArgumentParser(description="Test on CIFAR-10")
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnext18', 'se-resnet18', 'wide-resnet18', 'shake-wide-resnet18', 'preact-resnet18'], help="Model architecture to use")
    parser.add_argument('--batch', type=int, default=128, help="Batch size")
    parser.add_argument('--use_amp', action='store_true', help="Use automatic mixed precision")
    parser.add_argument('--use_mixup', action='store_true', help="Use mix up augmentation")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loader")
    parser.add_argument('--prefetch_factor', type=int, default=8, help="Prefetch factor for data")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--model_path', type=str, help="Path to best model")
    args = parser.parse_args()

    num_workers = args.num_workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalizer
    ])
    test_dataset = CIFAR10PredictionDataset(test_file, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=args.prefetch_factor)
    
    # load trained model
    model = model_factory(args.model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    predictions = []
    indices = []
    with torch.no_grad():
        for inputs, idx in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            indices.extend(idx.cpu().numpy())

    submission = pd.DataFrame({'ID': indices, 'Label': predictions})
    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    pred_csv = f"prediction/{args.model}_{timestamp}_submission.csv"
    submission.to_csv(pred_csv, index=False)
    print(f"\nTest predictions saved to {pred_csv}.")
    
if __name__ == '__main__':
    main()