import torch
import pandas as pd
from torch.utils.data import DataLoader
from utils import *
from config import *
from dataset import CIFAR10PredictionDataset
from model import ResNeXt_CIFAR
from torchvision.transforms import ToTensor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_dataset = CIFAR10PredictionDataset(test_file, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=8)
    
    # load trained model
    model = ResNeXt_CIFAR()
    model.load_state_dict(torch.load(best_model, map_location=device))
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
    submission.to_csv(pred_csv, index=False)
    print("\nTest predictions saved to prediction.csv")
    
if __name__ == '__main__':
    main()