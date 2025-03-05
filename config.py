import torch
import os

# Global flags and constants
USE_AMP = True       
USE_MIXUP = True     
NUM_WORKERS = 4      
BATCH_SIZE = 256     
PREFETCH_FACTOR = 8  

# paths
cifar10_dir = os.path.join('dataset', 'cifar-10-python', 'cifar-10-batches-py')
val_file = os.path.join('dataset', 'cifar-10-python', 'cifar-10-batches-py', 'test_batch')
train_files = [os.path.join(cifar10_dir, f"data_batch_{i}") for i in range(1, 6)]
test_file = 'dataset/cifar_test_nolabel.pkl'
best_model = 'logs/best_model.pth'
pred_csv = 'prediction/prediction.csv'

torch.backends.cudnn.benchmark = True
