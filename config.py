import torch
import os

# paths
cifar10_dir = os.path.join('dataset', 'cifar-10-python', 'cifar-10-batches-py')
val_file = os.path.join('dataset', 'cifar-10-python', 'cifar-10-batches-py', 'test_batch')
train_files = [os.path.join(cifar10_dir, f"data_batch_{i}") for i in range(1, 6)]
test_file = 'dataset/cifar_test_nolabel.pkl'

torch.backends.cudnn.benchmark = True
