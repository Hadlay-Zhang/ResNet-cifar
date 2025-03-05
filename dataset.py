import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch

# train_set
class CIFAR10Dataset(Dataset):
    def __init__(self, data_paths, transform):
        self.data = []
        self.labels = []
        for path in data_paths:
            batch = load_cifar_batch(path)
            self.data.append(batch[b'data'])
            self.labels.extend(batch[b'labels'])
        self.data = np.concatenate(self.data)
        # (N, 32, 32, 3)
        self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.transform = transform
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        pil_img = Image.fromarray(img)
        original = self.base_transform(pil_img)
        augmented = self.transform(pil_img)
        label = self.labels[idx]
        return augmented, original, label

# eval_set
class CIFAR10EvalDataset(Dataset):
    def __init__(self, file_path, transform=None):
        batch = load_cifar_batch(file_path)
        self.data = batch[b'data']
        # (N, 32, 32, 3)
        self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.labels = batch.get(b'labels', None)
        if self.labels is None:
            raise ValueError("Validation dataset must contain labels!")
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        pil_img = Image.fromarray(img)
        if self.transform:
            img_trans = self.transform(pil_img)
        else:
            img_trans = pil_img
        label = self.labels[idx]
        return img_trans, label


# test
class CIFAR10PredictionDataset(Dataset):
    def __init__(self, file_path, transform=None):
        batch = load_cifar_batch(file_path)
        self.data = batch[b'data']
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        pil_img = Image.fromarray(img)
        if self.transform:
            img_trans = self.transform(pil_img)
        else:
            img_trans = pil_img
        return img_trans, idx

