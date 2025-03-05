import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import USE_AMP, USE_MIXUP, BATCH_SIZE, NUM_WORKERS, PREFETCH_FACTOR
from model import ResNeXt_CIFAR
from utils import EarlyStopping, print_model
from dataset import CIFAR10EvalDataset

# mixup
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().to(x.device)
    else:
        lam = torch.tensor(1.0, device=x.device)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam.item()

# training for 1 epoch
def train_one_epoch(model, loader, criterion, optimizer, device, epoch_desc="Epoch"):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    loop = tqdm(loader, desc=epoch_desc, leave=False)
    for augmented, original, labels in loop:
        augmented, original, labels = augmented.to(device), original.to(device), labels.to(device)
        optimizer.zero_grad()
        if USE_MIXUP:
            augmented, targets_a, targets_b, lam = mixup_data(augmented, labels, alpha=1.0)
            if USE_AMP:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(augmented)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                outputs = model(augmented)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            if USE_AMP:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(augmented)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(augmented)
                loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            if USE_AMP:
                with torch.amp.autocast(device_type='cuda'):
                    clean_outputs = model(original)
                    clean_loss = criterion(clean_outputs, labels)
            else:
                clean_outputs = model(original)
                clean_loss = criterion(clean_outputs, labels)
        running_loss += clean_loss.item()
        _, preds = torch.max(clean_outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, epoch_time

# evaluation for one epoch
def val_one_epoch(model, loader, criterion, device, epoch_desc="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    loop = tqdm(loader, desc=epoch_desc, leave=False)
    with torch.no_grad():
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            if USE_AMP:
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

def train_and_validate(train_dataset, val_file, max_epochs=200, lr=0.1, patience=20, device=None, val_transform=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNeXt_CIFAR(num_classes=10, scale_factor=3.75, cardinality=8, block_counts=[3,3,3])
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=PREFETCH_FACTOR)
    val_dataset = CIFAR10EvalDataset(val_file, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=PREFETCH_FACTOR)
    
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        train_loss, train_acc, train_time = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                                             epoch_desc=f"Train Epoch {epoch+1}")
        val_loss, val_acc, val_time = val_one_epoch(model, val_loader, criterion, device,
                                                   epoch_desc=f"Val Epoch {epoch+1}")
        scheduler.step()
        print(f"Train  -> Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Time: {train_time:.2f}s")
        print(f"Val    -> Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Time: {val_time:.2f}s")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch+1
            best_model_state = model.state_dict()
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}. Best epoch: {best_epoch} with Val Acc: {best_val_acc:.2f}%")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model
