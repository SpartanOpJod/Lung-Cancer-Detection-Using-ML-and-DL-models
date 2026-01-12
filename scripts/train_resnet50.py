from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import torch.nn.functional as F


import sys, os
from pathlib import Path

PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from utils.dataset import CTDataset
from utils.transforms import train_transform, val_transform
import numpy as np

def focal_loss(inputs, targets, alpha=1.0, gamma=2.0):
    # inputs: raw logits (B, C), targets: (B,)
    ce = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce)
    loss = alpha * (1 - pt) ** gamma * ce
    return loss.mean()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

IMG_SIZE = 224
BATCH_SIZE = 32  # increased
EPOCHS = 30
NUM_WORKERS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# Use absolute paths for compatibility with Google Colab
DATA_META = PROJECT_ROOT / "data" / "meta"
MODEL_DIR = PROJECT_ROOT / "models" / "dl"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project Root: {PROJECT_ROOT}")
print(f"Data Meta: {DATA_META}")
print(f"Model Dir: {MODEL_DIR}")

def get_loaders(use_sampler=True):
    train_set = CTDataset(DATA_META / "train.csv", transform=train_transform)
    val_set   = CTDataset(DATA_META / "val.csv",   transform=val_transform)

    labels = train_set.get_labels_array()
    classes = sorted(list(train_set.classes))

    # class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array(classes), y=labels)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    # weighted sampler
    if use_sampler:
        label_to_index = {c: i for i, c in enumerate(classes)}
        sample_weights = [class_weights[label_to_index[l]] for l in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    else:
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, len(train_set.classes), class_weights_t


def train_model():
    train_loader, val_loader, num_classes, class_weights_t = get_loaders()

    print("Detected classes:", num_classes)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, num_classes)
    model = model.to(DEVICE)

    USE_FOCAL = False  # you can set True later if needed

    if not USE_FOCAL:
        # Use label smoothing for better regularization
        criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.1)
    else:
        criterion = None
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    best_acc = 0
    patience_counter = 0
    patience = 5

    for epoch in range(EPOCHS):
        model.train()
        total, correct = 0, 0
        total_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)

            if USE_FOCAL:
                loss = focal_loss(outputs, labels, alpha=1.0, gamma=2.0)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            
            optimizer.step()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        vtotal, vcorrect = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)

                vcorrect += (preds == labels).sum().item()
                vtotal += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = vcorrect / vtotal
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        
        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.4f} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {macro_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / "resnet50_best.pth")
            print(f"🔥 Saved new best model with Val Acc: {best_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Training complete. Best Validation Accuracy:", best_acc)

if __name__ == "__main__":
    train_model()
