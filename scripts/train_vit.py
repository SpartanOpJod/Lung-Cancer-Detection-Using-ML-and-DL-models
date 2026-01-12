from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import torch.nn.functional as F
import numpy as np

import sys, os
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from utils.dataset import CTDataset
from utils.transforms import train_transform, val_transform
import timm

def focal_loss(inputs, targets, alpha=1.0, gamma=2.0):
    # inputs: raw logits (B, C), targets: (B,)
    ce = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce)
    loss = alpha * (1 - pt) ** gamma * ce
    return loss.mean()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

IMG_SIZE = 224
BATCH_SIZE = 8  # Reduced from 16
EPOCHS = 30
NUM_WORKERS = 0  # Set to 0 for Windows compatibility
LEARNING_RATE = 1e-4  # Lower LR for transformers (much more stable)
WEIGHT_DECAY = 1e-5  # Reduced
GRAD_CLIP = 1.0

# Use absolute paths for compatibility with Google Colab
MODEL_DIR = PROJECT_ROOT / "models" / "dl"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_META = PROJECT_ROOT / "data" / "meta"

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

        train_loader = DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=False
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=False
        )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    return train_loader, val_loader, len(train_set.classes), class_weights_t

def build_model(num_classes):
    # ViT base patch16 224 pretrained
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'set_grad_checkpointing'):
        model.set_grad_checkpointing(True)
    return model.to(DEVICE)

def train():
    train_loader, val_loader, num_classes, class_weights_t = get_loaders()
    print(f"Detected classes: {num_classes}")
    print(f"Class weights: {class_weights_t}")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    model = build_model(num_classes)

    USE_FOCAL = False  # you can set True later if needed

    if not USE_FOCAL:
        criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.1)
    else:
        criterion = None
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()  # AMP scaler

    best_val = 0.0
    patience_counter = 0
    patience = 5
    for epoch in range(EPOCHS):
        model.train()
        total, correct = 0, 0
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(imgs)
                if USE_FOCAL:
                    loss = focal_loss(outputs, labels, alpha=1.0, gamma=2.0)
                else:
                    loss = criterion(outputs, labels)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
        train_acc = correct / total if total>0 else 0.0
        avg_loss = total_loss / len(train_loader)

        # validation
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
        val_acc = vcorrect / vtotal if vtotal>0 else 0.0
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.4f} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {macro_f1:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "val_acc": best_val,
                "epoch": epoch+1,
            }, MODEL_DIR / "vit_base_best.pth")
            print(f"🔥 Saved new best model with Val Acc: {best_val:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Training finished. Best Val Acc:", best_val)

if __name__ == "__main__":
    train()
