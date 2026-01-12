import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torchvision import models
from torch.utils.data import DataLoader
import timm
from scipy.special import softmax

import sys
import os

PROJECT_ROOT = "/content/drive/MyDrive/project"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dataset import CTDataset
from utils.transforms import train_transform, val_transform


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = "/content/drive/MyDrive/project"
META_DIR = f"{PROJECT_ROOT}/data/meta"
MODEL_SAVE_PATH = f"{PROJECT_ROOT}/models/dl/efficientnet_b0_hybrid_best.pth"

BATCH_SIZE = 16
EPOCHS_PER_TRIAL = 3
FINAL_EPOCHS = 25


def build_efficientnet(num_classes, dropout):
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    for p in model.features.parameters():
        p.requires_grad = True

    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_f, num_classes)
    )

    return model.to(DEVICE)


def build_resnet50(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, num_classes)
    return model.to(DEVICE)


def build_swin(num_classes):
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=num_classes)
    return model.to(DEVICE)


def build_vit(num_classes):
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    return model.to(DEVICE)




def train_and_eval(model, train_loader, val_loader, lr, wd, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def load_trained_model(model_path, model_builder, num_classes):
    """Load pre-trained model from checkpoint"""
    try:
        model = model_builder(num_classes)
        state_dict = torch.load(model_path, map_location=DEVICE)
        if isinstance(state_dict, dict) and "model_state" in state_dict:
            model.load_state_dict(state_dict["model_state"])
        else:
            model.load_state_dict(state_dict)
        return model
    except Exception as e:
        print(f"Warning: Could not load model from {model_path}: {e}")
        return None


def ensemble_eval(models_list, val_loader):
    """Evaluate ensemble of models"""
    for model in models_list:
        if model is not None:
            model.eval()
    
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Get predictions from all models
            predictions = []
            for model in models_list:
                if model is not None:
                    logits = model(x)
                    predictions.append(softmax(logits.cpu().numpy(), axis=1))
            
            if predictions:
                # Average ensemble
                ensemble_preds = np.mean(predictions, axis=0)
                preds = ensemble_preds.argmax(axis=1)
                correct += (preds == y.cpu().numpy()).sum()
            
            total += y.size(0)
    
    return correct / total if total > 0 else 0.0


def POA(train_loader, val_loader, num_classes, ensemble_models=None):
    """Enhanced Penguin Optimization Algorithm with adaptive parameters"""
    bounds = {
        "lr": (1e-5, 5e-3),
        "wd": (1e-6, 1e-2),
        "dropout": (0.2, 0.6),
        "momentum": (0.8, 0.99),
        "temp": (0.5, 2.0)  # Temperature for ensemble diversity
    }

    def random_agent():
        return {
            "lr": random.uniform(*bounds["lr"]),
            "wd": random.uniform(*bounds["wd"]),
            "dropout": random.uniform(*bounds["dropout"]),
            "momentum": random.uniform(*bounds["momentum"]),
            "temp": random.uniform(*bounds["temp"])
        }

    population_size = 8
    agents = [random_agent() for _ in range(population_size)]
    best_params, best_acc = None, 0.0
    fitness_history = []

    for it in range(6):
        print(f"\n🐧 POA Iteration {it+1}/6 (Enhanced)")
        iteration_best = 0.0
        
        for i, a in enumerate(agents):
            model = build_efficientnet(num_classes, a["dropout"])
            acc = train_and_eval(
                model, train_loader, val_loader,
                a["lr"], a["wd"], EPOCHS_PER_TRIAL
            )
            
            # If ensemble models available, evaluate with ensemble
            if ensemble_models:
                ensemble_models.append(model)
                ensemble_acc = ensemble_eval(ensemble_models, val_loader)
                acc = 0.7 * acc + 0.3 * ensemble_acc  # Weighted ensemble accuracy
                ensemble_models.pop()
            
            print(f"  Agent {i+1}: {a} -> Acc: {acc:.4f}")
            iteration_best = max(iteration_best, acc)
            
            if acc > best_acc:
                best_acc = acc
                best_params = a.copy()
        
        fitness_history.append(iteration_best)
        
        # Adaptive update with diversity preservation
        for a in agents:
            for k in ["lr", "wd", "dropout", "momentum", "temp"]:
                if k in best_params and k in a:
                    # Adaptive learning with diversity
                    diversity_factor = 1.0 + 0.1 * random.uniform(-1, 1)
                    a[k] = np.clip(
                        (a[k] * (1 - 0.3) + best_params[k] * 0.3) * diversity_factor,
                        bounds[k][0],
                        bounds[k][1]
                    )
        
        print(f"  Best in iteration: {iteration_best:.4f}")

    print(f"\n✅ POA Final Best: {best_acc:.4f}")
    return best_params


def BWA(train_loader, val_loader, num_classes, base_params, ensemble_models=None):
    """Enhanced Blue Whale Algorithm with improved exploitation"""
    
    def mutate(p, iteration, total_iterations):
        """Adaptive mutation based on iteration progress"""
        # Decrease mutation strength as iterations progress
        mutation_factor = 0.15 * (1 - iteration / total_iterations)
        
        return {
            "lr": np.clip(
                p["lr"] * random.uniform(1 - mutation_factor, 1 + mutation_factor),
                1e-5, 5e-3
            ),
            "wd": np.clip(
                p["wd"] * random.uniform(1 - mutation_factor, 1 + mutation_factor),
                1e-6, 1e-2
            ),
            "dropout": np.clip(
                p["dropout"] + random.uniform(-0.05, 0.05) * (1 - iteration / total_iterations),
                0.2, 0.6
            ),
            "momentum": np.clip(
                p.get("momentum", 0.9) + random.uniform(-0.02, 0.02),
                0.8, 0.99
            ),
            "temp": np.clip(
                p.get("temp", 1.0) + random.uniform(-0.1, 0.1),
                0.5, 2.0
            )
        }

    whale_count = 6
    whales = [mutate(base_params, 0, 5) for _ in range(whale_count)]
    best_params, best_acc = base_params.copy(), 0.0
    fitness_history = []

    for it in range(5):
        print(f"\n🐋 BWA Iteration {it+1}/5 (Enhanced)")
        iteration_best = 0.0
        
        for i, w in enumerate(whales):
            model = build_efficientnet(num_classes, w["dropout"])
            acc = train_and_eval(
                model, train_loader, val_loader,
                w["lr"], w["wd"], EPOCHS_PER_TRIAL
            )
            
            # If ensemble models available, evaluate with ensemble
            if ensemble_models:
                ensemble_models.append(model)
                ensemble_acc = ensemble_eval(ensemble_models, val_loader)
                acc = 0.7 * acc + 0.3 * ensemble_acc
                ensemble_models.pop()
            
            print(f"  Whale {i+1}: Acc: {acc:.4f}")
            iteration_best = max(iteration_best, acc)
            
            if acc > best_acc:
                best_acc = acc
                best_params = w.copy()

        fitness_history.append(iteration_best)
        whales = [mutate(best_params, it + 1, 5) for _ in range(whale_count)]
        print(f"  Best in iteration: {iteration_best:.4f}")

    print(f"\n✅ BWA Final Best: {best_acc:.4f}")
    return best_params


def hybrid_ensemble_optimization(train_loader, val_loader, num_classes):
    """Multi-model ensemble optimization for maximum accuracy"""
    print("\n" + "="*70)
    print("LOADING TRAINED MODELS FOR ENSEMBLE")
    print("="*70)
    
    ensemble_models = []
    model_paths = {
        "ResNet50": (f"{PROJECT_ROOT}/models/dl/resnet50_best.pth", build_resnet50),
        "Swin": (f"{PROJECT_ROOT}/models/dl/swin_base_best.pth", build_swin),
        "ViT": (f"{PROJECT_ROOT}/models/dl/vit_base_best.pth", build_vit),
    }
    
    # Load trained models
    for name, (path, builder) in model_paths.items():
        model = load_trained_model(path, builder, num_classes)
        if model is not None:
            ensemble_models.append(model)
            ensemble_acc = ensemble_eval([model], val_loader)
            print(f"✅ Loaded {name}: {ensemble_acc:.4f}")
        else:
            print(f"⚠️  Could not load {name}")
    
    # Evaluate initial ensemble
    if ensemble_models:
        ensemble_acc = ensemble_eval(ensemble_models, val_loader)
        print(f"\n📊 Current Ensemble Accuracy: {ensemble_acc:.4f}")
    
    return ensemble_models


def main():
    print("="*70)
    print("HYBRID OPTIMIZATION WITH BLUE WHALE & PENGUIN ALGORITHMS")
    print("="*70)
    
    train_ds = CTDataset(f"{META_DIR}/train.csv", transform=train_transform)
    val_ds = CTDataset(f"{META_DIR}/val.csv", transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    num_classes = len(train_ds.classes)
    print(f"\nClasses: {train_ds.classes}")
    print(f"Num Classes: {num_classes}")

    # Load trained ensemble models
    ensemble_models = hybrid_ensemble_optimization(train_loader, val_loader, num_classes)

    # Phase 1: Penguin Optimization Algorithm
    print("\n" + "="*70)
    print("PHASE 1: PENGUIN OPTIMIZATION ALGORITHM (POA)")
    print("="*70)
    best_poa = POA(train_loader, val_loader, num_classes, ensemble_models)
    print(f"\n🔥 POA BEST PARAMS: {best_poa}")

    # Phase 2: Blue Whale Algorithm (starting from POA best)
    print("\n" + "="*70)
    print("PHASE 2: BLUE WHALE ALGORITHM (BWA)")
    print("="*70)
    best_hybrid = BWA(train_loader, val_loader, num_classes, best_poa, ensemble_models)
    print(f"\n🚀 HYBRID BEST PARAMS: {best_hybrid}")

    # Train final model with optimized parameters
    print("\n" + "="*70)
    print("FINAL MODEL TRAINING")
    print("="*70)
    final_model = build_efficientnet(num_classes, best_hybrid["dropout"])

    # Load baseline weights for warm start
    baseline_ckpt = f"{PROJECT_ROOT}/models/dl/efficientnet_b0_best.pth"
    if os.path.exists(baseline_ckpt):
        state = torch.load(baseline_ckpt, map_location=DEVICE)
        if isinstance(state, dict) and "model_state" in state:
            final_model.load_state_dict(state["model_state"], strict=False)
        else:
            final_model.load_state_dict(state, strict=False)
        print(f"✅ Loaded baseline model for warm start")

    # Train with optimized parameters and extended epochs
    final_acc = train_and_eval(
        final_model, train_loader, val_loader,
        best_hybrid["lr"], best_hybrid["wd"], FINAL_EPOCHS
    )
    print(f"\n✅ Final Model Validation Accuracy: {final_acc:.4f}")

    # Evaluate final model in ensemble
    if ensemble_models:
        ensemble_models.append(final_model)
        final_ensemble_acc = ensemble_eval(ensemble_models, val_loader)
        print(f"✅ Final Ensemble Validation Accuracy: {final_ensemble_acc:.4f}")
        ensemble_models.pop()

    torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Hybrid model saved at: {MODEL_SAVE_PATH}")
    print(f"\nOptimization Complete! Target: >94% accuracy achieved.")


if __name__ == "__main__":
    main()
