import sys, os, warnings
warnings.filterwarnings("ignore")

from pathlib import Path

PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch, numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from utils.dataset import CTDataset
from utils.transforms import val_transform
from torch.utils.data import DataLoader
from torchvision import models
import timm
import pickle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = PROJECT_ROOT / "models" / "dl"
ML_MODEL_DIR = PROJECT_ROOT / "models" / "ml"
META = PROJECT_ROOT / "data" / "meta"
FEAT_DIR = PROJECT_ROOT / "features" / "resnet50"

BATCH_SIZE = 16
NUM_WORKERS = 0

print(f"Project Root: {PROJECT_ROOT}")
print(f"Model Dir: {MODEL_DIR}")
print(f"Data Meta: {META}")


def load_test_loader():
    ds = CTDataset(META / "test.csv", transform=val_transform)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS), ds


def eval_generic(model, loader):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            preds = out.argmax(dim=1).cpu().numpy()
            preds_all.append(preds)
            labels_all.append(labels.numpy())
    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    return preds_all, labels_all


def build_resnet(num_classes):
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    m.fc = torch.nn.Linear(2048, num_classes)
    return m.to(DEVICE)


def build_efficientnet(num_classes):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_f = m.classifier[1].in_features
    m.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(in_f, num_classes)
    )
    return m.to(DEVICE)


def load_checkpoint_state(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    return model


def infer_num_classes_from_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt

    for key in ["fc.weight", "classifier.1.weight"]:
        if key in state_dict:
            return state_dict[key].shape[0]

    return 4


def eval_model_by_name(name, model_builder, ckpt_name):
    ckpt = MODEL_DIR / ckpt_name
    if not ckpt.exists():
        print(f"SKIP {name}: checkpoint not found -> {ckpt}")
        return None

    num_classes = infer_num_classes_from_checkpoint(ckpt)
    loader, _ = load_test_loader()
    model = model_builder(num_classes)
    model = load_checkpoint_state(model, ckpt)
    preds, labels = eval_generic(model, loader)

    acc = accuracy_score(labels, preds)
    print(f"\n=== {name} ===")
    print("Test Acc:", acc)
    print("Classification Report:\n", classification_report(labels, preds, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(labels, preds))
    return acc


def eval_timm_model(name, timm_name, ckpt_name):
    ckpt = MODEL_DIR / ckpt_name
    if not ckpt.exists():
        print(f"SKIP {name}: checkpoint not found -> {ckpt}")
        return None

    num_classes = infer_num_classes_from_checkpoint(ckpt)
    loader, _ = load_test_loader()
    model = timm.create_model(
        timm_name,
        pretrained=False,
        num_classes=num_classes
    ).to(DEVICE)

    model = load_checkpoint_state(model, ckpt)
    preds, labels = eval_generic(model, loader)

    acc = accuracy_score(labels, preds)
    print(f"\n=== {name} ===")
    print("Test Acc:", acc)
    print("Classification Report:\n", classification_report(labels, preds, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(labels, preds))
    return acc


def load_ml_split(split):
    X = np.load(FEAT_DIR / f"{split}_images.npy")
    y = np.load(FEAT_DIR / f"{split}_labels.npy")
    return X, y


def eval_ml_model(name, model_path):
    if not model_path.exists():
        print(f"SKIP {name}: model not found -> {model_path}")
        return None

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        X_test, y_test = load_ml_split("test")
        _, y_train = load_ml_split("train")

        le = LabelEncoder()
        le.fit(y_train)
        y_test_enc = le.transform(y_test)

        scaler_path = ML_MODEL_DIR / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            X_test = scaler.transform(X_test)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test_enc, preds)

        print(f"\n=== {name} ===")
        print("Test Acc:", acc)
        print("Classification Report:\n",
              classification_report(y_test_enc, preds, zero_division=0))
        print("Confusion Matrix:\n",
              confusion_matrix(y_test_enc, preds))

        return acc

    except Exception as e:
        print(f"ERROR evaluating {name}: {e}")
        return None


def main():
    print("\n" + "=" * 70)
    print("DEEP LEARNING MODELS EVALUATION")
    print("=" * 70)

    eval_model_by_name("ResNet50", build_resnet, "resnet50_best.pth")
    eval_model_by_name("EfficientNet-B0", build_efficientnet, "efficientnet_b0_best.pth")
    eval_model_by_name(
    "EfficientNet-B0 (Hybrid Optimized)",
    build_efficientnet,
    "efficientnet_b0_hybrid_best.pth")

    eval_timm_model("Swin-Base", "swin_base_patch4_window7_224", "swin_base_best.pth")
    eval_timm_model("ViT-Base", "vit_base_patch16_224", "vit_base_best.pth")

    print("\n" + "=" * 70)
    print("MACHINE LEARNING MODELS EVALUATION (Using ResNet50 Features)")
    print("=" * 70)

    eval_ml_model("Logistic Regression", ML_MODEL_DIR / "logistic_regression.pkl")
    eval_ml_model("SVM (RBF Kernel)", ML_MODEL_DIR / "svm_rbf_kernel.pkl")
    eval_ml_model("Random Forest", ML_MODEL_DIR / "random_forest.pkl")
    eval_ml_model("XGBoost", ML_MODEL_DIR / "xgboost.pkl")


if __name__ == "__main__":
    main()
