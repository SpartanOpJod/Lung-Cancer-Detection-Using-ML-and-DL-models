from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from pathlib import Path
import pickle
import warnings
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")

# 🔑 COLAB-SAFE PROJECT ROOT (FIX)
PROJECT_ROOT = Path("/content/drive/MyDrive/project")

FEAT_DIR = PROJECT_ROOT / "features" / "resnet50"
MODEL_DIR = PROJECT_ROOT / "models" / "ml"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project Root: {PROJECT_ROOT}")
print(f"Features Dir: {FEAT_DIR}")
print(f"Model Dir: {MODEL_DIR}")

RANDOM_STATE = 42


def load_split(split):
    X = np.load(FEAT_DIR / f"{split}_images.npy")
    y = np.load(FEAT_DIR / f"{split}_labels.npy")
    return X, y


def train_and_save_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\n===== {name} =====")
    print(f"Training with {X_train.shape[1]} CNN features")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(y_test, preds))

    model_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    model_path = MODEL_DIR / f"{model_name}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to: {model_path}")
    return acc, macro_f1


def main():
    print("\nLoading ResNet50 CNN features...")
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "SVM (RBF Kernel)": SVC(
            kernel="rbf",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    accuracies = {}
    f1_scores = {}

    print("\nTraining ML models on CNN features...")

    for name, model in models.items():
        acc, f1 = train_and_save_model(
            name, model,
            X_train, y_train,
            X_test, y_test
        )
        accuracies[name] = acc
        f1_scores[name] = f1

    print("\nFINAL COMPARISON")
    for name in accuracies:
        print(f"{name:25} | Acc: {accuracies[name]:.4f} | F1: {f1_scores[name]:.4f}")

    print("\nAll ML models trained using CNN-extracted features only.")


if __name__ == "__main__":
    main()
