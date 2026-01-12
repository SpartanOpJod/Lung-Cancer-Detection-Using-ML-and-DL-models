import os, sys, importlib.util, traceback
from torch.utils.data import DataLoader

# === locate project root & utils files ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
utils_dir = os.path.join(PROJECT_ROOT, "utils")
dataset_file = os.path.join(utils_dir, "dataset.py")
transforms_file = os.path.join(utils_dir, "transforms.py")

for p in (PROJECT_ROOT, utils_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

def load_module_from_path(name, path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Module file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    ds_mod = load_module_from_path("utils_dataset", dataset_file)
    tr_mod = load_module_from_path("utils_transforms", transforms_file)
except Exception:
    traceback.print_exc()
    sys.exit(1)

# get classes/objects
CTDataset = getattr(ds_mod, "CTDataset")
train_transform = getattr(tr_mod, "train_transform")
val_transform = getattr(tr_mod, "val_transform")

# build dataloaders
import torch
train_dataset = CTDataset(os.path.join("data","meta","train.csv"), transform=train_transform)
val_dataset = CTDataset(os.path.join("data","meta","val.csv"), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))

imgs, labels = next(iter(train_loader))
print("Image batch shape:", imgs.shape, "dtype:", imgs.dtype)
print("Label batch shape:", labels.shape, "dtype:", labels.dtype)
print("Sample labels:", labels[:8].tolist())
