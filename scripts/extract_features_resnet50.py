import argparse, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Use absolute paths for compatibility with Google Colab
PROJECT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
META_DIR = PROJECT / "data" / "meta"
PROCESSED_ROOT = PROJECT / "data" / "processed"
OUT_ROOT = PROJECT / "features" / "resnet50"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

print(f"Project Root: {PROJECT}")
print(f"Features Output: {OUT_ROOT}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class SimpleImageDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.labels = list(self.df['label'].astype(str))
        self.paths = [str((PROJECT / p).resolve()) if not Path(p).is_absolute() else p for p in self.df['image'].tolist()]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

def make_model(device):
    # load pretrained ResNet50, remove final fc
    model = models.resnet50(pretrained=True)
    model.fc = nn.Identity()  # output will be 2048-d vector
    model = model.to(device)
    model.eval()
    return model

def extract_split(split, model, device, batch_size, img_size):
    csv_path = META_DIR / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    ds = SimpleImageDataset(csv_path, transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    feats = []
    labs  = []
    with torch.no_grad():
        for imgs, labels in tqdm(dl, desc=f"Extracting {split}", unit="batch"):
            imgs = imgs.to(device)
            out = model(imgs)  # shape [B, 2048]
            out = out.cpu().numpy()
            feats.append(out)
            labs.extend(labels)

    feats = np.vstack(feats)
    labs  = np.array(labs)
    np.save(OUT_ROOT / f"{split}_images.npy", feats)
    np.save(OUT_ROOT / f"{split}_labels.npy", labs)
    print(f"Wrote {feats.shape} features and {labs.shape} labels to {OUT_ROOT}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda (auto if None)")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = make_model(device)

    for split in ["train", "val", "test"]:
        extract_split(split, model, device, args.batch_size, args.img_size)

if __name__ == "__main__":
    main()
