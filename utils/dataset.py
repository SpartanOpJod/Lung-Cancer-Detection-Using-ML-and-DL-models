import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# 🔑 PROJECT ROOT (COLAB-SAFE)
PROJECT_ROOT = Path("/content/drive/MyDrive/project")

class CTDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.classes = sorted(self.df['label'].unique())
        self.label_map = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def get_labels_array(self):
        return self.df['label'].values

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row['image']
        img_path = PROJECT_ROOT / img_path   # 🔥 FIX

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = cv2.imread(str(img_path))
            if img is None:
                img = np.zeros((224, 224), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transform:
            aug = self.transform(image=img)
            img = aug['image']
        else:
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = torch.tensor(img, dtype=torch.float32)

        label = self.label_map[row['label']]
        return img, torch.tensor(label, dtype=torch.long)
