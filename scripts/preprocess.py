import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Config
RAW_DIR = Path("../data/raw")   # keep this (script run from project/scripts/)
OUT_DIR = Path("../data/processed")
META_DIR = Path("../data/meta")
IMG_SIZE = 224
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42
VALID_SPLIT_NAMES = {"train", "test", "valid", "val"}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

OUT_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

def find_images_and_labels(raw_dir: Path):
    files = list(raw_dir.rglob("*"))
    items = []
    for f in files:
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            # infer label: nearest ancestor folder that is NOT one of train/test/valid or raw
            ancestor_parts = f.resolve().parents
            label = None
            for anc in ancestor_parts:
                name = anc.name.lower()
                if name in {"raw", raw_dir.name.lower(), "data"}:
                    continue
                if name in VALID_SPLIT_NAMES:
                    continue
                # first reasonable anc is the class
                if anc == raw_dir:
                    continue
                label = anc.name
                break
            if label is None:
                # fallback: parent folder name
                label = f.parent.name
            items.append((str(f), label))
    return items

items = find_images_and_labels(RAW_DIR)
if len(items) == 0:
    print("ERROR: No image files found under", RAW_DIR.resolve())
    print("Please check the dataset path and that files have extensions:", ", ".join(sorted(IMG_EXTS)))
    raise SystemExit(1)

# quick summary
from collections import Counter
cnt = Counter([lbl for _, lbl in items])
print("Found total images:", len(items))
print("Classes detected and counts:")
for k,v in cnt.items():
    print(f"  {k}: {v}")

paths, labels = zip(*items)

# stratified split
try:
    train_p, test_p, train_l, test_l = train_test_split(paths, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE)
    train_p, val_p, train_l, val_l = train_test_split(train_p, train_l, test_size=VAL_SIZE/(1-TEST_SIZE), stratify=train_l, random_state=RANDOM_STATE)
except ValueError as e:
    print("Error during train/test split:", e)
    print("Attempting non-stratified split instead.")
    train_p, test_p, train_l, test_l = train_test_split(paths, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_p, val_p, train_l, val_l = train_test_split(train_p, train_l, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=RANDOM_STATE)

def write_split(paths_list, labels_list, split_name):
    out_split = OUT_DIR / split_name
    out_split.mkdir(parents=True, exist_ok=True)
    records = []
    idx = 0
    for src, lbl in tqdm(zip(paths_list, labels_list), total=len(paths_list), desc=f"Processing {split_name}"):
        try:
            img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # try reading as color and convert
                img = cv2.imread(src)
                if img is None:
                    print("WARN: could not read", src)
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            img_f = img.astype(np.float32) / 255.0
            save_name = f"{idx:06d}.png"
            save_path = out_split / save_name
            cv2.imwrite(str(save_path), (img_f * 255).astype(np.uint8))
            records.append({"image": str(save_path), "label": lbl})
            idx += 1
        except Exception as ex:
            print("WARN: failed processing", src, "->", ex)
    df = pd.DataFrame(records)
    csv_path = META_DIR / f"{split_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(records)} items to {out_split} and {csv_path}")

write_split(train_p, train_l, "train")
write_split(val_p, val_l, "val")
write_split(test_p, test_l, "test")

print("Preprocessing DONE.")
