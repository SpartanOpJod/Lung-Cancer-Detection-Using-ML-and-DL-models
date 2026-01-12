# run from project root: python scripts/merge_labels_to_base.py
import pandas as pd
from pathlib import Path
import shutil

ROOT = Path.cwd()
META_DIR = ROOT / "data" / "meta"
BACKUP_DIR = META_DIR / "backup_before_merge"
BACKUP_DIR.mkdir(exist_ok=True)

csv_files = ["train.csv", "val.csv", "test.csv"]

# backup originals
for f in csv_files:
    src = META_DIR / f
    if src.exists():
        shutil.copy(src, BACKUP_DIR / (f + ".bak"))

print("Backed up original meta CSVs to:", BACKUP_DIR)

def map_label(lbl: str) -> str:
    s = str(lbl).lower()
    if "adenocarcinoma" in s:
        return "adenocarcinoma"
    if "large.cell.carcinoma" in s or "large cell carcinoma" in s or "large.cell" in s:
        return "large.cell.carcinoma"
    if "squamous.cell.carcinoma" in s or "squamous" in s:
        return "squamous.cell.carcinoma"
    if "normal" in s:
        return "normal"
    # fallback: return original
    return lbl

for f in csv_files:
    path = META_DIR / f
    if not path.exists():
        print("Missing:", path)
        continue
    df = pd.read_csv(path)
    if 'label' not in df.columns:
        # guess column names
        print(f"Warning: {f} has columns {list(df.columns)}; expected 'label' column.")
        continue
    df['old_label'] = df['label']
    df['label'] = df['label'].apply(map_label)
    out = META_DIR / f  # overwrite (we backed up already)
    df.to_csv(out, index=False)
    print(f"Updated {out} — unique labels now:", sorted(df['label'].unique()))
    print(df['label'].value_counts().to_dict())
