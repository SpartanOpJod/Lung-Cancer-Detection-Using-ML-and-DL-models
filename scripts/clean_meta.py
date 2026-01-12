# scripts/clean_meta.py
import csv
from pathlib import Path
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
meta_dir = PROJECT / "data" / "meta"
processed_root = PROJECT / "data" / "processed"

def is_image_readable(p: Path):
    try:
        # quick read with PIL (more tolerant than cv2 for check)
        img = Image.open(p)
        img.verify()  # verify will raise if file is broken
        return True
    except Exception:
        return False

for csv_file in sorted(meta_dir.glob("*.csv")):
    rows = []
    removed = 0
    for row in csv.DictReader(csv_file.open()):
        img_path = Path(row["image"])
        # handle relative paths in CSV
        if not img_path.is_absolute():
            img_path = (PROJECT / img_path).resolve()
        if img_path.exists() and is_image_readable(img_path):
            rows.append(row)
        else:
            removed += 1
    out = meta_dir / f"{csv_file.stem}.cleaned.csv"
    if rows:
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    else:
        out.write_text("image,label\n")  # empty safe csv
    print(f"{csv_file.name}: kept={len(rows)} removed={removed} -> {out.name}")
print("Cleaning done. You can swap .cleaned.csv to .csv after checking.")
