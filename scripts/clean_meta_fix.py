# scripts/clean_meta_fix.py
import csv
from pathlib import Path
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
meta_dir = PROJECT / "data" / "meta"
processed_root = PROJECT / "data" / "processed"

def is_image_readable(p: Path):
    try:
        img = Image.open(p)
        img.verify()
        return True
    except Exception:
        return False

# build an index of available processed images by filename
available = {}
for p in processed_root.rglob("*.*"):
    if p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}:
        available[p.name] = p

print(f"Indexed processed images: {len(available)} files")

for csv_file in sorted(meta_dir.glob("*.csv")):
    rows = []
    removed = 0
    for row in csv.DictReader(csv_file.open()):
        raw_path = row["image"]
        candidate = Path(raw_path)
        resolved = None

        # 1) try path as-is (absolute or relative to project)
        p1 = candidate if candidate.is_absolute() else (PROJECT / candidate)
        if p1.exists() and is_image_readable(p1):
            resolved = p1
        else:
            # 2) try basename lookup in processed folder index
            fname = candidate.name
            if fname in available and is_image_readable(available[fname]):
                resolved = available[fname]
            else:
                # 3) try intelligent alternative: maybe path already points into data/processed with .. style
                # normalize backslashes, remove leading .. segments and try relative from project
                norm = Path(str(raw_path).replace("..\\", "").replace("../", "").lstrip("\\/"))
                p3 = PROJECT / norm
                if p3.exists() and is_image_readable(p3):
                    resolved = p3

        if resolved:
            # store path relative to project for portability
            rel = resolved.relative_to(PROJECT)
            rows.append({"image": str(rel), "label": row["label"]})
        else:
            removed += 1

    out = meta_dir / f"{csv_file.stem}.fixed.csv"
    with out.open("w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=["image","label"])
            writer.writeheader()
            writer.writerows(rows)
        else:
            f.write("image,label\n")
    print(f"{csv_file.name}: kept={len(rows)} removed={removed} -> {out.name}")

print("Fixed cleaning done. If counts look good, move .fixed.csv to replace original .csv files.")
