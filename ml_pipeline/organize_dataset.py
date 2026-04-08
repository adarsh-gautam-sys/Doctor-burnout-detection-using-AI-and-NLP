"""
organize_dataset.py
====================
Run once to consolidate all images from "CliniCare Dataset" into
"CliniCare v2/data/" with a unified JSON manifest for the webpage.

Structure produced:
  CliniCare v2/
    data/
      labeled/
        real/        ← 129 images (JPG) from real/ + real_labels.csv
        synthetic/   ← 1000 images (PNG) from synthetic/ + labels.csv
      unlabeled/     ← 384 images (JPG) from dataset_prescrition/ (no burnout labels)
      dataset.json   ← unified manifest the webpage can fetch

Folders removed after copy:
  - model_outputs/
  - model_outputs_v2/
  - saved_models/      (huge model weights not needed for frontend)
  - nb_dump.txt        (notebook text dump)
"""

import csv
import json
import os
import shutil
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────
SRC = Path(r"C:\Users\conta.LAPTOP-IR41J1UC\Desktop\CliniCare Dataset")
DST = Path(r"C:\Users\conta.LAPTOP-IR41J1UC\Desktop\CliniCare v2")

DATA        = DST / "data"
LABELED     = DATA / "labeled"
REAL_DST    = LABELED / "real"
SYNTH_DST   = LABELED / "synthetic"
UNLABELED   = DATA / "unlabeled"

# ── Create dirs ───────────────────────────────────────────
for d in [REAL_DST, SYNTH_DST, UNLABELED]:
    d.mkdir(parents=True, exist_ok=True)
    print(f"  ✅ Created {d.relative_to(DST)}")


# ── 1. Real labeled images (129) ──────────────────────────
print("\n── Processing REAL labeled images ──")
real_labels = {}
real_csv = SRC / "real" / "real_labels.csv"
with open(real_csv, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:
            fname, label = row[0].strip(), row[1].strip()
            real_labels[fname] = label

real_copied = 0
for fname, label in real_labels.items():
    src_path = SRC / "real" / fname
    if src_path.exists():
        shutil.copy2(src_path, REAL_DST / fname)
        real_copied += 1
print(f"  Copied {real_copied} real images → data/labeled/real/")


# ── 2. Synthetic labeled images (1000) ────────────────────
print("\n── Processing SYNTHETIC labeled images ──")
synth_labels = {}
labels_csv = SRC / "labels.csv"
with open(labels_csv, "r") as f:
    reader = csv.reader(f)
    header = next(reader)  # skip header row
    for row in reader:
        if len(row) >= 2:
            fname, label = row[0].strip(), row[1].strip()
            synth_labels[fname] = label

synth_copied = 0
for fname, label in synth_labels.items():
    src_path = SRC / "synthetic" / fname
    if src_path.exists():
        shutil.copy2(src_path, SYNTH_DST / fname)
        synth_copied += 1
    else:
        # Try without .png extension variations
        alt = SRC / "synthetic" / fname
        if alt.exists():
            shutil.copy2(alt, SYNTH_DST / fname)
            synth_copied += 1
print(f"  Copied {synth_copied} synthetic images → data/labeled/synthetic/")


# ── 3. Unlabeled prescription images (384) ────────────────
print("\n── Processing UNLABELED prescription images ──")
prescrip_dir = SRC / "dataset_prescrition"
unlabeled_copied = 0
for f in prescrip_dir.iterdir():
    if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
        shutil.copy2(f, UNLABELED / f.name)
        unlabeled_copied += 1
print(f"  Copied {unlabeled_copied} unlabeled images → data/unlabeled/")


# ── 4. Copy dashboard_data.json ───────────────────────────
print("\n── Copying dashboard data ──")
dash_src = SRC / "dashboard_data.json"
if dash_src.exists():
    shutil.copy2(dash_src, DATA / "dashboard_data.json")
    print("  ✅ dashboard_data.json → data/")


# ── 5. Copy best models only (keep it lean) ──────────────
print("\n── Copying essential model files ──")
models_dst = DATA / "models"
models_dst.mkdir(exist_ok=True)

# Best model from saved_models
best_model = SRC / "saved_models" / "best_model.pkl"
if best_model.exists():
    shutil.copy2(best_model, models_dst / "best_model.pkl")
    print("  ✅ best_model.pkl")

# Results CSVs
for csv_name in ["results.csv", "all_model_results.csv"]:
    csv_path = SRC / "saved_models" / csv_name
    if csv_path.exists():
        shutil.copy2(csv_path, models_dst / csv_name)
        print(f"  ✅ {csv_name}")

# Label encoder
le_path = SRC / "saved_models" / "label_encoder.pkl"
if le_path.exists():
    shutil.copy2(le_path, models_dst / "label_encoder.pkl")
    print("  ✅ label_encoder.pkl")


# ── 6. Build unified dataset.json ─────────────────────────
print("\n── Building unified dataset.json ──")

dataset = {
    "version": "1.0",
    "project": "BurnoutAI — Early Detection of Physician Burnout",
    "summary": {
        "total_labeled": real_copied + synth_copied,
        "total_unlabeled": unlabeled_copied,
        "total_images": real_copied + synth_copied + unlabeled_copied,
        "label_classes": ["Low", "Medium", "High"]
    },
    "labeled": {
        "real": {
            "count": real_copied,
            "format": "jpg",
            "directory": "data/labeled/real/",
            "images": []
        },
        "synthetic": {
            "count": synth_copied,
            "format": "png",
            "directory": "data/labeled/synthetic/",
            "images": []
        }
    },
    "unlabeled": {
        "count": unlabeled_copied,
        "directory": "data/unlabeled/",
        "images": []
    }
}

# Real entries
label_counts = {"Low": 0, "Medium": 0, "High": 0}
for fname, label in sorted(real_labels.items(), key=lambda x: x[0]):
    dataset["labeled"]["real"]["images"].append({
        "filename": fname,
        "label": label,
        "source": "real"
    })
    label_counts[label] = label_counts.get(label, 0) + 1

# Synthetic entries
for fname, label in sorted(synth_labels.items(), key=lambda x: x[0]):
    dataset["labeled"]["synthetic"]["images"].append({
        "filename": fname,
        "label": label,
        "source": "synthetic"
    })
    label_counts[label] = label_counts.get(label, 0) + 1

# Unlabeled entries
for f in sorted(UNLABELED.iterdir()):
    if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
        dataset["unlabeled"]["images"].append({
            "filename": f.name,
            "label": None,
            "source": "unlabeled_prescription"
        })

dataset["summary"]["label_distribution"] = label_counts

# Write JSON
json_path = DATA / "dataset.json"
with open(json_path, "w") as f:
    json.dump(dataset, f, indent=2)
print(f"  ✅ dataset.json written ({json_path.stat().st_size} bytes)")


# ── 7. Clean up unnecessary files ─────────────────────────
print("\n── Cleaning up unnecessary files ──")

# Remove nb_dump.txt from CliniCare v2
nb_dump = DST / "nb_dump.txt"
if nb_dump.exists():
    nb_dump.unlink()
    print("  🗑️  Removed nb_dump.txt")

# Remove heavy/duplicate output folders from original dataset
folders_to_remove = [
    SRC / "model_outputs",
    SRC / "model_outputs_v2",
    SRC / "saved_models",
]

for folder in folders_to_remove:
    if folder.exists():
        size_mb = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1_048_576
        shutil.rmtree(folder)
        print(f"  🗑️  Removed {folder.name}/ ({size_mb:.1f} MB)")


# ── Summary ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ORGANIZATION COMPLETE")
print("=" * 55)
print(f"""
  📁 data/labeled/real/       → {real_copied:>4} images (labeled)
  📁 data/labeled/synthetic/  → {synth_copied:>4} images (labeled)
  📁 data/unlabeled/          → {unlabeled_copied:>4} images (no labels)
  📁 data/models/             → essential model files only
  📄 data/dataset.json        → unified manifest
  📄 data/dashboard_data.json → precomputed dashboard data

  Label distribution (all labeled):
    Low:    {label_counts.get('Low', 0)}
    Medium: {label_counts.get('Medium', 0)}
    High:   {label_counts.get('High', 0)}

  🗑️  Removed: nb_dump.txt, model_outputs/, model_outputs_v2/, saved_models/
  ✅ Kept:    models/ (in source), real/, synthetic/, dataset_prescrition/
""")
