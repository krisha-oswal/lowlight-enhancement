"""
Dataset Diagnostics
====================
Run this FIRST to understand exactly what your ExDark folder
contains and what (if anything) is broken.

Usage:
  python tools/diagnose_dataset.py --root data/ExDark
"""

import sys
import argparse
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

EXDARK_CLASSES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car",
    "Cat", "Chair", "Cup", "Dog", "Motorbike", "People", "Table"
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


def scan_dir(path: Path, depth: int = 0, max_depth: int = 3, limit: int = 5):
    """Print tree of a directory."""
    if depth > max_depth:
        return
    try:
        entries = sorted(path.iterdir())
    except PermissionError:
        return
    shown = 0
    for entry in entries:
        if shown >= limit and depth > 0:
            print("  " * depth + f"... ({len(entries)-shown} more)")
            break
        icon = "📁" if entry.is_dir() else "📄"
        print("  " * depth + f"{icon} {entry.name}")
        if entry.is_dir():
            scan_dir(entry, depth + 1, max_depth, limit)
        shown += 1


def count_files(folder: Path, exts: set) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.rglob("*") if f.suffix in exts)


def check_annotation_format(txt_path: Path) -> str:
    """Return annotation format: 'exdark', 'yolo', 'empty', 'unknown'."""
    try:
        with open(txt_path, "r", errors="replace") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            return "empty"
        for line in lines:
            if line.startswith("%"):
                return "exdark"
        # Try YOLO
        parts = lines[0].split()
        if len(parts) == 5:
            try:
                int(parts[0])
                float(parts[1])
                return "yolo"
            except ValueError:
                pass
        return "unknown"
    except Exception:
        return "error"


def diagnose(root: Path):
    print("\n" + "="*65)
    print("  ExDark Dataset Diagnostics")
    print("="*65)

    if not root.exists():
        print(f"\n❌  ROOT NOT FOUND: {root}")
        print("   Check paths.dataset_root in configs/config.yaml")
        return

    print(f"\n📁  Root: {root.resolve()}\n")
    print("Directory tree (first 5 per level):")
    scan_dir(root, max_depth=2, limit=5)

    print("\n" + "-"*65)
    print("Class folder scan:")
    print("-"*65)

    # Check for images dir vs flat
    images_root = root / "images" if (root / "images").exists() else root

    ann_roots_to_check = [
        root / "Annotations",
        root / "annotations",
        root / "Groundtruth",
        root / "groundtruth",
        root / "labels",
        images_root,
        root,
    ]

    total_images = 0
    total_ann = 0
    class_stats = {}

    for cls in EXDARK_CLASSES:
        img_dir = images_root / cls
        n_images = count_files(img_dir, IMAGE_EXTS) if img_dir.exists() else 0

        # Find any annotation file for this class
        ann_found = False
        ann_format = "none"
        ann_location = None
        for ar in ann_roots_to_check:
            cls_ann_dir = ar / cls
            if not cls_ann_dir.is_dir():
                continue
            txts = list(cls_ann_dir.glob("*.txt"))
            if txts:
                ann_found = True
                ann_location = cls_ann_dir
                ann_format = check_annotation_format(txts[0])
                break

        status = "✅" if (n_images > 0 and ann_found) else \
                 "⚠️ " if (n_images > 0 and not ann_found) else \
                 "❌"

        class_stats[cls] = {
            "images": n_images, "has_ann": ann_found,
            "format": ann_format, "ann_dir": str(ann_location or "")
        }
        total_images += n_images
        if ann_found:
            total_ann += 1

        print(f"  {status} {cls:<12}: {n_images:4d} images  "
              f"ann={'YES ('+ann_format+')' if ann_found else 'MISSING'}")

    print("\n" + "-"*65)
    print(f"  Total images : {total_images}")
    print(f"  Classes with annotations: {total_ann}/12")

    # imageclasslist.txt
    icl = root / "imageclasslist.txt"
    if icl.exists():
        n_lines = sum(1 for _ in open(icl, errors="replace"))
        print(f"  imageclasslist.txt: FOUND ({n_lines} lines)")
    else:
        print(f"  imageclasslist.txt: not found")

    print("\n" + "="*65)

    if total_ann == 0:
        print("\n🔴  NO ANNOTATIONS FOUND\n")
        print("The pipeline will work but use pseudo full-image boxes.")
        print("mAP numbers will NOT be meaningful.")
        print()
        print("HOW TO FIX:")
        print("  Option 1 — Download official annotations (4 MB):")
        print("    1. Go to: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset")
        print("    2. Click the Groundtruth download link in the README")
        print("    3. Unzip. You should get folders: Bicycle/, Boat/, etc.")
        print("    4. Place them so the path is:")
        print("         data/ExDark/Annotations/Bicycle/img.jpg.txt")
        print()
        print("  Option 2 — Use Roboflow version (already converted, ~same images):")
        print("    pip install roboflow")
        print("    python tools/download_from_roboflow.py")
        print()
        print("  Option 3 — Run anyway (no annotations):")
        print("    The enhancement comparison IS still valid.")
        print("    mAP will be inflated but relative ranking between")
        print("    methods is preserved.")
    elif total_ann < 12:
        missing = [c for c, s in class_stats.items() if not s["has_ann"]]
        print(f"\n🟡  PARTIAL ANNOTATIONS: {12-total_ann} classes missing: {missing}")
        print("   Download Groundtruth.zip from ExDark GitHub to fix.")
    else:
        print("\n✅  All annotations found. You're good to go!")
        # Check annotation format consistency
        formats = set(s["format"] for s in class_stats.values())
        if len(formats) == 1:
            print(f"   Annotation format: {formats.pop()}")
        else:
            print(f"   ⚠️  Mixed annotation formats detected: {formats}")
            print("      This is OK — the loader handles both.")

    print("="*65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ExDark",
                        help="Path to ExDark root directory")
    args = parser.parse_args()
    diagnose(Path(args.root))
