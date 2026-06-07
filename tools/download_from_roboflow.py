"""
Download ExDark from Roboflow (pre-annotated, YOLO format)
============================================================
Roboflow hosts a converted version of ExDark with YOLO-format
annotations. This is the easiest way to get the dataset with
annotations if the official Groundtruth.zip link is broken.

Usage:
  pip install roboflow
  python tools/download_from_roboflow.py --dest data/ExDark_roboflow

Then update configs/config.yaml:
  paths:
    dataset_root: "data/ExDark_roboflow"
"""

import argparse
import sys
import shutil
from pathlib import Path


def download_roboflow(dest: str, api_key: str = None):
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Install roboflow first:  pip install roboflow")
        sys.exit(1)

    print(f"Downloading ExDark dataset to: {dest}")
    print("This may take a few minutes (~1.5 GB)...\n")

    # Public ExDark project on Roboflow Universe
    # No API key needed for public datasets
    rf = Roboflow(api_key=api_key or "")
    project = rf.workspace("project-h68de").project("exdark-kd37x")
    dataset = project.version(1).download("yolov8", location=dest)
    print(f"\nDownloaded to: {dataset.location}")
    print("\nNow update configs/config.yaml:")
    print(f"  paths:")
    print(f"    dataset_root: \"{dest}\"")
    print("\nAnd also set:")
    print("  dataset:")
    print("    annotation_format: yolo   # (already handled automatically)")


def reorganize_to_class_folders(roboflow_root: str, out_root: str):
    """
    Roboflow downloads as train/valid/test splits with flat image dirs.
    This script reorganizes into ExDark class-folder layout so the
    main loader works without changes.

    Roboflow YOLO format:
      <root>/train/images/Bicycle_img001.jpg
      <root>/train/labels/Bicycle_img001.txt
    """
    src = Path(roboflow_root)
    dst = Path(out_root)
    dst.mkdir(parents=True, exist_ok=True)

    EXDARK_CLASSES = [
        "Bicycle", "Boat", "Bottle", "Bus", "Car",
        "Cat", "Chair", "Cup", "Dog", "Motorbike", "People", "Table"
    ]

    moved = 0
    for split in ["train", "valid", "test"]:
        img_dir = src / split / "images"
        lbl_dir = src / split / "labels"
        if not img_dir.exists():
            continue

        for img_path in img_dir.glob("*"):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            # Try to infer class from filename prefix
            cls_name = None
            for cls in EXDARK_CLASSES:
                if img_path.stem.startswith(cls) or img_path.stem.startswith(cls.lower()):
                    cls_name = cls
                    break
            if cls_name is None:
                cls_name = "People"  # fallback

            # Create dest dirs
            (dst / "images" / cls_name).mkdir(parents=True, exist_ok=True)
            (dst / "labels" / cls_name).mkdir(parents=True, exist_ok=True)

            # Copy image
            shutil.copy2(img_path, dst / "images" / cls_name / img_path.name)

            # Copy label
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst / "labels" / cls_name / lbl_path.name)

            moved += 1

    print(f"Reorganized {moved} images into class folders at {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data/ExDark_roboflow")
    parser.add_argument("--api-key", default=None,
                        help="Roboflow API key (optional for public datasets)")
    parser.add_argument("--reorganize-only", action="store_true",
                        help="Only reorganize already-downloaded data")
    parser.add_argument("--src", default=None,
                        help="Source for --reorganize-only")
    parser.add_argument("--out", default="data/ExDark",
                        help="Output for --reorganize-only")
    args = parser.parse_args()

    if args.reorganize_only:
        if not args.src:
            print("--src required with --reorganize-only")
            sys.exit(1)
        reorganize_to_class_folders(args.src, args.out)
    else:
        download_roboflow(args.dest, args.api_key)
