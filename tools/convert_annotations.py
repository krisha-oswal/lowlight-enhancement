"""
Annotation Format Converter
=============================
Converts between annotation formats:
  - ExDark official (% lines) ↔ YOLO normalized xywh
  - Pascal VOC XML → ExDark
  - Fixes common issues: wrong extensions, missing folders, etc.

Usage:
  # Convert YOLO labels → ExDark format
  python tools/convert_annotations.py --from yolo --to exdark \
      --images data/ExDark/images --labels data/ExDark/labels \
      --output data/ExDark/Annotations

  # Check what you have and auto-fix paths
  python tools/convert_annotations.py --check-only --root data/ExDark
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

EXDARK_CLASSES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car",
    "Cat", "Chair", "Cup", "Dog", "Motorbike", "People", "Table"
]


def yolo_to_exdark(
    yolo_txt: Path, img_w: int, img_h: int, class_name: str
) -> List[str]:
    """Convert YOLO .txt to ExDark annotation lines."""
    lines = []
    try:
        with open(yolo_txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = [float(x) for x in parts[1:5]]

                # Denormalize
                l = int((cx - bw / 2) * img_w)
                t = int((cy - bh / 2) * img_h)
                w = int(bw * img_w)
                h = int(bh * img_h)

                # Clamp
                l = max(0, l)
                t = max(0, t)
                w = max(1, min(w, img_w - l))
                h = max(1, min(h, img_h - t))

                cls = EXDARK_CLASSES[cls_id] if 0 <= cls_id < len(EXDARK_CLASSES) else class_name
                lines.append(f"% {cls} {l} {t} {w} {h} 0 0 0")
    except Exception as e:
        logger.warning(f"Error converting {yolo_txt}: {e}")
    return lines


def exdark_to_yolo(
    exdark_txt: Path, img_w: int, img_h: int
) -> List[str]:
    """Convert ExDark .txt to YOLO normalized format."""
    lines = []
    try:
        with open(exdark_txt, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("%"):
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                cls_name = parts[1]
                if cls_name not in EXDARK_CLASSES:
                    continue
                cls_id = EXDARK_CLASSES.index(cls_name)
                try:
                    l, t, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                except ValueError:
                    continue

                cx = (l + w / 2) / img_w
                cy = (t + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    except Exception as e:
        logger.warning(f"Error converting {exdark_txt}: {e}")
    return lines


def convert_yolo_to_exdark(
    images_dir: str, labels_dir: str, output_dir: str
):
    """Batch convert YOLO labels directory → ExDark Annotations directory."""
    import cv2
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    n_converted = n_skipped = 0

    for cls in EXDARK_CLASSES:
        img_cls_dir = images_dir / cls
        lbl_cls_dir = labels_dir / cls
        out_cls_dir = output_dir / cls

        if not img_cls_dir.exists():
            continue
        out_cls_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(img_cls_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            # Find corresponding label
            lbl_candidates = [
                lbl_cls_dir / f"{img_path.stem}.txt",
                lbl_cls_dir / f"{img_path.name}.txt",
            ]
            lbl_path = next((p for p in lbl_candidates if p.exists()), None)

            if lbl_path is None:
                n_skipped += 1
                continue

            # Read image dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                n_skipped += 1
                continue
            h, w = img.shape[:2]

            # Convert
            exdark_lines = yolo_to_exdark(lbl_path, w, h, cls)
            if not exdark_lines:
                n_skipped += 1
                continue

            # Write to ExDark format: img.jpg.txt
            out_path = out_cls_dir / f"{img_path.name}.txt"
            header = f"% {img_path.name} - ExDark converted from YOLO\n"
            with open(out_path, "w") as f:
                f.write(header)
                f.write("\n".join(exdark_lines) + "\n")

            n_converted += 1

        logger.info(f"  {cls}: {n_converted} converted")

    logger.info(f"\nTotal: {n_converted} converted, {n_skipped} skipped")
    logger.info(f"Output: {output_dir}")


def check_and_report(root: str):
    """Quick sanity check — show what annotation files look like."""
    root = Path(root)
    logger.info(f"Scanning: {root}\n")

    for cls in EXDARK_CLASSES:
        for possible_ann_dir in [
            root / "Annotations" / cls,
            root / "annotations" / cls,
            root / "Groundtruth" / cls,
            root / cls,
            root / "images" / cls,
        ]:
            txts = list(possible_ann_dir.glob("*.txt"))
            if txts:
                sample = txts[0]
                with open(sample, errors="replace") as f:
                    first_lines = [l.strip() for l in f][:4]
                logger.info(f"{cls}: found {len(txts)} .txt files in {possible_ann_dir}")
                logger.info(f"  Sample ({sample.name}):")
                for line in first_lines:
                    logger.info(f"    {line}")
                break
        else:
            logger.info(f"{cls}: *** NO ANNOTATION FILES FOUND ***")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_fmt", choices=["yolo", "exdark"],
                        default="yolo")
    parser.add_argument("--to", dest="to_fmt", choices=["yolo", "exdark"],
                        default="exdark")
    parser.add_argument("--images", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--output", default="data/ExDark/Annotations")
    parser.add_argument("--root", default="data/ExDark")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        check_and_report(args.root)
    elif args.from_fmt == "yolo" and args.to_fmt == "exdark":
        if not args.images or not args.labels:
            print("--images and --labels required for YOLO→ExDark conversion")
            sys.exit(1)
        convert_yolo_to_exdark(args.images, args.labels, args.output)
    else:
        print("Currently supported: --from yolo --to exdark")
        print("Use --check-only to inspect your current annotations.")
