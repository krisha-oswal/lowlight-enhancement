"""
ExDark Dataset Loader — Robust Multi-Layout Version
=====================================================
Handles every common way people actually have ExDark on disk:

LAYOUT A — Official separate download (most common failure case):
  data/ExDark/images/Bicycle/2015_00001.jpg
  data/ExDark/Annotations/Bicycle/2015_00001.jpg.txt   ← note: .jpg.txt

LAYOUT B — Annotations inside images folder:
  data/ExDark/Bicycle/2015_00001.jpg
  data/ExDark/Bicycle/2015_00001.jpg.txt

LAYOUT C — imageclasslist.txt only (no per-image .txt files):
  → image-level label used as pseudo-annotation

LAYOUT D — YOLO format (.txt with normalized xywh, no % prefix):
  data/ExDark/images/Bicycle/img.jpg
  data/ExDark/labels/Bicycle/img.txt

LAYOUT E — No annotations at all (images only):
  → Falls back to class-folder name as pseudo full-image box.
    Detection still runs; mAP will be approximate but pipeline
    won't crash and enhancement comparison remains valid.

ExDark official .txt format (lines starting with %):
  % ClassName l t w h light_env reflect highlight
"""

import os
import glob
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ── ExDark class registry ──────────────────────────────────────────
EXDARK_CLASSES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car",
    "Cat", "Chair", "Cup", "Dog", "Motorbike", "People", "Table"
]
EXDARK_CLASS_TO_ID = {c: i for i, c in enumerate(EXDARK_CLASSES)}

_CLASS_ALIASES = {
    "bicycle": "Bicycle", "boat": "Boat", "bottle": "Bottle",
    "bus": "Bus", "car": "Car", "cat": "Cat", "chair": "Chair",
    "cup": "Cup", "dog": "Dog", "motorbike": "Motorbike",
    "motorcycle": "Motorbike",
    "person": "People", "people": "People", "human": "People",
    "table": "Table", "diningtable": "Table",
}


def _normalize_class(name: str) -> str:
    if name in EXDARK_CLASS_TO_ID:
        return name
    return _CLASS_ALIASES.get(name.lower(), name)


# ── Data classes ───────────────────────────────────────────────────
@dataclass
class BoundingBox:
    class_name: str
    class_id: int
    x1: int; y1: int; x2: int; y2: int
    light_env: int = 0

    @property
    def width(self):  return self.x2 - self.x1
    @property
    def height(self): return self.y2 - self.y1
    def to_xyxy(self): return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class ImageSample:
    image_id: str
    image_path: str
    class_folder: str
    annotations: List[BoundingBox] = field(default_factory=list)
    width: int = 0
    height: int = 0
    has_bbox: bool = True

    @property
    def num_objects(self): return len(self.annotations)

    def load_image(self) -> np.ndarray:
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load: {self.image_path}")
        return img

    def get_gt_boxes(self) -> List[Dict]:
        return [
            {"class_name": a.class_name, "class_id": a.class_id,
             "bbox": a.to_xyxy(), "light_env": a.light_env}
            for a in self.annotations
        ]


# ── Layout detector ────────────────────────────────────────────────
class LayoutDetector:
    def __init__(self, root: Path):
        self.root = root

    def detect(self):
        root = self.root
        # Determine images root
        images_dir = root / "images"
        if not images_dir.exists() or not self._has_class_dirs(images_dir):
            images_dir = root

        # Find annotation directory
        ann_candidates = [
            root / "Annotations", root / "annotations",
            root / "Groundtruth", root / "groundtruth",
            root / "labels", root / "Labels",
            images_dir,   # co-located
            root,
        ]
        ann_dir = None
        for c in ann_candidates:
            if c.exists() and self._has_ann_files(c):
                ann_dir = c
                break

        layout = (
            "no_annotations" if ann_dir is None else
            "D_yolo" if ann_dir.name.lower() in ("labels",) else
            "B_colocated" if ann_dir == images_dir else
            "A_separate"
        )
        logger.info(f"Layout '{layout}': images={images_dir}, annotations={ann_dir or 'NONE'}")
        return images_dir, ann_dir, layout

    def _has_class_dirs(self, base: Path) -> bool:
        return any((base / c).is_dir() for c in EXDARK_CLASSES)

    def _has_ann_files(self, path: Path) -> bool:
        for cls in EXDARK_CLASSES:
            if list((path / cls).glob("*.txt"))[:1]:
                return True
        return bool(list(path.glob("*.txt"))[:1])


# ── Parsers ────────────────────────────────────────────────────────





def _parse_exdark_txt(path: Path, default_class: str) -> List[BoundingBox]:
    boxes = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()

                # skip empty lines and header
                if not line or line.startswith("%"):
                    continue

                parts = line.split()

                if len(parts) < 5:
                    continue

                cls = _normalize_class(parts[0])
                if cls not in EXDARK_CLASS_TO_ID:
                    cls = _normalize_class(default_class)

                try:
                    l = int(float(parts[1]))
                    t = int(float(parts[2]))
                    w = int(float(parts[3]))
                    h = int(float(parts[4]))
                    light_env = int(float(parts[5])) if len(parts) > 5 else 0
                except:
                    continue

                x1 = max(0, l)
                y1 = max(0, t)
                x2 = x1 + max(1, w)
                y2 = y1 + max(1, h)

                boxes.append(BoundingBox(
                    class_name=cls,
                    class_id=EXDARK_CLASS_TO_ID.get(cls, 0),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    light_env=light_env,
                ))
    except Exception as e:
        logger.debug(f"ExDark parse error {path}: {e}")

    return boxes
def _parse_yolo_txt(path: Path, default_class: str, img_w: int, img_h: int) -> List[BoundingBox]:
    boxes = []
    try:
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = max(0, int((cx - bw / 2) * img_w))
                y1 = max(0, int((cy - bh / 2) * img_h))
                x2 = min(img_w, int((cx + bw / 2) * img_w))
                y2 = min(img_h, int((cy + bh / 2) * img_h))
                cls_name = EXDARK_CLASSES[cls_id] if 0 <= cls_id < len(EXDARK_CLASSES) \
                           else _normalize_class(default_class)
                boxes.append(BoundingBox(
                    class_name=cls_name, class_id=EXDARK_CLASS_TO_ID.get(cls_name, 0),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))
    except Exception as e:
        logger.debug(f"YOLO parse error {path}: {e}")
    return boxes


def _pseudo_annotation(class_name: str, img_w: int, img_h: int) -> List[BoundingBox]:
    """Full-image pseudo-box when no annotation file exists."""
    cls = _normalize_class(class_name)
    return [BoundingBox(
        class_name=cls, class_id=EXDARK_CLASS_TO_ID.get(cls, 0),
        x1=0, y1=0, x2=max(1, img_w - 1), y2=max(1, img_h - 1),
    )]


# ── Main dataset class ─────────────────────────────────────────────
class ExDarkDataset:
    """
    Robust ExDark loader. Auto-detects layout, handles all annotation
    formats, falls back gracefully when annotations are missing.
    """

    def __init__(self, config: dict):
        self.config = config
        self.root = Path(config["paths"]["dataset_root"])
        self.classes = config["dataset"]["classes"]
        self.max_images = config["dataset"].get("max_images", None)
        self.extensions = config["dataset"].get(
            "image_extensions", [".jpg", ".jpeg", ".png", ".bmp"])
        self.samples: List[ImageSample] = []
        self._load()

    def _load(self):
        if not self.root.exists():
            raise FileNotFoundError(
                f"\n\nDataset root not found: {self.root}\n"
                "Set paths.dataset_root in configs/config.yaml to your ExDark folder.\n"
            )

        detector = LayoutDetector(self.root)
        self.images_dir, self.ann_dir, self.layout = detector.detect()

        n_real = n_pseudo = 0

        for class_name in self.classes:
            class_img_dir = self.images_dir / class_name
            if not class_img_dir.is_dir():
                logger.warning(f"Class folder not found: {class_img_dir}")
                continue

            img_files = []
            for ext in self.extensions:
                img_files += glob.glob(str(class_img_dir / f"*{ext}"))
                img_files += glob.glob(str(class_img_dir / f"*{ext.upper()}"))
            img_files = sorted(set(img_files))

            for img_path_str in img_files:
                img_path = Path(img_path_str)
                img = cv2.imread(img_path_str)
                if img is None:
                    continue
                h, w = img.shape[:2]

                boxes, has_bbox = self._find_annotations(img_path, class_name, w, h)

                if has_bbox:
                    n_real += 1
                else:
                    n_pseudo += 1

                self.samples.append(ImageSample(
                    image_id=f"{class_name}/{img_path.stem}",
                    image_path=img_path_str,
                    class_folder=class_name,
                    annotations=boxes,
                    width=w, height=h,
                    has_bbox=has_bbox,
                ))

            logger.info(f"  {class_name}: {len(img_files)} images")

        logger.info(
            f"Total: {len(self.samples)} images | "
            f"Real bbox: {n_real} | Pseudo-bbox: {n_pseudo}"
        )

        if n_real == 0:
            logger.warning(
                "\n" + "="*60 +
                "\n⚠️  NO BOUNDING BOX ANNOTATIONS FOUND\n"
                "The pipeline will run using pseudo full-image boxes.\n"
                "Enhancement comparison is STILL VALID and meaningful.\n"
                "mAP scores will be inflated/approximate.\n\n"
                "To get proper mAP, download the Groundtruth zip:\n"
                "  https://github.com/cs-chan/Exclusively-Dark-Image-Dataset\n"
                "Then extract so annotations are at:\n"
                "  data/ExDark/Annotations/<ClassName>/<img>.jpg.txt\n"
                "="*60
            )
        elif n_pseudo > 0:
            logger.warning(
                f"{n_pseudo}/{len(self.samples)} images are missing annotations "
                "(pseudo full-image boxes used for those)."
            )

        if self.max_images:
            self.samples = self._stratified_subsample(self.max_images)

    def _find_annotations(
        self, img_path: Path, class_name: str, w: int, h: int
    ) -> Tuple[List[BoundingBox], bool]:
        stem = img_path.stem
        fname = img_path.name

        candidates = []
        if self.ann_dir:
            acd = self.ann_dir / class_name
            candidates += [
                acd / f"{fname}.txt",      # img.jpg.txt  ← official format
                acd / f"{stem}.txt",       # img.txt
                acd / f"{stem}.jpg.txt",   # sometimes doubled extension
            ]
        # Co-located
        candidates += [
            img_path.parent / f"{fname}.txt",
            img_path.parent / f"{stem}.txt",
        ]

        for cand in candidates:
            if not Path(cand).exists():
                continue
            # Detect format
            first_line = ""
            try:
                with open(cand, "r", errors="replace") as f:
                    for line in f:
                        s = line.strip()
                        if s:
                            first_line = s
                            break
            except Exception:
                continue

            if first_line.startswith("%"):
                boxes = _parse_exdark_txt(Path(cand), class_name)
            else:
                try:
                    int(first_line.split()[0])
                    boxes = _parse_yolo_txt(Path(cand), class_name, w, h)
                except (ValueError, IndexError):
                    boxes = _parse_exdark_txt(Path(cand), class_name)

            if boxes:
                return boxes, True

        return _pseudo_annotation(class_name, w, h), False

    def _stratified_subsample(self, n: int) -> List[ImageSample]:
        import random
        random.seed(42)
        by_class = defaultdict(list)
        for s in self.samples:
            by_class[s.class_folder].append(s)
        per_class = max(1, n // len(by_class))
        result = []
        for samps in by_class.values():
            result.extend(random.sample(samps, min(per_class, len(samps))))
        return result[:n]

    def __len__(self):           return len(self.samples)
    def __getitem__(self, i):    return self.samples[i]

    def get_class_distribution(self) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(s.class_folder for s in self.samples))

    def get_annotation_coverage(self) -> Dict[str, int]:
        real = sum(1 for s in self.samples if s.has_bbox)
        pseudo = len(self.samples) - real
        return {"real_bbox": real, "pseudo_bbox": pseudo, "total": len(self.samples)}

    def print_layout_summary(self):
        cov = self.get_annotation_coverage()
        print("\n" + "="*58)
        print("ExDark Dataset Summary")
        print("="*58)
        print(f"  Root dir  : {self.root}")
        print(f"  Layout    : {self.layout}")
        print(f"  Images    : {self.images_dir}")
        print(f"  Annotations: {self.ann_dir or '*** NOT FOUND ***'}")
        print(f"  Total imgs : {len(self.samples)}")
        print(f"  Real bbox  : {cov['real_bbox']}")
        print(f"  Pseudo-box : {cov['pseudo_bbox']}")
        if cov['pseudo_bbox'] > 0:
            print()
            print("  To fix: download Groundtruth.zip from ExDark GitHub")
            print("  Extract to: data/ExDark/Annotations/<Class>/<img>.jpg.txt")
        print("="*58 + "\n")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    import sys, logging
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml")
    ds = ExDarkDataset(cfg)
    ds.print_layout_summary()
