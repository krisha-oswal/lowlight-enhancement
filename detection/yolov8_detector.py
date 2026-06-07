"""
YOLOv8 Detection Pipeline
==========================
Wraps Ultralytics YOLOv8 for consistent inference across all
enhancement method variants.

Design decisions:
  - Single model loaded once, reused across all images/methods
  - Results stored as dicts for serialization
  - ExDark class names mapped to COCO IDs for YOLOv8 (pretrained on COCO)
  - Confidence and IoU thresholds fixed across all conditions
"""

import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

import numpy as np
import cv2

logger = logging.getLogger(__name__)


class YOLOv8Detector:
    """
    YOLOv8 object detector wrapper.

    Loads a single model instance and provides:
      - detect(): run inference on a single image
      - detect_batch(): run inference on a list of images

    Args:
        model_name: YOLOv8 variant (yolov8n/s/m/l/x.pt)
        conf_threshold: Minimum confidence score
        iou_threshold: NMS IoU threshold
        image_size: Input image size for YOLOv8
        device: 'cuda', 'cpu', or 'mps'
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model. Downloads pretrained weights if needed."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            # Move to device
            if self.device != "cpu":
                self.model.to(self.device)
            logger.info(f"YOLOv8 model loaded: {self.model_name} on {self.device}")
        except ImportError:
            raise ImportError(
                "ultralytics package not found. Install with: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv8 model: {e}")

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLOv8 on a single BGR image.

        Args:
            image: BGR uint8 numpy array

        Returns:
            List of detection dicts:
              {
                'bbox': [x1, y1, x2, y2],  # absolute pixel coords
                'confidence': float,
                'class_id': int,            # COCO class ID
                'class_name': str,
              }
        """
        if self.model is None:
            return []

        try:
            results = self.model.predict(
                source=image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                verbose=False,
                device=self.device,
            )

            detections = []
            for result in results:
                if result.boxes is None:
                    continue
                boxes = result.boxes
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = self.model.names.get(cls_id, str(cls_id))

                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": round(conf, 4),
                        "class_id": cls_id,
                        "class_name": cls_name,
                    })

            return detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    def detect_batch(
        self, images: List[np.ndarray]
    ) -> List[List[Dict[str, Any]]]:
        """Run detection on a list of images."""
        return [self.detect(img) for img in images]

    @property
    def coco_class_names(self) -> Dict[int, str]:
        """Return COCO class id → name mapping."""
        if self.model is not None:
            return self.model.names
        return {}


# ─────────────────────────────────────────────
# ExDark ↔ COCO class mapping utilities
# ─────────────────────────────────────────────

# ExDark class → closest COCO class ID (for YOLOv8 COCO pretrained)
EXDARK_TO_COCO = {
    "Bicycle":   1,   # bicycle
    "Boat":      8,   # boat
    "Bottle":    39,  # bottle
    "Bus":       5,   # bus
    "Car":       2,   # car
    "Cat":       15,  # cat
    "Chair":     56,  # chair
    "Cup":       41,  # cup
    "Dog":       16,  # dog
    "Motorbike": 3,   # motorcycle
    "People":    0,   # person
    "Table":     60,  # dining table
}

COCO_TO_EXDARK = {v: k for k, v in EXDARK_TO_COCO.items()}


def filter_detections_by_class(
    detections: List[Dict],
    allowed_coco_ids: List[int],
) -> List[Dict]:
    """Keep only detections whose COCO class is in allowed_coco_ids."""
    return [d for d in detections if d["class_id"] in allowed_coco_ids]


def map_coco_to_exdark(detections: List[Dict]) -> List[Dict]:
    """Add 'exdark_class' field to each detection if mapping exists."""
    mapped = []
    for d in detections:
        d = d.copy()
        d["exdark_class"] = COCO_TO_EXDARK.get(d["class_id"], None)
        mapped.append(d)
    return mapped
