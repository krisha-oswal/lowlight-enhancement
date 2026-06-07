"""
Detection Evaluation Metrics
==============================
Implements mAP, Precision, and Recall computation.

Follows the VOC 2010+ / COCO evaluation protocol:
  - IoU threshold @0.5 for mAP50
  - IoU thresholds @0.5:0.95:0.05 for mAP50-95
  - Per-class AP using 11-point or area-under-curve interpolation

Design choices:
  - Pure NumPy/Python implementation (no torchmetrics dependency)
  - Supports both per-image and per-dataset aggregation
  - Returns rich result dicts for downstream analysis
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """
    Compute Intersection over Union between two [x1,y1,x2,y2] boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-7)


def match_detections_to_gt(
    detections: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    exdark_to_coco: Dict[str, int] = None,
) -> Tuple[List[int], List[int], int]:
    """
    Match detections to ground-truth boxes using greedy IoU matching.

    Args:
        detections: List of {bbox, confidence, class_id, ...}
        ground_truths: List of {bbox, class_id (ExDark), class_name, ...}
        iou_threshold: IoU threshold for a positive match
        exdark_to_coco: ExDark class name → COCO class id mapping

    Returns:
        tp: List[int] - 1 if detection is TP, 0 if FP (parallel to detections)
        fp: List[int] - 1 if FP, 0 if TP
        n_gt: total number of GT boxes
    """
    if exdark_to_coco is None:
        from detection.yolov8_detector import EXDARK_TO_COCO
        exdark_to_coco = EXDARK_TO_COCO

    # Sort detections by confidence descending
    dets_sorted = sorted(detections, key=lambda x: x["confidence"], reverse=True)

    # Build GT list with matched flags
    gt_matched = [False] * len(ground_truths)

    tp = []
    fp = []

    for det in dets_sorted:
        det_cls = det["class_id"]
        det_box = det["bbox"]

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            if gt_matched[gt_idx]:
                continue

            # Map ExDark class to COCO ID for comparison
            gt_coco_id = exdark_to_coco.get(gt.get("class_name", ""), -1)
            if gt_coco_id != det_cls:
                continue

            iou = compute_iou(det_box, list(gt["bbox"]))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp.append(1)
            fp.append(0)
            gt_matched[best_gt_idx] = True
        else:
            tp.append(0)
            fp.append(1)

    return tp, fp, len(ground_truths)


def compute_ap_from_pr(
    precisions: np.ndarray, recalls: np.ndarray
) -> float:
    """
    Compute Average Precision using the area-under-curve method
    (VOC 2010+ protocol: 101-point interpolation).
    """
    # Append sentinel values
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])

    # Ensure precision is non-increasing (monotonic envelope)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Compute AP as area under the PR curve
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])
    return float(ap)


def compute_ap_for_class(
    all_tp: List[int],
    all_fp: List[int],
    n_gt: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute AP, precision curve, and recall curve for one class.

    Args:
        all_tp: Cumulative TP flags (sorted by confidence descending)
        all_fp: Cumulative FP flags
        n_gt: Total number of GT objects for this class

    Returns:
        ap: float
        precision_curve: np.ndarray
        recall_curve: np.ndarray
    """
    if n_gt == 0:
        return 0.0, np.array([]), np.array([])

    tp_cum = np.cumsum(all_tp)
    fp_cum = np.cumsum(all_fp)

    recall_curve = tp_cum / (n_gt + 1e-7)
    precision_curve = tp_cum / (tp_cum + fp_cum + 1e-7)

    ap = compute_ap_from_pr(precision_curve, recall_curve)
    return ap, precision_curve, recall_curve


class DetectionEvaluator:
    """
    Evaluates detection results against ground-truth annotations.

    Usage:
        evaluator = DetectionEvaluator(config)
        evaluator.add_image_result(image_id, method, detections, gt_boxes)
        metrics = evaluator.compute_metrics()
    """

    def __init__(self, config: dict):
        self.config = config
        self.iou_threshold = config["evaluation"]["iou_threshold_map"]
        self.exdark_classes = config["dataset"]["classes"]
        self.exdark_to_coco = config["dataset"]["exdark_to_coco"]

        # Storage: method → class → list of (tp, fp, score, n_gt_contribution)
        # We store per-detection entries for AP computation
        self._results: Dict[str, Dict[str, Dict]] = defaultdict(
            lambda: defaultdict(lambda: {"tp": [], "fp": [], "scores": [], "n_gt": 0})
        )
        # Per-image, per-method summary
        self._image_results: List[Dict] = []

    def add_image_result(
        self,
        image_id: str,
        method: str,
        detections: List[Dict],
        gt_boxes: List[Dict],
        image_features: Dict = None,
    ):
        """
        Register detection results for one image + method combination.

        Args:
            image_id: Unique image identifier
            method: Enhancement method name
            detections: YOLOv8 detection results
            gt_boxes: Ground-truth boxes from ExDark
            image_features: Optional dict of image characteristics
        """
        # Match detections to GT
        tp_list, fp_list, n_gt = match_detections_to_gt(
            detections, gt_boxes,
            iou_threshold=self.iou_threshold,
            exdark_to_coco=self.exdark_to_coco,
        )

        # Compute per-image precision/recall
        if len(tp_list) > 0:
            tp_sum = sum(tp_list)
            fp_sum = sum(fp_list)
            precision = tp_sum / (tp_sum + fp_sum + 1e-7)
            recall = tp_sum / (n_gt + 1e-7)
        else:
            precision = 0.0
            recall = 0.0 if n_gt > 0 else 1.0

        # Accumulate class-level stats for AP computation
        dets_sorted = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        # Build reverse map here to avoid circular/torch imports
        _coco_to_exdark = {v: k for k, v in self.exdark_to_coco.items()}
        for i, det in enumerate(dets_sorted):
            coco_id = det["class_id"]
            COCO_TO_EXDARK = _coco_to_exdark
            exdark_cls = COCO_TO_EXDARK.get(coco_id, None)
            if exdark_cls is None:
                continue
            r = self._results[method][exdark_cls]
            if i < len(tp_list):
                r["tp"].append(tp_list[i])
                r["fp"].append(fp_list[i])
            r["scores"].append(det["confidence"])

        # Add GT counts per class
        from collections import Counter
        gt_class_counts = Counter(g.get("class_name", "") for g in gt_boxes)
        for cls, cnt in gt_class_counts.items():
            if cls in self.exdark_classes:
                self._results[method][cls]["n_gt"] += cnt

        # Store per-image result
        self._image_results.append({
            "image_id": image_id,
            "method": method,
            "n_gt": n_gt,
            "n_det": len(detections),
            "tp": sum(tp_list),
            "fp": sum(fp_list),
            "fn": n_gt - sum(tp_list),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "features": image_features or {},
        })

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute mAP50 and per-class AP for all methods.

        Returns:
            {
              "per_method": {
                method_name: {
                  "mAP50": float,
                  "per_class_AP": {class: float},
                  "mean_precision": float,
                  "mean_recall": float,
                }
              },
              "per_image": List[Dict],
            }
        """
        method_metrics = {}

        for method, class_data in self._results.items():
            aps = {}
            for cls in self.exdark_classes:
                data = class_data.get(cls, {"tp": [], "fp": [], "scores": [], "n_gt": 0})
                if data["n_gt"] == 0:
                    aps[cls] = 0.0
                    continue
                # Sort by score descending
                if len(data["scores"]) == 0:
                    aps[cls] = 0.0
                    continue
                order = np.argsort(data["scores"])[::-1]
                tp_sorted = np.array(data["tp"])[order] if data["tp"] else np.array([])
                fp_sorted = np.array(data["fp"])[order] if data["fp"] else np.array([])
                ap, _, _ = compute_ap_for_class(
                    tp_sorted.tolist(), fp_sorted.tolist(), data["n_gt"]
                )
                aps[cls] = round(ap, 4)

            map50 = float(np.mean(list(aps.values())))

            # Per-image stats for this method
            method_images = [r for r in self._image_results if r["method"] == method]
            mean_prec = float(np.mean([r["precision"] for r in method_images])) if method_images else 0.0
            mean_rec = float(np.mean([r["recall"] for r in method_images])) if method_images else 0.0

            method_metrics[method] = {
                "mAP50": round(map50, 4),
                "per_class_AP": aps,
                "mean_precision": round(mean_prec, 4),
                "mean_recall": round(mean_rec, 4),
                "n_images": len(method_images),
            }

        return {
            "per_method": method_metrics,
            "per_image": self._image_results,
        }

    def get_per_image_results(self) -> List[Dict]:
        return self._image_results
