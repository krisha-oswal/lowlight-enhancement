"""
I/O Utilities
==============
Handles result caching, serialization, and efficient recomputation avoidance.

Key design principles:
  - Enhanced images cached as .jpg to disk (avoid recomputing)
  - Detection results cached as .json per image per method
  - Final aggregated results saved as .json and .csv
  - All paths derived from config for reproducibility
"""

import os
import json
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import cv2
import pandas as pd

logger = logging.getLogger(__name__)


def ensure_dirs(config: dict):
    """Create all output directories from config."""
    paths = config.get("paths", {})
    for key, path in paths.items():
        if path and not path.endswith((".pt", ".pth", ".yaml")):
            Path(path).mkdir(parents=True, exist_ok=True)
    logger.info("Output directories created.")


def get_cache_path(
    cache_dir: str, image_id: str, method: str, ext: str = ".jpg"
) -> str:
    """Build a deterministic cache path for an enhanced image."""
    # Replace path separators in image_id
    safe_id = image_id.replace("/", "__").replace("\\", "__")
    return str(Path(cache_dir) / method / f"{safe_id}{ext}")


def save_enhanced_image(
    image: np.ndarray, cache_dir: str, image_id: str, method: str
) -> str:
    """Save enhanced image to cache. Returns path."""
    path = get_cache_path(cache_dir, image_id, method)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def load_enhanced_image(
    cache_dir: str, image_id: str, method: str
) -> Optional[np.ndarray]:
    """Load enhanced image from cache. Returns None if not cached."""
    path = get_cache_path(cache_dir, image_id, method)
    if not Path(path).exists():
        return None
    img = cv2.imread(path)
    return img


def get_detection_cache_path(
    det_dir: str, image_id: str, method: str
) -> str:
    """Path for cached detection results (.json)."""
    safe_id = image_id.replace("/", "__").replace("\\", "__")
    return str(Path(det_dir) / method / f"{safe_id}.json")


def save_detections(
    detections: List[Dict],
    det_dir: str,
    image_id: str,
    method: str,
):
    """Cache detection results as JSON."""
    path = get_detection_cache_path(det_dir, image_id, method)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(detections, f)


def load_detections(
    det_dir: str, image_id: str, method: str
) -> Optional[List[Dict]]:
    """Load cached detection results. Returns None if not cached."""
    path = get_detection_cache_path(det_dir, image_id, method)
    if not Path(path).exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def save_results(results: Dict, output_path: str):
    """Save result dict as JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializer)
    logger.info(f"Results saved to {output_path}")


def load_results(input_path: str) -> Optional[Dict]:
    """Load results from JSON."""
    if not Path(input_path).exists():
        return None
    with open(input_path, "r") as f:
        return json.load(f)


def save_dataframe(df: pd.DataFrame, output_path: str):
    """Save DataFrame as CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"DataFrame saved to {output_path}")


def load_dataframe(input_path: str) -> Optional[pd.DataFrame]:
    """Load DataFrame from CSV."""
    if not Path(input_path).exists():
        return None
    return pd.read_csv(input_path)


def save_model(model: Any, output_path: str):
    """Save sklearn/ML model with pickle."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {output_path}")


def load_model(input_path: str) -> Optional[Any]:
    """Load pickled model."""
    if not Path(input_path).exists():
        return None
    with open(input_path, "rb") as f:
        return pickle.load(f)


def per_image_results_to_dataframe(per_image_results: List[Dict]) -> pd.DataFrame:
    """
    Convert list of per-image result dicts to a flat DataFrame.
    Flattens nested 'features' and 'quality_metrics' dicts.
    """
    rows = []
    for r in per_image_results:
        row = {k: v for k, v in r.items() if k not in ("features",)}
        # Flatten features
        for feat_key, feat_val in r.get("features", {}).items():
            row[f"feat_{feat_key}"] = feat_val
        # Flatten quality metrics
        for qk, qv in r.get("quality_metrics", {}).items():
            row[qk] = qv
        rows.append(row)
    return pd.DataFrame(rows)


def _json_serializer(obj):
    """Handle non-serializable types in JSON dumps."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
