"""
Main Experiment Orchestrator
==============================
Runs the full detection pipeline:
  1. Load ExDark dataset
  2. For each image × each enhancement method:
     a. Enhance the image (with caching)
     b. Run YOLOv8 detection (with caching)
     c. Extract image features
     d. Compute PSNR/SSIM vs original
  3. Accumulate results in DetectionEvaluator
  4. Compute and save final metrics

Usage:
  python run_experiment.py --config configs/config.yaml [--force] [--limit 100]
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── project imports ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from data.exdark_loader import ExDarkDataset, load_config
from preprocessing.enhancer_factory import build_enhancers
from detection.yolov8_detector import YOLOv8Detector, filter_detections_by_class
from evaluation.metrics import DetectionEvaluator
from evaluation.image_characterizer import (
    extract_image_features, compute_quality_metrics
)
from utils.io_utils import (
    ensure_dirs, save_enhanced_image, load_enhanced_image,
    save_detections, load_detections,
    save_results, per_image_results_to_dataframe, save_dataframe,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiment")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Low-light detection experiment pipeline"
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if cached results exist"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of images (for debugging)"
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Override enhancement methods to run"
    )
    return parser.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def run_experiment(config: dict, force: bool = False, limit: int = None, methods_override=None):
    """
    Full experiment pipeline.

    Returns:
        dict with 'metrics', 'per_image_df' paths
    """
    set_seed(config.get("seed", 42))
    ensure_dirs(config)

    paths = config["paths"]
    det_cfg = config["detection"]
    exdark_cfg = config["dataset"]

    # ── 1. Load dataset ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading ExDark dataset")
    logger.info("=" * 60)

    if limit:
        config["dataset"]["max_images"] = limit

    dataset = ExDarkDataset(config)
    logger.info(f"Dataset: {len(dataset)} images")
    logger.info(f"Class distribution: {dataset.get_class_distribution()}")

    # ── 2. Build enhancers ────────────────────────────────────────
    logger.info("\nSTEP 2: Loading enhancement methods")
    if methods_override:
        config["enhancement"]["methods"] = methods_override
    enhancers = build_enhancers(config)
    methods = list(enhancers.keys())
    logger.info(f"Methods: {methods}")

    # ── 3. Load YOLOv8 ───────────────────────────────────────────
    logger.info("\nSTEP 3: Loading YOLOv8 detector")
    detector = YOLOv8Detector(
        model_name=det_cfg["model"],
        conf_threshold=det_cfg["confidence_threshold"],
        iou_threshold=det_cfg["iou_threshold"],
        image_size=det_cfg["image_size"],
        device=det_cfg["device"],
    )

    # Allowed COCO class IDs (ExDark classes only)
    allowed_coco_ids = list(exdark_cfg["exdark_to_coco"].values())

    # ── 4. Evaluator ─────────────────────────────────────────────
    evaluator = DetectionEvaluator(config)

    # ── 5. Main loop ─────────────────────────────────────────────
    logger.info("\nSTEP 4: Running detection pipeline")
    total_steps = len(dataset) * len(methods)
    pbar = tqdm(total=total_steps, desc="Processing", unit="img-method")

    t_start = time.time()
    n_cached_enh = 0
    n_cached_det = 0

    for sample in dataset:
        image_id = sample.image_id
        gt_boxes = sample.get_gt_boxes()

        # Load original image once
        try:
            original_img = sample.load_image()
        except FileNotFoundError as e:
            logger.warning(f"Skipping {image_id}: {e}")
            pbar.update(len(methods))
            continue

        # Extract features from ORIGINAL image (used by selector)
        image_features = extract_image_features(original_img)

        for method_name, enhancer in enhancers.items():

            # ── a. Enhance (with cache) ──────────────────────────
            enhanced_img = None
            if not force:
                enhanced_img = load_enhanced_image(
                    paths["enhanced_cache_dir"], image_id, method_name
                )
                if enhanced_img is not None:
                    n_cached_enh += 1

            if enhanced_img is None:
                try:
                    enhanced_img = enhancer._safe_enhance(original_img)
                    save_enhanced_image(
                        enhanced_img, paths["enhanced_cache_dir"],
                        image_id, method_name
                    )
                except Exception as e:
                    logger.error(f"Enhancement failed [{method_name}] {image_id}: {e}")
                    enhanced_img = original_img.copy()

            # ── b. Detect (with cache) ───────────────────────────
            detections = None
            if not force:
                detections = load_detections(
                    paths["detection_results_dir"], image_id, method_name
                )
                if detections is not None:
                    n_cached_det += 1

            if detections is None:
                try:
                    raw_dets = detector.detect(enhanced_img)
                    # Filter to ExDark-relevant COCO classes
                    detections = filter_detections_by_class(raw_dets, allowed_coco_ids)
                    save_detections(
                        detections, paths["detection_results_dir"],
                        image_id, method_name
                    )
                except Exception as e:
                    logger.error(f"Detection failed [{method_name}] {image_id}: {e}")
                    detections = []

            # ── c. Quality metrics ───────────────────────────────
            if method_name != "baseline":
                quality = compute_quality_metrics(original_img, enhanced_img)
            else:
                quality = {"psnr": 100.0, "ssim": 1.0}

            # ── d. Register result ───────────────────────────────
            # Merge features and quality into a single dict
            combined_features = {**image_features, **quality}

            evaluator.add_image_result(
                image_id=image_id,
                method=method_name,
                detections=detections,
                gt_boxes=gt_boxes,
                image_features=combined_features,
            )

            pbar.update(1)

    pbar.close()
    elapsed = time.time() - t_start
    logger.info(f"\nPipeline complete in {elapsed:.1f}s")
    logger.info(f"  Cache hits — enhanced: {n_cached_enh}, detections: {n_cached_det}")

    # ── 6. Compute metrics ────────────────────────────────────────
    logger.info("\nSTEP 5: Computing metrics")
    all_metrics = evaluator.compute_metrics()
    per_image_results = evaluator.get_per_image_results()

    # ── 7. Save results ───────────────────────────────────────────
    metrics_path = str(Path(paths["metrics_dir"]) / "all_metrics.json")
    save_results(all_metrics, metrics_path)

    df = per_image_results_to_dataframe(per_image_results)
    df_path = str(Path(paths["metrics_dir"]) / "per_image_results.csv")
    save_dataframe(df, df_path)

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    header = f"{'Method':<20} {'mAP@0.5':>9} {'Precision':>10} {'Recall':>8}"
    logger.info(header)
    logger.info("-" * 50)
    for method, m in all_metrics["per_method"].items():
        logger.info(
            f"{method:<20} {m['mAP50']:>9.4f} {m['mean_precision']:>10.4f} {m['mean_recall']:>8.4f}"
        )

    return {
        "metrics_path": metrics_path,
        "df_path": df_path,
        "metrics": all_metrics,
    }


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    run_experiment(
        config,
        force=args.force,
        limit=args.limit,
        methods_override=args.methods,
    )
