"""
Demo / Smoke Test
==================
Runs the full pipeline on synthetic dummy data (no ExDark download required).
Validates that every module works end-to-end.

Usage:
  python demo.py
"""

import sys
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("demo")


# ── 1. Test all enhancement methods ──────────────────────────────
def test_enhancers():
    logger.info("\n── Testing Enhancement Methods ──")
    from preprocessing.enhancer_factory import build_enhancers

    config = {
        "enhancement": {
            "methods": ["baseline", "histogram_eq", "clahe",
                        "gamma_correction", "retinex", "zero_dce"],
            "clahe": {"clip_limit": 2.0, "tile_grid_size": [8, 8]},
            "gamma_correction": {"gamma": 1.5},
            "retinex": {"sigma_list": [15, 80, 250], "G": 192, "b": -30,
                        "alpha": 125, "beta": 46, "low_clip": 0.01, "high_clip": 0.99},
            "zero_dce": {"n_filters": 32, "n_iters": 8},
        },
        "detection": {"device": "cpu"},
        "paths": {"zero_dce_weights": "nonexistent.pth"},
    }

    enhancers = build_enhancers(config)
    # Create a synthetic dark image (mean brightness ~40)
    test_img = (np.random.randint(0, 80, (240, 320, 3), dtype=np.uint8))

    for name, enhancer in enhancers.items():
        try:
            result = enhancer._safe_enhance(test_img)
            assert result.shape == test_img.shape, f"Shape mismatch for {name}"
            assert result.dtype == np.uint8, f"dtype mismatch for {name}"
            brightness_in = test_img.mean()
            brightness_out = result.mean()
            logger.info(f"  ✓ {name:<20}: {brightness_in:.1f} → {brightness_out:.1f} brightness")
        except Exception as e:
            logger.error(f"  ✗ {name}: {e}")


# ── 2. Test image characterizer ───────────────────────────────────
def test_characterizer():
    logger.info("\n── Testing Image Characterizer ──")
    from evaluation.image_characterizer import (
        extract_image_features, compute_quality_metrics, get_feature_names
    )

    img = np.random.randint(0, 80, (240, 320, 3), dtype=np.uint8)
    enhanced = np.clip(img.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)

    features = extract_image_features(img)
    quality = compute_quality_metrics(img, enhanced)

    logger.info(f"  ✓ Extracted {len(features)} features")
    logger.info(f"    brightness_mean={features['brightness_mean']:.2f}")
    logger.info(f"    entropy={features['entropy']:.4f}")
    logger.info(f"    PSNR={quality['psnr']:.2f} dB, SSIM={quality['ssim']:.4f}")

    expected_features = get_feature_names()
    assert all(k in features for k in expected_features), "Missing features"
    logger.info(f"  ✓ All {len(expected_features)} expected features present")


# ── 3. Test metrics computation ───────────────────────────────────
def test_metrics():
    logger.info("\n── Testing Detection Metrics ──")
    from evaluation.metrics import DetectionEvaluator, compute_iou

    # IoU test
    box1 = [0, 0, 100, 100]
    box2 = [50, 50, 150, 150]
    iou = compute_iou(box1, box2)
    expected_iou = 2500 / (10000 + 10000 - 2500)
    assert abs(iou - expected_iou) < 0.001, f"IoU mismatch: {iou} vs {expected_iou}"
    logger.info(f"  ✓ compute_iou: {iou:.4f} (expected {expected_iou:.4f})")

    config = {
        "evaluation": {"iou_threshold_map": 0.5,
                       "iou_thresholds_coco": [0.5]},
        "dataset": {
            "classes": ["Car", "People"],
            "exdark_to_coco": {"Car": 2, "People": 0},
        },
    }
    evaluator = DetectionEvaluator(config)

    # Simulate 3 images, 2 methods
    for i in range(3):
        for method in ["baseline", "clahe"]:
            evaluator.add_image_result(
                image_id=f"image_{i}",
                method=method,
                detections=[
                    {"bbox": [10, 10, 90, 90], "confidence": 0.9,
                     "class_id": 2, "class_name": "car"},
                ],
                gt_boxes=[
                    {"bbox": (15, 15, 85, 85), "class_name": "Car",
                     "class_id": 0, "light_env": 1},
                ],
                image_features={"feat_brightness_mean": 40.0},
            )

    all_metrics = evaluator.compute_metrics()
    assert "per_method" in all_metrics
    assert "baseline" in all_metrics["per_method"]
    baseline_map = all_metrics["per_method"]["baseline"]["mAP50"]
    logger.info(f"  ✓ DetectionEvaluator: baseline mAP50={baseline_map:.4f}")
    logger.info(f"  ✓ {len(all_metrics['per_image'])} per-image results recorded")


# ── 4. Test selector ─────────────────────────────────────────────
def test_selector():
    logger.info("\n── Testing Enhancement Selector ──")
    from selector.enhancement_selector import (
        EnhancementSelector, assign_oracle_labels, build_selector_dataset
    )
    import pandas as pd

    config = {
        "selector": {
            "models": ["random_forest"],
            "test_size": 0.3,
            "random_state": 42,
            "cross_val_folds": 3,
            "random_forest": {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2},
            "mlp": {"hidden_layer_sizes": [16], "max_iter": 50, "early_stopping": False},
        }
    }

    # Synthetic per-image results
    methods = ["baseline", "clahe", "gamma_correction"]
    n_images = 60
    per_image_results = []
    rng = np.random.RandomState(42)

    for i in range(n_images):
        for m in methods:
            per_image_results.append({
                "image_id": f"img_{i}",
                "method": m,
                "recall": float(rng.uniform(0.2, 0.9)),
                "precision": float(rng.uniform(0.2, 0.9)),
                "feat_brightness_mean": float(rng.uniform(20, 100)),
                "feat_entropy": float(rng.uniform(3, 7)),
                "feat_contrast_std": float(rng.uniform(10, 60)),
                "feat_dark_pixel_ratio": float(rng.uniform(0, 1)),
            })

    oracle_labels = assign_oracle_labels(per_image_results, methods)
    assert len(oracle_labels) == n_images

    X, y = build_selector_dataset(per_image_results, oracle_labels, feature_prefix="feat_")
    assert len(X) == n_images
    logger.info(f"  ✓ Oracle labels: {dict(y.value_counts())}")

    selector = EnhancementSelector(config)
    selector.fit(X, y)
    assert selector.best_model_name is not None
    logger.info(f"  ✓ Selector trained: best={selector.best_model_name}")

    # Predict
    sample_feat = dict(zip(X.columns, X.iloc[0].values))
    pred = selector.predict_best(sample_feat)
    assert pred in methods
    logger.info(f"  ✓ Prediction for sample: {pred}")


# ── 5. Test IO utilities ──────────────────────────────────────────
def test_io():
    logger.info("\n── Testing I/O Utilities ──")
    import tempfile
    import cv2
    from utils.io_utils import (
        save_enhanced_image, load_enhanced_image,
        save_detections, load_detections,
        save_results, load_results,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Enhanced image cache
        path = save_enhanced_image(img, tmpdir, "test/img001", "clahe")
        loaded = load_enhanced_image(tmpdir, "test/img001", "clahe")
        assert loaded is not None, "Image not loaded from cache"
        logger.info(f"  ✓ Enhanced image cache: saved & loaded ({loaded.shape})")

        # Detection cache
        dets = [{"bbox": [0, 0, 10, 10], "confidence": 0.9,
                 "class_id": 0, "class_name": "person"}]
        save_detections(dets, tmpdir, "test/img001", "clahe")
        loaded_dets = load_detections(tmpdir, "test/img001", "clahe")
        assert loaded_dets is not None and len(loaded_dets) == 1
        logger.info(f"  ✓ Detection cache: saved & loaded ({len(loaded_dets)} dets)")

        # Results JSON
        results = {"mAP50": 0.432, "method": "clahe"}
        out_path = str(Path(tmpdir) / "results.json")
        save_results(results, out_path)
        loaded_results = load_results(out_path)
        assert loaded_results["mAP50"] == 0.432
        logger.info(f"  ✓ Results JSON: saved & loaded")


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Low-Light Detection Framework — Demo / Smoke Test")
    logger.info("=" * 60)

    errors = []
    for test_fn in [test_enhancers, test_characterizer, test_metrics,
                    test_selector, test_io]:
        try:
            test_fn()
        except Exception as e:
            logger.error(f"FAILED: {test_fn.__name__}: {e}")
            errors.append((test_fn.__name__, e))

    logger.info("\n" + "=" * 60)
    if errors:
        logger.error(f"DEMO FAILED: {len(errors)} test(s) failed:")
        for name, e in errors:
            logger.error(f"  ✗ {name}: {e}")
        sys.exit(1)
    else:
        logger.info("ALL TESTS PASSED ✓")
        logger.info("\nTo run the full experiment:")
        logger.info("  1. Download ExDark → data/ExDark/")
        logger.info("  2. python run_experiment.py --config configs/config.yaml")
        logger.info("  3. python train_selector.py  --config configs/config.yaml")
        logger.info("  4. python analyze_results.py --config configs/config.yaml")
