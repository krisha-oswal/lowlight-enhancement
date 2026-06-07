"""
Train Enhancement Selector
===========================
Loads per-image results, assigns oracle labels, builds feature dataset,
trains classifiers, evaluates, and saves the best model.

Usage:
  python train_selector.py --config configs/config.yaml
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data.exdark_loader import load_config
from selector.enhancement_selector import (
    EnhancementSelector,
    assign_oracle_labels,
    build_selector_dataset,
)
from utils.io_utils import (
    load_results, load_dataframe, save_results, save_model,
    per_image_results_to_dataframe,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_selector")


def parse_args():
    parser = argparse.ArgumentParser(description="Train enhancement selector")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--metric", default="recall",
        choices=["recall", "precision"],
        help="Metric used to assign oracle labels"
    )
    return parser.parse_args()


def train_selector(config: dict, metric: str = "recall"):
    paths = config["paths"]
    sel_dir = paths["selector_dir"]
    metrics_dir = paths["metrics_dir"]

    # ── 1. Load per-image results ─────────────────────────────────
    logger.info("Loading per-image results...")
    df_path = str(Path(metrics_dir) / "per_image_results.csv")
    df = load_dataframe(df_path)
    if df is None:
        raise FileNotFoundError(
            f"Per-image results not found at {df_path}. "
            "Run run_experiment.py first."
        )
    logger.info(f"Loaded {len(df)} rows from {df_path}")

    # Convert DataFrame back to list of dicts for selector
    per_image_results = df.to_dict(orient="records")

    # ── 2. Assign oracle labels ───────────────────────────────────
    logger.info(f"Assigning oracle labels (metric={metric})...")
    methods = config["enhancement"]["methods"]

    oracle_labels = assign_oracle_labels(
        per_image_results, methods, metric=metric
    )

    # Save oracle labels
    oracle_path = str(Path(sel_dir) / "oracle_labels.json")
    save_results(oracle_labels, oracle_path)

    # Distribution of oracle labels
    from collections import Counter
    dist = Counter(oracle_labels.values())
    logger.info(f"Oracle label distribution: {dict(dist)}")

    # ── 3. Build feature dataset ──────────────────────────────────
    logger.info("Building feature dataset...")
    X, y = build_selector_dataset(per_image_results, oracle_labels)

    feature_names = list(X.columns)
    logger.info(f"Feature matrix: {X.shape}, Labels: {y.shape}")

    # ── 4. Train selector ─────────────────────────────────────────
    logger.info("Training classifiers...")
    selector = EnhancementSelector(config)
    selector.fit(X, y)

    # ── 5. Compute performance comparison ─────────────────────────
    logger.info("Computing strategy performance comparison...")
    comparison = selector.compute_performance_comparison(
        per_image_results, oracle_labels, X, y, metric=metric
    )
    logger.info(f"Performance comparison: {comparison}")

    # ── 6. Feature importance ─────────────────────────────────────
    feat_importance = selector.get_feature_importance()

    # ── 7. Save everything ────────────────────────────────────────
    # Save selector results
    selector_results_path = str(Path(sel_dir) / "selector_results.json")
    save_results(selector.results, selector_results_path)

    comparison_path = str(Path(sel_dir) / "strategy_comparison.json")
    save_results(comparison, comparison_path)

    # Save best model
    model_path = str(Path(sel_dir) / "best_selector_model.pkl")
    save_model(selector.best_model, model_path)

    # Save feature importances
    if feat_importance is not None:
        fi_df = pd.DataFrame({
            "feature": feature_names[:len(feat_importance)],
            "importance": feat_importance.values
        }).sort_values("importance", ascending=False)
        fi_path = str(Path(sel_dir) / "feature_importances.csv")
        fi_df.to_csv(fi_path, index=False)
        logger.info(f"Feature importances saved to {fi_path}")

    # ── 8. Print summary ──────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("SELECTOR RESULTS SUMMARY")
    logger.info("=" * 60)
    for model_name, res in selector.results.items():
        logger.info(
            f"{model_name:<20}: CV={res['cv_mean']:.3f}±{res['cv_std']:.3f}, "
            f"TestAcc={res['test_accuracy']:.3f}"
        )

    logger.info(f"\nBest model: {selector.best_model_name}")
    logger.info("\nStrategy comparison (mean recall):")
    logger.info(f"  Oracle:       {comparison['oracle_mean']:.4f}  (upper bound)")
    logger.info(f"  Selector:     {comparison['selector_mean']:.4f}")
    logger.info(f"  Best Fixed ({comparison['best_fixed_method']}): {comparison['best_fixed_mean']:.4f}")
    logger.info(f"  Baseline:     {comparison['baseline_mean']:.4f}")

    return {
        "selector": selector,
        "oracle_labels": oracle_labels,
        "feature_names": feature_names,
        "feat_importance": feat_importance,
        "comparison": comparison,
    }


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train_selector(config, metric=args.metric)
