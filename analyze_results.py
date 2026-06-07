"""
Analyze Results & Generate All Plots
======================================
Loads saved metrics and generates all publication-quality plots
and a summary report.

Usage:
  python analyze_results.py --config configs/config.yaml
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data.exdark_loader import load_config
from visualization.plot_results import generate_all_plots
from utils.io_utils import load_results, load_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("analyze_results")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def analyze(config: dict):
    paths = config["paths"]
    plots_dir = paths["plots_dir"]
    metrics_dir = paths["metrics_dir"]
    sel_dir = paths["selector_dir"]

    # ── 1. Load metrics ───────────────────────────────────────────
    metrics_path = str(Path(metrics_dir) / "all_metrics.json")
    all_metrics = load_results(metrics_path)
    if all_metrics is None:
        raise FileNotFoundError(f"Metrics not found at {metrics_path}. Run run_experiment.py first.")

    method_metrics = all_metrics["per_method"]

    # ── 2. Load per-image DataFrame ───────────────────────────────
    df_path = str(Path(metrics_dir) / "per_image_results.csv")
    df = load_dataframe(df_path)
    if df is None:
        raise FileNotFoundError(f"DataFrame not found at {df_path}")
    logger.info(f"Loaded {len(df)} rows")

    # ── 3. Load selector results (optional) ───────────────────────
    selector_results = load_results(str(Path(sel_dir) / "selector_results.json")) or {}
    comparison = load_results(str(Path(sel_dir) / "strategy_comparison.json")) or {}

    # Feature importances
    fi_path = str(Path(sel_dir) / "feature_importances.csv")
    fi_df = None
    feat_importances = None
    feature_names = []
    if Path(fi_path).exists():
        fi_df = pd.read_csv(fi_path)
        feature_names = fi_df["feature"].tolist()
        feat_importances = fi_df["importance"].values

    # ── 4. Generate all plots ─────────────────────────────────────
    classes = config["dataset"]["classes"]
    generate_all_plots(
        method_metrics=method_metrics,
        df=df,
        selector_results=selector_results,
        comparison=comparison,
        feature_names=feature_names,
        feature_importances=feat_importances,
        classes=classes,
        plots_dir=plots_dir,
    )

    # ── 5. Print correlation analysis ─────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("CORRELATION ANALYSIS: Perceptual Quality vs Detection")
    logger.info("=" * 60)
    for metric in ["recall", "precision"]:
        if metric in df.columns:
            for quality_metric in ["psnr", "ssim"]:
                if quality_metric in df.columns:
                    corr = df[quality_metric].corr(df[metric])
                    logger.info(f"  {quality_metric.upper()} vs {metric}: r = {corr:.4f}")

    # ── 6. Failure analysis summary ───────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FAILURE ANALYSIS SUMMARY")
    logger.info("=" * 60)
    if "method" in df.columns and "recall" in df.columns:
        baseline_df = df[df["method"] == "baseline"][["image_id", "recall"]]
        baseline_df = baseline_df.rename(columns={"recall": "recall_baseline"})

        from preprocessing.enhancer_factory import METHOD_NAMES
        for method in [m for m in METHOD_NAMES if m != "baseline"]:
            sub = df[df["method"] == method][["image_id", "recall"]]
            merged = sub.merge(baseline_df, on="image_id")
            delta = merged["recall"] - merged["recall_baseline"]
            n_failures = int((delta < -0.05).sum())
            n_improvements = int((delta > 0.05).sum())
            mean_delta = float(delta.mean())
            logger.info(
                f"  {method:<20}: +improved={n_improvements:4d}, -hurt={n_failures:4d}, "
                f"mean_delta={mean_delta:+.4f}"
            )

    # ── 7. Per-class best method ──────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PER-CLASS BEST ENHANCEMENT METHOD")
    logger.info("=" * 60)
    for cls in config["dataset"]["classes"]:
        class_aps = {
            m: method_metrics[m]["per_class_AP"].get(cls, 0.0)
            for m in method_metrics
        }
        best_method = max(class_aps, key=class_aps.get)
        baseline_ap = class_aps.get("baseline", 0.0)
        best_ap = class_aps[best_method]
        delta = best_ap - baseline_ap
        logger.info(
            f"  {cls:<12}: best={best_method:<20} AP={best_ap:.4f} "
            f"(baseline={baseline_ap:.4f}, Δ={delta:+.4f})"
        )

    logger.info(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    analyze(config)
