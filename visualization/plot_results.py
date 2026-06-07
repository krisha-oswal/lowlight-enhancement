"""
Visualization Module
=====================
Generates all publication-quality plots for the research paper.

Plots generated:
  1.  mAP comparison bar chart (all methods)
  2.  Per-class AP heatmap
  3.  Precision–Recall curves
  4.  PSNR vs mAP scatter (correlation study)
  5.  SSIM vs mAP scatter
  6.  Image feature distributions (brightness, entropy, etc.)
  7.  Enhancement effect grid (visual comparison)
  8.  Failure case analysis (where enhancement hurts)
  9.  Selector confusion matrix
  10. Feature importance bar chart
  11. Performance comparison: oracle vs selector vs fixed vs baseline
  12. Per-class enhancement benefit analysis
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)

# ── Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

METHOD_COLORS = {
    "baseline":        "#4C72B0",
    "histogram_eq":    "#DD8452",
    "clahe":           "#55A868",
    "gamma_correction":"#C44E52",
    "retinex":         "#8172B3",
    "zero_dce":        "#937860",
}

METHOD_LABELS = {
    "baseline":        "Baseline",
    "histogram_eq":    "HE",
    "clahe":           "CLAHE",
    "gamma_correction":"Gamma",
    "retinex":         "Retinex",
    "zero_dce":        "Zero-DCE",
}

METHODS_ORDER = list(METHOD_COLORS.keys())


def _save(fig, plots_dir: str, name: str):
    """Save figure to plots directory."""
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(plots_dir) / f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot: {path}")
    return path


# ─────────────────────────────────────────────
# 1. mAP Comparison Bar Chart
# ─────────────────────────────────────────────

def plot_map_comparison(
    method_metrics: Dict[str, Dict], plots_dir: str
) -> str:
    """Bar chart comparing mAP50 across all enhancement methods."""
    methods = [m for m in METHODS_ORDER if m in method_metrics]
    maps = [method_metrics[m]["mAP50"] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, maps, color=colors, edgecolor="white", linewidth=0.8)

    # Annotate values
    for bar, val in zip(bars, maps):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    # Highlight best
    best_idx = int(np.argmax(maps))
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(2)

    ax.set_xlabel("Enhancement Method")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Object Detection Performance by Enhancement Method")
    ax.set_ylim(0, max(maps) * 1.15 + 0.02)
    ax.axhline(maps[0], color=METHOD_COLORS["baseline"], linestyle="--",
               alpha=0.5, linewidth=1, label="Baseline")
    ax.legend()
    sns.despine(ax=ax)

    return _save(fig, plots_dir, "01_map_comparison")


# ─────────────────────────────────────────────
# 2. Per-Class AP Heatmap
# ─────────────────────────────────────────────

def plot_per_class_heatmap(
    method_metrics: Dict[str, Dict],
    classes: List[str],
    plots_dir: str,
) -> str:
    """Heatmap of AP per class × per method."""
    methods = [m for m in METHODS_ORDER if m in method_metrics]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    data = np.zeros((len(classes), len(methods)))
    for j, method in enumerate(methods):
        per_class_ap = method_metrics[method].get("per_class_AP", {})
        for i, cls in enumerate(classes):
            data[i, j] = per_class_ap.get(cls, 0.0)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        data, xticklabels=labels, yticklabels=classes,
        annot=True, fmt=".2f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, vmin=0, vmax=1,
        cbar_kws={"label": "AP@0.5"},
    )
    ax.set_title("Per-Class Average Precision by Enhancement Method")
    ax.set_xlabel("Enhancement Method")
    ax.set_ylabel("Object Class")

    return _save(fig, plots_dir, "02_perclass_heatmap")


# ─────────────────────────────────────────────
# 3. PSNR vs mAP Scatter
# ─────────────────────────────────────────────

def plot_psnr_map_correlation(
    df: pd.DataFrame, plots_dir: str
) -> str:
    """Scatter plot: PSNR vs per-image recall, colored by method."""
    if "psnr" not in df.columns:
        logger.warning("No PSNR column found.")
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric in zip(axes, ["recall", "precision"]):
        if metric not in df.columns:
            continue
        for method in METHODS_ORDER:
            sub = df[df["method"] == method]
            if len(sub) == 0:
                continue
            ax.scatter(
                sub["psnr"], sub[metric],
                c=METHOD_COLORS[method],
                label=METHOD_LABELS.get(method, method),
                alpha=0.5, s=20,
            )

        # Add regression line
        valid = df[["psnr", metric]].dropna()
        if len(valid) > 2:
            z = np.polyfit(valid["psnr"], valid[metric], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid["psnr"].min(), valid["psnr"].max(), 100)
            ax.plot(x_line, p(x_line), "k--", alpha=0.7, linewidth=1.5,
                    label=f"Trend (r={valid['psnr'].corr(valid[metric]):.2f})")

        ax.set_xlabel("PSNR (dB)")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"PSNR vs {metric.capitalize()}")
        ax.legend(fontsize=8, ncol=2)
        sns.despine(ax=ax)

    fig.suptitle("Do Perceptual Quality Metrics Predict Detection Performance?",
                 fontsize=13)
    plt.tight_layout()
    return _save(fig, plots_dir, "03_psnr_map_correlation")


# ─────────────────────────────────────────────
# 4. SSIM vs Recall Scatter
# ─────────────────────────────────────────────

def plot_ssim_correlation(df: pd.DataFrame, plots_dir: str) -> str:
    """Scatter plot: SSIM vs recall."""
    if "ssim" not in df.columns:
        return ""

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in METHODS_ORDER:
        sub = df[df["method"] == method]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["ssim"], sub["recall"],
            c=METHOD_COLORS[method],
            label=METHOD_LABELS.get(method, method),
            alpha=0.5, s=20,
        )

    valid = df[["ssim", "recall"]].dropna()
    if len(valid) > 2:
        corr = valid["ssim"].corr(valid["recall"])
        ax.set_title(f"SSIM vs Detection Recall (r={corr:.2f})")

    ax.set_xlabel("SSIM")
    ax.set_ylabel("Recall")
    ax.legend(fontsize=8)
    sns.despine(ax=ax)
    return _save(fig, plots_dir, "04_ssim_recall_scatter")


# ─────────────────────────────────────────────
# 5. Image Feature Distributions
# ─────────────────────────────────────────────

def plot_feature_distributions(df: pd.DataFrame, plots_dir: str) -> str:
    """Violin/box plots of key image features grouped by oracle method."""
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    plot_features = [
        "feat_brightness_mean", "feat_entropy",
        "feat_contrast_std", "feat_noise_variance",
        "feat_dark_pixel_ratio", "feat_colorfulness",
    ]
    plot_features = [f for f in plot_features if f in feature_cols]
    if not plot_features:
        logger.warning("No feature columns found in DataFrame.")
        return ""

    n = len(plot_features)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    # Use oracle_method if available, else method
    group_col = "oracle_method" if "oracle_method" in df.columns else "method"
    if group_col not in df.columns:
        group_col = "method"

    for ax, feat in zip(axes, plot_features):
        feat_label = feat.replace("feat_", "").replace("_", " ").title()
        sub = df[["method", feat]].dropna()
        methods_present = [m for m in METHODS_ORDER if m in sub["method"].values]
        data_by_method = [sub[sub["method"] == m][feat].values for m in methods_present]
        labels = [METHOD_LABELS.get(m, m) for m in methods_present]
        colors = [METHOD_COLORS[m] for m in methods_present]

        parts = ax.violinplot(
            data_by_method, positions=range(len(methods_present)),
            showmedians=True, showextrema=False,
        )
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(methods_present)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(feat_label)
        sns.despine(ax=ax)

    fig.suptitle("Image Feature Distributions by Enhancement Method", fontsize=13)
    plt.tight_layout()
    return _save(fig, plots_dir, "05_feature_distributions")


# ─────────────────────────────────────────────
# 6. Failure Case Analysis
# ─────────────────────────────────────────────

def plot_failure_analysis(
    df: pd.DataFrame,
    plots_dir: str,
    top_n: int = 5,
) -> str:
    """
    Bar chart showing cases where enhancement hurts (recall drops below baseline).
    """
    if "method" not in df.columns or "recall" not in df.columns:
        return ""

    baseline_df = df[df["method"] == "baseline"][["image_id", "recall"]]
    baseline_df = baseline_df.rename(columns={"recall": "recall_baseline"})

    failure_counts = {}
    degradation_amounts = {}

    for method in [m for m in METHODS_ORDER if m != "baseline"]:
        method_df = df[df["method"] == method][["image_id", "recall"]]
        merged = method_df.merge(baseline_df, on="image_id")
        delta = merged["recall"] - merged["recall_baseline"]
        failures = delta[delta < -0.05]  # >5% drop is a failure
        failure_counts[method] = len(failures)
        degradation_amounts[method] = float(failures.mean()) if len(failures) > 0 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    methods = [m for m in METHODS_ORDER if m != "baseline"]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]

    # Failure counts
    counts = [failure_counts.get(m, 0) for m in methods]
    axes[0].bar(labels, counts, color=colors, edgecolor="white")
    axes[0].set_title("Number of Images Where Enhancement Hurts\n(Recall drop > 5%)")
    axes[0].set_ylabel("# Failure Images")
    axes[0].set_xlabel("Enhancement Method")

    # Mean degradation
    degs = [abs(degradation_amounts.get(m, 0.0)) for m in methods]
    axes[1].bar(labels, degs, color=colors, edgecolor="white")
    axes[1].set_title("Mean Recall Degradation\n(on failure images)")
    axes[1].set_ylabel("Mean |ΔRecall|")
    axes[1].set_xlabel("Enhancement Method")

    for ax in axes:
        sns.despine(ax=ax)

    fig.suptitle("Failure Case Analysis: When Enhancement Degrades Detection", fontsize=13)
    plt.tight_layout()
    return _save(fig, plots_dir, "06_failure_analysis")


# ─────────────────────────────────────────────
# 7. Selector Confusion Matrix
# ─────────────────────────────────────────────

def plot_selector_confusion_matrix(
    selector_results: Dict[str, Any],
    plots_dir: str,
) -> str:
    """Confusion matrix for the best selector model."""
    if not selector_results:
        return ""

    best_name = max(
        selector_results.keys(),
        key=lambda k: selector_results[k].get("test_accuracy", 0)
    )
    best = selector_results[best_name]
    cm = np.array(best["confusion_matrix"])
    classes = best.get("classes", [])

    fig, ax = plt.subplots(figsize=(8, 6))
    # Normalize row-wise
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-7)
    sns.heatmap(
        cm_norm, annot=cm, fmt="d",
        xticklabels=[METHOD_LABELS.get(c, c) for c in classes],
        yticklabels=[METHOD_LABELS.get(c, c) for c in classes],
        cmap="Blues", ax=ax,
        cbar_kws={"label": "Normalized frequency"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (Oracle)")
    ax.set_title(
        f"Enhancement Selector Confusion Matrix\n"
        f"({best_name}, Acc={best['test_accuracy']:.3f})"
    )
    return _save(fig, plots_dir, "07_selector_confusion_matrix")


# ─────────────────────────────────────────────
# 8. Feature Importance
# ─────────────────────────────────────────────

def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    plots_dir: str,
    top_n: int = 15,
) -> str:
    """Horizontal bar chart of top feature importances."""
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i].replace("feat_", "").replace("_", " ").title()
                 for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(top_names[::-1], top_vals[::-1], color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Top Feature Importances for Enhancement Selector\n(Random Forest)")
    sns.despine(ax=ax)
    plt.tight_layout()
    return _save(fig, plots_dir, "08_feature_importance")


# ─────────────────────────────────────────────
# 9. Oracle vs Selector vs Fixed vs Baseline
# ─────────────────────────────────────────────

def plot_performance_comparison(
    comparison: Dict[str, float], plots_dir: str
) -> str:
    """
    Bar chart comparing oracle / selector / best-fixed / baseline performance.
    """
    strategies = ["Oracle", "Selector", f"Best Fixed\n({comparison.get('best_fixed_method','')})", "Baseline"]
    scores = [
        comparison.get("oracle_mean", 0),
        comparison.get("selector_mean", 0),
        comparison.get("best_fixed_mean", 0),
        comparison.get("baseline_mean", 0),
    ]
    colors = ["#55A868", "#4C72B0", "#DD8452", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(strategies, scores, color=colors, edgecolor="white")
    for bar, val in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_ylabel("Mean Recall")
    ax.set_title("Enhancement Strategy Performance Comparison")
    ax.set_ylim(0, max(scores) * 1.15 + 0.01)
    sns.despine(ax=ax)

    # Selector gap annotation
    oracle = comparison.get("oracle_mean", 0)
    selector = comparison.get("selector_mean", 0)
    if oracle > 0:
        gap = (oracle - selector) / oracle * 100
        ax.text(0.98, 0.95, f"Selector gap: {gap:.1f}% below oracle",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="gray")

    return _save(fig, plots_dir, "09_strategy_comparison")


# ─────────────────────────────────────────────
# 10. Correlation Table Heatmap
# ─────────────────────────────────────────────

def plot_correlation_heatmap(
    df: pd.DataFrame, plots_dir: str
) -> str:
    """Pearson correlation between image features and detection metrics."""
    metric_cols = ["recall", "precision"]
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    if not feat_cols:
        return ""

    plot_cols = feat_cols[:12]  # top 12 features
    corr_data = {}
    for mc in metric_cols:
        if mc in df.columns:
            corr_data[mc] = df[plot_cols + [mc]].corr()[mc][plot_cols]

    if not corr_data:
        return ""

    corr_df = pd.DataFrame(corr_data)
    corr_df.index = [i.replace("feat_", "").replace("_", " ").title()
                     for i in corr_df.index]

    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(
        corr_df, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, ax=ax, linewidths=0.5,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Pearson Correlation:\nImage Features vs Detection Metrics")
    ax.set_xlabel("Detection Metric")
    plt.tight_layout()
    return _save(fig, plots_dir, "10_feature_metric_correlation")


# ─────────────────────────────────────────────
# Master plot runner
# ─────────────────────────────────────────────

def generate_all_plots(
    method_metrics: Dict,
    df: pd.DataFrame,
    selector_results: Dict,
    comparison: Dict,
    feature_names: List[str],
    feature_importances: Optional[np.ndarray],
    classes: List[str],
    plots_dir: str,
):
    """Generate all plots for the research paper."""
    logger.info("Generating all plots...")
    generated = []

    try:
        generated.append(plot_map_comparison(method_metrics, plots_dir))
    except Exception as e:
        logger.error(f"plot_map_comparison failed: {e}")

    try:
        generated.append(plot_per_class_heatmap(method_metrics, classes, plots_dir))
    except Exception as e:
        logger.error(f"plot_per_class_heatmap failed: {e}")

    try:
        generated.append(plot_psnr_map_correlation(df, plots_dir))
    except Exception as e:
        logger.error(f"plot_psnr_map_correlation failed: {e}")

    try:
        generated.append(plot_ssim_correlation(df, plots_dir))
    except Exception as e:
        logger.error(f"plot_ssim_correlation failed: {e}")

    try:
        generated.append(plot_feature_distributions(df, plots_dir))
    except Exception as e:
        logger.error(f"plot_feature_distributions failed: {e}")

    try:
        generated.append(plot_failure_analysis(df, plots_dir))
    except Exception as e:
        logger.error(f"plot_failure_analysis failed: {e}")

    if selector_results:
        try:
            generated.append(plot_selector_confusion_matrix(selector_results, plots_dir))
        except Exception as e:
            logger.error(f"plot_selector_confusion_matrix failed: {e}")

    if feature_importances is not None:
        try:
            generated.append(plot_feature_importance(feature_names, feature_importances, plots_dir))
        except Exception as e:
            logger.error(f"plot_feature_importance failed: {e}")

    if comparison:
        try:
            generated.append(plot_performance_comparison(comparison, plots_dir))
        except Exception as e:
            logger.error(f"plot_performance_comparison failed: {e}")

    try:
        generated.append(plot_correlation_heatmap(df, plots_dir))
    except Exception as e:
        logger.error(f"plot_correlation_heatmap failed: {e}")

    logger.info(f"Generated {len([g for g in generated if g])} plots in {plots_dir}")
