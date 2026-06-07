"""
Image Characterizer
====================
Extracts low-level features from images for the enhancement selector
and correlation analysis.

Features extracted:
  1. Brightness       — mean pixel intensity in grayscale
  2. Contrast         — std dev of pixel intensities (global contrast)
  3. Entropy          — Shannon entropy of grayscale histogram
  4. Noise (variance) — estimated noise variance via Laplacian
  5. Histogram stats  — skewness, kurtosis, dark/mid/bright pixel ratios
  6. Colorfulness     — Hasler & Süsstrunk colorfulness metric
  7. Blur             — Laplacian variance (focus measure)
  8. PSNR/SSIM        — Quality metrics between original and enhanced
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import cv2
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


def extract_image_features(image: np.ndarray) -> Dict[str, float]:
    """
    Extract characterization features from a single BGR image.

    Args:
        image: BGR uint8 numpy array

    Returns:
        Dict of feature_name → float value
    """
    if image is None or image.size == 0:
        return _empty_features()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    n_pixels = h * w

    features = {}

    # ── 1. Brightness ──────────────────────────────────────────
    features["brightness_mean"] = float(np.mean(gray))
    features["brightness_median"] = float(np.median(gray))
    features["brightness_std"] = float(np.std(gray))

    # ── 2. Contrast ────────────────────────────────────────────
    features["contrast_std"] = float(np.std(gray))
    features["contrast_range"] = float(gray.max() - gray.min())
    # RMS contrast
    features["contrast_rms"] = float(np.sqrt(np.mean((gray - np.mean(gray)) ** 2)))

    # ── 3. Shannon Entropy ─────────────────────────────────────
    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0, 256])
    hist = hist.flatten() / n_pixels  # normalize to probability
    hist = hist[hist > 0]  # remove zeros for log
    features["entropy"] = float(-np.sum(hist * np.log2(hist)))

    # ── 4. Noise (Laplacian-based variance estimate) ────────────
    # Laplacian highlights rapid intensity changes (noise)
    laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
    features["noise_variance"] = float(np.var(laplacian))
    features["noise_std"] = float(np.std(laplacian))

    # ── 5. Histogram Distribution Statistics ───────────────────
    pixel_vals = gray.flatten()
    features["hist_skewness"] = float(scipy_stats.skew(pixel_vals))
    features["hist_kurtosis"] = float(scipy_stats.kurtosis(pixel_vals))

    # Pixel ratio by region
    features["dark_pixel_ratio"] = float(np.sum(gray < 64) / n_pixels)
    features["mid_pixel_ratio"] = float(
        np.sum((gray >= 64) & (gray < 192)) / n_pixels
    )
    features["bright_pixel_ratio"] = float(np.sum(gray >= 192) / n_pixels)

    # Percentiles
    features["p5"] = float(np.percentile(gray, 5))
    features["p25"] = float(np.percentile(gray, 25))
    features["p75"] = float(np.percentile(gray, 75))
    features["p95"] = float(np.percentile(gray, 95))

    # ── 6. Colorfulness (Hasler & Süsstrunk, 2003) ─────────────
    b, g, r = cv2.split(image.astype(np.float32))
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + \
                   0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    features["colorfulness"] = float(colorfulness)

    # ── 7. Blur Metric (Laplacian variance — high = sharp) ─────
    features["blur_laplacian"] = float(np.var(laplacian))

    # ── 8. Local contrast (block std average) ──────────────────
    block_size = 16
    local_stds = []
    for row in range(0, h - block_size, block_size):
        for col in range(0, w - block_size, block_size):
            block = gray[row:row + block_size, col:col + block_size]
            local_stds.append(np.std(block))
    features["local_contrast_mean"] = float(np.mean(local_stds)) if local_stds else 0.0
    features["local_contrast_std"] = float(np.std(local_stds)) if local_stds else 0.0

    return features


def compute_psnr(original: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between original and enhanced.
    Higher PSNR → enhanced is more similar to original.

    Note: In low-light enhancement, very high PSNR means less change;
    this is why PSNR alone doesn't capture enhancement quality.
    """
    original_f = original.astype(np.float64)
    enhanced_f = enhanced.astype(np.float64)
    mse = np.mean((original_f - enhanced_f) ** 2)
    if mse < 1e-10:
        return 100.0  # Identical images
    psnr = 20.0 * np.log10(255.0 / np.sqrt(mse))
    return float(np.clip(psnr, 0, 100))


def compute_ssim(original: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    Uses OpenCV-based grayscale SSIM approximation.

    SSIM ∈ [-1, 1]; 1 = identical, higher is more structurally similar.
    """
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        gray1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        ssim_val = sk_ssim(gray1, gray2, data_range=255)
        return float(ssim_val)
    except Exception:
        # Fallback: OpenCV-based approximation
        return _ssim_opencv(original, enhanced)


def _ssim_opencv(img1: np.ndarray, img2: np.ndarray) -> float:
    """Manual SSIM computation as fallback."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1_f = img1.astype(np.float64)
    img2_f = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1_f, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2_f, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1_f ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2_f ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1_f * img2_f, (11, 11), 1.5) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / (den + 1e-7)

    return float(np.mean(ssim_map))


def compute_quality_metrics(
    original: np.ndarray, enhanced: np.ndarray
) -> Dict[str, float]:
    """Compute PSNR and SSIM between original and enhanced."""
    return {
        "psnr": compute_psnr(original, enhanced),
        "ssim": compute_ssim(original, enhanced),
    }


def _empty_features() -> Dict[str, float]:
    """Return zero-filled feature dict for invalid images."""
    keys = [
        "brightness_mean", "brightness_median", "brightness_std",
        "contrast_std", "contrast_range", "contrast_rms",
        "entropy", "noise_variance", "noise_std",
        "hist_skewness", "hist_kurtosis",
        "dark_pixel_ratio", "mid_pixel_ratio", "bright_pixel_ratio",
        "p5", "p25", "p75", "p95",
        "colorfulness", "blur_laplacian",
        "local_contrast_mean", "local_contrast_std",
    ]
    return {k: 0.0 for k in keys}


def get_feature_names() -> list:
    """Return list of feature names in canonical order."""
    return list(_empty_features().keys())
