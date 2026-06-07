"""
Enhancer Factory
================
Creates and manages all enhancement method instances.
Uses the Factory pattern to decouple instantiation from usage.
"""

from typing import Dict, List
import logging

from preprocessing.base_enhancer import BaseEnhancer
from preprocessing.baseline import BaselineEnhancer
from preprocessing.histogram_eq import HistogramEqualizationEnhancer
from preprocessing.clahe import CLAHEEnhancer
from preprocessing.gamma_correction import GammaCorrectionEnhancer
from preprocessing.retinex import RetinexEnhancer
from preprocessing.zero_dce import ZeroDCEEnhancer

logger = logging.getLogger(__name__)

# Canonical method names
METHOD_NAMES = [
    "baseline",
    "histogram_eq",
    "clahe",
    "gamma_correction",
    "retinex",
    "zero_dce",
]


def build_enhancers(config: dict) -> Dict[str, BaseEnhancer]:
    """
    Instantiate all configured enhancers.

    Args:
        config: Full experiment config dict

    Returns:
        dict mapping method_name → enhancer instance
    """
    enh_config = config.get("enhancement", {})
    det_config = config.get("detection", {})
    methods = enh_config.get("methods", METHOD_NAMES)
    device = det_config.get("device", "cpu")
    weights_path = config.get("paths", {}).get("zero_dce_weights", None)

    enhancers: Dict[str, BaseEnhancer] = {}

    for method in methods:
        try:
            if method == "baseline":
                enhancers[method] = BaselineEnhancer()

            elif method == "histogram_eq":
                enhancers[method] = HistogramEqualizationEnhancer()

            elif method == "clahe":
                cfg = enh_config.get("clahe", {})
                enhancers[method] = CLAHEEnhancer(
                    clip_limit=cfg.get("clip_limit", 2.0),
                    tile_grid_size=cfg.get("tile_grid_size", [8, 8]),
                )

            elif method == "gamma_correction":
                cfg = enh_config.get("gamma_correction", {})
                enhancers[method] = GammaCorrectionEnhancer(
                    gamma=cfg.get("gamma", 1.5),
                    adaptive=cfg.get("adaptive", False),
                )

            elif method == "retinex":
                cfg = enh_config.get("retinex", {})
                enhancers[method] = RetinexEnhancer(
                    sigma_list=cfg.get("sigma_list", [15, 80, 250]),
                    G=cfg.get("G", 192.0),
                    b=cfg.get("b", -30.0),
                    alpha=cfg.get("alpha", 125.0),
                    beta=cfg.get("beta", 46.0),
                    low_clip=cfg.get("low_clip", 0.01),
                    high_clip=cfg.get("high_clip", 0.99),
                )

            elif method == "zero_dce":
                cfg = enh_config.get("zero_dce", {})
                enhancers[method] = ZeroDCEEnhancer(
                    weights_path=weights_path,
                    n_filters=cfg.get("n_filters", 32),
                    n_iters=cfg.get("n_iters", 8),
                    device=device,
                )

            else:
                logger.warning(f"Unknown enhancement method: {method}")
                continue

            logger.info(f"  Loaded enhancer: {method}")

        except Exception as e:
            logger.error(f"Failed to load enhancer '{method}': {e}")
            # Fallback to baseline so experiments can continue
            enhancers[method] = BaselineEnhancer()
            enhancers[method].name = method

    return enhancers


def get_method_names() -> List[str]:
    """Return canonical list of all method names."""
    return METHOD_NAMES.copy()
