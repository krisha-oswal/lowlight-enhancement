"""
Gamma Correction
=================
Applies a power-law transformation: I_out = I_in ^ (1/gamma)

Design choice: gamma > 1 brightens dark images by compressing the
upper range and expanding the lower range. We use an adaptive variant
that estimates optimal gamma from image brightness when adaptive=True.

Operating on luminance only prevents hue shift.
"""

import numpy as np
import cv2
from preprocessing.base_enhancer import BaseEnhancer


class GammaCorrectionEnhancer(BaseEnhancer):
    """
    Gamma correction for low-light enhancement.

    Args:
        gamma: Gamma value. Values > 1 brighten the image.
               Typical range for low-light: 1.5 – 3.0
        adaptive: If True, estimate gamma from mean brightness.

    Strengths:
      - Simple, fast, interpretable
      - Adaptive variant handles varying darkness levels

    Weaknesses:
      - Global: no spatial adaptivity
      - Fixed gamma may over-brighten partially lit images
    """

    def __init__(self, gamma: float = 1.5, adaptive: bool = False):
        super().__init__(name="gamma_correction")
        self.gamma = gamma
        self.adaptive = adaptive
        # Pre-build lookup table for fixed gamma (fast uint8 mapping)
        self._lut = self._build_lut(gamma)

    def _build_lut(self, gamma: float) -> np.ndarray:
        """Build uint8 lookup table for given gamma."""
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ], dtype=np.uint8)
        return table

    def _estimate_gamma(self, image: np.ndarray) -> float:
        """
        Estimate gamma from mean brightness using logarithmic formula.
        Brighter images need less correction (gamma closer to 1).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray) / 255.0
        if mean_brightness <= 0:
            return self.gamma
        # log(0.5) / log(mean) maps mean brightness to target 0.5
        gamma_est = np.log(0.5) / (np.log(mean_brightness) + 1e-7)
        # Clamp to sensible range
        return float(np.clip(gamma_est, 0.5, 5.0))

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Apply gamma correction via lookup table (fast) or adaptive."""
        image = self._validate_input(image)

        if self.adaptive:
            gamma = self._estimate_gamma(image)
            lut = self._build_lut(gamma)
        else:
            lut = self._lut

        enhanced = cv2.LUT(image, lut)
        return enhanced
