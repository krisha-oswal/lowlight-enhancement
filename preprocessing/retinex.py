"""
Retinex-Based Enhancement
==========================
Implements Multi-Scale Retinex with Color Restoration (MSRCR).

Theory (Land & McCann, 1971):
  An image I(x,y) = L(x,y) * R(x,y)
  where L = illumination, R = reflectance (true scene color)
  Retinex estimates log(R) = log(I) - log(L_estimated)
  Illumination L is estimated by Gaussian blurring.

MSR uses multiple scales (sigmas) to capture both fine detail
and large illumination gradients.

MSRCR adds color restoration to prevent gray-world color cast.

References:
  - Jobson et al., "A multiscale retinex for bridging the gap
    between color images and the human observation of scenes," 1997.
"""

import numpy as np
import cv2
from preprocessing.base_enhancer import BaseEnhancer


class RetinexEnhancer(BaseEnhancer):
    """
    Multi-Scale Retinex with Color Restoration (MSRCR).

    Args:
        sigma_list: Gaussian blur scales for illumination estimation
        G, b, alpha, beta: MSRCR color restoration parameters
        low_clip, high_clip: Percentile normalization clips

    Strengths:
      - Physically motivated model of human visual system
      - Handles both global and local illumination variations
      - Preserves color fidelity via color restoration

    Weaknesses:
      - Computationally heavier than histogram methods
      - Parameter sensitive; may produce halos near edges
    """

    def __init__(
        self,
        sigma_list: list = None,
        G: float = 192.0,
        b: float = -30.0,
        alpha: float = 125.0,
        beta: float = 46.0,
        low_clip: float = 0.01,
        high_clip: float = 0.99,
    ):
        super().__init__(name="retinex")
        self.sigma_list = sigma_list or [15, 80, 250]
        self.G = G
        self.b = b
        self.alpha = alpha
        self.beta = beta
        self.low_clip = low_clip
        self.high_clip = high_clip

    def _single_scale_retinex(
        self, image_float: np.ndarray, sigma: float
    ) -> np.ndarray:
        """
        Single Scale Retinex (SSR) for one sigma.
        SSR(x,y) = log(I(x,y)) - log(F(x,y) * I(x,y))
        where F is a Gaussian filter.
        """
        # Estimate illumination via Gaussian blur
        blur = cv2.GaussianBlur(image_float, (0, 0), sigma)
        # Log-domain subtraction; add 1 to avoid log(0)
        retinex = np.log1p(image_float) - np.log1p(blur)
        return retinex

    def _multi_scale_retinex(self, image_float: np.ndarray) -> np.ndarray:
        """MSR = average of SSR across all scales."""
        msr = np.zeros_like(image_float)
        for sigma in self.sigma_list:
            msr += self._single_scale_retinex(image_float, sigma)
        msr /= len(self.sigma_list)
        return msr

    def _color_restoration(self, image_float: np.ndarray) -> np.ndarray:
        """
        Color Restoration Function (CRF):
        CRF(x,y) = beta * (log(alpha * I_c(x,y)) - log(sum_c(I_c(x,y))))
        Prevents the gray-world effect in MSR.
        """
        img_sum = np.sum(image_float, axis=2, keepdims=True) + 1e-6
        crf = self.beta * (np.log1p(self.alpha * image_float) - np.log1p(img_sum))
        return crf

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        """Percentile-based normalization to [0, 255]."""
        low = np.percentile(array, self.low_clip * 100)
        high = np.percentile(array, self.high_clip * 100)
        array = np.clip(array, low, high)
        # Scale to [0, 255]
        array = (array - low) / (high - low + 1e-6) * 255.0
        return array

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Full MSRCR pipeline:
        1. Convert to float, add small offset to avoid log(0)
        2. Compute MSR in log domain
        3. Multiply by Color Restoration Function
        4. Scale G and offset b
        5. Percentile normalize and clip to uint8
        """
        image = self._validate_input(image)

        # Convert to float with small offset
        image_float = image.astype(np.float64) + 1.0

        # Multi-scale retinex
        msr = self._multi_scale_retinex(image_float)

        # Color restoration
        crf = self._color_restoration(image_float)

        # MSRCR = G * (MSR * CRF + b)
        msrcr = self.G * (msr * crf + self.b)

        # Normalize each channel independently
        for c in range(3):
            msrcr[:, :, c] = self._normalize(msrcr[:, :, c])

        enhanced = np.clip(msrcr, 0, 255).astype(np.uint8)
        return enhanced


class SSREnhancer(RetinexEnhancer):
    """Single Scale Retinex (simplified variant)."""

    def __init__(self, sigma: float = 80.0, **kwargs):
        super().__init__(sigma_list=[sigma], **kwargs)
        self.name = "retinex_ssr"
