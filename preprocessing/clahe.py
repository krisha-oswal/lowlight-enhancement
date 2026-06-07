"""
CLAHE — Contrast Limited Adaptive Histogram Equalization
==========================================================
Applies CLAHE on the luminance channel in LAB color space.

Design choice: CLAHE divides the image into tiles and equalizes each
locally, then uses bilinear interpolation to remove tile borders.
The clip_limit prevents over-amplification of noise.

LAB color space is preferred over YCrCb for CLAHE because L channel
is perceptually uniform, making the enhancement more visually natural.
"""

import numpy as np
import cv2
from preprocessing.base_enhancer import BaseEnhancer


class CLAHEEnhancer(BaseEnhancer):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Args:
        clip_limit: Threshold for contrast limiting (higher = more contrast)
        tile_grid_size: Size of grid for histogram equalization (rows, cols)

    Strengths:
      - Local adaptivity handles non-uniform illumination
      - Clip limit prevents noise amplification
      - Widely used, well-validated in medical and night imaging

    Weaknesses:
      - Tile artifacts possible if grid too coarse
      - Still a contrast method; does not model illumination physics
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        super().__init__(name="clahe")
        self.clip_limit = clip_limit
        self.tile_grid_size = tuple(tile_grid_size)
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=self.tile_grid_size
        )

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Steps:
        1. Convert BGR → LAB
        2. Apply CLAHE to L channel
        3. Convert LAB → BGR
        """
        image = self._validate_input(image)

        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L (lightness) channel
        l_enhanced = self.clahe.apply(l_channel)

        # Merge and convert back
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return enhanced
