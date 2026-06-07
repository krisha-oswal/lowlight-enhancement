"""
Histogram Equalization (HE)
============================
Applies global histogram equalization to the luminance (Y) channel
in YCrCb color space to avoid color distortion.

Design choice: Operating in YCrCb rather than directly on RGB prevents
color shifting artifacts while boosting global contrast.
"""

import numpy as np
import cv2
from preprocessing.base_enhancer import BaseEnhancer


class HistogramEqualizationEnhancer(BaseEnhancer):
    """
    Global Histogram Equalization applied on luminance channel.

    Strengths:
      - Simple, deterministic, fast
      - Works well when histogram is spread across full range

    Weaknesses:
      - Global: over-enhances already bright regions
      - Can amplify noise in very dark areas
      - No local adaptivity
    """

    def __init__(self):
        super().__init__(name="histogram_eq")

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Steps:
        1. Convert BGR → YCrCb
        2. Equalize the Y (luma) channel
        3. Convert back YCrCb → BGR
        """
        image = self._validate_input(image)

        # Convert to YCrCb (luma + chroma)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        channels = list(cv2.split(ycrcb))

        # Apply histogram equalization to Y channel only
        channels[0] = cv2.equalizeHist(channels[0])

        # Merge and convert back
        ycrcb_eq = cv2.merge(channels)
        enhanced = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

        return enhanced
