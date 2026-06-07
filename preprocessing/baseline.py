"""
Baseline Enhancer — No Enhancement
====================================
Returns the original image unchanged.
Serves as the control condition in all experiments.
"""

import numpy as np
from preprocessing.base_enhancer import BaseEnhancer


class BaselineEnhancer(BaseEnhancer):
    """
    Identity enhancer: returns the original image unchanged.
    Used as the control/baseline in all comparative experiments.
    """

    def __init__(self):
        super().__init__(name="baseline")

    def enhance(self, image: np.ndarray) -> np.ndarray:
        return image.copy()
