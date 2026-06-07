"""
Base Enhancer Interface
=======================
Abstract base class for all enhancement methods.
Ensures a consistent API across classical and learned methods.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import cv2


class BaseEnhancer(ABC):
    """
    Abstract interface for low-light image enhancement.

    All enhancers accept a BGR uint8 NumPy array and return
    a BGR uint8 NumPy array of the same spatial dimensions.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance a single low-light image.

        Args:
            image: BGR uint8 numpy array, shape (H, W, 3)

        Returns:
            Enhanced BGR uint8 numpy array, shape (H, W, 3)
        """
        pass

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Allow instances to be called directly."""
        return self.enhance(image)

    def _validate_input(self, image: np.ndarray) -> np.ndarray:
        """Ensure image is BGR uint8 (H, W, 3)."""
        if image is None or image.size == 0:
            raise ValueError("Empty image passed to enhancer")
        if image.ndim == 2:
            # Grayscale → BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel image, got shape {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _safe_enhance(self, image: np.ndarray) -> np.ndarray:
        """Validate input, enhance, and validate output."""
        image = self._validate_input(image)
        result = self.enhance(image)
        # Ensure output is valid
        if result is None or result.size == 0:
            return image
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
