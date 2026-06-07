"""
Zero-DCE — Zero-Reference Deep Curve Estimation
================================================
Zero-DCE enhances images by estimating pixel-wise light-enhancement
curves using a lightweight CNN, without requiring paired training data.

Paper: "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement"
       Guo et al., CVPR 2020. https://arxiv.org/abs/2001.06826

Architecture:
  - 7-layer CNN with skip connections
  - Input: low-light image (3 channels)
  - Output: 24 channel maps → 8 quadratic curve parameter maps (3 × 8)
  - Enhancement: iterative curve application (n_iters times)

Curve function (per pixel, per channel):
  LE(I, alpha) = I + alpha * I * (1 - I)
  Applied iteratively to progressively brighten the image.

We include the full model architecture so this runs without
downloading external weights. For best results, load pretrained
weights from the original paper (see comments below).
"""

import os
import logging
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing.base_enhancer import BaseEnhancer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Zero-DCE Network Architecture
# ─────────────────────────────────────────────

class ZeroDCENet(nn.Module):
    """
    Lightweight CNN for Zero-Reference Deep Curve Estimation.

    Architecture (as in original paper):
      - 7 convolutional layers, each 32 filters, 3×3 kernel, ReLU
      - Skip connections: layer 1–6, layer 2–5, layer 3–4
      - Final layer: 24 output channels (8 curve maps × 3 channels)
      - Tanh activation on output (curves in [-1, 1])
    """

    def __init__(self, n_filters: int = 32, n_iters: int = 8):
        super().__init__()
        self.n_iters = n_iters

        # Encoder path
        self.conv1 = nn.Conv2d(3, n_filters, 3, padding=1)
        self.conv2 = nn.Conv2d(n_filters, n_filters, 3, padding=1)
        self.conv3 = nn.Conv2d(n_filters, n_filters, 3, padding=1)
        self.conv4 = nn.Conv2d(n_filters, n_filters, 3, padding=1)

        # Decoder path (with skip connections)
        self.conv5 = nn.Conv2d(n_filters * 2, n_filters, 3, padding=1)
        self.conv6 = nn.Conv2d(n_filters * 2, n_filters, 3, padding=1)
        self.conv7 = nn.Conv2d(n_filters * 2, 24, 3, padding=1)  # 8 iters × 3 channels

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        Returns:
          enhanced: Final enhanced image (B, 3, H, W) in [0,1]
          curves: List of curve parameter maps
        """
        # Encoder
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))

        # Decoder with skip connections (concat)
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        x_r = self.tanh(self.conv7(torch.cat([x1, x6], dim=1)))

        # Split into 8 curve parameter maps (each 3-channel)
        # x_r shape: (B, 24, H, W) → 8 × (B, 3, H, W)
        curve_maps = torch.split(x_r, 3, dim=1)  # 8 maps of shape (B,3,H,W)

        # Apply curve iteratively
        enhanced = x
        curves = []
        for alpha in curve_maps:
            # LE(I, alpha) = I + alpha * I * (1 - I)
            enhanced = enhanced + alpha * (enhanced - enhanced * enhanced)
            enhanced = torch.clamp(enhanced, 0, 1)
            curves.append(alpha)

        return enhanced, curves


# ─────────────────────────────────────────────
# Zero-DCE Enhancer Wrapper
# ─────────────────────────────────────────────

class ZeroDCEEnhancer(BaseEnhancer):
    """
    Zero-DCE enhancer wrapping the ZeroDCENet model.

    Args:
        weights_path: Path to pretrained .pth weights file.
                      If None or not found, uses randomly initialized
                      weights (results will be suboptimal but runnable).
        n_filters: Number of filters in each conv layer
        n_iters: Number of curve application iterations
        device: 'cuda' or 'cpu'

    Note on weights:
      Pretrained weights can be obtained from the official repo:
      https://github.com/Li-Chongyi/Zero-DCE
      Save as 'preprocessing/zero_dce_weights.pth'
      The model expects the state_dict key format from that repo.
    """

    def __init__(
        self,
        weights_path: str = None,
        n_filters: int = 32,
        n_iters: int = 8,
        device: str = "cpu",
    ):
        super().__init__(name="zero_dce")
        self.device = torch.device(device)
        self.model = ZeroDCENet(n_filters=n_filters, n_iters=n_iters).to(self.device)
        self.model.eval()

        self._weights_loaded = False
        if weights_path and Path(weights_path).exists():
            self._load_weights(weights_path)
        else:
            logger.warning(
                "Zero-DCE: No pretrained weights found. "
                "Results will use randomly initialized network. "
                "Download weights from https://github.com/Li-Chongyi/Zero-DCE "
                "and set paths.zero_dce_weights in config.yaml"
            )

    def _load_weights(self, weights_path: str):
        """Load pretrained weights with key remapping for compatibility."""
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            # Handle different checkpoint formats
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Try direct load, then with prefix strip
            try:
                self.model.load_state_dict(state_dict, strict=False)
            except RuntimeError:
                # Strip common prefixes
                new_dict = {}
                for k, v in state_dict.items():
                    new_key = k.replace("module.", "").replace("net.", "")
                    new_dict[new_key] = v
                self.model.load_state_dict(new_dict, strict=False)

            self._weights_loaded = True
            logger.info(f"Zero-DCE weights loaded from {weights_path}")
        except Exception as e:
            logger.error(f"Failed to load Zero-DCE weights: {e}")

    @torch.no_grad()
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image using Zero-DCE.
        Steps:
        1. BGR → RGB, normalize to [0,1], add batch dim
        2. Forward pass through ZeroDCENet
        3. Remove batch dim, scale to [0,255], convert to BGR uint8
        """
        image = self._validate_input(image)
        h, w = image.shape[:2]

        # Preprocess: BGR → RGB → float tensor [0,1]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0

        # HWC → CHW → BCHW
        tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)

        # Forward pass
        enhanced_tensor, _ = self.model(tensor)

        # Postprocess: BCHW → HWC → [0,255] uint8
        enhanced_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced_np = np.clip(enhanced_np * 255.0, 0, 255).astype(np.uint8)
        enhanced_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)

        return enhanced_bgr
