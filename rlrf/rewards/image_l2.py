"""
rlrf/rewards/image_l2.py
========================
Image Reconstruction Reward (R_img) — paper §3.2, Eq. 5.

Two variants are implemented:
    1. L2       — pixel-level L2 on normalised images.
    2. L2 Canny — same L2 on Canny edge maps (edge-aware variant).

Formula (paper Eq. 5):
    I_norm = (I − μ) / σ        (per-image normalisation)
    R_img  = clip(1 − (1/N) ‖I_in^norm − I_pred^norm‖₂², −1, 1)

The normalisation makes the reward:
    • scale-invariant (brightness / contrast shifts don't dominate),
    • bounded in [−1, 1] by the hard clip.

Canny variant (paper §3.2):
    Apply Canny edge detector, then dilate (3×3 kernel, 1 iter)
    and Gaussian blur (kernel 13) to both images before L2.
    This "enhances structural alignment based on visual fidelity."
"""

from __future__ import annotations

import numpy as np
import cv2

from .base import RewardFn


def _normalize(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize image to zero mean and unit variance (paper Eq. 5)."""
    img = img.astype(np.float32)
    mu  = img.mean()
    sig = img.std()
    return (img - mu) / (sig + eps)


def _l2_reward(ref: np.ndarray, pred: np.ndarray) -> float:
    """Core L2 reward on (already normalised) float arrays."""
    n = ref.size  # total pixels × channels
    sq_dist = np.sum((ref - pred) ** 2) / n
    reward = 1.0 - sq_dist
    return float(np.clip(reward, -1.0, 1.0))


def _canny_edge(
    img: np.ndarray,
    low: int = 50,
    high: int = 150,
    dil_kernel: int = 3,
    dil_iters: int = 1,
    blur_kernel: int = 13,
) -> np.ndarray:
    """Apply Canny + dilation + Gaussian blur (paper §3.2).

    Args:
        img: RGB uint8 [H, W, 3].
        low, high: Canny thresholds.
        dil_kernel: Dilation kernel size (paper: 3).
        dil_iters:  Dilation iterations (paper: 1).
        blur_kernel: Gaussian blur kernel size (paper: 13).

    Returns:
        Float32 edge map [H, W, 3] (replicated across channels).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)

    # Dilate edges to give slight tolerance (paper §3.2)
    kernel = np.ones((dil_kernel, dil_kernel), np.uint8)
    edges  = cv2.dilate(edges, kernel, iterations=dil_iters)

    # Gaussian blur to smooth edges
    edges = cv2.GaussianBlur(edges, (blur_kernel, blur_kernel), 0)

    # Expand to 3 channels to match the base reward interface
    edges_3c = np.stack([edges, edges, edges], axis=-1).astype(np.float32)
    return edges_3c


# ---------------------------------------------------------------------------
# Reward classes
# ---------------------------------------------------------------------------

class L2Reward(RewardFn):
    """Pixel-level L2 image reconstruction reward (paper §3.2 Eq. 5).

    R_img = clip(1 − (1/N) ‖I_in^norm − I_pred^norm‖₂², −1, 1)
    """

    def __call__(self, ref: np.ndarray, pred: np.ndarray, **kwargs) -> float:
        """
        Args:
            ref:  Ground-truth rendered image, uint8 [H, W, 3].
            pred: Model-generated rendered image, uint8 [H, W, 3].

        Returns:
            float in [−1, 1].
        """
        ref_norm  = _normalize(ref)
        pred_norm = _normalize(pred)
        return _l2_reward(ref_norm, pred_norm)


class L2CannyReward(RewardFn):
    """Edge-aware L2 reward using Canny edge detection (paper §3.2).

    Applies Canny → dilation → Gaussian blur to both images before
    computing the normalised L2 distance, emphasising structural and
    contour alignment over flat colour regions.
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        dil_kernel: int = 3,
        dil_iters: int = 1,
        blur_kernel: int = 13,
    ) -> None:
        self.canny_low  = canny_low
        self.canny_high = canny_high
        self.dil_kernel = dil_kernel
        self.dil_iters  = dil_iters
        self.blur_kernel = blur_kernel

    def __call__(self, ref: np.ndarray, pred: np.ndarray, **kwargs) -> float:
        ref_edge  = _canny_edge(ref,  self.canny_low, self.canny_high,
                                self.dil_kernel, self.dil_iters, self.blur_kernel)
        pred_edge = _canny_edge(pred, self.canny_low, self.canny_high,
                                self.dil_kernel, self.dil_iters, self.blur_kernel)
        # Normalise the edge maps
        ref_norm  = _normalize(ref_edge)
        pred_norm = _normalize(pred_edge)
        return _l2_reward(ref_norm, pred_norm)
