"""
rlrf/rewards/base.py
====================
Abstract base class for reward functions.

All rewards must:
    - Return a float in [-1, 1] (paper convention).
    - Accept (ref_image, pred_image) as numpy uint8 arrays [H, W, 3].
    - Be stateless or carry read-only state (e.g. pretrained weights).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class RewardFn(ABC):
    """Abstract base class for individual reward functions."""

    @abstractmethod
    def __call__(self, ref: np.ndarray, pred: np.ndarray, **kwargs) -> float:
        """Compute a scalar reward in [-1, 1].

        Args:
            ref:  Reference (ground-truth) image, shape (H, W, 3) uint8.
            pred: Predicted (rendered) image, shape (H, W, 3) uint8.
            **kwargs: Optional extra data (e.g. svg_text, gt_length).

        Returns:
            float in [-1, 1] where higher is better.
        """
        ...

    def batch(
        self,
        refs: list[np.ndarray],
        preds: list[np.ndarray],
        **kwargs,
    ) -> list[float]:
        """Compute rewards for a batch. Default: loop over pairs."""
        return [self(r, p, **kwargs) for r, p in zip(refs, preds)]
