"""
rlrf/rewards/length.py
======================
Code Efficiency Reward (R_len) — paper §3.2.

Formula (paper):
    R_len = clip(1 − ((1/L_gt) · max(0, L_pred − L_gt/2))², −1, 1)

Intuition:
    • No penalty if L_pred ≤ L_gt/2  (the model is very concise).
    • Quadratic penalty as L_pred grows beyond half the ground-truth length.
    • Allows moderate variation while discouraging overly long/redundant SVGs.
    • Hard clipped to [−1, 1].

Note: L_pred and L_gt are measured in **tokens** (using the model tokenizer).
For simplicity during reward computation, we use character count as a proxy
if the tokenizer is unavailable; pass `use_char_count=True`.
"""

from __future__ import annotations

import numpy as np
from .base import RewardFn


class LengthReward(RewardFn):
    """Code efficiency reward penalising excessively long SVG outputs.

    Args:
        use_char_count: If True, measure length in characters instead of
            tokens (faster, no tokenizer dependency; good approximation).
    """

    def __init__(self, use_char_count: bool = True) -> None:
        self.use_char_count = use_char_count

    def __call__(
        self,
        ref: np.ndarray,   # unused — length is a code property
        pred: np.ndarray,  # unused — length is a code property
        svg_pred: str = "",
        gt_length: int = 500,  # default from data-curation threshold
        **kwargs,
    ) -> float:
        """Compute length reward.

        Args:
            ref, pred: Image arrays (not used by this reward).
            svg_pred:  Generated SVG string.
            gt_length: Ground-truth SVG length in tokens (or chars).

        Returns:
            float in [−1, 1].
        """
        if not svg_pred or gt_length <= 0:
            return 0.0

        l_pred = len(svg_pred) if self.use_char_count else len(svg_pred.split())
        l_gt   = gt_length

        # Paper formula
        excess = max(0, l_pred - l_gt / 2)
        penalty = (excess / l_gt) ** 2
        reward = 1.0 - penalty
        return float(np.clip(reward, -1.0, 1.0))


def compute_dynamic_max_length(
    gt_lengths: list[int],
    threshold: int = 128,
) -> int:
    """Compute per-batch dynamic maximum generation length (App. C.3).

    Paper: "For each batch, estimate the required output length using the
    ground truth SVGs and set the maximum length to the longest sample
    plus a small threshold t."

    Args:
        gt_lengths: List of ground-truth SVG token counts in the batch.
        threshold:  Safety margin added to the max length.

    Returns:
        int: Maximum number of new tokens to generate for this batch.
    """
    if not gt_lengths:
        return 512
    return int(max(gt_lengths)) + threshold
