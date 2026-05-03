"""
rlrf/rewards/composite.py
=========================
Composite reward aggregation — paper §3.2.

Formula (paper):
    R_total = Σ_{i=1}^{K} w_i · R_i

Each R_i ∈ [−1, 1] and the weights w_i are configurable via RewardConfig.
The final R_total is NOT clipped (the paper sums weighted rewards; the
individual components are already bounded).

This module:
    1. Instantiates all reward components from a RewardConfig.
    2. Renders the generated SVG with SVGRenderer.
    3. Calls each reward component in sequence.
    4. Returns the weighted sum R_total.

The CompositeReward object is the single entry point used by the
GRPO training loop (rlrf/training/rlrf.py).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

from ..config import RewardConfig
from ..rendering import SVGRenderer
from .base import RewardFn
from .image_l2 import L2Reward, L2CannyReward
from .semantic import DreamSimReward, LPIPSReward
from .length import LengthReward

logger = logging.getLogger(__name__)


class CompositeReward:
    """Aggregates multiple reward signals into a single scalar per rollout.

    Args:
        config:     RewardConfig with weights and flags.
        renderer:   SVGRenderer instance (shared, thread-safe for reads).
        device:     Torch device string for perceptual models.
    """

    def __init__(
        self,
        config: RewardConfig,
        renderer: Optional[SVGRenderer] = None,
        device: str = "cuda",
    ) -> None:
        self.config   = config
        self.renderer = renderer or SVGRenderer(enforce_viewbox=True)
        self.device   = device

        # ── Instantiate active reward components ──────────────────────────
        self._components: list[tuple[float, str, RewardFn]] = []

        if config.w_img_l2 > 0:
            self._components.append((config.w_img_l2, "l2", L2Reward()))

        if config.w_img_l2_canny > 0:
            canny_rw = L2CannyReward(
                canny_low=config.canny_low,
                canny_high=config.canny_high,
                dil_kernel=config.dilation_kernel,
                dil_iters=config.dilation_iters,
                blur_kernel=config.blur_kernel,
            )
            self._components.append((config.w_img_l2_canny, "l2_canny", canny_rw))

        if config.w_dreamsim > 0 and config.use_dreamsim:
            self._components.append(
                (config.w_dreamsim, "dreamsim",
                 DreamSimReward(device=device, use_canny=False))
            )

        if config.w_dreamsim_canny > 0 and config.use_dreamsim:
            self._components.append(
                (config.w_dreamsim_canny, "dreamsim_canny",
                 DreamSimReward(device=device, use_canny=True,
                                canny_low=config.canny_low,
                                canny_high=config.canny_high,
                                dil_kernel=config.dilation_kernel,
                                dil_iters=config.dilation_iters,
                                blur_kernel=config.blur_kernel))
            )

        if config.use_lpips and not config.use_dreamsim:
            # LPIPS as semantic fallback when DreamSim is disabled
            w = config.w_dreamsim + config.w_dreamsim_canny  # absorb both weights
            if w > 0:
                self._components.append(
                    (w, "lpips", LPIPSReward(device=device))
                )

        if config.w_length > 0:
            self._components.append(
                (config.w_length, "length", LengthReward())
            )

        names = [name for _, name, _ in self._components]
        logger.info("CompositeReward components: %s", names)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        ref_image: np.ndarray,
        svg_pred: str,
        gt_length: int = 500,
    ) -> dict[str, float]:
        """Compute composite reward for a single (ref_image, svg_pred) pair.

        Steps:
            1. Render svg_pred → pred_image via CairoSVG.
            2. Compute each component reward.
            3. Return weighted sum and per-component breakdown.

        Args:
            ref_image: Ground-truth image, uint8 [H, W, 3].
            svg_pred:  Generated SVG string (raw model output).
            gt_length: Ground-truth SVG token/char count (for R_len).

        Returns:
            dict with keys:
                "total"        – weighted sum R_total
                "<component>"  – individual reward value for each component
        """
        pred_image = self.renderer.render(svg_pred)

        breakdown: dict[str, float] = {}
        total = 0.0

        for weight, name, fn in self._components:
            if name == "length":
                # Length reward doesn't use images
                val = fn(ref_image, pred_image,
                         svg_pred=svg_pred, gt_length=gt_length)
            else:
                val = fn(ref_image, pred_image)

            breakdown[name] = val
            total += weight * val

        breakdown["total"] = total
        return breakdown

    def reward_scalar(
        self,
        ref_image: np.ndarray,
        svg_pred: str,
        gt_length: int = 500,
    ) -> float:
        """Convenience method: return only R_total as a float."""
        return self(ref_image, svg_pred, gt_length)["total"]

    def batch_rewards(
        self,
        ref_images: list[np.ndarray],
        svg_preds: list[str],
        gt_lengths: list[int],
    ) -> list[float]:
        """Compute R_total for a list of (ref, pred) pairs.

        Used by the GRPO training loop to score all G rollouts in a group.
        """
        return [
            self.reward_scalar(ref, svg, gtl)
            for ref, svg, gtl in zip(ref_images, svg_preds, gt_lengths)
        ]
