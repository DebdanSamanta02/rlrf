"""
rlrf/utils/metrics.py
=====================
Evaluation metrics for Im2SVG (paper §4.2).

Metrics reported in the paper (Table 1):
    • MSE          – Mean Squared Error (pixel-level, ↓ better)
    • SSIM         – Structural Similarity Index (↑ better)
    • DINO Score   – DINOv2 cosine similarity (↑ better)
    • LPIPS        – Learned Perceptual Image Patch Similarity (↓ better)
    • Code Efficiency – negated mean token-count delta vs ground truth

All image metrics operate on uint8 [H, W, 3] numpy arrays.
Images are resized to a common size (224×224) before comparison.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as _ssim_fn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _resize(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize uint8 [H, W, 3] → [size, size, 3]."""
    pil = Image.fromarray(img).resize((size, size), Image.LANCZOS)
    return np.array(pil)


def _to_float(img: np.ndarray) -> np.ndarray:
    """Convert uint8 [0,255] → float32 [0,1]."""
    return img.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def mse(ref: np.ndarray, pred: np.ndarray, size: int = 224) -> float:
    """Mean Squared Error between reference and predicted images.

    Paper reports MSE × 100 (percentage scale); we return raw [0,1] range.
    Multiply by 100 to match Table 1 values.

    Args:
        ref, pred: uint8 [H, W, 3] images.
        size:      Resize target before comparison.

    Returns:
        float — lower is better.
    """
    r = _to_float(_resize(ref,  size))
    p = _to_float(_resize(pred, size))
    return float(np.mean((r - p) ** 2))


def ssim(ref: np.ndarray, pred: np.ndarray, size: int = 224) -> float:
    """Structural Similarity Index (SSIM).

    Returns:
        float in [−1, 1] — higher is better. Paper reports as percentage.
    """
    r = _resize(ref,  size)
    p = _resize(pred, size)
    val, _ = _ssim_fn(r, p, full=True, channel_axis=-1,
                      data_range=255)
    return float(val)


def dino_score(
    ref: np.ndarray,
    pred: np.ndarray,
    size: int = 224,
    device: str = "cuda",
    _model_cache: dict = {},
) -> float:
    """DINOv2 cosine similarity (perceptual metric).

    Paper uses DINOv2 ViT-B/14 features.

    Returns:
        float in [0, 1] — higher is better.
    """
    import torch
    import torchvision.transforms as T

    if "model" not in _model_cache:
        try:
            _model_cache["model"] = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True
            ).to(device).eval()
            _model_cache["transform"] = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        except Exception as exc:
            logger.warning("Could not load DINOv2: %s. Returning 0.", exc)
            return 0.0

    model     = _model_cache["model"]
    transform = _model_cache["transform"]

    def embed(img_np: np.ndarray) -> torch.Tensor:
        pil = Image.fromarray(img_np)
        t   = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(t)
        return feat / feat.norm(dim=-1, keepdim=True)

    f_ref  = embed(_resize(ref,  224))
    f_pred = embed(_resize(pred, 224))
    return float((f_ref * f_pred).sum().item())


def lpips_score(
    ref: np.ndarray,
    pred: np.ndarray,
    device: str = "cuda",
    _model_cache: dict = {},
) -> float:
    """LPIPS perceptual distance.

    Returns:
        float — lower is better.
    """
    import torch

    if "fn" not in _model_cache:
        try:
            import lpips
            _model_cache["fn"] = lpips.LPIPS(net="vgg").to(device).eval()
        except ImportError:
            logger.warning("lpips not installed. Returning 0.")
            return 0.0

    fn = _model_cache["fn"]

    def to_t(img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(_resize(img, 224)).float() / 127.5 - 1.0
        return t.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        dist = fn(to_t(ref), to_t(pred)).item()
    return float(dist)


def code_efficiency(
    pred_len: int,
    gt_len: int,
) -> float:
    """Code Efficiency metric (paper §4.2).

    Defined as the negated mean difference between ground-truth and
    predicted SVG token counts.  Positive = model is more compact than GT.
    Values near zero are ideal; large negatives = too verbose.

    Args:
        pred_len: Predicted SVG token count.
        gt_len:   Ground-truth SVG token count.

    Returns:
        float: gt_len − pred_len  (positive = compact, negative = verbose)
    """
    return float(gt_len - pred_len)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    refs: list[np.ndarray],
    preds: list[np.ndarray],
    pred_svgs: Optional[list[str]] = None,
    gt_svgs: Optional[list[str]] = None,
    device: str = "cuda",
    compute_dino: bool = True,
    compute_lpips: bool = True,
) -> dict[str, float]:
    """Compute all metrics over a batch and return averages.

    Args:
        refs:         List of ground-truth images.
        preds:        List of rendered predicted images.
        pred_svgs:    Generated SVG strings (for code efficiency).
        gt_svgs:      Ground-truth SVG strings (for code efficiency).
        device:       Torch device.
        compute_dino: Include DINO Score (loads ~300 MB model).
        compute_lpips:Include LPIPS (loads ~100 MB model).

    Returns:
        dict with keys: mse, ssim, dino_score, lpips, code_efficiency.
        All values are floats; paper-style scale (MSE ×100).
    """
    mse_vals, ssim_vals, dino_vals, lpips_vals, ce_vals = [], [], [], [], []

    for i, (ref, pred) in enumerate(zip(refs, preds)):
        mse_vals.append(mse(ref, pred) * 100)   # paper scale
        ssim_vals.append(ssim(ref, pred) * 100)  # paper scale (%)

        if compute_dino:
            dino_vals.append(dino_score(ref, pred, device=device) * 100)
        if compute_lpips:
            lpips_vals.append(lpips_score(ref, pred, device=device) * 100)

        if pred_svgs and gt_svgs:
            ce_vals.append(code_efficiency(len(pred_svgs[i]), len(gt_svgs[i])))

    result: dict[str, float] = {
        "mse":  float(np.mean(mse_vals))  if mse_vals  else 0.0,
        "ssim": float(np.mean(ssim_vals)) if ssim_vals else 0.0,
    }
    if dino_vals:
        result["dino_score"] = float(np.mean(dino_vals))
    if lpips_vals:
        result["lpips"] = float(np.mean(lpips_vals))
    if ce_vals:
        result["code_efficiency"] = float(np.mean(ce_vals))

    return result
