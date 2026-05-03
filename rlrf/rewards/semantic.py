"""
rlrf/rewards/semantic.py
========================
Semantic Similarity Rewards (paper §3.2).

Im2SVG mode (this file):
    • DreamSim    — concatenated CLIP+OpenCLIP+DINOv2 ViT-B/16 features
                    → linear projection → cosine similarity.
                    sim ∈ [0, 2]; R_sim = 1 − sim ∈ [−1, 1].
    • DreamSim Canny — same but applied to Canny edge maps of both images.
    • LPIPS fallback — used when DreamSim is too memory-intensive (Kaggle).

Text2SVG mode (not primary here but included for completeness):
    • CLIPReward  — cosine similarity between text prompt and rendered image.

DreamSim reference: Fu et al., 2023. https://github.com/ssundaram21/dreamsim
LPIPS reference: Zhang et al., 2018. https://github.com/richzhang/PerceptualSimilarity
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import RewardFn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DreamSim reward
# ---------------------------------------------------------------------------

class DreamSimReward(RewardFn):
    """Semantic reward via DreamSim (paper §3.2).

    DreamSim encodes each image using three ViT-B/16 backbones
    (CLIP, OpenCLIP, DINOv2), concatenates their features, passes
    through a linear projection, and computes cosine similarity.

    score  = 1 − cosine_similarity(embed_ref, embed_pred) ∈ [0, 2]
    R_sim  = 1 − score ∈ [−1, 1]   (higher = more similar)

    Args:
        device: "cuda" or "cpu".
        use_canny: If True, apply Canny pre-processing (DreamSim Canny variant).
        canny_low, canny_high: Canny thresholds.
        dil_kernel, dil_iters, blur_kernel: Dilation / blur params (paper §3.2).
    """

    def __init__(
        self,
        device: str = "cuda",
        use_canny: bool = False,
        canny_low: int = 50,
        canny_high: int = 150,
        dil_kernel: int = 3,
        dil_iters:  int = 1,
        blur_kernel: int = 13,
    ) -> None:
        self.device = device
        self.use_canny = use_canny
        self.canny_low  = canny_low
        self.canny_high = canny_high
        self.dil_kernel  = dil_kernel
        self.dil_iters   = dil_iters
        self.blur_kernel = blur_kernel

        self._model = None  # lazy load to avoid import cost at startup

    def _load(self):
        """Lazy-load DreamSim model (downloads ~1 GB on first call)."""
        if self._model is not None:
            return
        try:
            from dreamsim import dreamsim
            self._model, self._preprocess = dreamsim(
                pretrained=True, device=self.device
            )
            self._model.eval()
            logger.info("DreamSim loaded on %s.", self.device)
        except ImportError:
            raise ImportError(
                "dreamsim is not installed. Install with: pip install dreamsim\n"
                "Or set use_dreamsim=False in RewardConfig to use LPIPS instead."
            )

    def _embed(self, img_np: np.ndarray) -> torch.Tensor:
        """Preprocess numpy image → DreamSim embedding."""
        import cv2
        if self.use_canny:
            gray  = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, self.canny_low, self.canny_high)
            kernel = np.ones((self.dil_kernel, self.dil_kernel), np.uint8)
            edges  = cv2.dilate(edges, kernel, iterations=self.dil_iters)
            edges  = cv2.GaussianBlur(edges, (self.blur_kernel, self.blur_kernel), 0)
            img_np = np.stack([edges, edges, edges], axis=-1)

        pil = Image.fromarray(img_np.astype(np.uint8))
        tensor = self._preprocess(pil).to(self.device)   # (1, C, H, W)
        with torch.no_grad():
            emb = self._model.embed(tensor)
        return emb  # (1, D)

    def __call__(self, ref: np.ndarray, pred: np.ndarray, **kwargs) -> float:
        self._load()
        emb_ref  = self._embed(ref)
        emb_pred = self._embed(pred)
        # DreamSim returns distance in [0, 2]; convert to similarity in [-1, 1]
        sim_dist = self._model.dist(emb_ref, emb_pred).item()  # distance ∈ [0, 2]
        reward = 1.0 - sim_dist   # ∈ [-1, 1]
        return float(np.clip(reward, -1.0, 1.0))


# ---------------------------------------------------------------------------
# LPIPS reward (Kaggle-friendly fallback)
# ---------------------------------------------------------------------------

class LPIPSReward(RewardFn):
    """Perceptual similarity reward via LPIPS (fallback for DreamSim).

    LPIPS distance d ∈ [0, ∞). We convert to a bounded reward:
        R = clip(1 − 2*d, −1, 1)
    so d=0 → R=1 (identical), d=0.5 → R=0, d≥1 → R=−1.

    References: Zhang et al. (2018).
    """

    def __init__(self, net: str = "vgg", device: str = "cuda") -> None:
        self.net    = net
        self.device = device
        self._fn    = None

    def _load(self):
        if self._fn is not None:
            return
        try:
            import lpips
            self._fn = lpips.LPIPS(net=self.net).to(self.device)
            self._fn.eval()
        except ImportError:
            raise ImportError("lpips not installed. Run: pip install lpips")

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert uint8 [H,W,3] → float32 [-1,1] tensor [1,3,H,W]."""
        t = torch.from_numpy(img).float() / 127.5 - 1.0
        return t.permute(2, 0, 1).unsqueeze(0).to(self.device)

    def __call__(self, ref: np.ndarray, pred: np.ndarray, **kwargs) -> float:
        self._load()
        t_ref  = self._to_tensor(ref)
        t_pred = self._to_tensor(pred)
        with torch.no_grad():
            dist = self._fn(t_ref, t_pred).item()
        reward = 1.0 - 2.0 * dist
        return float(np.clip(reward, -1.0, 1.0))


# ---------------------------------------------------------------------------
# CLIP reward (Text2SVG — included for completeness)
# ---------------------------------------------------------------------------

class CLIPReward(RewardFn):
    """Cosine similarity between text prompt and rendered SVG image (§3.2).

    Used in Text2SVG mode. For Im2SVG the semantic reward is DreamSim.

    Args:
        model_name: OpenCLIP model identifier (default: ViT-B-32).
        device: "cuda" or "cpu".
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device     = device
        self._model     = None
        self._preprocess = None
        self._tokenize  = None

    def _load(self):
        if self._model is not None:
            return
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        model.eval()
        self._model      = model
        self._preprocess = preprocess
        self._tokenize   = open_clip.get_tokenizer(self.model_name)

    def __call__(
        self,
        ref: np.ndarray,          # unused for Text2SVG (no reference image)
        pred: np.ndarray,
        text_prompt: str = "",
        **kwargs,
    ) -> float:
        """
        Args:
            ref:  Ignored in Text2SVG mode.
            pred: Rendered SVG image uint8 [H, W, 3].
            text_prompt: The conditioning text string.
        """
        if not text_prompt:
            return 0.0
        self._load()

        pil = Image.fromarray(pred.astype(np.uint8))
        img_t  = self._preprocess(pil).unsqueeze(0).to(self.device)
        tok    = self._tokenize([text_prompt]).to(self.device)

        with torch.no_grad():
            img_feat  = self._model.encode_image(img_t)
            text_feat = self._model.encode_text(tok)
            img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat * text_feat).sum(dim=-1).item()

        # CLIP cosine sim ∈ [−1, 1]; already bounded
        return float(np.clip(sim, -1.0, 1.0))
