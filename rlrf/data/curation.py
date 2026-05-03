"""
rlrf/data/curation.py
=====================
High-entropy data filtering for the RLRF training set (paper §4.1).

Paper: "We begin by filtering the SVG-Stack dataset to select 20k
high-entropy samples that are rich in visual detail and SVG complexity
(each with over 500 tokens)."

Two filtering criteria:
    1. Token count threshold (min_gt_tokens ≥ 500): ensures the SVG is
       complex enough to benefit from rendering feedback.
    2. Entropy-based visual filtering: estimate image entropy as a proxy
       for visual complexity (avoids simple, uniform-color SVGs).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token / character count filter
# ---------------------------------------------------------------------------

def filter_by_length(
    examples: list[dict],
    svg_key: str = "svg",
    min_tokens: int = 500,
    tokenizer: Optional[Callable] = None,
) -> list[dict]:
    """Keep only examples whose SVG meets the minimum length threshold.

    Args:
        examples:   List of dataset dicts, each containing a SVG string.
        svg_key:    Key for the SVG field in each dict.
        min_tokens: Minimum number of tokens (or chars if tokenizer is None).
        tokenizer:  Optional HF tokenizer. If None, character count is used.

    Returns:
        Filtered list.
    """
    kept = []
    for ex in examples:
        svg = ex.get(svg_key, "")
        if not svg:
            continue
        if tokenizer is not None:
            length = len(tokenizer(svg, add_special_tokens=False)["input_ids"])
        else:
            length = len(svg)  # character count as proxy
        if length >= min_tokens:
            kept.append({**ex, "_svg_length": length})

    logger.info(
        "Length filter: kept %d / %d (min_tokens=%d)",
        len(kept), len(examples), min_tokens,
    )
    return kept


# ---------------------------------------------------------------------------
# Visual entropy filter
# ---------------------------------------------------------------------------

def image_entropy(img: np.ndarray) -> float:
    """Compute Shannon entropy of the grayscale pixel histogram.

    Higher entropy → more complex, visually diverse image.
    Used to exclude trivially simple (blank, uniform) SVGs.
    """
    gray = np.mean(img, axis=-1).astype(np.uint8)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist / (hist.sum() + 1e-8)
    # Shannon entropy: H = -Σ p log2(p)
    nonzero = hist[hist > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def filter_by_entropy(
    examples: list[dict],
    image_key: str = "image",
    min_entropy: float = 4.0,
) -> list[dict]:
    """Remove low-entropy (visually trivial) samples.

    Args:
        examples:    Dataset dicts with a PIL Image or np.ndarray in image_key.
        image_key:   Key for the image field.
        min_entropy: Minimum Shannon entropy (bits). Default 4.0 keeps ~top 80%.

    Returns:
        Filtered list.
    """
    kept = []
    for ex in examples:
        img = ex.get(image_key)
        if img is None:
            continue
        if isinstance(img, Image.Image):
            arr = np.array(img.convert("RGB"))
        else:
            arr = np.array(img)
        H = image_entropy(arr)
        if H >= min_entropy:
            kept.append({**ex, "_entropy": H})

    logger.info(
        "Entropy filter: kept %d / %d (min_entropy=%.2f)",
        len(kept), len(examples), min_entropy,
    )
    return kept


# ---------------------------------------------------------------------------
# Combined curation pipeline
# ---------------------------------------------------------------------------

def curate_dataset(
    examples: list[dict],
    svg_key: str = "svg",
    image_key: str = "image",
    min_tokens: int = 500,
    min_entropy: float = 4.0,
    max_samples: Optional[int] = None,
    tokenizer: Optional[Callable] = None,
    skip_entropy: bool = False,
) -> list[dict]:
    """Run the full curation pipeline: length filter → entropy filter → cap.

    Args:
        examples:     Raw dataset records.
        svg_key:      Field name for SVG strings.
        image_key:    Field name for images.
        min_tokens:   Minimum SVG token count (paper: 500).
        min_entropy:  Minimum image entropy (visual complexity proxy).
        max_samples:  Cap on output size (paper Im2SVG RLRF: 20k).
        tokenizer:    Optional HF tokenizer for precise token counting.
        skip_entropy: If True, skip entropy filter (useful when images
                      are not pre-loaded into memory).

    Returns:
        Curated list of dicts, sorted by descending entropy.
    """
    # Step 1: Token length filter
    curated = filter_by_length(examples, svg_key, min_tokens, tokenizer)

    # Step 2: Visual entropy filter (requires images in memory)
    if not skip_entropy and image_key in (curated[0] if curated else {}):
        curated = filter_by_entropy(curated, image_key, min_entropy)

    # Step 3: Sort by entropy descending (prefer richer images)
    curated.sort(key=lambda x: x.get("_entropy", 0), reverse=True)

    # Step 4: Cap at max_samples
    if max_samples is not None:
        curated = curated[:max_samples]

    logger.info("Curation complete: %d samples.", len(curated))
    return curated
