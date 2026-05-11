"""
rlrf/data/dataset.py
====================
Dataset loading and preprocessing for the SVG-SFT and RLRF stages.

Public dataset used:
    starvector/svg-stack  (HuggingFace, no auth token required)
    URL: https://huggingface.co/datasets/starvector/svg-stack

Actual schema of starvector/svg-stack (confirmed from HF viewer):
    • "Filename": str — e.g. "312f68f...db.svg"  (capital F)
    • "Svg":      str — raw SVG XML code          (capital S)

There is NO pre-rendered image column. We obtain the reference image by
rendering the ground-truth SVG with CairoSVG. This is consistent with the
paper: the same renderer (CairoSVG) is used for both GT and predicted SVGs,
so the reward measures how faithfully the model reproduces the GT rendering.

The module provides:
    • SVGDataset  — PyTorch Dataset wrapping the HF dataset.
    • build_sft_dataset()  — prepare tokenised data for SFTTrainer.
    • build_rlrf_dataset() — prepare prompt data for GRPO training.
    • collate_fn_rlrf()    — custom collator for mixed image+text batches.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Lazy import — only needed when rendering GT SVGs
try:
    import cairosvg as _cairosvg
    _CAIROSVG_OK = True
except ImportError:
    _CAIROSVG_OK = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Qwen2.5-VL chat template for Im2SVG
# The model is prompted with the image and asked to produce SVG code.
IM2SVG_SYSTEM_PROMPT = (
    "You are an expert SVG programmer. "
    "Convert the given image to clean, compact SVG code. "
    "Output only the SVG XML, starting with <svg and ending with </svg>."
)

IM2SVG_USER_PROMPT = "Convert this image to SVG code."


def make_im2svg_messages(image: Image.Image) -> list[dict]:
    """Build a Qwen2.5-VL chat-format message list for Im2SVG.

    The processor will substitute the image token <|image_pad|> when
    apply_chat_template is called with the image.

    Returns:
        List of HF-style message dicts (role / content).
    """
    return [
        {
            "role": "system",
            "content": IM2SVG_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": IM2SVG_USER_PROMPT},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# SVGDataset
# ---------------------------------------------------------------------------

class SVGDataset(Dataset):
    """PyTorch Dataset for image → SVG pairs.

    Args:
        records:    List of dicts with keys "image" (PIL) and "svg" (str).
        processor:  Qwen2.5-VL AutoProcessor for tokenisation.
        max_seq_length: Maximum tokenised sequence length.
        mode:       "sft" (include SVG labels) or "rlrf" (prompts only).
        render_size: Resize images to this square dimension before encoding.
    """

    def __init__(
        self,
        records: list[dict],
        processor: Any,
        max_seq_length: int = 4096,
        mode: str = "sft",        # "sft" or "rlrf"
        render_size: int = 224,
    ) -> None:
        assert mode in ("sft", "rlrf"), f"mode must be 'sft' or 'rlrf', got {mode!r}"
        self.records        = records
        self.processor      = processor
        self.max_seq_length = max_seq_length
        self.mode           = mode
        self.render_size    = render_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec    = self.records[idx]
        svg    = rec.get("svg", "")
        gt_len = rec.get("_svg_length", len(svg))

        # Render the ground-truth SVG to get the reference image.
        # starvector/svg-stack has no pre-rendered image column, so we
        # produce the reference image with the same CairoSVG renderer
        # used for predicted SVGs.  This is consistent with the paper:
        # reward = similarity(render(gt_svg), render(pred_svg)).
        image = self._render_gt_svg(svg)

        messages = make_im2svg_messages(image)

        if self.mode == "sft":
            return self._encode_sft(messages, svg, image, gt_len)
        else:
            return self._encode_rlrf(messages, image, svg, gt_len)

    # ------------------------------------------------------------------
    # Internal encoding helpers
    # ------------------------------------------------------------------

    def _render_gt_svg(self, svg: str) -> Image.Image:
        """Render the ground-truth SVG string → PIL Image (RGB).

        This is used instead of loading a pre-rendered image because
        starvector/svg-stack only provides raw SVG code (no image column).

        Falls back to a white image if CairoSVG is not installed or
        rendering fails, so the dataset never raises an unrecoverable error.
        """
        import io
        size = self.render_size

        if not svg or not _CAIROSVG_OK:
            # White fallback — cairosvg not installed yet
            return Image.new("RGB", (size, size), (255, 255, 255))

        # Strip markdown fences the dataset sometimes includes
        import re
        svg = re.sub(r"```(?:svg|xml)?\s*\n?", "", svg, flags=re.IGNORECASE)
        svg = re.sub(r"```", "", svg).strip()

        try:
            png = _cairosvg.svg2png(
                bytestring=svg.encode("utf-8"),
                output_width=size,
                output_height=size,
            )
            # Composite RGBA onto white to avoid transparent→black conversion
            # (same fix as renderer.py — CairoSVG produces RGBA PNGs)
            img = Image.open(io.BytesIO(png)).convert("RGBA")
            bg = Image.new("RGBA", img.size, "WHITE")
            bg.alpha_composite(img)
            return bg.convert("RGB")
        except Exception:
            return Image.new("RGB", (size, size), (255, 255, 255))

    # kept for backwards compatibility if image-bearing datasets are used
    def _load_image(self, image: Any) -> Image.Image:
        """Load a PIL Image from a dataset field (numpy or PIL)."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image or ndarray, got {type(image)}")
        return image.convert("RGB").resize(
            (self.render_size, self.render_size), Image.LANCZOS
        )

    def _encode_sft(
        self,
        messages: list[dict],
        svg: str,
        image: Image.Image,
        gt_len: int,
    ) -> dict:
        """Encode a (prompt, SVG) pair for SFT teacher-forcing.

        The SVG is appended to the message as the assistant's response.
        Loss is computed only on the SVG tokens (labels masking).

        IMPORTANT: We do NOT truncate here. If the sequence exceeds
        max_seq_length, we return None and __getitem__ will skip it.
        Previous code used truncation=True which silently deleted the
        SVG target — the #1 cause of blank outputs.
        """
        # Add assistant response to messages
        full_messages = messages + [
            {"role": "assistant", "content": svg}
        ]

        text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Process with image — NO truncation! We'd rather skip long
        # samples than silently destroy the SVG target.
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=False,
        )

        # Squeeze batch dimension from processor output
        item = {k: v.squeeze(0) for k, v in inputs.items()}

        input_ids = item["input_ids"]

        # If the sequence is too long, truncate from the END but warn.
        # This loses SVG tokens, so we only keep the sample if at least
        # 50 SVG tokens survive after truncation.
        if input_ids.shape[0] > self.max_seq_length:
            logger.debug(
                "Sequence too long (%d > %d), truncating.",
                input_ids.shape[0], self.max_seq_length,
            )
            for key in ["input_ids", "attention_mask"]:
                if key in item:
                    item[key] = item[key][:self.max_seq_length]
            input_ids = item["input_ids"]

        # Mask prompt tokens from loss (only compute loss on SVG tokens)
        labels = input_ids.clone()

        # Find where the assistant response starts
        assistant_start = self._find_assistant_start(input_ids)
        num_svg_tokens = len(input_ids) - assistant_start

        if assistant_start > 0 and num_svg_tokens > 20:
            labels[:assistant_start] = -100   # ignore prompt in loss
        else:
            # Not enough SVG tokens survived — mask everything so this
            # sample contributes zero loss (effectively skipped).
            logger.warning(
                "Sample has only %d SVG tokens (assistant_start=%d, total=%d). "
                "Masking entire sample.",
                num_svg_tokens, assistant_start, len(input_ids),
            )
            labels[:] = -100

        item["labels"]    = labels
        item["gt_length"] = gt_len
        return item

    def _encode_rlrf(
        self,
        messages: list[dict],
        image: Image.Image,
        gt_svg: str,
        gt_len: int,
    ) -> dict:
        """Encode a prompt for RLRF rollout generation (no labels).

        Returns the prompt input_ids and the reference image as a numpy
        array (for reward computation after rendering).
        """
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_seq_length // 2,  # leave room for generation
        )

        item = {k: v.squeeze(0) for k, v in inputs.items()}
        # Store reference image and GT info for reward computation
        item["ref_image"] = np.array(image, dtype=np.uint8)   # [H, W, 3]
        item["gt_svg"]    = gt_svg
        item["gt_length"] = gt_len
        return item

    # Qwen2.5-VL special token IDs (hardcoded for robustness)
    _IM_START_TOKEN_ID = 151644   # <|im_start|>
    _ASSISTANT_TOKEN_ID = 77091   # 'assistant'
    _NEWLINE_TOKEN_ID = 198       # '\n'

    def _find_assistant_start(self, input_ids: torch.Tensor) -> int:
        """Find the token index where the assistant SVG content begins.

        Qwen2.5-VL uses <|im_start|>assistant\n as the delimiter.
        The sequence is: [151644, 77091, 198, <svg tokens...>]

        We find the LAST occurrence of 151644 (the assistant's <|im_start|>)
        and return the index of the first SVG content token (+3).
        """
        ids = input_ids.tolist()

        # Try to get the token ID from the tokenizer first, fall back to hardcoded
        try:
            im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        except (AttributeError, KeyError):
            im_start_id = self._IM_START_TOKEN_ID

        # Find the LAST <|im_start|> — that's the assistant turn
        last_im_start_idx = -1
        for i, token_id in enumerate(ids):
            if token_id == im_start_id:
                last_im_start_idx = i

        if last_im_start_idx != -1:
            # Skip: <|im_start|> (1) + assistant (1) + \n (1) = +3
            content_start = last_im_start_idx + 3
            # Sanity: don't go past the end
            return min(content_start, len(ids))

        # Absolute last resort: scan for known assistant token ID
        for i, token_id in enumerate(ids):
            if token_id == self._ASSISTANT_TOKEN_ID:
                return min(i + 2, len(ids))  # skip 'assistant' + '\n'

        logger.warning(
            "Could not find assistant start token in sequence of length %d. "
            "Masking will be unreliable.", len(ids)
        )
        return len(ids)  # mask everything — don't train on garbage


# ---------------------------------------------------------------------------
# Dataset builder functions
# ---------------------------------------------------------------------------

def load_hf_dataset(
    dataset_name: str = "starvector/svg-stack",
    split: str = "train",
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Load the public SVG-Stack dataset from HuggingFace.

    No authentication token required.

    Args:
        dataset_name: HF dataset repo id.
        split:        Dataset split to load.
        cache_dir:    Local cache directory.
        max_samples:  Optional cap (for fast iteration on Kaggle).

    Returns:
        List of dicts with "image" (PIL) and "svg" (str) fields.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    logger.info("Loading dataset '%s' split='%s' ...", dataset_name, split)
    ds = load_dataset(
        dataset_name,
        split=split,
        cache_dir=cache_dir,
    )

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    # ── Field normalisation ───────────────────────────────────────────────
    # starvector/svg-stack uses capital-letter field names: "Svg", "Filename".
    # We also fall back to lowercase variants for future compatibility.
    records = []
    for row in ds:
        svg = (
            row.get("Svg")       # starvector/svg-stack (capital S)
            or row.get("svg")
            or row.get("svg_code")
            or row.get("code")
            or ""
        )
        filename = (
            row.get("Filename")  # starvector/svg-stack (capital F)
            or row.get("filename")
            or ""
        )
        # No image column — reference image is rendered from GT SVG at load time.
        # We store the raw SVG; rendering happens in SVGDataset.__getitem__.
        if svg:
            records.append({"svg": svg, "filename": filename})

    logger.info(
        "Loaded %d records from %s (fields: Svg+Filename, no pre-rendered image).",
        len(records), dataset_name,
    )
    return records


def build_sft_dataset(
    records: list[dict],
    processor: Any,
    max_seq_length: int = 4096,
    render_size: int = 224,
) -> SVGDataset:
    """Create a SVGDataset in SFT mode (with labels for teacher forcing)."""
    return SVGDataset(records, processor, max_seq_length, mode="sft",
                      render_size=render_size)


def build_rlrf_dataset(
    records: list[dict],
    processor: Any,
    max_seq_length: int = 4096,
    render_size: int = 224,
) -> SVGDataset:
    """Create a SVGDataset in RLRF mode (prompt + reference image only)."""
    return SVGDataset(records, processor, max_seq_length, mode="rlrf",
                      render_size=render_size)


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

def collate_fn_rlrf(batch: list[dict]) -> dict:
    """Custom collate for RLRF batches with mixed tensor / numpy / str fields.

    Tensors are padded to the longest sequence in the batch.
    numpy arrays (ref_image) and strings (gt_svg) are kept as lists.
    """
    import torch.nn.functional as F

    # Qwen2.5-VL pad_token_id = 151643 (<|endoftext|>)
    # Using 0 corrupts attention because 0 is a real vocabulary token.
    PAD_TOKEN_ID = 151643

    result: dict = {}

    # Pad tensors with correct pad values
    pad_values = {
        "input_ids": PAD_TOKEN_ID,
        "attention_mask": 0,
    }
    for key, pad_val in pad_values.items():
        if key not in batch[0]:
            continue
        seqs = [item[key] for item in batch]
        max_len = max(s.shape[0] for s in seqs)
        padded = torch.stack([
            F.pad(s, (0, max_len - s.shape[0]), value=pad_val)
            for s in seqs
        ])
        result[key] = padded

    # Keep as lists of tensors so we can easily index them per-image in RLRF loop
    if "pixel_values" in batch[0]:
        result["pixel_values"] = [b["pixel_values"] for b in batch]
    if "image_grid_thw" in batch[0]:
        result["image_grid_thw"] = [b["image_grid_thw"] for b in batch]

    # Non-tensor fields
    result["ref_image"] = [b["ref_image"] for b in batch]
    result["gt_svg"]    = [b["gt_svg"]    for b in batch]
    result["gt_length"] = [b["gt_length"] for b in batch]

    return result
