"""
rlrf/model/vlm.py
=================
Model loading utilities for Qwen2.5-VL with QLoRA (4-bit) + LoRA adapters.

The paper uses Qwen2.5-VL-3B-Instruct and Qwen2.5-VL-7B-Instruct.
For Kaggle T4 (16 GB VRAM), we default to the 3B model with 4-bit quantisation.

Loading strategy:
    1. Load base model in 4-bit via BitsAndBytes (NF4 quantisation).
    2. Wrap with PEFT LoRA adapters targeting attention projection layers.
    3. Load AutoProcessor (tokenizer + image processor) for Qwen2.5-VL.

References:
    Paper: §C.1 (Multimodal Architecture)
    Qwen2.5-VL: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from ..config import ModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_processor(
    config: ModelConfig,
    device_map: str = "auto",
) -> Tuple[object, object]:
    """Load Qwen2.5-VL model and processor with QLoRA.

    Args:
        config:     ModelConfig with model name, quantisation, and LoRA settings.
        device_map: "auto" for multi-GPU, "cuda:0" for single GPU.

    Returns:
        (model, processor) tuple ready for training.
    """
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    logger.info("Loading model: %s (4-bit=%s)", config.model_name, config.load_in_4bit)

    # ── BitsAndBytes quantisation config ──────────────────────────────────
    bnb_config = None
    if config.load_in_4bit:
        compute_dtype = (
            torch.bfloat16
            if config.bnb_4bit_compute_dtype == "bfloat16"
            else torch.float16
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,   # nested quant saves ~0.4 bpp
        )

    # ── Load base model ───────────────────────────────────────────────────
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if not config.load_in_4bit else None,
        trust_remote_code=True,
    )

    # ── Prepare for k-bit training (gradient checkpointing + cast norms) ─
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    # ── Attach LoRA adapters ──────────────────────────────────────────────
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Load processor (tokenizer + image processor) ──────────────────────
    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    # Qwen2.5-VL uses <|endoftext|> as pad; ensure left-padding for generation
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"  # required for batch generation

    logger.info("Model and processor loaded.")
    return model, processor


def load_reference_model(
    config: ModelConfig,
    device_map: str = "auto",
) -> object:
    """Load a frozen reference (SFT) model for KL computation.

    The reference model is identical in architecture to the policy model
    but its weights are frozen. Used in GRPO Eq. 4 for KL regularisation
    (note: paper sets β=0 so KL is not computed, but we include this for
    completeness / future use).

    Args:
        config: ModelConfig (same as policy model).

    Returns:
        Frozen base model (no LoRA, no quantisation for fp precision).
    """
    from transformers import Qwen2_5_VLForConditionalGeneration

    logger.info("Loading frozen reference model: %s", config.model_name)
    ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Freeze all parameters
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()
    logger.info("Reference model frozen.")
    return ref_model


def get_trainable_param_count(model: object) -> str:
    """Return a human-readable summary of trainable vs total parameters."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / (total + 1e-8)
    return (
        f"Trainable: {trainable:,} / {total:,} params ({pct:.2f}%)"
    )
