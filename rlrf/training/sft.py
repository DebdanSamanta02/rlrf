"""
rlrf/training/sft.py
====================
Stage 1: Supervised Fine-Tuning on Images and SVGs (SVG-SFT).

Paper §3.1, Equation 1:
    L_SFT(θ) = E_{x_c ~ D} [-log p_θ(x_s | x_c)]
             = E_{x_c ~ D} [-Σ_{l=1}^{L} log p_θ(x_{s,l} | x_{s,<l}, x_c)]

This is standard teacher-forcing NLL — the model receives the full
ground-truth SVG prefix at each step ("completion" objective).

Implementation:
    • Uses Hugging Face Trainer with a custom data collator.
    • LoRA/QLoRA handles memory; gradient checkpointing reduces VRAM.
    • Loss is masked so that only the assistant's SVG tokens contribute
      (prompt and image tokens are labelled −100 and ignored).

Paper hyperparameters (§4.1):
    lr = 1e-5, batch = 1024 (effective), epochs = 3, context = 32k.
    Scaled for Kaggle T4: epochs=1, eff. batch ≈ 8, context = 4096.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader

from ..config import Config, SFTConfig
from ..data import build_sft_dataset, load_hf_dataset, curate_dataset
from ..model import load_model_and_processor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

class SVGDataCollator:
    """Pads a batch of SFT examples to the longest sequence.

    Pads input_ids with pad_token_id, attention_mask with 0,
    and labels with −100 (ignored by cross-entropy loss).
    """

    def __init__(self, pad_token_id: int = 0) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch.nn.functional as F

        max_len = max(item["input_ids"].shape[0] for item in batch)
        out: dict = {}

        for key, pad_val in [
            ("input_ids",      self.pad_token_id),
            ("attention_mask", 0),
            ("labels",         -100),
        ]:
            if key not in batch[0]:
                continue
            seqs = [item[key] for item in batch]
            out[key] = torch.stack([
                F.pad(s, (0, max_len - s.shape[0]), value=pad_val)
                for s in seqs
            ])

        if "pixel_values" in batch[0]:
            out["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])

        return out


# ---------------------------------------------------------------------------
# SFT trainer
# ---------------------------------------------------------------------------

def run_sft(cfg: Config, resume_from: Optional[str] = None) -> None:
    """Run Stage 1 SVG-SFT training.

    Args:
        cfg:          Master Config object.
        resume_from:  Path to a checkpoint to resume from (optional).
    """
    from transformers import Trainer, TrainingArguments

    # ── Load model ────────────────────────────────────────────────────────
    model, processor = load_model_and_processor(cfg.model, device_map=cfg.device_map)
    model.train()

    # ── Load and curate dataset ───────────────────────────────────────────
    raw = load_hf_dataset(
        cfg.data.dataset_name,
        split=cfg.data.dataset_split,
        cache_dir=cfg.data.cache_dir,
        max_samples=cfg.data.max_train_samples * 5,  # over-sample before filter
    )
    curated = curate_dataset(
        raw,
        min_tokens=cfg.data.min_gt_tokens,
        max_samples=cfg.data.max_train_samples,
        skip_entropy=True,   # skip entropy filter during SFT (no images pre-loaded)
    )

    # Train / eval split
    n_eval  = min(cfg.data.max_test_samples, len(curated) // 10)
    train_r = curated[n_eval:]
    eval_r  = curated[:n_eval]

    train_ds = build_sft_dataset(train_r, processor,
                                  cfg.model.max_seq_length, cfg.model.render_size)
    eval_ds  = build_sft_dataset(eval_r,  processor,
                                  cfg.model.max_seq_length, cfg.model.render_size)

    logger.info("SFT dataset — train: %d, eval: %d", len(train_ds), len(eval_ds))

    # ── Training arguments ────────────────────────────────────────────────
    sft: SFTConfig = cfg.sft
    training_args = TrainingArguments(
        output_dir=sft.output_dir,
        num_train_epochs=sft.num_train_epochs,
        per_device_train_batch_size=sft.per_device_train_batch_size,
        per_device_eval_batch_size=sft.per_device_train_batch_size,
        gradient_accumulation_steps=sft.gradient_accumulation_steps,
        learning_rate=sft.learning_rate,
        lr_scheduler_type=sft.lr_scheduler_type,
        warmup_ratio=sft.warmup_ratio,
        weight_decay=sft.weight_decay,
        max_grad_norm=sft.max_grad_norm,
        fp16=sft.fp16,
        bf16=sft.bf16,
        logging_steps=sft.logging_steps,
        save_steps=sft.save_steps,
        eval_steps=sft.eval_steps,
        evaluation_strategy="steps",
        save_total_limit=sft.save_total_limit,
        dataloader_num_workers=sft.dataloader_num_workers,
        report_to=sft.report_to,
        remove_unused_columns=False,
        label_names=["labels"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    collator = SVGDataCollator(
        pad_token_id=processor.tokenizer.pad_token_id or 0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    logger.info("Starting SVG-SFT training …")
    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save final checkpoint ─────────────────────────────────────────────
    trainer.save_model(sft.output_dir)
    processor.save_pretrained(sft.output_dir)
    logger.info("SVG-SFT complete. Checkpoint saved to %s", sft.output_dir)
