#!/usr/bin/env python3
"""
scripts/train_sft.py
====================
CLI script for Stage 1: SVG-SFT (Supervised Fine-Tuning).

Usage (Kaggle notebook cell):
    !python scripts/train_sft.py \
        --model_name Qwen/Qwen2.5-VL-3B-Instruct \
        --max_train_samples 1000 \
        --num_epochs 1 \
        --output_dir ./checkpoints/svg_sft

Usage (local):
    python scripts/train_sft.py --help
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rlrf.config import Config, ModelConfig, DataConfig, SFTConfig
from rlrf.training import run_sft

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1: SVG-SFT training")

    # Model
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--max_seq_length", type=int, default=4096)

    # Data
    p.add_argument("--dataset_name", default="starvector/svg-stack")
    p.add_argument("--max_train_samples", type=int, default=2000)
    p.add_argument("--min_gt_tokens", type=int, default=500)
    p.add_argument("--cache_dir", default=None)

    # Training
    p.add_argument("--output_dir", default="./checkpoints/svg_sft")
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--resume_from", default=None)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config(
        model=ModelConfig(
            model_name=args.model_name,
            load_in_4bit=args.load_in_4bit,
            lora_r=args.lora_r,
            max_seq_length=args.max_seq_length,
        ),
        data=DataConfig(
            dataset_name=args.dataset_name,
            max_train_samples=args.max_train_samples,
            min_gt_tokens=args.min_gt_tokens,
            cache_dir=args.cache_dir,
        ),
        sft=SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
        ),
        seed=args.seed,
    )

    logger.info("SVG-SFT configuration: %s", cfg)
    run_sft(cfg, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
