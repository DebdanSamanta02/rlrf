#!/usr/bin/env python3
"""
scripts/train_rlrf.py
=====================
CLI script for Stage 2: RLRF training (GRPO with rendering reward).

Usage (Kaggle notebook cell):
    !python scripts/train_rlrf.py \
        --sft_checkpoint ./checkpoints/svg_sft \
        --G 4 \
        --max_steps 500 \
        --output_dir ./checkpoints/rlrf

All hyperparameters default to paper values (§4.1).
Kaggle-specific defaults: G=4, max_train_samples=2000.
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rlrf.config import (
    Config, ModelConfig, DataConfig, SFTConfig, RLRFConfig, RewardConfig
)
from rlrf.training import run_rlrf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: RLRF / GRPO training")

    # Model
    p.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--sft_checkpoint", default="./checkpoints/svg_sft",
                   help="Path to Stage 1 SFT checkpoint (start of Stage 2).")

    # Data
    p.add_argument("--dataset_name", default="starvector/svg-stack")
    p.add_argument("--max_train_samples", type=int, default=2000)
    p.add_argument("--min_gt_tokens", type=int, default=500)
    p.add_argument("--cache_dir", default=None)

    # GRPO hyperparameters (paper §4.1)
    p.add_argument("--G", type=int, default=4,
                   help="Rollouts per image. Paper: 64. Kaggle: 4.")
    p.add_argument("--epsilon", type=float, default=0.4,
                   help="PPO clipping threshold (paper: 0.4).")
    p.add_argument("--kl_coeff", type=float, default=0.0,
                   help="KL regularisation β. Paper: 0 (disabled).")
    p.add_argument("--temperature", type=float, default=1.1,
                   help="Sampling temperature (paper: 1.1).")
    p.add_argument("--max_steps", type=int, default=500,
                   help="Total GRPO update steps (paper: 500).")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--dynamic_len_threshold", type=int, default=128)

    # Reward weights
    p.add_argument("--w_l2",    type=float, default=1.0)
    p.add_argument("--w_canny", type=float, default=0.5)
    p.add_argument("--w_len",   type=float, default=0.5)
    p.add_argument("--use_lpips", action="store_true", default=True)
    p.add_argument("--w_lpips", type=float, default=1.0,
                   help="Absorbed into dreamsim weight slot when DreamSim disabled.")
    p.add_argument("--use_dreamsim", action="store_true", default=False)

    # Training
    p.add_argument("--output_dir", default="./checkpoints/rlrf")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--resume_step", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config(
        model=ModelConfig(model_name=args.model_name),
        data=DataConfig(
            dataset_name=args.dataset_name,
            max_train_samples=args.max_train_samples,
            min_gt_tokens=args.min_gt_tokens,
            cache_dir=args.cache_dir,
        ),
        rlrf=RLRFConfig(
            output_dir=args.output_dir,
            G=args.G,
            epsilon=args.epsilon,
            kl_coeff=args.kl_coeff,
            temperature=args.temperature,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            max_new_tokens=args.max_new_tokens,
            dynamic_len_threshold=args.dynamic_len_threshold,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
        ),
        reward=RewardConfig(
            w_img_l2=args.w_l2,
            w_img_l2_canny=args.w_canny,
            w_length=args.w_len,
            use_lpips=args.use_lpips,
            use_dreamsim=args.use_dreamsim,
            w_dreamsim=args.w_lpips,      # absorbs LPIPS weight
        ),
        seed=args.seed,
    )

    logger.info("RLRF configuration: G=%d, ε=%.2f, temp=%.2f, β=%.2f",
                cfg.rlrf.G, cfg.rlrf.epsilon,
                cfg.rlrf.temperature, cfg.rlrf.kl_coeff)

    run_rlrf(cfg, sft_checkpoint=args.sft_checkpoint,
             resume_step=args.resume_step)


if __name__ == "__main__":
    main()
