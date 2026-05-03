"""
rlrf/config.py
==============
Single source of truth for every hyperparameter used in the RLRF pipeline.
All values default to those reported in the paper (arXiv:2505.20793v2).
Override at runtime via dataclass field assignment or CLI scripts.

Kaggle-specific defaults (G=4 instead of 64) are noted inline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Architecture and loading settings."""

    # HuggingFace model ID — public, no token required
    # Paper uses Qwen2.5-VL-7B; 3B recommended for Kaggle T4 (16 GB).
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # 4-bit QLoRA quantization (required for Kaggle T4)
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"   # bfloat16 or float16
    bnb_4bit_quant_type: str = "nf4"           # nf4 or fp4

    # LoRA adapter settings (PEFT)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Maximum context length.
    # Paper uses 32k (supports 128k but limited for memory).
    max_seq_length: int = 4096   # Reduced for Kaggle; paper: 32768

    # Render resolution (pixels). Paper: 224×224 or adaptive.
    render_size: int = 224


# ---------------------------------------------------------------------------
# Dataset / data curation configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Dataset loading and high-entropy filtering settings."""

    # Public HuggingFace dataset (no auth token needed).
    dataset_name: str = "starvector/svg-stack"
    dataset_split: str = "train"

    # High-entropy filtering threshold (§4.1):
    # Only keep samples whose ground-truth SVG has ≥ 500 tokens.
    min_gt_tokens: int = 500

    # Maximum number of RLRF training samples (paper uses 20k;
    # reduce for Kaggle experimentation).
    max_train_samples: int = 2000   # paper: 20000

    # Test split size
    max_test_samples: int = 200

    # Cache directory for downloaded datasets
    cache_dir: Optional[str] = None

    # Number of dataloader workers
    num_workers: int = 2


# ---------------------------------------------------------------------------
# SFT (Stage 1) configuration
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    """Supervised fine-tuning hyperparameters (Stage 1).

    Paper: Qwen2.5-VL trained on 1.7M pairs, 3 epochs, lr=1e-5,
           batch 1024, context 32k, 4×8 H100 GPUs ~4 days.
    Scaled for Kaggle single-node T4.
    """

    output_dir: str = "./checkpoints/svg_sft"
    num_train_epochs: int = 1          # paper: 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # effective batch ~8
    learning_rate: float = 1e-5           # paper: 1e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 2
    dataloader_num_workers: int = 2
    report_to: str = "none"             # set "wandb" for experiment tracking


# ---------------------------------------------------------------------------
# RLRF (Stage 2) configuration
# ---------------------------------------------------------------------------

@dataclass
class RLRFConfig:
    """GRPO / RLRF hyperparameters (Stage 2).

    All values default to paper values (§4.1, App. C.3).
    Kaggle-specific overrides are documented.
    """

    output_dir: str = "./checkpoints/rlrf"

    # ── GRPO core (paper §3.1, Eq. 4) ──────────────────────────────────────
    # Number of rollouts per input image.
    # Paper: G=64. Kaggle T4: G=4 (memory limit).
    G: int = 4                          # paper: 64

    # PPO-style clipping threshold ε (paper: 0.4)
    epsilon: float = 0.4

    # KL regularization coefficient β.
    # Paper explicitly disables KL (β=0) as it improves reward learning.
    kl_coeff: float = 0.0              # paper: 0 (disabled)

    # ── Sampling (paper §4.1, App. C.3) ─────────────────────────────────────
    # Temperature for rollout generation (paper: 1.1).
    temperature: float = 1.1

    # top-p for inference best-of-n selection (App. C.3)
    top_p: float = 0.9

    # Best-of-n candidates during inference (App. C.3): pick lowest MSE
    best_of_n: int = 5

    # ── Training schedule (paper §4.1) ───────────────────────────────────────
    max_steps: int = 70                # 3-hour limit on Kaggle
    learning_rate: float = 1e-5        # paper: 1e-5

    # LR decay: 70% every 100 steps (paper §4.1)
    lr_decay_factor: float = 0.70
    lr_decay_steps: int = 100

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1   # Update weights after every image (faster steps)
    max_grad_norm: float = 1.0
    bf16: bool = True
    fp16: bool = False

    # ── Dynamic max length (App. C.3) ────────────────────────────────────────
    # Per-batch max generation length = max(gt_lengths) + threshold.
    # Prevents infinite loops; "naturally fades" as training progresses.
    dynamic_len_threshold: int = 128

    # Hard cap on generation length (tokens)
    max_new_tokens: int = 1024   # reduce for Kaggle memory

    # ── Logging & saving ─────────────────────────────────────────────────────
    logging_steps: int = 1
    save_steps: int = 10
    eval_steps: int = 10
    save_total_limit: int = 2
    report_to: str = "none"


# ---------------------------------------------------------------------------
# Reward configuration
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """Weights and settings for each reward component.

    Paper §3.2: R_total = Σ w_i R_i.
    Weights are tunable; defaults match the paper's primary configuration
    (L2 + DreamSim Canny + R_len).
    """

    # Image reconstruction reward weight (§3.2 Eq. 5)
    w_img_l2: float = 1.0

    # L2 Canny (edge-aware) reward weight (§3.2)
    w_img_l2_canny: float = 0.5

    # DreamSim semantic reward weight (§3.2)
    # Set to 0 to disable DreamSim (saves ~1 GB VRAM on Kaggle)
    w_dreamsim: float = 0.5

    # DreamSim Canny (edge-aware semantic) weight (§3.2)
    w_dreamsim_canny: float = 0.5

    # Code efficiency / length reward weight (§3.2)
    w_length: float = 0.5

    # Whether to use DreamSim; if False, uses LPIPS as semantic reward instead
    use_dreamsim: bool = False   # disabled by default for Kaggle memory

    # Whether to use LPIPS as fallback semantic reward
    use_lpips: bool = True

    # Canny edge detection thresholds
    canny_low: int = 50
    canny_high: int = 150

    # Dilation kernel size and iterations (paper: 3×3, 1 iteration)
    dilation_kernel: int = 3
    dilation_iters: int = 1

    # Gaussian blur kernel size (paper: 13)
    blur_kernel: int = 13


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Top-level config bundling all sub-configs."""

    model: ModelConfig   = field(default_factory=ModelConfig)
    data: DataConfig     = field(default_factory=DataConfig)
    sft: SFTConfig       = field(default_factory=SFTConfig)
    rlrf: RLRFConfig     = field(default_factory=RLRFConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Random seed for reproducibility
    seed: int = 42

    # Device map for multi-GPU Kaggle sessions ("auto" = both T4s)
    device_map: str = "auto"

    def __post_init__(self):
        # Sync render_size across components
        assert self.model.render_size > 0
