# RLRF — Reinforcement Learning from Rendering Feedback for SVG Generation

Faithful implementation of the RLRF paper:
> **"Reinforcement Learning from Rendering Feedback for SVG Generation"**  
> arXiv:2505.20793v2

---

## Overview

RLRF trains an autoregressive Vision-Language Model (VLM) to generate
Scalable Vector Graphics (SVG) code from images. It uses a **two-stage pipeline**:

| Stage | Name | What it does |
|---|---|---|
| 1 | **SVG-SFT** | Supervised fine-tuning on image→SVG pairs (teacher forcing, NLL loss) |
| 2 | **RLRF** | GRPO-based RL with composite visual reward from CairoSVG rendering |

---

## Paper Hyperparameters (§4.1)

| Parameter | Paper Value | This Repo (Kaggle) |
|---|---|---|
| Base model | Qwen2.5-VL-7B | Qwen2.5-VL-3B (public) |
| Rollouts G | 64 | **4** (T4 VRAM limit) |
| Clip ε | 0.4 | 0.4 |
| Temperature | 1.1 | 1.1 |
| KL coeff β | 0 (disabled) | 0 |
| Learning rate | 1e-5 | 1e-5 |
| LR decay | 70% / 100 steps | 70% / 100 steps |
| RLRF steps | 500 | 100–500 |
| Context length | 32k | 2k–4k (memory limit) |
| Min SVG tokens | 500 | 500 |

---

## Reward Functions (§3.2)

| Reward | Formula | Weight |
|---|---|---|
| `R_img` (L2) | `clip(1 - (1/N)‖I_norm - Î_norm‖₂², -1, 1)` | 1.0 |
| `R_img` (L2 Canny) | Same on Canny edge maps | 0.5 |
| `R_sim` (DreamSim) | `1 - DreamSim_dist ∈ [-1,1]` | 0.5 (disabled on Kaggle) |
| `R_sim` (LPIPS fallback) | `clip(1 - 2·LPIPS, -1, 1)` | 1.0 |
| `R_len` | `clip(1 - ((max(0, L_pred - L_gt/2) / L_gt))², -1, 1)` | 0.5 |

**Total**: `R_total = Σ wᵢ Rᵢ`

---

## Guardrails (§5)

1. **ViewBox enforcement**: Force SVG render at reference size regardless of predicted `viewBox` (prevents reward hacking).
2. **Text stripping** (Text2SVG only): Remove `<text>` elements before rendering.
3. **Dynamic max length** (App. C.3): Per-batch `max_tokens = max(gt_lengths) + threshold`.
4. **Graceful failure**: Blank image returned on render error → minimum reward, no crash.

---

## Project Structure

```
rlrf/
├── config.py              # All hyperparameters (dataclasses)
├── rendering/
│   └── renderer.py        # CairoSVG rasterizer + guardrails
├── rewards/
│   ├── base.py            # Abstract RewardFn
│   ├── image_l2.py        # R_img (L2, L2 Canny)
│   ├── semantic.py        # DreamSim, LPIPS, CLIP
│   ├── length.py          # R_len + dynamic_max_length()
│   └── composite.py       # R_total aggregator
├── data/
│   ├── dataset.py         # SVGDataset, HF loader, chat template
│   └── curation.py        # Length + entropy filtering
├── model/
│   └── vlm.py             # QLoRA model + processor loading
├── training/
│   ├── sft.py             # Stage 1: teacher-forcing SFT
│   └── rlrf.py            # Stage 2: GRPO training loop
└── utils/
    └── metrics.py         # MSE, SSIM, DINO, LPIPS, Code Efficiency
scripts/
├── train_sft.py           # CLI for Stage 1
├── train_rlrf.py          # CLI for Stage 2
└── evaluate.py            # Evaluation (paper Table 1 metrics)
notebooks/
└── kaggle_rlrf.py         # Kaggle notebook (copy cells to notebook)
```

---

## Kaggle Quick Start

### 1. Install dependencies
```python
!pip install -q cairosvg transformers peft trl bitsandbytes accelerate \
    datasets opencv-python-headless scikit-image lpips open-clip-torch einops
```

### 2. Clone / upload this repo to Kaggle
```bash
# In Kaggle terminal:
git clone <your-repo-url> /kaggle/working/RLRF
```

### 3. Stage 1 — SFT
```bash
python /kaggle/working/RLRF/scripts/train_sft.py \
    --max_train_samples 500 \
    --num_epochs 1 \
    --output_dir /kaggle/working/checkpoints/svg_sft
```

### 4. Stage 2 — RLRF
```bash
python /kaggle/working/RLRF/scripts/train_rlrf.py \
    --sft_checkpoint /kaggle/working/checkpoints/svg_sft \
    --G 4 \
    --max_steps 100 \
    --output_dir /kaggle/working/checkpoints/rlrf
```

### 5. Evaluate
```bash
python /kaggle/working/RLRF/scripts/evaluate.py \
    --checkpoint /kaggle/working/checkpoints/rlrf/step_100 \
    --num_samples 50 \
    --compute_lpips \
    --output_csv /kaggle/working/results/eval.csv
```

---

## Enabling DreamSim (optional)

DreamSim requires ~1 GB download. Enable on machines with more memory:

```python
from rlrf.config import RewardConfig
cfg.reward.use_dreamsim = True   # downloads checkpoint on first call
cfg.reward.use_lpips = False
```

---

## Differences from the Paper

| Aspect | Paper | This Implementation |
|---|---|---|
| G (rollouts) | 64 | 4 (Kaggle T4) |
| Training samples | 20k | 500–2000 (Kaggle) |
| Context length | 32k tokens | 2k–4k tokens |
| GPU | 4×8 H100 | 2×T4 (Kaggle free) |
| DreamSim | ✓ | Optional (LPIPS fallback) |
| KL penalty | 0 (disabled) | 0 (matches paper) |
| ε, temp, lr | Paper values | All match paper |

---

## Citation

```bibtex
@article{rodriguez2025rlrf,
  title={Reinforcement Learning from Rendering Feedback for SVG Generation},
  author={Rodriguez, Juan A. and others},
  journal={arXiv preprint arXiv:2505.20793},
  year={2025}
}
```
