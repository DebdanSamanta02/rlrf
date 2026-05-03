# RLRF: Reinforcement Learning from Rendering Feedback — Implementation Plan

## Overview

Implementation of the two-stage RLRF training pipeline from the paper *"Reinforcement Learning from Rendering Feedback for SVG Generation"* (arXiv:2505.20793v2). The system fine-tunes an autoregressive VLM to generate high-quality SVG code by incorporating visual feedback from rendered images via GRPO.

The implementation is organized as a **modular Python package** targeting local/cloud GPU training (H100 / A100 / consumer GPU). We implement a faithful, well-documented version of every component described in the paper, including all reward functions, training strategies, and guardrails.

---

## Open Questions

> [!IMPORTANT]
> **Scale & Hardware**: The paper uses 4×8 or 8×8 H100 GPUs. For local implementation we will use a smaller model (StarVector-1B or Qwen2.5-VL-3B) and reduce batch sizes. Please confirm your target GPU budget.

> [!IMPORTANT]
> **Base model**: The paper primarily experiments with Qwen2.5-VL-3B/7B (for Im2SVG) and Qwen3-8B (for Text2SVG). Do you want to target Im2SVG, Text2SVG, or both?

> [!NOTE]
> **DreamSim dependency**: The semantic reward for Im2SVG relies on the DreamSim library. It requires downloading a ~1GB checkpoint. This is included in the plan.

---

## Proposed Changes

### Project Layout (`NEW`)

```
/Users/debdan/Documents/Programs/Jupyter/SVG/RLRF/
├── rlrf/
│   ├── __init__.py
│   ├── config.py              # All hyperparameters (dataclass)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py         # SVGDataset, data collation
│   │   └── curation.py        # High-entropy filtering (≥500 tokens)
│   ├── model/
│   │   ├── __init__.py
│   │   └── vlm.py             # Model loader (Qwen2.5-VL / StarVector)
│   ├── rewards/
│   │   ├── __init__.py
│   │   ├── base.py            # RewardFn abstract class
│   │   ├── image_l2.py        # R_img (L2 + Canny variant)
│   │   ├── semantic.py        # R_sim via DreamSim / CLIP
│   │   ├── length.py          # R_len code efficiency
│   │   └── composite.py       # R_total = Σ w_i R_i
│   ├── rendering/
│   │   ├── __init__.py
│   │   └── renderer.py        # CairoSVG rasterizer + guardrails
│   ├── training/
│   │   ├── __init__.py
│   │   ├── sft.py             # Stage 1: SVG-SFT trainer
│   │   └── rlrf.py            # Stage 2: GRPO trainer
│   └── utils/
│       ├── __init__.py
│       └── metrics.py         # MSE, SSIM, DINO, LPIPS evaluation
├── scripts/
│   ├── train_sft.py
│   ├── train_rlrf.py
│   └── evaluate.py
├── notebooks/
│   └── demo.ipynb
├── requirements.txt
└── README.md
```

---

### `rlrf/config.py` [NEW]

Central dataclass holding **all hyperparameters** from the paper:

| Param | Value | Source |
|---|---|---|
| `G` (rollouts) | 64 | §4.1 |
| `epsilon` (clip) | 0.4 | §4.1 |
| `temperature` | 1.1 | §4.1 |
| `kl_coeff` β | 0.0 | §4.1 (disabled) |
| `lr` | 1e-5 | §4.1 |
| `lr_decay` | 70% every 100 steps | §4.1 |
| `max_rlrf_steps` | 500 | §4.1 |
| `min_gt_tokens` | 500 | §4.1 |
| `render_size` | 224×224 | §4.1 |
| `dynamic_len_threshold` | configurable | App. C.3 |
| `top_p` (inference) | 0.9 | App. C.3 |
| `temperature_inference` | 0.5 | App. C.3 |
| `best_of_n` | 5 | App. C.3 |

---

### `rlrf/rendering/renderer.py` [NEW]

**CairoSVG-based rasterizer** with paper guardrails:

- Render all SVGs at fixed reference size (224×224 or input size), **ignoring viewBox** manipulation to prevent reward hacking (§5 Constraints).
- For Text2SVG: strip `<text>` elements before rendering to prevent prompt-copying (§5 Constraints).
- Handle rendering failures gracefully (return blank image / zero reward).
- Dynamic max length: per-batch, set max tokens = `max(gt_lengths_in_batch) + threshold`.

---

### `rlrf/rewards/` [NEW]

#### `image_l2.py` — Image Reconstruction Reward ($R_{img}$)

```python
# Normalize both images to zero mean, unit variance
I_norm = (I - I.mean()) / (I.std() + eps)
# L2 distance, clipped to [-1, 1]
R_img = clip(1 - (1/N) * ||I_in_norm - I_pred_norm||²₂, -1, 1)
```

Also implements **L2 Canny** variant: apply `cv2.Canny`, dilate (3×3, 1 iter), Gaussian blur (σ=13), then compute L2.

#### `semantic.py` — Semantic Similarity Reward ($R_{sim}$)

- **Im2SVG**: DreamSim (concatenated CLIP+OpenCLIP+DINOv2 ViT-B/16 features → linear → cosine sim). `sim ∈ [0,2]`, so `R_sim = 1 - sim ∈ [-1,1]`.
- **DreamSim Canny**: edge-detect → DreamSim (emphasizes geometry/contours).
- **Text2SVG**: CLIP cosine similarity between text prompt and rendered image. Also VLM-as-judge reward using Qwen2.5-VL-7B with Prompts 2 & 3 from App. C.2.

#### `length.py` — Code Efficiency Reward ($R_{len}$)

```python
R_len = clip(1 - ((1/L_gt) * max(0, L_pred - L_gt/2))², -1, 1)
```
Quadratic penalty only when predicted length exceeds half of ground-truth.

#### `composite.py` — Final Reward Aggregation

```python
R_total = sum(w_i * R_i for i in rewards)
```
Configurable weight dictionary.

---

### `rlrf/training/sft.py` [NEW]

Stage 1: Standard teacher-forcing fine-tuning.

- Loss: negative log-likelihood over ground-truth SVG token sequence.
- Uses HuggingFace `Trainer` or `trl.SFTTrainer`.
- Context length capped at 32k tokens.

### `rlrf/training/rlrf.py` [NEW] — Core GRPO Loop

This is the main contribution. Per training step:

1. Sample a batch of `B` input images from filtered dataset.
2. For each image, generate `G=64` rollout SVG sequences with temperature=1.1.
3. Render each rollout SVG with CairoSVG.
4. Compute per-rollout reward `R(xc, oi)`.
5. Compute group-centered advantage: `A_i = R_i - mean(R_j for j in group)`.
6. Compute probability ratio `r_i = π_θ(o_i|x_c) / π_θ_old(o_i|x_c)`.
7. Compute clipped GRPO loss: `min(r_i * A_i, clip(r_i, 1-ε, 1+ε) * A_i)`.
8. No KL penalty (`β=0`).
9. Update model with gradient step.
10. LR schedule: decay 70% every 100 steps from lr=1e-5.

**Dynamic max length**: Before rollout, compute `max(len(gt_svg) for sample in batch) + threshold` and set as the generation max length.

---

### `rlrf/data/curation.py` [NEW]

High-entropy data filtering:
- Keep samples with ≥ 500 ground-truth SVG tokens.
- Select visually complex, diverse SVGs.
- Target: 20k training samples for RLRF phase.

---

### `scripts/` [NEW]

- `train_sft.py`: Launch Stage 1 with CLI args.
- `train_rlrf.py`: Launch Stage 2 with CLI args.
- `evaluate.py`: Compute MSE, SSIM, DINO Score, LPIPS, Code Efficiency on test set.

---

### `requirements.txt` [NEW]

```
torch>=2.2
transformers>=4.40
trl>=0.8
cairosvg
opencv-python
dreamsim
open_clip_torch
Pillow
datasets
accelerate
peft
scikit-image
lpips
numpy
```

---

## Verification Plan

### Automated Tests

1. **Unit tests per reward**: feed synthetic image pairs with known distances; assert outputs within `[-1, 1]` and monotonicity.
2. **Renderer test**: render a known SVG string; assert output shape is `(224, 224, 3)`.
3. **GRPO advantage test**: verify `sum(A_i) ≈ 0` for any reward batch.
4. **Dynamic max length test**: verify max tokens is set correctly per batch.
5. **End-to-end smoke test**: 2-step RLRF on 2 images with `G=2` rollouts — assert loss decreases or no crash.

### Manual Verification

- Render sample SVGs before and after RLRF fine-tuning; visually compare quality.
- Plot reward curves over training steps (expected: upward trend matching Fig. 4/5 in paper).
- Run `evaluate.py` on 10 test samples; compare MSE/SSIM to paper Table 1 baselines.
