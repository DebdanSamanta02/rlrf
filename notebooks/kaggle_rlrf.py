"""
notebooks/kaggle_rlrf.py
========================
Kaggle-ready training script for RLRF Im2SVG.

Run this as a Kaggle notebook (copy each section into a cell).
GPU: T4 x2 (16 GB each) — free tier Kaggle.

Pipeline:
    1. Install dependencies
    2. Load and curate SVG-Stack dataset
    3. Stage 1: SVG-SFT (optional skip if starting from SFT checkpoint)
    4. Stage 2: RLRF (GRPO with rendering reward)
    5. Evaluate and visualise results
"""

# ============================================================================
# CELL 1 — Kaggle Setup: Where does the code go?
# ============================================================================
#
# DIRECTORY LAYOUT ON KAGGLE
# ───────────────────────────
#   /kaggle/working/   ← WRITABLE. Put all your code here. This is where
#                        checkpoints, results, and logs are also saved.
#   /kaggle/input/     ← READ-ONLY. Only for datasets you attach in the
#                        "Data" panel. Do NOT put training code here.
#
# RECOMMENDED: clone this repo directly into /kaggle/working/ (Cell 1a below).
# ALTERNATIVE: upload a zip via the Kaggle "Data" panel, then copy it (Cell 1b).
# ============================================================================

# ── CELL 1a: Clone from GitHub (recommended) ─────────────────────────────
# Run this in your Kaggle notebook if you push this repo to GitHub:
#
# !git clone https://github.com/DebdanSamanta02/rlrf.git /kaggle/working/RLRF
#
# After cloning, your layout will be:
#   /kaggle/working/
#   └── RLRF/
#       ├── rlrf/          ← the Python package (imported as `import rlrf`)
#       ├── scripts/       ← train_sft.py, train_rlrf.py, evaluate.py
#       ├── notebooks/     ← this file
#       └── requirements.txt

# ── CELL 1b: Manual upload fallback ──────────────────────────────────────
# If you don't use GitHub:
#   1. Zip the entire RLRF/ folder on your machine.
#   2. In Kaggle → "Data" panel → "+ Add Data" → "Upload" → upload the zip.
#      The zip will appear at: /kaggle/input/rlrf/RLRF.zip  (read-only)
#   3. Run the cell below to unzip it into /kaggle/working/:
#
# !unzip -q /kaggle/input/rlrf/RLRF.zip -d /kaggle/working/
#
# After unzipping your layout will be identical to Cell 1a above.

# ── CELL 1c: Install dependencies ────────────────────────────────────────
# Run AFTER cloning/unzipping so the code is in place.

# !pip install -q \
#     cairosvg \
#     "transformers>=4.49.0" \
#     "peft>=0.10.0" \
#     "trl>=0.8.6" \
#     "bitsandbytes>=0.43.0" \
#     "accelerate>=0.30.0" \
#     "datasets>=2.18.0" \
#     opencv-python-headless \
#     scikit-image \
#     lpips \
#     open-clip-torch \
#     einops

# ── Tell Python where to find the `rlrf` package ─────────────────────────
# /kaggle/working/RLRF contains the `rlrf/` sub-directory (the package).
# Adding its parent to sys.path lets us do `import rlrf` anywhere.

import sys
import os

REPO_ROOT = "/kaggle/working/RLRF"   # adjust only if you used a different name
assert os.path.isdir(REPO_ROOT), (
    f"Could not find {REPO_ROOT}. "
    "Did you run the git clone / unzip step above?"
)
sys.path.insert(0, REPO_ROOT)
print(f"sys.path updated. REPO_ROOT = {REPO_ROOT}")
print(f"rlrf package found: {os.path.isdir(os.path.join(REPO_ROOT, 'rlrf'))}")

# ============================================================================
# CELL 2 — Imports and configuration
# ============================================================================

import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from rlrf.config import Config, ModelConfig, DataConfig, SFTConfig, RLRFConfig, RewardConfig
from rlrf.rendering import SVGRenderer
from rlrf.rewards import CompositeReward

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ── Kaggle-scaled configuration ─────────────────────────────────────────────
cfg = Config(
    model=ModelConfig(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        load_in_4bit=True,
        lora_r=16,
        lora_alpha=32,
        max_seq_length=2048,          # reduce for T4 VRAM
        render_size=224,
    ),
    data=DataConfig(
        dataset_name="starvector/svg-stack",
        dataset_split="train",
        min_gt_tokens=500,            # paper threshold
        max_train_samples=500,        # scale down for quick Kaggle experiment
        max_test_samples=50,
    ),
    sft=SFTConfig(
        output_dir="/kaggle/working/checkpoints/svg_sft",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,           # paper: 1e-5
    ),
    rlrf=RLRFConfig(
        output_dir="/kaggle/working/checkpoints/rlrf",
        G=4,                          # paper: 64; T4 budget: 4
        epsilon=0.4,                  # paper: 0.4
        kl_coeff=0.0,                 # paper: 0 (disabled)
        temperature=1.1,              # paper: 1.1
        max_steps=100,                # paper: 500; quick experiment: 100
        learning_rate=1e-5,
        lr_decay_factor=0.70,         # paper: 70% every 100 steps
        lr_decay_steps=100,
        max_new_tokens=512,
        dynamic_len_threshold=128,    # paper: App. C.3
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
    ),
    reward=RewardConfig(
        w_img_l2=1.0,         # primary reconstruction reward
        w_img_l2_canny=0.5,   # edge-aware variant
        w_length=0.5,         # code efficiency
        use_dreamsim=False,   # disabled — requires 1 GB download
        use_lpips=True,       # lighter perceptual reward fallback
        w_dreamsim=1.0,       # absorbed by LPIPS when dreamsim=False
    ),
    seed=42,
    device_map="auto",        # uses both T4s if available
)

print("Configuration set.")
print(f"  Model:       {cfg.model.model_name}")
print(f"  G rollouts:  {cfg.rlrf.G}")
print(f"  RLRF steps:  {cfg.rlrf.max_steps}")
print(f"  ε (clip):    {cfg.rlrf.epsilon}")
print(f"  temperature: {cfg.rlrf.temperature}")
print(f"  KL coeff β:  {cfg.rlrf.kl_coeff}  (disabled per paper)")
print(f"  Rewards:     L2={cfg.reward.w_img_l2}, L2_Canny={cfg.reward.w_img_l2_canny}, "
      f"Len={cfg.reward.w_length}, LPIPS={cfg.reward.w_dreamsim}")

# ============================================================================
# CELL 3 — Load and curate dataset
# ============================================================================

from rlrf.data import load_hf_dataset, curate_dataset

print("Loading SVG-Stack dataset (this may take a few minutes)...")
raw = load_hf_dataset(
    cfg.data.dataset_name,
    split=cfg.data.dataset_split,
    cache_dir="/kaggle/working/.cache",
    max_samples=cfg.data.max_train_samples * 5,
)
print(f"Raw samples: {len(raw)}")

# High-entropy filtering (paper §4.1: ≥500 tokens, visually complex)
curated = curate_dataset(
    raw,
    min_tokens=cfg.data.min_gt_tokens,
    max_samples=cfg.data.max_train_samples,
    skip_entropy=True,   # skip when images not pre-loaded
)
print(f"Curated: {len(curated)} samples (min_tokens={cfg.data.min_gt_tokens})")

# Train/eval split
n_eval   = cfg.data.max_test_samples
train_r  = curated[n_eval:]
eval_r   = curated[:n_eval]
print(f"Train: {len(train_r)}, Eval: {len(eval_r)}")

# Sanity check: display one sample
sample = train_r[0]
print(f"\nSample SVG length: {len(sample['svg'])} chars")
print(f"Sample filename: {sample.get('filename', 'N/A')}")
# (Note: starvector/svg-stack has no pre-rendered images.
# SVGDataset renders them on-the-fly using CairoSVG.)


# ============================================================================
# CELL 4 — Quick reward sanity check (no GPU needed)
# ============================================================================

renderer = SVGRenderer(size=cfg.model.render_size, enforce_viewbox=True)

# Test 1: render a known SVG
test_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="224" height="224">
  <rect x="10" y="10" width="200" height="200" fill="steelblue" rx="20"/>
  <circle cx="112" cy="112" r="60" fill="gold"/>
</svg>"""
rendered = renderer.render(test_svg)
print(f"Renderer output shape: {rendered.shape}, dtype: {rendered.dtype}")
assert rendered.shape == (224, 224, 3), "Renderer size mismatch!"

# Test 2: blank SVG → white image
blank_rendered = renderer.render("")
assert blank_rendered.mean() > 200, "Blank SVG should give near-white image"

# Test 3: reward on identical images → should be near 1.0
from rlrf.rewards import L2Reward, LengthReward
l2 = L2Reward()
r_identical = l2(rendered, rendered)
print(f"L2 reward (identical): {r_identical:.4f}  (should be ≈ 1.0)")
assert abs(r_identical - 1.0) < 0.01

# Test 4: length reward
lr = LengthReward()
r_len = lr(rendered, rendered, svg_pred="x" * 300, gt_length=500)
print(f"Length reward (300 chars, gt=500): {r_len:.4f}  (should be ~1.0)")

# Test 5: group-centred advantage sums to zero
rewards_test = [0.5, 0.3, 0.8, 0.4]
mean_r = sum(rewards_test) / len(rewards_test)
advantages = [r - mean_r for r in rewards_test]
assert abs(sum(advantages)) < 1e-9, "Advantages must sum to zero!"
print(f"Advantage sum: {sum(advantages):.2e}  (should be ≈ 0)")

print("\nAll sanity checks passed ✓")


# ============================================================================
# CELL 5 — Stage 1: SVG-SFT (Supervised Fine-Tuning)
# ============================================================================
# Skip this cell if you already have an SFT checkpoint.

from rlrf.training import run_sft

# Uncomment to train:
# run_sft(cfg)

# Or use the script from terminal:
# !python /kaggle/working/RLRF/scripts/train_sft.py \
#     --max_train_samples 500 \
#     --num_epochs 1 \
#     --output_dir /kaggle/working/checkpoints/svg_sft

print("SFT cell ready. Uncomment run_sft(cfg) to train.")
print("SFT checkpoint will be saved to:", cfg.sft.output_dir)


# ============================================================================
# CELL 6 — Stage 2: RLRF Training (GRPO)
# ============================================================================

from rlrf.training import run_rlrf

# Uncomment to train:
# run_rlrf(
#     cfg,
#     sft_checkpoint=cfg.sft.output_dir,
#     resume_step=0,
# )

# Or use the script:
# !python /kaggle/working/RLRF/scripts/train_rlrf.py \
#     --sft_checkpoint /kaggle/working/checkpoints/svg_sft \
#     --G 4 \
#     --max_steps 100 \
#     --output_dir /kaggle/working/checkpoints/rlrf

print("RLRF cell ready. Uncomment run_rlrf(...) to train.")
print("RLRF checkpoint will be saved to:", cfg.rlrf.output_dir)


# ============================================================================
# CELL 7 — Visualise reward components during training
# ============================================================================

def plot_training_curves(log_file: str = None):
    """Plot reward curve from training log (if wandb/logging was enabled)."""
    # Stub: replace with actual logged values
    steps   = list(range(0, 100, 5))
    rewards = [float(np.random.uniform(0.2, 0.8)) for _ in steps]  # placeholder

    plt.figure(figsize=(10, 4))
    plt.plot(steps, rewards, marker="o", label="Mean Reward")
    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("RLRF Step")
    plt.ylabel("Composite Reward")
    plt.title("RLRF Training Reward Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/kaggle/working/reward_curve.png", dpi=150)
    plt.show()
    print("Saved reward_curve.png")

plot_training_curves()


# ============================================================================
# CELL 8 — Inference and visualisation
# ============================================================================

def inference_demo(
    model, processor, renderer, image: Image.Image,
    max_new_tokens: int = 512, best_of_n: int = 5, device: str = "cuda"
) -> tuple[str, np.ndarray]:
    """Run best-of-N inference (App. C.3).

    Generate N SVG candidates and return the one with the lowest MSE
    relative to the input image.
    """
    from rlrf.data import make_im2svg_messages
    from rlrf.utils.metrics import mse

    ref_arr  = np.array(image.resize((224, 224)))
    messages = make_im2svg_messages(image)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    best_svg, best_mse = "", float("inf")
    with torch.no_grad():
        for _ in range(best_of_n):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            svg = processor.tokenizer.decode(
                out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            pred_arr = renderer.render(svg)
            m = mse(ref_arr, pred_arr) * 100
            if m < best_mse:
                best_mse, best_svg = m, svg

    return best_svg, renderer.render(best_svg)


def show_comparison(ref_image: Image.Image, pred_arr: np.ndarray, title: str = "RLRF"):
    """Side-by-side display of reference and predicted SVG rendering."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(np.array(ref_image.resize((224, 224))))
    axes[0].set_title("Reference Image")
    axes[0].axis("off")
    axes[1].imshow(pred_arr)
    axes[1].set_title(f"{title} Prediction")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(f"/kaggle/working/comparison_{title}.png", dpi=150)
    plt.show()

print("Inference demo functions defined.")
print("After training, call:")
print("  svg, pred = inference_demo(model, processor, renderer, sample_image)")
print("  show_comparison(sample_image, pred, title='RLRF')")


# ============================================================================
# CELL 9 — Evaluation
# ============================================================================

# !python /kaggle/working/RLRF/scripts/evaluate.py \
#     --checkpoint /kaggle/working/checkpoints/rlrf/step_100 \
#     --num_samples 50 \
#     --compute_lpips \
#     --output_csv /kaggle/working/results/eval.csv

print("Run the evaluate.py script to get paper-style metrics.")
print("Expected metrics (Table 1 scale):")
print("  MSE (↓), SSIM (↑), DINO Score (↑), LPIPS (↓), Code Efficiency (near 0)")

# ============================================================================
# CELL 9 — Load Latest Checkpoint & Run Inference
# ============================================================================

def load_and_infer(image_index=0):
    """Load the latest saved checkpoint from disk and test it!"""
    import os
    import glob
    from rlrf.model.vlm import load_model_and_processor
    from rlrf.rendering import SVGRenderer
    from peft import PeftModel
    from datasets import load_dataset
    
    # 1. Find the latest saved checkpoint
    ckpts = glob.glob(cfg.rlrf.output_dir + "/checkpoint-*")
    if not ckpts:
        print(f"No checkpoints found in {cfg.rlrf.output_dir}! Did it reach Step 10?")
        return
        
    latest_ckpt = max(ckpts, key=os.path.getmtime)
    print(f"Loading latest checkpoint: {latest_ckpt}")
    
    # 2. Load 4-bit base model & processor
    base_model, processor = load_model_and_processor(cfg.model, cfg.device_map)
    
    # 3. Apply the RLRF LoRA weights you just trained!
    model = PeftModel.from_pretrained(base_model, latest_ckpt)
    
    # 4. Setup renderer & grab a sample image
    renderer = SVGRenderer(device="cuda")
    ds = load_dataset(cfg.data.dataset_name, split=cfg.data.dataset_split)
    sample_image = ds[image_index]["image"].convert("RGB")
    
    # 5. Run Best-of-N inference!
    print("Running inference (Best of 5)... this will take a moment.")
    svg, pred_arr = inference_demo(model, processor, renderer, sample_image, best_of_n=5)
    
    step_num = latest_ckpt.split('-')[-1]
    show_comparison(sample_image, pred_arr, title=f"RLRF (Step {step_num})")

print("Added load_and_infer() helper function!")
# Uncomment to test the first image in the dataset:
# load_and_infer(image_index=0)
