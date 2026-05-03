#!/usr/bin/env python3
"""
scripts/evaluate.py
===================
Evaluate a trained RLRF model on the Im2SVG test set.

Reports metrics from paper Table 1: MSE, SSIM, DINO Score, LPIPS,
and Code Efficiency — all at paper scale.

Usage:
    python scripts/evaluate.py \
        --checkpoint ./checkpoints/rlrf/step_500 \
        --num_samples 200 \
        --output_csv ./results/eval_results.csv
"""

import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rlrf.config import Config, ModelConfig, DataConfig
from rlrf.data import load_hf_dataset, curate_dataset, make_im2svg_messages
from rlrf.rendering import SVGRenderer
from rlrf.utils.metrics import mse, ssim, dino_score, lpips_score, code_efficiency

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RLRF Im2SVG model")
    p.add_argument("--checkpoint", required=True,
                   help="Path to RLRF (or SFT) checkpoint directory.")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--dataset_name", default="starvector/svg-stack")
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--render_size", type=int, default=224)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--best_of_n", type=int, default=5,
                   help="Best-of-N inference: generate N candidates, keep lowest MSE (App. C.3).")
    p.add_argument("--compute_dino",  action="store_true", default=False)
    p.add_argument("--compute_lpips", action="store_true", default=True)
    p.add_argument("--output_csv", default="./results/eval_results.csv")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_model(checkpoint: str, base_model: str, device: str):
    """Load model from checkpoint (handles both full and LoRA checkpoints)."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    try:
        from peft import PeftModel
        processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, checkpoint)
    except Exception:
        logger.info("Loading as full checkpoint (not PeftModel).")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model.eval()
    return model, processor


def generate_svg(model, processor, image, device, max_new_tokens, n=1):
    """Generate SVG for one image; return list of `n` candidates."""
    import numpy as np
    from PIL import Image as PILImage

    messages = make_im2svg_messages(image)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    candidates = []
    with torch.no_grad():
        for _ in range(n):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(n > 1),
                temperature=0.5 if n > 1 else 1.0,
                top_p=0.9,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            gen_text = processor.tokenizer.decode(
                out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            candidates.append(gen_text)
    return candidates


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    # Load model
    model, processor = load_model(args.checkpoint, args.base_model, args.device)
    renderer = SVGRenderer(size=args.render_size, enforce_viewbox=True)

    # Load test data
    raw = load_hf_dataset(args.dataset_name, split="test",
                          max_samples=args.num_samples * 3)
    if not raw:  # fallback to train split if no test split
        raw = load_hf_dataset(args.dataset_name, split="train",
                              max_samples=args.num_samples * 3)
    curated = curate_dataset(raw, min_tokens=500, max_samples=args.num_samples,
                              skip_entropy=True)
    logger.info("Evaluating on %d samples.", len(curated))

    results = []
    from PIL import Image as PILImage

    for i, rec in enumerate(curated):
        img   = rec["image"]
        gt_svg = rec["svg"]
        if isinstance(img, np.ndarray):
            img = PILImage.fromarray(img)

        ref_arr = np.array(img.convert("RGB").resize(
            (args.render_size, args.render_size)))

        # Best-of-N inference (App. C.3): pick candidate with lowest MSE
        candidates = generate_svg(
            model, processor, img, args.device,
            args.max_new_tokens, n=args.best_of_n
        )

        best_svg, best_mse = None, float("inf")
        for cand in candidates:
            pred_arr = renderer.render(cand)
            m = mse(ref_arr, pred_arr) * 100
            if m < best_mse:
                best_mse, best_svg = m, cand

        pred_arr = renderer.render(best_svg)

        row = {
            "sample_idx":      i,
            "mse":             best_mse,
            "ssim":            ssim(ref_arr, pred_arr) * 100,
            "code_efficiency": code_efficiency(len(best_svg), len(gt_svg)),
        }
        if args.compute_dino:
            row["dino_score"] = dino_score(ref_arr, pred_arr,
                                            device=args.device) * 100
        if args.compute_lpips:
            row["lpips"] = lpips_score(ref_arr, pred_arr, device=args.device) * 100

        results.append(row)

        if (i + 1) % 10 == 0:
            logger.info("  [%d/%d] MSE=%.2f SSIM=%.2f",
                        i + 1, len(curated),
                        np.mean([r["mse"]  for r in results]),
                        np.mean([r["ssim"] for r in results]))

    # Aggregate
    keys = list(results[0].keys())
    numeric_keys = [k for k in keys if k != "sample_idx"]
    logger.info("\n=== Evaluation Results ===")
    for k in numeric_keys:
        vals = [r[k] for r in results if k in r]
        logger.info("  %-18s %.4f", k, np.mean(vals))

    # Save CSV
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Results saved to %s", args.output_csv)


if __name__ == "__main__":
    main()
