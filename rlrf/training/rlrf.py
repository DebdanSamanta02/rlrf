"""
rlrf/training/rlrf.py
=====================
Stage 2: Reinforcement Learning from Rendering Feedback (RLRF).

Implements GRPO — Group Relative Policy Optimization (paper §3.1, Eq. 3-4).

Algorithm (per training step):
    1. Sample a batch of B input images from the curated dataset.
    2. For each image, generate G rollout SVG sequences using the current
       policy with temperature=1.1 (paper §4.1).
    3. Render each rollout with CairoSVG (paper §3.2).
    4. Compute composite reward R(x_c, o_i) for each rollout (paper §3.2).
    5. Compute group-centred advantage A_i = R_i − (1/G) Σ_j R_j  (Eq. 3).
    6. Compute token-level probability ratio r_t (Eq. 3).
    7. Compute clipped surrogate objective (Eq. 4):
           min(r_t · A_i, clip(r_t, 1−ε, 1+ε) · A_i)
    8. Average over tokens and rollouts; backpropagate; update policy.
    9. LR decay: 70% every 100 steps (paper §4.1).

KL regularisation: disabled (β=0), consistent with paper §4.1 and §5.

Dynamic max length (App. C.3):
    Per batch, generation max_new_tokens = max(gt_lengths_in_batch) + threshold.

Kaggle constraints:
    G=4 (vs 64 in paper), batch_size=1, gradient_accumulation=4.
    All other hyperparameters match the paper.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..config import Config, RLRFConfig
from ..data import build_rlrf_dataset, load_hf_dataset, curate_dataset, collate_fn_rlrf
from ..model import load_model_and_processor
from ..rewards import CompositeReward
from ..rewards.length import compute_dynamic_max_length

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LR scheduler: 70% decay every 100 steps (paper §4.1)
# ---------------------------------------------------------------------------

def make_rlrf_lr_scheduler(
    optimizer: AdamW,
    decay_factor: float = 0.70,
    decay_steps: int = 100,
) -> LambdaLR:
    """Create a step-decay LR scheduler matching the paper (§4.1).

    LR at step t:  lr_0 × decay_factor^(t // decay_steps)
    """
    def lr_lambda(step: int) -> float:
        exponent = step // decay_steps
        return decay_factor ** exponent

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# Log-probability computation
# ---------------------------------------------------------------------------

def compute_sequence_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    generated_ids: torch.Tensor,
    is_reference: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Compute per-token log probabilities for a generated sequence.

    Args:
        model:          Policy (or reference) model.
        input_ids:      Prompt token IDs [B, L_prompt].
        attention_mask: Prompt attention mask [B, L_prompt].
        generated_ids:  Generated token IDs [B, L_gen].
        is_reference:   If True, use torch.no_grad().

    Returns:
        Tensor of shape [B, L_gen] — log p_θ(o_t | o_{<t}, x_c).
    """
    # Concatenate prompt + generated for a full forward pass
    full_ids  = torch.cat([input_ids,  generated_ids],  dim=1)
    gen_mask  = torch.ones(generated_ids.shape, dtype=attention_mask.dtype,
                           device=attention_mask.device)
    full_mask = torch.cat([attention_mask, gen_mask], dim=1)

    ctx = torch.no_grad() if is_reference else contextlib.nullcontext()
    with ctx:
        outputs = model(input_ids=full_ids, attention_mask=full_mask, **kwargs)
        logits  = outputs.logits  # [B, L_prompt+L_gen, V]

    # We only want the logits that predict the generated tokens:
    # logits at position (L_prompt-1) predicts token L_prompt, etc.
    L_prompt = input_ids.shape[1]
    gen_logits = logits[:, L_prompt - 1 : L_prompt - 1 + generated_ids.shape[1], :]

    log_probs = F.log_softmax(gen_logits, dim=-1)  # [B, L_gen, V]
    token_log_probs = log_probs.gather(
        -1, generated_ids.unsqueeze(-1)
    ).squeeze(-1)  # [B, L_gen]

    return token_log_probs


# ---------------------------------------------------------------------------
# GRPO loss (Eq. 4)
# ---------------------------------------------------------------------------

def grpo_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.4,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the clipped GRPO surrogate loss (paper Eq. 4).

    All inputs are per-token over the generated sequence.

    Args:
        log_probs_new: log p_θ(o_t | ...)     shape [N_tokens,] flattened
        log_probs_old: log p_θ_old(o_t | ...) shape [N_tokens,]
        advantages:    A_i replicated per token shape [N_tokens,]
        epsilon:       Clip threshold (paper: 0.4).
        attention_mask: Optional mask for padding tokens.

    Returns:
        Scalar loss (negated objective, for gradient descent).
    """
    # Ratio r_t = exp(log p_new - log p_old)   (numerically stable)
    log_ratio = log_probs_new - log_probs_old
    ratio     = torch.exp(log_ratio)

    # Clipped surrogate (PPO-style)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages

    obj = torch.min(surr1, surr2)  # shape [N_tokens,]

    if attention_mask is not None:
        obj = obj * attention_mask
        loss = -obj.sum() / (attention_mask.sum() + 1e-8)
    else:
        loss = -obj.mean()

    return loss


# ---------------------------------------------------------------------------
# Main RLRF trainer
# ---------------------------------------------------------------------------

class RLRFTrainer:
    """GRPO-based RLRF trainer.

    Implements the full RLRF training loop as described in the paper §3.1-3.2.

    Args:
        model:       Policy model (with LoRA adapters, in train mode).
        processor:   AutoProcessor for tokenisation and image processing.
        reward_fn:   CompositeReward instance.
        cfg:         Master Config object.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor: object,
        reward_fn: CompositeReward,
        cfg: Config,
    ) -> None:
        self.model      = model
        self.processor  = processor
        self.reward_fn  = reward_fn
        self.cfg        = cfg
        self.rlrf_cfg   = cfg.rlrf
        self.device     = next(model.parameters()).device

        # Optimiser
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.rlrf_cfg.learning_rate,
        )
        # LR scheduler: 70% decay every 100 steps (paper §4.1)
        self.scheduler = make_rlrf_lr_scheduler(
            self.optimizer,
            decay_factor=self.rlrf_cfg.lr_decay_factor,
            decay_steps=self.rlrf_cfg.lr_decay_steps,
        )

        self.global_step = 0

    # ------------------------------------------------------------------
    # Training entry point
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        resume_step: int = 0,
    ) -> None:
        """Run the full RLRF training loop.

        Args:
            train_dataset: RLRF-mode SVGDataset.
            eval_dataset:  Optional evaluation dataset.
            resume_step:   Step to resume from (for checkpoint restarts).
        """
        rlrf = self.rlrf_cfg

        loader = DataLoader(
            train_dataset,
            batch_size=rlrf.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collate_fn_rlrf,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
        )

        self.global_step = resume_step
        os.makedirs(rlrf.output_dir, exist_ok=True)

        logger.info(
            "Starting RLRF training: %d steps, G=%d rollouts, ε=%.2f, temp=%.2f",
            rlrf.max_steps, rlrf.G, rlrf.epsilon, rlrf.temperature,
        )

        accum_loss  = 0.0
        accum_steps = 0
        reward_history: list[float] = []

        for batch in self._cycle(loader):
            if self.global_step >= rlrf.max_steps:
                break

            step_loss, step_rewards = self._train_step(batch)
            accum_loss  += step_loss
            accum_steps += 1
            reward_history.extend(step_rewards)

            # Gradient accumulation
            if accum_steps % rlrf.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), rlrf.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                avg_loss   = accum_loss / rlrf.gradient_accumulation_steps
                avg_reward = float(np.mean(reward_history[-rlrf.G * 10:])) \
                             if reward_history else 0.0
                accum_loss = 0.0

                self.global_step += 1

                if self.global_step % rlrf.logging_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        "Step %4d | loss=%.4f | reward=%.4f | lr=%.2e",
                        self.global_step, avg_loss, avg_reward, lr,
                    )

                if (
                    eval_dataset is not None
                    and self.global_step % rlrf.eval_steps == 0
                ):
                    self._evaluate(eval_dataset)

                if self.global_step % rlrf.save_steps == 0:
                    self._save(rlrf.output_dir)

        # Final save
        self._save(rlrf.output_dir)
        logger.info("RLRF training complete.")

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def _train_step(self, batch: dict) -> tuple[float, list[float]]:
        """Execute one GRPO step over a batch.

        Returns:
            (loss_value, list_of_rewards_for_all_rollouts)
        """
        rlrf = self.rlrf_cfg

        input_ids     = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values_list = batch.get("pixel_values")
        image_grid_thw_list = batch.get("image_grid_thw")
        ref_images    = batch["ref_image"]      # list[np.ndarray]
        gt_svgs       = batch["gt_svg"]         # list[str]
        gt_lengths    = batch["gt_length"]      # list[int]

        B = input_ids.shape[0]

        # ── Dynamic max length (App. C.3) ─────────────────────────────────
        max_new = compute_dynamic_max_length(
            list(gt_lengths), rlrf.dynamic_len_threshold
        )
        max_new = min(max_new, rlrf.max_new_tokens)

        # ── Rollout generation (frozen policy snapshot) ───────────────────
        self.model.eval()
        all_gen_ids:    list[torch.Tensor] = []  # [G*B, L_gen] after flatten
        all_gen_texts:  list[str]          = []
        all_advantages: list[float]        = []
        all_rewards:    list[float]         = []

        with torch.no_grad():
            for b in range(B):
                group_texts: list[str]  = []
                group_rewards: list[float] = []
                group_ids: list[torch.Tensor] = []

                for _ in range(rlrf.G):
                    # Build single-item inputs
                    gen_kwargs: dict = dict(
                        input_ids=input_ids[b : b + 1],
                        attention_mask=attention_mask[b : b + 1],
                        max_new_tokens=max_new,
                        do_sample=True,
                        temperature=rlrf.temperature,
                        top_p=rlrf.top_p,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                    )
                    if pixel_values_list is not None:
                        gen_kwargs["pixel_values"] = pixel_values_list[b].to(self.device)
                    if image_grid_thw_list is not None:
                        gen_kwargs["image_grid_thw"] = image_grid_thw_list[b].unsqueeze(0).to(self.device)

                    output_ids = self.model.generate(**gen_kwargs)

                    # Isolate only the newly generated tokens
                    prompt_len = input_ids.shape[1]
                    gen_ids    = output_ids[:, prompt_len:]   # [1, L_gen]
                    gen_text   = self.processor.tokenizer.decode(
                        gen_ids[0], skip_special_tokens=True
                    )

                    # Compute reward by rendering
                    reward_val = self.reward_fn.reward_scalar(
                        ref_images[b], gen_text, int(gt_lengths[b])
                    )

                    group_ids.append(gen_ids.squeeze(0))   # [L_gen]
                    group_texts.append(gen_text)
                    group_rewards.append(reward_val)

                # Group-centred advantage (paper Eq. 3)
                mean_r  = float(np.mean(group_rewards))
                adv     = [r - mean_r for r in group_rewards]

                all_gen_ids.extend(group_ids)
                all_gen_texts.extend(group_texts)
                all_advantages.extend(adv)
                all_rewards.extend(group_rewards)

        # ── Compute GRPO loss ─────────────────────────────────────────────
        self.model.train()
        
        # Count valid pairs to scale the loss correctly
        n_pairs = sum(1 for gen_ids in all_gen_ids if gen_ids.numel() > 0)
        total_loss_val = 0.0

        for idx, (gen_ids, adv_val) in enumerate(
            zip(all_gen_ids, all_advantages)
        ):
            b_idx = idx // rlrf.G   # which image in the batch

            if gen_ids.numel() == 0:
                continue

            gen_ids_2d = gen_ids.unsqueeze(0).to(self.device)   # [1, L_gen]
            inp_2d     = input_ids[b_idx : b_idx + 1]
            mask_2d    = attention_mask[b_idx : b_idx + 1]
            
            vision_kwargs = {}
            if pixel_values_list is not None:
                vision_kwargs["pixel_values"] = pixel_values_list[b_idx].to(self.device)
            if image_grid_thw_list is not None:
                vision_kwargs["image_grid_thw"] = image_grid_thw_list[b_idx].unsqueeze(0).to(self.device)

            # Current policy log probs
            lp_new = compute_sequence_log_probs(
                self.model, inp_2d, mask_2d, gen_ids_2d,
                is_reference=False, **vision_kwargs
            ).squeeze(0)  # [L_gen]

            # Old policy log probs (computed without gradient at generation time;
            # re-use the same frozen snapshot for efficiency)
            with torch.no_grad():
                lp_old = compute_sequence_log_probs(
                    self.model, inp_2d, mask_2d, gen_ids_2d,
                    is_reference=True, **vision_kwargs
                ).squeeze(0).detach()

            adv_tensor = torch.full_like(lp_new, adv_val)

            loss_i = grpo_loss(lp_new, lp_old, adv_tensor, rlrf.epsilon)
            
            # Backpropagate immediately to free the computational graph and save VRAM
            scaled_loss = loss_i / n_pairs / self.rlrf_cfg.gradient_accumulation_steps
            scaled_loss.backward()
            
            total_loss_val += float(loss_i.item()) / n_pairs

        return total_loss_val, all_rewards

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, eval_dataset: torch.utils.data.Dataset) -> dict:
        """Run a quick reward evaluation on the eval split."""
        loader = DataLoader(
            eval_dataset,
            batch_size=1,
            collate_fn=collate_fn_rlrf,
        )
        rewards = []
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= 20:   # limit eval to 20 examples
                    break
                input_ids     = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                ref_image  = batch["ref_image"][0]
                gt_length  = int(batch["gt_length"][0])

                gen_kwargs = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.rlrf_cfg.max_new_tokens,
                    do_sample=False,       # greedy for eval
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
                if "pixel_values" in batch:
                    gen_kwargs["pixel_values"] = batch["pixel_values"][0].to(self.device)
                if "image_grid_thw" in batch:
                    gen_kwargs["image_grid_thw"] = batch["image_grid_thw"][0].unsqueeze(0).to(self.device)

                output_ids = self.model.generate(**gen_kwargs)
                gen_text   = self.processor.tokenizer.decode(
                    output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
                )
                r = self.reward_fn.reward_scalar(ref_image, gen_text, gt_length)
                rewards.append(r)

        mean_r = float(np.mean(rewards)) if rewards else 0.0
        logger.info("Eval step %d | mean_reward=%.4f", self.global_step, mean_r)
        self.model.train()
        return {"eval_reward": mean_r}

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def _save(self, output_dir: str) -> None:
        """Save LoRA adapter weights and processor."""
        ckpt_dir = os.path.join(output_dir, f"step_{self.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model.save_pretrained(ckpt_dir)
        self.processor.save_pretrained(ckpt_dir)
        logger.info("Checkpoint saved to %s", ckpt_dir)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cycle(loader: DataLoader):
        """Infinite cycle over a DataLoader."""
        while True:
            yield from loader


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def run_rlrf(
    cfg: Config,
    sft_checkpoint: Optional[str] = None,
    resume_step: int = 0,
) -> None:
    """Run Stage 2 RLRF training.

    Args:
        cfg:            Master Config object.
        sft_checkpoint: Path to SFT checkpoint to start from
                        (defaults to cfg.sft.output_dir).
        resume_step:    Step number to resume from.
    """
    from peft import PeftModel
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    rlrf_cfg = cfg.rlrf

    # ── Load SFT checkpoint or base model ────────────────────────────────
    ckpt = sft_checkpoint or cfg.sft.output_dir
    logger.info("Loading policy model from: %s", ckpt)

    try:
        # If ckpt is a PeftModel directory
        processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model.model_name,
            torch_dtype=torch.bfloat16,
            device_map=cfg.device_map,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, ckpt)
        model = model.merge_and_unload()   # merge LoRA into base weights
        # Re-apply fresh LoRA for RLRF fine-tuning
        from peft import get_peft_model, LoraConfig
        from ..model.vlm import load_model_and_processor
        model, processor = load_model_and_processor(cfg.model, cfg.device_map)
    except Exception:
        # Fallback: load from scratch
        logger.warning("Could not load SFT checkpoint; starting from base model.")
        model, processor = load_model_and_processor(cfg.model, cfg.device_map)

    model.train()

    # ── Load and curate RLRF dataset ──────────────────────────────────────
    raw = load_hf_dataset(
        cfg.data.dataset_name,
        split=cfg.data.dataset_split,
        cache_dir=cfg.data.cache_dir,
        max_samples=cfg.data.max_train_samples * 3,
    )
    curated = curate_dataset(
        raw,
        min_tokens=cfg.data.min_gt_tokens,
        max_samples=cfg.data.max_train_samples,
        skip_entropy=True,
    )

    n_eval   = min(cfg.data.max_test_samples, len(curated) // 10)
    train_r  = curated[n_eval:]
    eval_r   = curated[:n_eval]

    train_ds = build_rlrf_dataset(train_r, processor,
                                  cfg.model.max_seq_length, cfg.model.render_size)
    eval_ds  = build_rlrf_dataset(eval_r,  processor,
                                  cfg.model.max_seq_length, cfg.model.render_size)

    logger.info("RLRF dataset — train: %d, eval: %d", len(train_ds), len(eval_ds))

    # ── Reward function ───────────────────────────────────────────────────
    from ..rendering import SVGRenderer
    renderer  = SVGRenderer(size=cfg.model.render_size, enforce_viewbox=True)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    reward_fn = CompositeReward(cfg.reward, renderer=renderer, device=device_str)

    # ── Run training ──────────────────────────────────────────────────────
    trainer = RLRFTrainer(model, processor, reward_fn, cfg)
    trainer.train(train_ds, eval_ds, resume_step=resume_step)
