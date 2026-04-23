#!/usr/bin/env python3
"""
OpenVLA fine-tuning on LIBERO dataset using LoRA.

Reference: Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model" (2024)
           https://arxiv.org/abs/2406.09246

Usage:
    # Single GPU
    python train.py --dataset_root /path/to/dataset --output_dir ./runs/exp01

    # Multi-GPU (via run_train.sh which calls torchrun)
    torchrun --nproc_per_node=4 train.py --dataset_root /path/to/dataset --output_dir ./runs/exp01

    # Override config hyperparams
    python train.py --dataset_root /data --output_dir ./runs/exp01 \\
        --max_steps 100000 --batch_size 32 --lr 1e-5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

from src.action_tokenizer import ActionTokenizer
from src.dataset import DataCollatorForOpenVLA, LIBERODataset


# ──────────────────────────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────────────────────────

class _Namespace:
    """Recursive dict-to-attribute converter."""
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, _Namespace(v) if isinstance(v, dict) else v)
    def get(self, key, default=None):
        return getattr(self, key, default)


def load_config(path: str) -> _Namespace:
    with open(path) as f:
        return _Namespace(yaml.safe_load(f))


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="OpenVLA LoRA fine-tuning on LIBERO")
    p.add_argument("--config",       default="configs/libero_lora.yaml")
    p.add_argument("--dataset_root", required=True,
                   help="Root of lerobot-format LIBERO dataset")
    p.add_argument("--output_dir",   required=True,
                   help="Directory for checkpoints and logs")

    # CLI overrides (all optional — fall back to config values)
    p.add_argument("--pretrained",   default=None,  help="HF model ID or local path")
    p.add_argument("--max_steps",    type=int,   default=None)
    p.add_argument("--batch_size",   type=int,   default=None,
                   help="Per-device train batch size")
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--lora_rank",    type=int,   default=None)
    p.add_argument("--lora_alpha",   type=int,   default=None)
    p.add_argument("--save_steps",   type=int,   default=None)
    p.add_argument("--camera_key",   default=None,
                   help="Camera key, e.g. observation.images.image")
    p.add_argument("--wandb",        action="store_true",
                   help="Enable Weights & Biases logging")
    p.add_argument("--run_name",     default=None)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    pretrained   = args.pretrained  or cfg.model.pretrained
    max_steps    = args.max_steps   or cfg.training.max_steps
    batch_size   = args.batch_size  or cfg.training.per_device_train_batch_size
    lr           = args.lr          or cfg.training.learning_rate
    lora_rank    = args.lora_rank   or cfg.lora.rank
    lora_alpha   = args.lora_alpha  or cfg.lora.alpha
    save_steps   = args.save_steps  or cfg.training.save_steps
    camera_key   = args.camera_key  or cfg.data.camera_key

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Print run config ──────────────────────────────────────────────
    print("=" * 60)
    print("  OpenVLA LoRA fine-tuning — LIBERO")
    print("=" * 60)
    print(f"  Pretrained   : {pretrained}")
    print(f"  Dataset root : {args.dataset_root}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Max steps    : {max_steps}")
    print(f"  Batch/GPU    : {batch_size}")
    print(f"  LR           : {lr}")
    print(f"  LoRA rank    : {lora_rank} / alpha: {lora_alpha}")
    print(f"  Camera       : {camera_key}")
    print("=" * 60)

    # ── Load processor ────────────────────────────────────────────────
    print("\n[1/4] Loading processor ...")
    processor = AutoProcessor.from_pretrained(
        pretrained, trust_remote_code=True
    )
    # OpenVLA uses left-padding
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ── Setup action tokenizer ────────────────────────────────────────
    action_tokenizer = ActionTokenizer(
        processor.tokenizer, n_bins=cfg.data.action_bins
    )

    # ── Load model ────────────────────────────────────────────────────
    print("[2/4] Loading model ...")
    model = AutoModelForVision2Seq.from_pretrained(
        pretrained,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=cfg.model.attn_implementation,
    )

    # Resize embeddings to include newly added action tokens
    # pad_to_multiple_of=64 keeps matmul-friendly sizes
    model.resize_token_embeddings(
        len(processor.tokenizer), pad_to_multiple_of=64
    )

    # ── Apply LoRA ────────────────────────────────────────────────────
    print("[3/4] Applying LoRA ...")
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=cfg.lora.target_modules,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────
    print("[4/4] Building dataset ...")
    train_dataset = LIBERODataset(
        dataset_root=args.dataset_root,
        processor=processor,
        action_tokenizer=action_tokenizer,
        camera_key=camera_key,
        image_size=cfg.model.image_size,
        norm_mode=cfg.data.norm_mode,
    )
    print(f"  Dataset size: {len(train_dataset)} frames")

    collator = DataCollatorForOpenVLA(processor.tokenizer)

    # ── Training arguments ────────────────────────────────────────────
    report_to = "wandb" if args.wandb else "none"
    run_name  = args.run_name or f"openvla_libero_lora_r{lora_rank}"

    warmup_steps = cfg.training.warmup_steps
    decay_steps  = max(1, max_steps - warmup_steps)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # Steps
        max_steps=max_steps,
        # Batch
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        # Optimizer (AdamW, β1=0.9, β2=0.95, paper §B.2)
        learning_rate=lr,
        lr_scheduler_type=cfg.training.lr_scheduler,
        warmup_steps=warmup_steps,
        weight_decay=cfg.training.weight_decay,
        adam_beta1=cfg.training.adam_beta1,
        adam_beta2=cfg.training.adam_beta2,
        max_grad_norm=cfg.training.max_grad_norm,
        # Precision
        bf16=cfg.training.bf16,
        tf32=True,          # Ampere+ tensor cores
        # Save / log
        save_steps=save_steps,
        save_total_limit=cfg.training.save_total_limit,
        logging_steps=cfg.training.logging_steps,
        logging_first_step=True,
        # Misc
        seed=cfg.training.seed,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=report_to,
        run_name=run_name,
    )

    # ── Train ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    print("\nTraining start ...")
    trainer.train()

    # ── Save final checkpoint ─────────────────────────────────────────
    final_dir = Path(args.output_dir) / "final_lora"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"\nSaved final LoRA checkpoint → {final_dir}")

    # Save run metadata
    meta = {
        "pretrained": pretrained,
        "dataset_root": args.dataset_root,
        "max_steps": max_steps,
        "lr": lr,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "batch_size": batch_size,
    }
    with open(Path(args.output_dir) / "run_config.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
