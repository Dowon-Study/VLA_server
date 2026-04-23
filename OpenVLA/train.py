#!/usr/bin/env python3
"""
OpenVLA full fine-tuning on LIBERO dataset.

Reference: Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model" (2024)
           https://arxiv.org/abs/2406.09246

Usage:
    # Single GPU
    python train.py --dataset_root /path/to/dataset --output_dir ./runs/exp01

    # Multi-GPU with FSDP (B200 × 16, via run_train.sh)
    accelerate launch --config_file configs/fsdp_config.yaml train.py \\
        --dataset_root /path/to/dataset --output_dir ./runs/exp01

    # CLI overrides
    python train.py --dataset_root /data --output_dir ./runs/exp01 \\
        --max_steps 20000 --batch_size 32 --lr 1e-5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
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
    p = argparse.ArgumentParser(description="OpenVLA full fine-tuning on LIBERO")
    p.add_argument("--config",       default="configs/libero_full.yaml")
    p.add_argument("--dataset_root", required=True,
                   help="Root of lerobot-format LIBERO dataset")
    p.add_argument("--output_dir",   required=True,
                   help="Directory for checkpoints and logs")

    # CLI overrides
    p.add_argument("--pretrained",   default=None,  help="HF model ID or local path")
    p.add_argument("--max_steps",    type=int,   default=None)
    p.add_argument("--batch_size",   type=int,   default=None, help="Per-device train batch size")
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--save_steps",   type=int,   default=None)
    p.add_argument("--camera_key",   default=None)
    p.add_argument("--wandb",        action="store_true")
    p.add_argument("--run_name",     default=None)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    pretrained = args.pretrained or cfg.model.pretrained
    max_steps  = args.max_steps  or cfg.training.max_steps
    batch_size = args.batch_size or cfg.training.per_device_train_batch_size
    lr         = args.lr         or cfg.training.learning_rate
    save_steps = args.save_steps or cfg.training.save_steps
    camera_key = args.camera_key or cfg.data.camera_key

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Print run config ──────────────────────────────────────────────
    num_gpus = torch.cuda.device_count()
    eff_batch = batch_size * max(num_gpus, 1) * cfg.training.gradient_accumulation_steps
    print("=" * 60)
    print("  OpenVLA Full Fine-tuning — LIBERO")
    print("=" * 60)
    print(f"  Pretrained   : {pretrained}")
    print(f"  Dataset root : {args.dataset_root}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Max steps    : {max_steps}")
    print(f"  Batch/GPU    : {batch_size}  (effective: {eff_batch})")
    print(f"  LR           : {lr}")
    print(f"  GPUs         : {num_gpus}")
    print(f"  Camera       : {camera_key}")
    print("=" * 60)

    # ── Load processor ────────────────────────────────────────────────
    print("\n[1/3] Loading processor ...")
    processor = AutoProcessor.from_pretrained(
        pretrained, trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ── Setup action tokenizer ────────────────────────────────────────
    action_tokenizer = ActionTokenizer(
        processor.tokenizer, n_bins=cfg.data.action_bins
    )

    # ── Load model ────────────────────────────────────────────────────
    print("[2/3] Loading model ...")
    model = AutoModelForVision2Seq.from_pretrained(
        pretrained,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=cfg.model.attn_implementation,
    )

    # Resize embeddings for newly added action tokens
    model.resize_token_embeddings(
        len(processor.tokenizer), pad_to_multiple_of=64
    )

    # Gradient checkpointing: 활성화 메모리 절약 (full FT 필수)
    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Full fine-tuning: 모든 파라미터 학습
    for param in model.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {total_params / 1e9:.2f}B (100%)")

    # ── Dataset ───────────────────────────────────────────────────────
    print("[3/3] Building dataset ...")
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
    run_name  = args.run_name or "openvla_libero_full"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # Steps
        max_steps=max_steps,
        # Batch
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        # Optimizer (AdamW β1=0.9, β2=0.95, paper §B.2)
        learning_rate=lr,
        lr_scheduler_type=cfg.training.lr_scheduler,
        warmup_steps=cfg.training.warmup_steps,
        weight_decay=cfg.training.weight_decay,
        adam_beta1=cfg.training.adam_beta1,
        adam_beta2=cfg.training.adam_beta2,
        max_grad_norm=cfg.training.max_grad_norm,
        # Precision
        bf16=cfg.training.bf16,
        tf32=True,
        # Memory: gradient checkpointing은 model에서 이미 활성화
        gradient_checkpointing=False,   # Trainer 중복 설정 방지
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
        # FSDP: accelerate config로 제어 (여기선 비워둠)
        ddp_find_unused_parameters=False,
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
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"\nSaved final checkpoint → {final_dir}")

    # Save run metadata
    meta = {
        "pretrained": pretrained,
        "dataset_root": args.dataset_root,
        "max_steps": max_steps,
        "lr": lr,
        "batch_size": batch_size,
        "effective_batch": eff_batch,
        "fine_tuning": "full",
    }
    with open(Path(args.output_dir) / "run_config.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
