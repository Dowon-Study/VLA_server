# OpenVLA Fine-tuning on LIBERO

LoRA fine-tuning of [OpenVLA-7B](https://huggingface.co/openvla/openvla-7b) on LIBERO robot manipulation tasks.

Hyperparameters follow **Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model" (2024)** — [arXiv:2406.09246](https://arxiv.org/abs/2406.09246).

---

## Setup

```bash
git clone <this-repo>
cd OpenVLA
bash setup.sh
conda activate openvla
```

`setup.sh` creates a `conda` environment, installs PyTorch cu128 (B200 / CUDA 12.8), and all required packages. flash-attention-2 is built from source for maximum throughput.

---

## Dataset format

Expects a **lerobot v3** dataset directory:

```
dataset_root/
├── meta/
│   ├── info.json          # dataset metadata
│   ├── stats.json         # action normalization statistics
│   ├── tasks.jsonl        # task_index → language instruction
│   └── episodes.jsonl
├── data/
│   └── chunk-000/
│       ├── file-000.parquet   # episode 0  (action, state, indices)
│       └── ...
└── videos/
    └── observation.images.image/
        └── chunk-000/
            ├── file-000.mp4   # episode 0 frames
            └── ...
```

> **Compatible with** the merged dataset produced by `merge_with_hf_libero.py` from the SmolVLA training pipeline (same format, no conversion needed).

---

## Training

### Quick start (single GPU)

```bash
python train.py \
    --dataset_root /path/to/dataset_hf_aug_mix \
    --output_dir   ./runs/exp01
```

### Multi-GPU on B200 server

```bash
bash scripts/run_train.sh \
    --dataset_root /path/to/dataset_hf_aug_mix \
    --ngpu 4
```

### Key CLI overrides

| Flag | Default (config) | Description |
|------|-----------------|-------------|
| `--max_steps` | 50 000 | Total training steps |
| `--batch_size` | 16 | Per-device batch size |
| `--lr` | 2e-5 | Learning rate |
| `--lora_rank` | 32 | LoRA rank |
| `--camera_key` | `observation.images.image` | Camera to use |
| `--wandb` | off | Enable W&B logging |

---

## Hyperparameters (OpenVLA paper §B.2)

| Hyperparameter | Value |
|----------------|-------|
| Base model | `openvla/openvla-7b` |
| Fine-tuning | LoRA (rank 32, α 16) |
| LoRA target | All linear layers (q/k/v/o/gate/up/down) |
| Optimizer | AdamW β₁=0.9, β₂=0.95 |
| Learning rate | 2 × 10⁻⁵ |
| LR schedule | Constant with 500-step linear warmup |
| Batch size (per GPU) | 16 |
| Gradient clipping | 1.0 |
| Weight decay | 0.1 |
| Action bins | 256 per dimension |
| Image size | 224 × 224 |
| Precision | BF16 |

---

## Action tokenization

Each of the 7 action dimensions is normalized to [−1, 1] using dataset `stats.json` (min/max), then discretized into one of 256 bins and mapped to a special vocabulary token `<ACTION_0>` … `<ACTION_255>`.

Output prompt format:
```
In: What action should the robot take to {instruction}?
Out: <ACTION_128><ACTION_50><ACTION_200><ACTION_100><ACTION_130><ACTION_110><ACTION_255>
```

---

## Project structure

```
OpenVLA/
├── configs/
│   └── libero_lora.yaml   # all hyperparameters
├── src/
│   ├── action_tokenizer.py
│   └── dataset.py
├── scripts/
│   └── run_train.sh       # torchrun launcher
├── train.py               # main training script (HF Trainer)
├── requirements.txt
└── setup.sh
```
