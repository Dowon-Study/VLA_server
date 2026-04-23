# OpenVLA Full Fine-tuning on LIBERO

[OpenVLA-7B](https://huggingface.co/openvla/openvla-7b)을 LIBERO 로봇 조작 데이터셋으로 **Full Fine-tuning**하는 코드입니다.

> **왜 Full Fine-tuning인가?**  
> 기존 연구들은 메모리 제약으로 LoRA(rank 32)를 사용했습니다.  
> B200 × 16 GPU(총 2,880 GB VRAM) 환경에서는 7B 파라미터 전체를 학습할 수 있으며,  
> LoRA 대비 더 높은 성능을 기대할 수 있습니다.

---

## 모델 개요

**OpenVLA** (Kim et al., 2024 — [arXiv:2406.09246](https://arxiv.org/abs/2406.09246))는  
SigLIP + DINOv2 비전 인코더와 Llama-2-7B 언어 모델을 결합한 7B VLA 모델입니다.

- 입력: RGB 이미지 (224×224) + 자연어 명령
- 출력: 7차원 로봇 액션 (x, y, z, rx, ry, rz, gripper)
- 액션 표현: 각 차원을 256개 bin으로 이산화 → 어휘 토큰으로 예측

---

## 학습 환경

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA B200 × 16 |
| GPU당 VRAM | 180 GB HBM3e |
| 총 VRAM | **2,880 GB** |
| CUDA | 12.8+ |
| 병렬화 | FSDP ZeRO-3 (파라미터 + 옵티마이저 상태 분산) |
| 정밀도 | BF16 + TF32 |
| Attention | Flash Attention 2 |

### Full Fine-tuning 메모리 분석

OpenVLA-7B (BF16 기준):

| 항목 | 메모리 |
|------|--------|
| 모델 파라미터 (BF16) | ~14 GB |
| 옵티마이저 상태 (FP32 Adam) | ~28 GB |
| 그라디언트 (BF16) | ~14 GB |
| Activation (배치=16, GC on) | ~10 GB |
| **합계 (single GPU)** | **~66 GB** |
| **FSDP ZeRO-3 (÷16 GPU)** | **~4~5 GB/GPU** |

→ B200(180GB/GPU) 기준 VRAM 사용률 **3% 미만**, 배치 확장 여유 충분

---

## 학습 설정

### 핵심 하이퍼파라미터

| 파라미터 | 값 | 근거 |
|----------|-----|------|
| **Learning Rate** | **2e-5** | Full FT 표준 (LoRA 5e-4의 1/25) |
| **LR 스케줄** | Constant + 500-step warmup | OpenVLA 논문 §B.2 |
| **배치 (GPU당)** | **16** | 논문과 동일한 per-GPU 배치 |
| **유효 배치** | **256** (16 × 16 GPU) | 논문 128(8GPU)의 2배 |
| **학습 스텝** | **50,000** | 논문 기준 (Spatial/Object) |
| LIBERO-10 스텝 | 80,000 | 논문 기준 (장기 태스크) |
| **옵티마이저** | AdamW | β₁=0.9, β₂=0.95 |
| **Weight decay** | 0.1 | 논문 §B.2 |
| **Gradient clipping** | 1.0 | 표준값 |
| **Gradient checkpointing** | ON | activation 메모리 절약 |
| **병렬화** | FSDP ZeRO-3 | 16 GPU 메모리 분산 |

### LoRA vs Full Fine-tuning 비교

| 항목 | LoRA (논문) | **Full FT (본 실험)** |
|------|------------|----------------------|
| 학습 파라미터 | ~1% (26M) | **100% (7B)** |
| Learning Rate | 5e-4 | **2e-5** |
| 필요 VRAM | ~72 GB (8×A100) | ~66 GB (single B200) |
| FSDP | 불필요 | **ZeRO-3** |
| 기대 성능 | 논문 기준선 | 논문 기준선 이상 |

### 액션 토큰화

7차원 연속 액션을 다음과 같이 이산화합니다:

```
raw action (7-dim)
    → stats.json min/max로 [-1, 1] 정규화
    → 256 bins으로 이산화
    → <ACTION_0> ~ <ACTION_255> 토큰으로 변환
    → LLM이 7개 토큰 순차 예측
```

입력 프롬프트 형식:
```
In: What action should the robot take to {instruction}?
Out: <ACTION_128><ACTION_050><ACTION_200><ACTION_100><ACTION_130><ACTION_110><ACTION_255>
```

---

## 실험 설계 (증강 기법 검증)

본 코드는 **IK 기반 노이즈 상태 복구 증강 기법**의 효과를 검증하기 위해 설계되었습니다.

```
동일 모델 (OpenVLA Full FT) + 동일 하이퍼파라미터
         │
         ├── Condition A: HF + GT 데이터만  ← 대조군 (aug 없음)
         │
         └── Condition B: HF + GT + 증강 데이터  ← 실험군 (aug 있음)

평가:
  ① noise_std = 0.0   — 일반 환경
  ② noise_std = 0.05  — 노이즈 환경  ← 증강 효과 핵심 지표
  ③ noise_std = 0.10  — 강한 노이즈 환경
```

Condition A vs B의 noisy 평가 성능 차이가 **증강 기법의 효과를 직접 증명**합니다.

---

## 빠른 시작

### 1. 환경 설치

```bash
git clone https://github.com/Dowon-Study/VLA_server.git
cd VLA_server/OpenVLA
bash setup.sh
conda activate openvla
```

### 2. 학습 실행

```bash
# 대조군 (aug 없음)
bash scripts/run_train.sh \
    --dataset_root /path/to/dataset_no_aug \
    --run_name full_ft_baseline

# 실험군 (aug 있음)
bash scripts/run_train.sh \
    --dataset_root /path/to/dataset_with_aug \
    --run_name full_ft_aug
```

### 3. LIBERO-10 (더 긴 학습)

```bash
bash scripts/run_train.sh \
    --dataset_root /path/to/dataset \
    --max_steps 80000 \
    --run_name full_ft_libero10
```

### 4. 주요 CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--dataset_root` | 필수 | 데이터셋 경로 |
| `--max_steps` | 50,000 | 학습 스텝 수 |
| `--batch_size` | 16 | GPU당 배치 크기 |
| `--lr` | 2e-5 | 학습률 |
| `--ngpu` | 자동감지 | 사용 GPU 수 |
| `--run_name` | 타임스탬프 | 실험 이름 |
| `--wandb` | off | W&B 로깅 활성화 |

---

## 데이터셋 형식

lerobot v3 포맷을 사용합니다 (SmolVLA 학습용 병합 데이터셋과 동일 포맷):

```
dataset_root/
├── meta/
│   ├── info.json        # 데이터셋 메타데이터
│   ├── stats.json       # 액션 정규화 통계 (min/max)
│   ├── tasks.jsonl      # task_index → 언어 명령
│   └── episodes.jsonl
├── data/
│   └── chunk-000/
│       ├── file-000.parquet   # 에피소드별 액션/상태 데이터
│       └── ...
└── videos/
    └── observation.images.image/
        └── chunk-000/
            ├── file-000.mp4   # 에피소드별 이미지 프레임
            └── ...
```

---

## 프로젝트 구조

```
OpenVLA/
├── configs/
│   ├── libero_full.yaml    # 학습 하이퍼파라미터 (Full FT)
│   └── fsdp_config.yaml    # FSDP ZeRO-3 설정 (16 GPU)
├── src/
│   ├── action_tokenizer.py # 액션 256-bin 이산화
│   └── dataset.py          # LIBERO parquet+mp4 데이터 로더
├── scripts/
│   └── run_train.sh        # B200 × 16 GPU 학습 런처
├── train.py                # 메인 학습 스크립트 (HF Trainer + FSDP)
├── requirements.txt
└── setup.sh                # 환경 설치 스크립트
```

---

## 참고 문헌

- Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model" (2024) — [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
- Hejna et al., "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success" (OpenVLA-OFT, 2025) — [arXiv:2502.19645](https://arxiv.org/abs/2502.19645)
