# VLA Server

B200 서버에서 실행하는 VLA(Vision-Language-Action) 모델 학습 코드 모음입니다.  
각 폴더는 독립적인 실험 단위로 구성되며, `git clone` 후 바로 실행 가능하도록 설계되어 있습니다.

---

## 현재 구성

| 폴더 | 설명 |
|------|------|
| [`OpenVLA/`](OpenVLA/) | OpenVLA-7B LoRA 파인튜닝 on LIBERO |

---

## OpenVLA — LIBERO 파인튜닝

[`OpenVLA/`](OpenVLA/) 폴더는 **OpenVLA-7B** 모델을 LIBERO 로봇 조작 데이터셋으로 파인튜닝하는 코드입니다.

### 모델 개요

[OpenVLA](https://arxiv.org/abs/2406.09246) (Kim et al., 2024)는 7B 파라미터 기반의 오픈소스 VLA 모델입니다.  
SigLIP + DINOv2 비전 인코더와 Llama-2-7B 언어 모델을 결합한 구조로,  
로봇 팔의 액션(7차원)을 언어 명령과 이미지만으로 직접 예측합니다.

### 파인튜닝 방식

- **LoRA** (rank 32, alpha 16) — 전체 파라미터의 약 1%만 학습
- 액션 이산화: 각 액션 차원을 256개의 bin으로 나눠 토큰으로 변환
- 입력 프롬프트 형식:
  ```
  In: What action should the robot take to {instruction}?
  Out: <ACTION_128><ACTION_050>...  (7개 토큰)
  ```

### 주요 하이퍼파라미터 (논문 §B.2 기준)

| 항목 | 값 |
|------|----|
| Learning rate | 2e-5 |
| LR 스케줄 | Constant + 500-step warmup |
| 배치 크기 (GPU당) | 16 |
| LoRA rank / alpha | 32 / 16 |
| 옵티마이저 | AdamW (β₁=0.9, β₂=0.95) |
| 학습 스텝 | 50,000 |
| 액션 bins | 256 |
| 이미지 해상도 | 224 × 224 |
| 정밀도 | BF16 |

### 빠른 시작

```bash
git clone https://github.com/Dowon-Study/VLA_server.git
cd VLA_server/OpenVLA
bash setup.sh
conda activate openvla

bash scripts/run_train.sh \
    --dataset_root /path/to/dataset \
    --ngpu 4
```

자세한 내용은 [`OpenVLA/README.md`](OpenVLA/README.md)를 참고하세요.
