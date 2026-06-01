# SAR-RNSD: Diffusion 기반 SAR Single-Look 노이즈 합성기 구현 명세서

> Claude Code(Linux)가 이 문서만 보고 데이터 준비 → 학습 → 추론 전체 파이프라인을 구현할 수 있도록 작성된 작업 명세서입니다.
> 원본 아이디어 출처: Wu et al., "Realistic Noise Synthesis with Diffusion Models" (RNSD, AAAI 2025).
> 본 프로젝트는 RNSD를 **SAR (Synthetic Aperture Radar)** 도메인으로 이식한 것입니다.

---

## 0. 프로젝트 개요

### 0.1 목표
주어진 깨끗한 SAR 영상(temporal averaged)으로부터, **그 장면에 어울리는 single-look noise가 낀 영상**을 합성하는 conditional diffusion model을 학습/배포한다.

### 0.2 핵심 아이디어 (RNSD 매핑)
| RNSD (RGB) | 본 프로젝트 (SAR) | 비고 |
|---|---|---|
| `x₀` = real noisy RGB image | `x₀` = single-look SAR image (`y`) | Diffusion target을 noisy 분포로 재정의 |
| `s` = clean image | `s` = temporal-averaged SAR image | MCAM의 content guidance |
| `cs` = camera settings vector | `cs` = denoised ratio image | **공간 정보 conditioning** (image-level, scalar 아님) |
| TCCAM (scalar metadata → affine) | (현 단계 비활성) timestep embedding만 사용 | 추후 이중 편파 시 편파 인코딩에 활용 |
| MCAM (clean s → 다중 스케일 주입) | **MCAM-SAR**: `s`와 `cs` 두 image를 다중 스케일 주입 | 핵심 변경점 |
| DIPS (가속 샘플링) | DIPS 그대로 채택 | Basic 우선, Advanced는 추후 |

### 0.3 현재 단계 범위
- **VV 단일 편파 모델**만 제작 (이중 편파는 후속 확장)
- 입력 디렉토리, ratio denoise 모델 경로 등은 사용자가 추후 제공 → 코드에서는 config로 분리

---

## 1. 데이터 정의

### 1.1 세 가지 입력 영상

| 기호 | 이름 | 정의 | 도메인 |
|---|---|---|---|
| `y` | single-look image | 실제 노이즈가 낀 1-look SAR 영상 | linear (raw) |
| `s` | temporal averaged image | 동일 장면의 다중 시점 평균 → 깨끗한 참조 | linear (raw) |
| `cs` | ratio image (denoised) | `y / s`를 ratio denoise model로 처리한 결과 | normalized log-domain |

### 1.2 `cs` (ratio image) 생성 파이프라인 — 매우 중요
사용자 정의:
```
raw_ratio  = single_look / temporal_avg          # linear scale, 변화 없는 곳 ≈ 1
log_ratio  = log10(raw_ratio)                    # 변화 없는 곳 ≈ 0
norm_ratio = norm_config의 ratio 스케일로 [0,1] 정규화  # 변화 없는 곳 ≈ 0.5
cs         = ratio_denoise_model(norm_ratio)     # pretrained, freeze
```
- ratio denoise 모델은 **pretrained, 학습 중 freeze**.
- 추론 시 변화 없는 장면을 합성하려면 `cs = 0.5` 상수 이미지 사용.

### 1.3 데이터 전처리 규칙 (모든 영상 공통)
```
linear_image
  → log10(image + eps)        # amplitude/intensity 구분 없이 그냥 log10
  → normalize per norm_config  # → [0, 1]
  → 모델 입력
```
- `eps`는 underflow 방지용 작은 값 (예 `1e-8`).
- `10 * log10`이 아니라 **순수 `log10`** 사용 (사용자 명시).
- normalize는 `norm_config.py`에서 정의된 (min, max) 또는 (mean, std) 방식을 따른다 → 본 명세서 §3.2 참조.

### 1.4 추론 시 후처리
학습은 normalized log domain에서만 이루어지지만, **추론 결과물은 linear domain으로 복원**해야 한다:
```
model_output (∈ [0,1], log-normalized)
  → denormalize (norm_config 역연산)  # → log10 domain
  → 10^x                              # → linear domain
  → 저장 (single-look noisy SAR)
```

---

## 2. 디렉토리 구조 (생성할 것)

```
sar_rnsd/
├── README.md                          # 사용자용 짧은 사용 가이드
├── SAR_RNSD_SPEC.md                   # 본 명세서 (참조용)
├── requirements.txt                   # 의존성
├── norm_config.py                     # 정규화 설정 (사용자가 채울 placeholder)
├── configs/
│   └── vv_default.yaml                # 학습/추론 하이퍼파라미터
├── data/
│   ├── __init__.py
│   ├── dataset.py                     # SARTripletDataset (y, s, cs)
│   ├── transforms.py                  # log10, normalize, crop, flip
│   └── ratio_builder.py               # cs 생성 파이프라인
├── models/
│   ├── __init__.py
│   ├── unet.py                        # MCAM-SAR가 통합된 UNet
│   ├── mcam.py                        # Multi-scale Content-Aware Module (s, cs 동시 주입)
│   ├── tccam.py                       # 현 단계: time-only embedding (추후 편파 확장 자리)
│   └── ratio_denoiser.py              # pretrained ratio denoise model 래퍼 (구조는 사용자 제공 시 채움)
├── diffusion/
│   ├── __init__.py
│   ├── schedule.py                    # β_t, ᾱ_t 등 noise schedule
│   ├── trainer.py                     # Algorithm 1 구현
│   └── sampler.py                     # DIPS-Basic 샘플러 (Algorithm 2)
├── scripts/
│   ├── prepare_data.py                # raw → (y, s, cs) 페어 생성, npy/tiff로 저장
│   ├── train.py                       # 학습 진입점
│   ├── infer.py                       # 추론 진입점 (single image)
│   └── infer_batch.py                 # 디렉토리 단위 일괄 추론
├── utils/
│   ├── __init__.py
│   ├── io.py                          # SAR 영상 I/O (tiff, npy, raw 등)
│   ├── logger.py                      # tensorboard / wandb
│   └── viz.py                         # 디버그용 시각화
└── checkpoints/                       # 학습 결과 저장 (gitignore)
```

---

## 3. 핵심 구현 명세

### 3.1 `norm_config.py` (사용자 placeholder)
**역할**: 영상 종류별 (min, max) 또는 (mean, std)를 보관하는 단일 진실 공급원(SSOT).
값은 사용자가 제공 — 본 파일은 dict 골격만 만들어두고 주석으로 채우는 위치를 표시.

```python
# norm_config.py
"""
모든 정규화 파라미터의 단일 진실 공급원.
값은 데이터 통계 분석 후 사용자가 채워 넣는다.
정규화 방식: minmax(default) 또는 zscore 중 선택.
"""

NORM_CONFIG = {
    "vv": {
        # image = log10(Re²+Im²) intensity, filtering INTENSITY_NORM["VV"] 와 통일
        "image": {
            "mode": "minmax",
            "log_min": -12.0,
            "log_max":  10.0,
        },
        # ratio 영상 전용 (변화 없는 곳이 0.5가 되도록)
        "ratio": {
            "mode": "minmax_symmetric",
            "log_abs_max": 2.5,  # log10(ratio) ∈ [-2.5, 2.5] → norm [0, 1], 중심 0.5
        },
    },
    # 실제 config 엔 "vh"(image -14/8, ratio L=2.9), "cross"(image -8.3/2.2, ratio L=1.5) 포함
}

def normalize_image(x_log, pol="vv"):
    cfg = NORM_CONFIG[pol]["image"]
    if cfg["mode"] == "minmax":
        return (x_log - cfg["log_min"]) / (cfg["log_max"] - cfg["log_min"])
    raise NotImplementedError

def denormalize_image(x_norm, pol="vv"):
    cfg = NORM_CONFIG[pol]["image"]
    if cfg["mode"] == "minmax":
        return x_norm * (cfg["log_max"] - cfg["log_min"]) + cfg["log_min"]
    raise NotImplementedError

def normalize_ratio(r_log, pol="vv"):
    cfg = NORM_CONFIG[pol]["ratio"]
    # 변화 없는 곳(log10=0) → 0.5
    m = cfg["log_abs_max"]
    return (r_log + m) / (2 * m)

def denormalize_ratio(r_norm, pol="vv"):
    cfg = NORM_CONFIG[pol]["ratio"]
    m = cfg["log_abs_max"]
    return r_norm * (2 * m) - m
```

### 3.2 `data/transforms.py`
- `to_log10(x_linear, eps=1e-8)` — `log10(x + eps)`
- `from_log10(x_log)` — `10 ** x_log`
- `random_crop`, `random_hflip`, `random_vflip` (3개 입력 동기화)
- 텐서화: `[H, W] → [1, H, W]` float32

### 3.3 `data/ratio_builder.py`
```python
def build_ratio_cs(y_linear, s_linear, ratio_denoiser, eps=1e-8, pol="vv"):
    """
    1. raw_ratio = y / (s + eps)
    2. log_ratio = log10(raw_ratio + eps)  # 0이 변화 없음
    3. norm_ratio = normalize_ratio(log_ratio, pol)  # 0.5가 변화 없음
    4. cs = ratio_denoiser(norm_ratio)  # pretrained, freeze, eval mode
    5. return cs  # [0, 1] 범위, 학습/추론 모두 이 형태
    """
```
- `ratio_denoiser`는 `models/ratio_denoiser.py`에서 로드.
- 학습 중에는 GPU에 올린 채로 forward만, gradient는 끊는다.

### 3.4 `data/dataset.py` — `SARTripletDataset`
```python
class SARTripletDataset(Dataset):
    """
    한 sample = (y_norm, s_norm, cs_norm)
    모두 log-normalized [0, 1], shape [1, H, W].

    옵션:
      - 미리 cs까지 계산해서 디스크에 저장해두고 로드 (빠름)
      - 매 iter마다 즉석에서 cs 계산 (메모리/디스크 절약, 느림)
    """
    def __init__(self, root, pol="vv", patch_size=128, mode="cache_cs",
                 ratio_denoiser=None, augment=True):
        ...
    def __getitem__(self, idx):
        # 1. y_linear, s_linear 로드
        # 2. log10 + normalize → y_norm, s_norm
        # 3. cs_norm 로드 또는 즉석 계산
        # 4. random crop / flip (3개 동기화)
        # 5. return dict(y=..., s=..., cs=...)
```

### 3.5 `scripts/prepare_data.py`
사용자가 한 번만 실행:
1. 입력 디렉토리(사용자 제공) 스캔 → (y, s) pair 목록 작성.
2. 각 pair에 대해 `ratio_builder.build_ratio_cs`로 `cs` 계산.
3. 결과를 `data_root/<scene_id>/{y.npy, s.npy, cs.npy}`로 저장.
4. 학습/검증 split 파일 (`splits/train.txt`, `splits/val.txt`) 생성.

CLI 예시:
```bash
python scripts/prepare_data.py \
    --y_dir /path/to/single_look \
    --s_dir /path/to/temporal_avg \
    --ratio_ckpt /path/to/pretrained_ratio_denoiser.pth \
    --out_dir ./data_root \
    --pol vv
```

### 3.6 `models/mcam.py` — MCAM-SAR (핵심)
원 RNSD의 MCAM은 `s` 하나를 받지만, 본 모델은 **`s`와 `cs` 두 image conditioning**을 받는다.

```python
class MCAMEncoder(nn.Module):
    """
    s, cs를 각각 별도 encoder(non-shared weights)로 다중 스케일 feature 추출.
    3개 downsampling stage (RNSD와 동일).
    """
    def __init__(self, in_ch, base_ch=64):
        ...
    def forward(self, x):
        # returns [F_1, F_2, F_3]  (3 scales)
        ...

class MCAM(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        self.enc_s  = MCAMEncoder(in_ch=1, base_ch=base_ch)
        self.enc_cs = MCAMEncoder(in_ch=1, base_ch=base_ch)
    def forward(self, s, cs):
        F_s  = self.enc_s(s)
        F_cs = self.enc_cs(cs)
        return F_s, F_cs   # 각각 [F_1, F_2, F_3]
```
- UNet decoder의 i번째 upsampling stage에서 `concat([upsampled, skip_i, F_s[i], F_cs[i]])` 형태로 주입.
- 채널 수가 늘어나므로 decoder의 첫 conv는 그에 맞게 조정.

### 3.7 `models/tccam.py` — 현 단계 단순화 버전
원 RNSD는 (timestep + camera settings) → MLP → affine (γ, β)이지만, 본 단계에서는 camera settings 자리에 넣을 명시적 metadata가 없다.

```python
class TimeEmbedding(nn.Module):
    """현재 단계: timestep만 sinusoidal → MLP. 추후 편파 정보 합류 자리."""
    def __init__(self, dim, hidden):
        ...
    def forward(self, t, extra=None):
        # extra: 추후 편파 one-hot 등을 받기 위한 슬롯
        ...
```
- UNet의 각 ResBlock에 표준 방식(채널별 bias 추가)으로 주입.
- 추후 편파 확장 시 이 모듈만 교체.

### 3.8 `models/unet.py`
- 입력 채널: 1 (x_t는 single channel SAR).
- 출력 채널: 1 (예측 noise ε).
- Conditioning:
  - `t` → TimeEmbedding → 각 ResBlock
  - `(s, cs)` → MCAM → 각 upsampling stage에 concat
- 구조 가이드:
  - Base channels: 64 (config로 조정 가능)
  - Downsampling: 3회 (RNSD와 동일)
  - Middle block: ResBlock × 2 (+ optional attention)
  - Upsampling: 3회, 각 단계에서 skip + F_s[i] + F_cs[i] concat
  - 마지막 1×1 conv → 1 channel ε 예측

### 3.9 `diffusion/schedule.py`
- DDPM linear β schedule: `β_1 = 1e-4`, `β_T = 0.02`, `T = 1000`.
- 사전 계산: `α_t = 1 - β_t`, `ᾱ_t = ∏α_s`, `√ᾱ_t`, `√(1-ᾱ_t)`.

### 3.10 `diffusion/trainer.py` — Algorithm 1
```python
def training_step(batch, model, schedule, device):
    y, s, cs = batch["y"], batch["s"], batch["cs"]   # all [B,1,H,W], all in [0,1] log-norm
    x0 = y

    B = x0.shape[0]
    t = torch.randint(0, schedule.T, (B,), device=device)
    eps = torch.randn_like(x0)

    sqrt_ab   = schedule.sqrt_alpha_bar[t].view(B,1,1,1)
    sqrt_1_ab = schedule.sqrt_one_minus_alpha_bar[t].view(B,1,1,1)
    x_t = sqrt_ab * x0 + sqrt_1_ab * eps

    eps_hat = model(x_t, t, s, cs)
    loss = F.mse_loss(eps_hat, eps)
    return loss
```

### 3.11 `diffusion/sampler.py` — DIPS-Basic (Algorithm 2의 단순 버전)
```python
def dips_basic_schedule(T=1000, S=30, t_last=4, r=10.0):
    """
    t_i = t_last + (T - t_last) * (exp(r * (i-1)/(S-1)) - 1) / (exp(r) - 1)
    i = S, ..., 1  →  반환은 큰 t부터 작은 t로 (역순)
    마지막에 0을 붙여 [t_S, ..., t_1, 0] 형태.
    """
    ...

@torch.no_grad()
def sample(model, schedule, s, cs, shape, device, S=30, eta=0.0):
    """
    DDIM-style deterministic update (eta=0).
    return: x0 (model output, log-normalized [0,1])
    """
    x_t = torch.randn(shape, device=device)
    ts = dips_basic_schedule(T=schedule.T, S=S)
    for i in range(len(ts) - 1):
        t      = ts[i]
        t_next = ts[i+1]
        # broadcast
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps_hat = model(x_t, t_tensor, s, cs)
        ab_t      = schedule.alpha_bar[t]
        ab_next   = schedule.alpha_bar[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
        x0_pred   = (x_t - (1 - ab_t).sqrt() * eps_hat) / ab_t.sqrt()
        x0_pred   = x0_pred.clamp(0.0, 1.0)   # log-norm 범위 강제
        dir_xt    = (1 - ab_next).sqrt() * eps_hat
        x_t       = ab_next.sqrt() * x0_pred + dir_xt
    return x_t   # ≈ x_0
```
- `clamp(0,1)`은 SAR가 log-norm 범위 밖으로 폭주하는 것을 방지 (디버깅 단계에서만 사용, 학습이 안정되면 제거 가능).
- DIPS-Advanced (one-step distillation)은 후속 단계.

### 3.12 `scripts/train.py`
- config 로드 (`configs/vv_default.yaml`).
- `SARTripletDataset` + DataLoader (`num_workers`, `pin_memory`).
- 모델: UNet(in=1, out=1) + MCAM + TimeEmbedding.
- Optimizer: Adam, `lr=8e-5`, gradient accumulation=2 (RNSD와 동일).
- EMA decay 0.995 (추론 시 사용할 weight).
- 매 epoch마다 validation: 작은 patch에서 sample 1개 뽑아 보고 (linear domain 복원 후) 시각화 저장.
- Checkpoint: `last.pt`, `best.pt`, `ema.pt` 분리 저장.

### 3.13 `scripts/infer.py` — 단일 영상 추론
입력:
- clean `s` 영상 경로 (linear).
- (옵션) `cs` 영상 경로. 생략 시 `cs = 0.5` 상수 이미지 사용 → "변화 없음" 시나리오.

처리:
```
1. s_linear 로드
2. s_log = log10(s_linear + eps)
3. s_norm = normalize_image(s_log, "vv")  → [1,1,H,W]
4. cs_norm = 0.5 * ones_like(s_norm)   (또는 입력에서 로드)
5. x_T = randn_like(s_norm)
6. x_0_norm = sampler.sample(model, schedule, s_norm, cs_norm, shape=s_norm.shape, ...)
7. y_log    = denormalize_image(x_0_norm, "vv")
8. y_linear = 10 ** y_log
9. save y_linear (tiff/npy)
```

CLI 예시:
```bash
python scripts/infer.py \
    --ckpt checkpoints/ema.pt \
    --s /path/to/clean.tif \
    --out /path/to/synth_noisy.tif \
    --pol vv \
    --steps 30
```

### 3.14 `scripts/infer_batch.py`
- 디렉토리 단위로 clean 영상들을 일괄 변환 → noisy 영상 데이터셋 생성용.
- 대용량 영상은 overlap-tile 방식 (예: 256×256 patch, 32px overlap, blending).

---

## 4. 설정 파일 (`configs/vv_default.yaml`)

```yaml
exp_name: vv_rnsd_baseline
seed: 42
pol: vv

data:
  root: ./data_root            # prepare_data.py 결과 경로
  patch_size: 128
  batch_size: 16
  num_workers: 4
  augment: true

model:
  base_ch: 64
  ch_mults: [1, 2, 4, 8]
  num_res_blocks: 2
  attn_resolutions: [16]
  in_ch: 1
  out_ch: 1
  use_mcam: true
  use_tccam: false             # 현 단계 비활성, 추후 편파 확장 시 true

diffusion:
  T: 1000
  beta_start: 1.0e-4
  beta_end: 2.0e-2
  schedule: linear
  loss: l2

train:
  lr: 8.0e-5
  optimizer: adam
  grad_accum_steps: 2
  total_iters: 200000
  ema_decay: 0.995
  log_every: 100
  val_every: 2000
  ckpt_every: 5000

sampling:
  method: dips_basic
  steps: 30
  t_last: 4
  r: 10.0
  eta: 0.0
```

---

## 5. 학습/추론 워크플로우 (사용자 입장)

### 5.1 1회성 환경 셋업
```bash
cd sar_rnsd
pip install -r requirements.txt
# norm_config.py의 TODO 값을 데이터 통계로 채움
# configs/vv_default.yaml의 data.root, 사용자 경로 등 확인
```

### 5.2 데이터 준비
```bash
python scripts/prepare_data.py \
    --y_dir   /USER/PROVIDED/single_look_vv \
    --s_dir   /USER/PROVIDED/temporal_avg_vv \
    --ratio_ckpt /USER/PROVIDED/ratio_denoiser_vv.pth \
    --out_dir ./data_root \
    --pol vv
```

### 5.3 학습
```bash
python scripts/train.py --config configs/vv_default.yaml
```

### 5.4 추론 (변화 없는 가정)
```bash
python scripts/infer.py \
    --ckpt    checkpoints/vv_rnsd_baseline/ema.pt \
    --s       sample_clean_vv.tif \
    --out     sample_synth_noisy_vv.tif \
    --pol     vv \
    --steps   30
```

### 5.5 추론 (사용자 정의 ratio 적용)
```bash
python scripts/infer.py \
    --ckpt    checkpoints/vv_rnsd_baseline/ema.pt \
    --s       sample_clean_vv.tif \
    --cs      custom_ratio_normalized.tif \
    --out     sample_synth_noisy_with_change.tif \
    --pol     vv
```

---

## 6. 검증 체크리스트 (Claude Code가 구현 후 자체 점검)

### 6.1 데이터 파이프라인
- [ ] `to_log10(0)`이 NaN/inf 없이 처리되는가? (eps 적용 확인)
- [ ] `normalize → denormalize` 왕복 시 픽셀 단위 |오차| < 1e-5인가?
- [ ] `normalize_ratio(log10(1.0))`이 정확히 0.5인가?
- [ ] random crop이 (y, s, cs) 세 영상에 **동일한 좌표**로 적용되는가?

### 6.2 모델
- [ ] UNet forward 한 번 통과 시 출력 shape이 입력과 동일한가?
- [ ] MCAM의 두 encoder가 **가중치 공유하지 않는가**? (`id(self.enc_s) != id(self.enc_cs)`)
- [ ] `model(x_t, t, s, cs)` 에서 t를 바꾸면 출력이 달라지는가?
- [ ] `model(x_t, t, s, cs1)`과 `model(x_t, t, s, cs2)`이 다른가? (cs 영향 검증)
- [ ] `ratio_denoiser`가 `eval()` + `requires_grad=False` 상태인가?

### 6.3 학습
- [ ] 첫 100 iter에서 loss가 단조 감소 경향을 보이는가?
- [ ] EMA weight가 별도로 갱신·저장되는가?
- [ ] OOM 없이 patch 128, batch 16이 RTX 2080Ti급에서 돌아가는가?

### 6.4 추론
- [ ] `cs = 0.5` 입력 시 출력이 입력 `s`와 비슷한 통계지만 noise 분산이 더 크게 나오는가?
- [ ] 출력 linear 영상이 음수가 아닌가?
- [ ] DIPS-Basic 30-step과 (가능하면) DDPM 1000-step의 결과가 시각적으로 유사한가?

---

## 7. 향후 확장 (현 단계에서는 자리만 마련)

1. **이중 편파 (VV + VH/HV)**
   - UNet `in_ch`, `out_ch`를 2로 변경.
   - MCAM encoder도 2-channel 입력.
   - `tccam.py`에 편파 one-hot/embedding을 합류 → 진짜 TCCAM 활성화.
   - `norm_config.py`에 `vh`, `hh`, `hv` 키 추가.

2. **DIPS-Advanced (one-step distillation)**
   - `diffusion/distill.py` 추가.
   - 사전학습된 ε_θ를 freeze, ψ_θ를 학습해 `x_T → x_N` (N=200) 단일 step 압축.
   - 5-step 샘플링 지원.

3. **부가 metadata conditioning**
   - 입사각, look 수 등 scalar metadata가 있다면 TCCAM 자리에 주입.

4. **평가 지표**
   - SAR 도메인의 AKLD 대응: ENL (Equivalent Number of Looks), speckle index, M-index 등.
   - 이로 학습된 denoiser의 SAR 벤치마크 성능으로 간접 평가도 가능.

---

## 8. 코딩 컨벤션 (Claude Code에게)

- **PyTorch 2.x**, Python 3.10+.
- 모든 모듈은 `torch.nn.Module` 상속, `forward`에 docstring 필수 (입출력 shape 명시).
- 모든 텐서 연산에서 명시적 dtype (`float32`)과 device 관리.
- 입출력 형식은 `[B, C, H, W]` 일관 유지. SAR I/O는 utils/io에 격리.
- Config는 YAML, 코드에서 `omegaconf` 또는 `pyyaml`로 로드.
- 로깅은 `logging` 모듈 + tensorboard. wandb는 옵션.
- TODO/사용자 입력 위치에는 명확한 주석:
  ```python
  # TODO(user): set this path / value
  ```

---

## 9. 첫 작업 순서 권장 (Claude Code에게)

1. 디렉토리 골격 + `requirements.txt` 생성.
2. `norm_config.py` placeholder + `data/transforms.py` 작성 후 단위 테스트 (log10 왕복 등).
3. `diffusion/schedule.py` 작성 후 ᾱ_t shape/값 sanity check.
4. `models/unet.py` 골격 작성 (MCAM 없이 먼저), 더미 입력 forward 확인.
5. `models/mcam.py` 작성, UNet에 통합, forward shape 재확인.
6. `data/ratio_builder.py` + `models/ratio_denoiser.py` placeholder. (사용자가 ratio 모델 제공 시 채움)
7. `data/dataset.py` 작성, dummy npy로 1-step 로딩 검증.
8. `scripts/prepare_data.py` 작성.
9. `diffusion/trainer.py` + `scripts/train.py` 작성, 더미 데이터 1 batch overfit 테스트.
10. `diffusion/sampler.py` + `scripts/infer.py` 작성, 임의 weight로 shape-correctness 확인.
11. 실제 사용자 데이터/경로 도착 시: `norm_config.py` 값 채우기 → 데이터 준비 → 학습 → 추론.

---

**문서 끝.** 추가 정보(입력 디렉토리 구조, ratio denoiser 아키텍처/체크포인트, 정규화 통계값 등)는 사용자가 제공하는 대로 본 문서 또는 해당 파일을 갱신.
