# renoise — SAR speckle 합성 (speckle-residual 설계)

> 2026-06-11 전환. 이전 "clean→noisy 전체 생성"(s/cs MCAM, image-norm) 설계는 폐기.
> 현행 코드/스키마는 모두 이 문서 기준.

## 1. 무엇을 하는 모델인가

깨끗한 구조(시간평균 μ) 위에 **물리적으로 맞는 single-look speckle 을 생성**한다.
diffusion 으로 **speckle 자체**(r)를 만들고, 구조는 조건 μ 가 결정한다.

- **target** `x0 = r = log10(y_t / μ)` — = speckle (multiplicative noise 를 log 도메인으로).
  `y_t` = 한 날짜 single-look intensity, `μ` = 100-date 시간평균 intensity(≈clean, ENL~100).
- **복원** `ŷ = μ · 10^r̂` — 구조·밝기는 μ 가 보장(드리프트 0), 모델은 speckle 만.
- **조건(conditioning)** `x_t` 에 **3채널 input-concat** (`in_ch = 1 + 3 = 4`):
  1. `μ` (normalize_mu_intensity) — 밝기/구조.
  2. `D_A = 1/MSR` (normalize_da) — 픽셀별 temporal 안정도. 낮으면 PS(점산란체, speckle≈0),
     높으면 distributed(full speckle). μ(밝기)+D_A(분산)로 픽셀별 speckle 분포를 조건화.
  3. `valid/shadow` — 유효 마스크(1=정상).
- **loss**: masked ε-MSE — shadow/change/no-data 픽셀은 `cmask=0` 으로 제외.

### 왜 이렇게?
- speckle 은 곱셈노이즈 → log 에서 `log(y/μ)=log(speckle)`. **target 이 곧 speckle** 이라 정규화가
  신호를 압축하지 않음(구 image-norm 의 문제 해결).
- 구조를 μ 가 담당 → 노이즈에서 구조를 재생성하다 생기는 **드리프트 제거**.
- 같은 평균밝기라도 PS(밝고 안정) vs distributed(불안정)는 다른 speckle → `μ`(=평균)+`D_A`(=분산)
  둘 다 줘야 픽셀별 분포가 결정됨. D_A 단독은 밝기를, μ 단독은 안정도를 못 줌.

## 2. 파이프라인 (raw SLC → 학습/추론)

조건·타깃은 모두 같은 radar 격자(예 korea: 13124×68647, big-endian float32 `.img`).

### (a) conditioning 입력 만들기 — `scripts/data_prep/`
- **μ (시간평균 intensity)** = `compute_ave_cov.py` → `C11.img`(=mean|VV|²), C22/C12_mag 도.
  (스택을 메모리에 안 올리고 azimuth row-block 스트리밍.)
- **D_A 맵** = GAMMA IPTA `pwr_stat <SLC_tab> <ref.par> MSR.img plist ...` → **MSR.img**(=mean/sigma=1/D_A,
  연속 FLOAT). PS 후보 점리스트 `plist` 도. (D_A = 1/MSR.)
- **valid/shadow 마스크** = GAMMA `gc_map2` 의 `ls_map_rdc`(UCHAR, MLI range_looks=3) →
  range ×3 nearest 업샘플 → SLC 격자. 값: 0=no-data, 1=정상, &4=layover, &16=shadow.
  최종 valid = ls==1 **AND** μ>0.

### (b) 데이터셋 빌드 — `scripts/data_prep/build_speckle_ds.py`
30date(균등) 에 대해 `r = log10(y_t/μ)` 패치(512, stride512)를 만들되 **change-free 만 채택**:
- `r` 의 **ML4(4×4)·ML16(16×16)** 으로 speckle 을 눌러 변화 검출.
- per-patch 채택 = `|mean(ML4)−date중앙값| < τ_μ` (균일변화) **AND**
  `std(ML4)/std(ML16)` 가 speckle 처럼 큼(분산 persistence, 구조변화 없음). 임계는 per-date 자동보정.
- per-pixel `cmask` = valid(ls&μ>0) **AND** 그 날짜 데이터(y>0&μ>0) **AND** ~(ML4 셀 이탈=change).
- (참고: 엣지 직접검출은 잔여 speckle 에 묻혀 무의미 → 분산-persistence 로 대체.)

**출력 `speckle_ds/`** (korea VV 30date: 65,976샘플, train59379/val6597, 88GB):
```
mu_patches.npy        (N,512,512) μ linear        — 공유(날짜무관), coords idx 로 매핑
da_patches.npy        (N,512,512) D_A=1/MSR        — 공유
valid_patches.npy     (N,512,512) valid 0/1        — 공유 (azimuth 4100~4700 제외=정합오차)
r_patches/<date>.npy      (Nk,512,512) raw log10(y/μ)   — 정규화는 dataset load 시
cmask_patches/<date>.npy  (Nk,512,512) uint8 loss mask
idx_patches/<date>.npy     (Nk,)  공유 cond 의 coords idx
coords.csv, splits/{train,val}.txt   # split = "date:k" per line
```

## 3. 정규화 (`src/nstsr/config/norm_config.py` → `NORM_SPECKLE`)
speckle_ds 실분포로 확정:
- `r_absmax=2.5` : `r_norm = clip(r/2.5, -1, 1)` (0=무변동, clip~0.5%).
- `mu_log_min=-2.5, mu_log_max=1.0` : μ → log10 → [0,1].
- `da_max=3.0` : `clip(D_A,0,3)/3` → [0,1].
새 지역/편파면 분포 측정해 갱신(구 `NORM_CONFIG`/`ratio` L 은 폐기설계용).

## 4. 모델·학습·추론
- 모델: `UNet(in_ch=4, use_mcam=false, forward(x_t, t, cond))` — cond=[μ,D_A,shadow] concat.
- 데이터셋: `SARSpeckleDataset` (r=normalize_r, cond 3ch, cmask).
- loss: `trainer.training_step` masked ε-MSE. 샘플러: `sampler.sample(model, sched, cond, ...)`,
  **eta>0 stochastic 기본**(speckle 텍스처).

**학습** (config: `configs/vv_default.yaml`, `data.root` 를 speckle_ds 경로로):
```
cd src && python main.py train --config ../configs/vv_default.yaml --gpu 0
```
val 마다 `r_gt vs r̂ std` 로깅 + `val/step_*.png`(μ / r_gt / r̂).

**추론** (대상 장면의 μ/D_A/shadow 필요 → 스택 있어야 함):
```
cd src && python main.py infer --ckpt <ema.pt> \
    --mu μ.img --da D_A.img --shadow valid.img --out ŷ.img \
    --img_shape H W --eta 1.0 --steps 30
```
출력 `ŷ = μ·10^r̂` (single-look noisy intensity), shadow/no-data=0.

## 5. 검증 / 상태
- `python scripts/sanity_check.py` (CPU, 데이터 불필요) → 정규화·모델·masked diffusion·샘플러·dataset, 16/16.
- 미완: `infer_batch_pipeline.py`(디렉토리 배치, 보조) 구설계; 실제 학습·ckpt 추론 검증.
- 추론 제약: D_A·μ 가 시간스택을 요구 → 임의 단일 clean 영상엔 일반화 안 됨(스택 있는 장면용).
