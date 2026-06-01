# SAR-RNSD — Diffusion 기반 SAR single-look 노이즈 합성기

깨끗한 temporal-averaged SAR(`s`)와 denoised ratio(`cs`)로부터 현실적인 single-look
노이즈 영상(`y`)을 합성하는 conditional diffusion 모델. 상세 설계는 `SAR_RNSD_SPEC.md`.

## 1. 설치 (어느 머신에서든)

```bash
git clone <this-repo> && cd nst-sar-renoise
pip install -r requirements.txt        # torch는 머신에 맞는 빌드 권장 (CPU/CUDA)
```

GPU가 있으면 자동 사용, 없으면 CPU로 fallback.

## 2. 데이터 (`data_root`) — git에 없음, 별도로 옮길 것

학습은 **미리 만들어둔 `data_root`** 만 있으면 됩니다 (ratio 모델/원본 .img 불필요).
stacked 레이아웃:

```
data_root/
├── y_patches/<date>.npy    # (N, 512, 512) linear single-look
├── cs_patches/<date>.npy   # (N, 512, 512) [0,1] denoised ratio
├── s_patches.npy           # (N, 512, 512) linear temporal-avg (모든 날짜 공유)
├── coords.csv              # patch 좌표
└── splits/{train,val}.txt  # "date:idx"
```

**옮기는 법 2가지**
- (간단) 만들어둔 `data_root/` 통째로 복사.
- (재생성) 원본 `.img` + ratio 체크포인트를 옮긴 뒤 아래 prepare 실행:

```bash
python src/main.py prepare \
  --y_dir <single_look_dir> --s_dir <temporal_avg_dir> \
  --ratio_arch i2i_unetsar --ratio_ckpt <i2i_ratio_C11_ft/best_model.pth> --ratio_base_ch 32 \
  --out_dir ./data_root \
  --pol vv --channel C11 --img_shape 3680 12960 \
  --patch 512 --patch_stride 256 --land_ref_date <YYYYMMDD> --land_min 0.70 \
  --device cuda            # 또는 cpu
  # 날짜 일부만: --limit 10  또는  --dates 20180104 20180116 ...
```

## 3. 학습

```bash
python src/main.py train --config configs/vv_default.yaml --data_root /path/to/data_root
```
- `--data_root` 로 yaml 수정 없이 데이터 경로 지정 (생략 시 yaml의 `data.root=./data_root`).
- 체크포인트는 `checkpoints/<exp_name>/{last.pt, ema.pt}` 에 저장.
- 학습 patch_size=128 → 저장된 512 패치에서 random-crop.

## 4. 추론

```bash
python src/main.py infer --ckpt checkpoints/<exp>/ema.pt \
  --s clean.tif --out synth_noisy.tif --pol vv --steps 30
# --cs 생략 시 cs=0.5 (변화 없음) 가정
```

## 5. 검증 (선택)

```bash
python scripts/sanity_check.py          # 모델/정규화/diffusion 자체 점검 (CPU)
# ratio denoiser 섹션까지 보려면:
RATIO_CKPT=/path/best_model.pth python scripts/sanity_check.py
```

## 참고
- 정규화 정의는 `src/nstsr/config/norm_config.py` 단일 기준 (image: log10+minmax, ratio: 대칭 minmax, 0.5=무변화). nst-sar-filtering 과 동일 값.
- ratio denoiser(`i2i_unetsar`)는 **prepare 단계의 cs 생성에만** 필요. 학습/추론은 불필요.
- `scripts/` 의 데이터 빌드 스크립트(build_pair_index 등)는 경로가 하드코딩된 1회성 도구.
