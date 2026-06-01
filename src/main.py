"""
NARASPACE SAR-Renoise (SAR-RNSD) Command Line Interface

서브커맨드:
    prepare       — raw (y, s) → (y, s, cs) triple 데이터셋 생성
    train         — diffusion 모델 학습
    infer         — 단일 영상 추론 (clean s → synthetic noisy y)
    infer-batch   — 디렉토리 단위 추론 (overlap-tile blending)

기본 추론 예시:
    python main.py infer \\
        --ckpt checkpoints/vv_rnsd_baseline/ema.pt \\
        --s    sample_clean_vv.tif \\
        --out  sample_synth_noisy_vv.tif \\
        --pol  vv --steps 30

학습 예시:
    python main.py train --config configs/vv_default.yaml

데이터 준비 예시:
    python main.py prepare \\
        --y_dir /USER/single_look_vv \\
        --s_dir /USER/temporal_avg_vv \\
        --out_dir ./data_root --pol vv
"""
import argparse
import sys


SUBCOMMANDS = ("prepare", "train", "infer", "infer-batch")


def _dispatch(cmd: str, argv):
    if cmd == "prepare":
        from nstsr.process.prepare_pipeline import PreparePipeline
        proc = PreparePipeline()
    elif cmd == "train":
        from nstsr.process.train_pipeline import TrainPipeline
        proc = TrainPipeline()
    elif cmd == "infer":
        from nstsr.process.infer_pipeline import InferPipeline
        proc = InferPipeline()
    elif cmd == "infer-batch":
        from nstsr.process.infer_batch_pipeline import InferBatchPipeline
        proc = InferBatchPipeline()
    else:
        raise ValueError(f"unknown subcommand: {cmd}")

    args = proc.build_parser().parse_args(argv)
    proc.run(args)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print(f"available subcommands: {', '.join(SUBCOMMANDS)}")
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    cmd = sys.argv[1]
    if cmd not in SUBCOMMANDS:
        print(f"unknown subcommand: {cmd}")
        print(f"available: {', '.join(SUBCOMMANDS)}")
        sys.exit(1)

    _dispatch(cmd, sys.argv[2:])


if __name__ == "__main__":
    main()
