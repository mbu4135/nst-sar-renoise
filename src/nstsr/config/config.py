"""
경로 상수 + YAML config 로더.

PROJECT_ROOT 는 본 파일 기준 4단계 위 — nst-sar-renoise/ 를 가리킨다.
구조:
    nst-sar-renoise/                          <-- PROJECT_ROOT
        configs/
        checkpoints/
        src/
            main.py
            nstsr/
                config/config.py              <-- 본 파일
"""
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml


PROJECT_ROOT  = Path(__file__).resolve().parents[3]
CONFIGS_DIR   = PROJECT_ROOT / "configs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_CONFIG = CONFIGS_DIR / "vv_default.yaml"


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path: Path | str | None = None) -> SimpleNamespace:
    """YAML config 로드. path 가 None 이면 DEFAULT_CONFIG."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)
    cfg = _to_namespace(raw)
    cfg._path = str(cfg_path)
    cfg._raw = raw
    return cfg


def exp_dir(cfg: SimpleNamespace) -> Path:
    """checkpoints/<exp_name>/ — 학습 산출물 디렉토리."""
    d = CHECKPOINT_DIR / cfg.exp_name
    d.mkdir(parents=True, exist_ok=True)
    return d
