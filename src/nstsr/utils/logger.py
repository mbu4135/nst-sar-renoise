"""
경량 logger — stdlib logging + (옵션) tensorboard.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "nstsr", level: int = logging.INFO, log_file: Optional[str | Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class TBWriter:
    """tensorboard SummaryWriter wrapper. tensorboard 가 없으면 no-op."""

    def __init__(self, log_dir: str | Path):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(log_dir))
            self.enabled = True
        except ImportError:
            self._writer = None
            self.enabled = False

    def add_scalar(self, tag: str, value, step: int):
        if self.enabled:
            self._writer.add_scalar(tag, value, step)

    def add_image(self, tag: str, img, step: int):
        if self.enabled:
            self._writer.add_image(tag, img, step)

    def close(self):
        if self.enabled:
            self._writer.close()
