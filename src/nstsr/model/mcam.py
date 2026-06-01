"""
MCAM-SAR — Multi-scale Content-Aware Module (s, cs 동시 주입).

원 RNSD 의 MCAM 은 `s` 한 image 만 받지만 본 모델은 `s` 와 `cs`
**두 image** 를 받는다. 가중치를 **공유하지 않는** 두 인코더로 다중 스케일 feature 를 산출한다.

Encoder 는 3 단계 다운샘플링 (입력 해상도 포함 → 3개 스케일 feature).
UNet decoder 의 각 upsampling stage 에서
    concat([upsampled, skip_i, F_s[i], F_cs[i]])
형태로 주입된다.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.SiLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.SiLU(),
    )


class MCAMEncoder(nn.Module):
    """
    단일 image (s 또는 cs) → 다중 스케일 feature pyramid.

    ch_mults 가 [1, 2, 4, 8] 이라면 본 인코더는 앞 3개 항목을 사용해
    [base_ch, 2*base_ch, 4*base_ch] 채널의 3 단계 feature 를 만든다.

    Returns
    -------
    [F_0, F_1, F_2]
        F_0 : full resolution  @ base_ch * ch_mults[0]
        F_1 : 1/2 resolution   @ base_ch * ch_mults[1]
        F_2 : 1/4 resolution   @ base_ch * ch_mults[2]
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 64, ch_mults: Sequence[int] = (1, 2, 4)):
        super().__init__()
        assert len(ch_mults) == 3, "MCAMEncoder expects exactly 3 scales"
        c0 = base_ch * ch_mults[0]
        c1 = base_ch * ch_mults[1]
        c2 = base_ch * ch_mults[2]
        self.block0 = _conv_block(in_ch, c0)
        self.down01 = nn.Conv2d(c0, c0, kernel_size=3, stride=2, padding=1)
        self.block1 = _conv_block(c0, c1)
        self.down12 = nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1)
        self.block2 = _conv_block(c1, c2)
        self.out_channels = (c0, c1, c2)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        x : [B, in_ch, H, W]
        return : list of 3 tensors at scales [1, 1/2, 1/4].
        """
        f0 = self.block0(x)
        f1 = self.block1(self.down01(f0))
        f2 = self.block2(self.down12(f1))
        return [f0, f1, f2]


class MCAM(nn.Module):
    """
    s, cs 각각 독립적인 (non-shared) encoder 로 multi-scale feature 추출.

    forward:
        s, cs : [B, 1, H, W]
        return: (F_s, F_cs) — 각 list of 3 tensors.
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 64, ch_mults: Sequence[int] = (1, 2, 4)):
        super().__init__()
        self.enc_s  = MCAMEncoder(in_ch=in_ch, base_ch=base_ch, ch_mults=ch_mults)
        self.enc_cs = MCAMEncoder(in_ch=in_ch, base_ch=base_ch, ch_mults=ch_mults)
        self.out_channels = self.enc_s.out_channels  # tuple (c0, c1, c2)

    def forward(self, s: torch.Tensor, cs: torch.Tensor):
        F_s  = self.enc_s(s)
        F_cs = self.enc_cs(cs)
        return F_s, F_cs
