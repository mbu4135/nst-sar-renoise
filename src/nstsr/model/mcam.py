"""
MCAM-SAR — Multi-scale Content-Aware Module (s, cs 동시 주입).

원 RNSD 의 MCAM 은 `s` 한 image 만 받는다. 본 모델은 `s` 와 `cs` 를 받되 **역할이 다르다**:
- `s` (full-res clean): MCAMEncoder 로 3 스케일 feature → UNet decoder up-path 각 stage 에
  concat([upsampled, skip_i, F_s[i]]) 형태로 주입.
- `cs` (저해상 16x multilook, coarse 조건): CSEncoder 로 native 저해상에서 conv 후
  bottleneck 해상도로 resize → UNet middle 에서 1회 concat. (cs 는 fine 정보가 없어
  full-res conv/upsample 이 낭비라 저해상 그대로 처리.)
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


class CSEncoder(nn.Module):
    """저해상 cs(16x multilook) 전용 인코더 — bottleneck 1회 주입용.

    cs 는 y/s 보다 looks 배 coarse(예: 128 patch → 8x8)라 fine 스케일 정보가 없다.
    그래서 다운샘플 없이 native 해상도에서 conv 만 돌린 뒤(=full-res conv 낭비 제거),
    forward 시점에 bottleneck 해상도(target_hw)로 nearest resize 해서 한 번만 주입한다.
    cs 입력 해상도는 임의여도 됨(학습 crop·추론 타일 모두 target_hw 로 맞춰짐).

    Returns: [B, out_ch, *target_hw].
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 256):
        super().__init__()
        self.net = _conv_block(in_ch, out_ch)
        self.out_ch = out_ch

    def forward(self, cs: torch.Tensor, target_hw) -> torch.Tensor:
        f = self.net(cs)
        if f.shape[-2:] != tuple(target_hw):
            f = F.interpolate(f, size=tuple(target_hw), mode="nearest")
        return f


class MCAM(nn.Module):
    """
    s 는 multi-scale(non-shared) 인코더로 up-path 3 스케일 주입,
    cs 는 저해상 전용 CSEncoder 로 bottleneck 1회 주입.

    forward:
        s  : [B, 1, H, W]      → F_s (list of 3, scales [1, 1/2, 1/4])
        cs : [B, 1, Hc, Wc]    (저해상) + target_hw → cs_feat [B, cs_ch, *target_hw]
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 64, ch_mults: Sequence[int] = (1, 2, 4),
                 cs_ch: int | None = None):
        super().__init__()
        self.enc_s  = MCAMEncoder(in_ch=in_ch, base_ch=base_ch, ch_mults=ch_mults)
        self.out_channels = self.enc_s.out_channels  # tuple (c0, c1, c2) — F_s 용
        self.cs_ch = cs_ch if cs_ch is not None else base_ch * ch_mults[-1]
        self.enc_cs = CSEncoder(in_ch=in_ch, out_ch=self.cs_ch)

    def encode_s(self, s: torch.Tensor):
        return self.enc_s(s)

    def encode_cs(self, cs: torch.Tensor, target_hw) -> torch.Tensor:
        return self.enc_cs(cs, target_hw)
