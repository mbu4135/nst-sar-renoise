"""
MCAM-SAR 가 통합된 conditional UNet for diffusion.

Inputs
------
x_t : [B, 1, H, W]   noisy single-look SAR (log-normalized).
t   : [B]            timestep (long).
s   : [B, 1, H, W]   clean (temporal-averaged) SAR  (log-normalized).
cs  : [B, 1, H, W]   denoised ratio image          ([0, 1], 변화 없음 = 0.5).

Output
------
eps_hat : [B, 1, H, W] 예측 noise.

Structure (ch_mults = [1, 2, 4, 8]):
                   level 0           level 1           level 2           level 3 (bottom)
    encoder    in→64 ──down──→ 128 ──down──→ 256 ──down──→ 512
                  │              │              │              │
                 skip0          skip1          skip2         middle
                  │              │              │              │
    MCAM(s,cs)   F0[64]         F1[128]        F2[256]
                  │              │              │              │
    decoder   ←─up─ 64  ←──up── 128  ←──up── 256  ←──up── 512
                       (concat [up, skip, F_s, F_cs] at each level i ∈ {0,1,2})
"""
from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from nstsr.model.mcam import MCAM
from nstsr.model.tccam import TimeEmbedding


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)


class ResBlock(nn.Module):
    """conv-norm-act ResBlock with timestep embedding bias."""

    def __init__(self, in_ch: int, out_ch: int, t_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.t_proj = nn.Linear(t_emb_dim, out_ch)
        self.norm2 = _norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """단일 head self-attention (간단 구현)."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = _norm(ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, kernel_size=1)
        self.proj = nn.Conv2d(ch, ch, kernel_size=1)
        self.scale = ch ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(b, 3, c, h * w).unbind(dim=1)
        attn = torch.softmax(torch.einsum("bci,bcj->bij", q, k) * self.scale, dim=-1)
        out = torch.einsum("bij,bcj->bci", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# UNet
# ─────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Conditional UNet for SAR diffusion (RNSD-style).

    Args
    ----
    in_ch, out_ch       : 입력/출력 채널 (현 단계 1).
    base_ch             : 기본 채널.
    ch_mults            : 각 level 채널 배수. 길이 4 권장 ([1,2,4,8] → 3 downsamples).
    num_res_blocks      : 각 level 의 ResBlock 수.
    attn_resolutions    : 자기-attention 을 삽입할 spatial resolution 들 (입력 patch_size 기준).
    use_mcam            : True 이면 (s, cs) MCAM injection.
    t_emb_dim           : sinusoidal embedding dim.
    t_hidden            : time MLP hidden / ResBlock 에 들어가는 embedding dim.
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        base_ch: int = 64,
        ch_mults: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (16,),
        use_mcam: bool = True,
        t_emb_dim: int = 128,
        t_hidden: int = 256,
        input_resolution: int = 128,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.base_ch = base_ch
        self.ch_mults = tuple(ch_mults)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = set(attn_resolutions)
        self.use_mcam = use_mcam

        # ── time embedding ─────────────────────────────────────────────
        self.time_emb = TimeEmbedding(dim=t_emb_dim, hidden=t_hidden)

        # ── mcam (s, cs) — 첫 3 level 만 ────────────────────────────────
        if use_mcam:
            self.mcam = MCAM(in_ch=1, base_ch=base_ch, ch_mults=ch_mults[:3])
            mcam_chs = self.mcam.out_channels  # (c0, c1, c2)
        else:
            self.mcam = None
            mcam_chs = (0, 0, 0)

        # ── input projection ───────────────────────────────────────────
        self.in_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # ── encoder ────────────────────────────────────────────────────
        self.down_blocks = nn.ModuleList()
        self.down_resolutions: List[int] = []
        cur_ch = base_ch
        cur_res = input_resolution
        skip_channels: List[int] = [base_ch]  # for initial in_conv output (level 0)
        for level, mult in enumerate(ch_mults):
            out_c = base_ch * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(cur_ch, out_c, t_hidden))
                cur_ch = out_c
                if cur_res in self.attn_resolutions:
                    self.down_blocks.append(SelfAttention2d(cur_ch))
                skip_channels.append(cur_ch)
            if level != len(ch_mults) - 1:
                self.down_blocks.append(Downsample(cur_ch))
                skip_channels.append(cur_ch)
                cur_res //= 2
            self.down_resolutions.append(cur_res)

        # ── middle ─────────────────────────────────────────────────────
        self.mid_block1 = ResBlock(cur_ch, cur_ch, t_hidden)
        self.mid_attn   = SelfAttention2d(cur_ch) if cur_res in self.attn_resolutions else nn.Identity()
        self.mid_block2 = ResBlock(cur_ch, cur_ch, t_hidden)

        # ── decoder ────────────────────────────────────────────────────
        # 각 level (top-down 순) 에서 MCAM features 가 더해질 채널 수:
        #   level 0,1,2 → 2 * mcam_chs[level]   (F_s + F_cs)
        #   level 3 (bottom) → 0
        self.up_blocks = nn.ModuleList()
        self.up_specs: List[dict] = []  # 각 entry: 어떤 작업인지 메타
        for level in reversed(range(len(ch_mults))):
            out_c = base_ch * ch_mults[level]
            mcam_c = (2 * mcam_chs[level]) if (use_mcam and level < 3) else 0
            for i in range(num_res_blocks + 1):
                skip_c = skip_channels.pop()
                self.up_blocks.append(ResBlock(cur_ch + skip_c + mcam_c, out_c, t_hidden))
                self.up_specs.append({"type": "res", "level": level, "i": i, "mcam_c": mcam_c})
                cur_ch = out_c
                # MCAM 은 각 level 의 첫 ResBlock 에서만 concat 하고, 이후는 0
                mcam_c = 0
                if cur_res in self.attn_resolutions:
                    self.up_blocks.append(SelfAttention2d(cur_ch))
                    self.up_specs.append({"type": "attn", "level": level})
            if level != 0:
                self.up_blocks.append(Upsample(cur_ch))
                self.up_specs.append({"type": "up", "level": level})
                cur_res *= 2

        # ── output ─────────────────────────────────────────────────────
        self.out_norm = _norm(cur_ch)
        self.out_conv = nn.Conv2d(cur_ch, out_ch, kernel_size=3, padding=1)

        # store for forward
        self._skip_channels_for_check = skip_channels  # 비어있어야 함

    # ─────────────────────────────────────────────────────────────────
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor | None = None,
        cs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x_t : [B, 1, H, W]
        t   : [B] long
        s   : [B, 1, H, W] or None  (use_mcam=True 이면 필수)
        cs  : [B, 1, H, W] or None  (use_mcam=True 이면 필수)
        """
        if self.use_mcam:
            assert s is not None and cs is not None, "MCAM 활성 시 s, cs 가 필요"
            F_s, F_cs = self.mcam(s, cs)   # 각 list[3]
        else:
            F_s = F_cs = [None, None, None]

        t_emb = self.time_emb(t)

        # ── encode ───────────────────────────────────────────────────
        h = self.in_conv(x_t)
        skips: List[torch.Tensor] = [h]
        for block in self.down_blocks:
            if isinstance(block, ResBlock):
                h = block(h, t_emb)
                skips.append(h)
            elif isinstance(block, SelfAttention2d):
                h = block(h)
            elif isinstance(block, Downsample):
                h = block(h)
                skips.append(h)
            else:
                raise RuntimeError(f"unexpected block {type(block)}")

        # ── middle ───────────────────────────────────────────────────
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h) if not isinstance(self.mid_attn, nn.Identity) else h
        h = self.mid_block2(h, t_emb)

        # ── decode ───────────────────────────────────────────────────
        for block, spec in zip(self.up_blocks, self.up_specs):
            kind = spec["type"]
            if kind == "res":
                skip = skips.pop()
                cond_extra = []
                if spec["mcam_c"] > 0:
                    level = spec["level"]
                    cond_extra = [F_s[level], F_cs[level]]
                h_in = torch.cat([h, skip] + cond_extra, dim=1)
                h = block(h_in, t_emb)
            elif kind == "attn":
                h = block(h)
            elif kind == "up":
                h = block(h)

        out = self.out_conv(F.silu(self.out_norm(h)))
        return out
