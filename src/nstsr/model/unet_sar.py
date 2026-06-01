"""
nstsf/model/unet_sar.py — SAR despeckling 용 경량 U-Net.

입력 : (B, 1, H, W)  — log10(Re²) 또는 log10(Im²), 정규화 [0,1]
출력 : (B, 1, H, W)  — 예측된 성분, sigmoid → [0,1]
"""

import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSAR(nn.Module):
    """
    4-level U-Net for SAR despeckling.
    base_ch=32 → ~7.8M params (in_ch=1).
    in_ch / out_ch: 멀티채널 입출력 지원 (기본값 1로 기존 호환 유지).
    """

    def __init__(self, base_ch: int = 32, in_ch: int = 1, out_ch: int = 1):
        super().__init__()
        b = base_ch
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = _DoubleConv(in_ch, b)
        self.enc2 = _DoubleConv(b,    b * 2)
        self.enc3 = _DoubleConv(b*2,  b * 4)
        self.enc4 = _DoubleConv(b*4,  b * 8)

        # Bottleneck
        self.bottleneck = _DoubleConv(b*8, b * 16)

        # Decoder
        self.up4  = nn.ConvTranspose2d(b*16, b*8, 2, stride=2)
        self.dec4 = _DoubleConv(b*16, b*8)
        self.up3  = nn.ConvTranspose2d(b*8,  b*4, 2, stride=2)
        self.dec3 = _DoubleConv(b*8,  b*4)
        self.up2  = nn.ConvTranspose2d(b*4,  b*2, 2, stride=2)
        self.dec2 = _DoubleConv(b*4,  b*2)
        self.up1  = nn.ConvTranspose2d(b*2,  b,   2, stride=2)
        self.dec1 = _DoubleConv(b*2,  b)

        self.head = nn.Conv2d(b, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.head(d1))   # [0, 1]
