"""
Definizione dell'architettura 3D U-Net-GAN.
- UNet3DGenerator: Generatore basato su 3D U-Net (Input/Output: [B, 1, 10, 128, 128])
- CNN3DDiscriminator: Discriminatore CNN 3D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3DGenerator(nn.Module):
    """
    Generatore 3D U-Net.
    Input : [B, 1, 10, 128, 128]
    Output: [B, 1, 10, 128, 128] con attivazione Sigmoid in [0, 1]
    """
    def __init__(self):
        super().__init__()
        # Encoder Level 1: [B, 1, 10, 128, 128] -> [B, 64, 10, 128, 128]
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # -> [B, 64, 5, 64, 64]

        # Encoder Level 2: [B, 64, 5, 64, 64] -> [B, 128, 5, 64, 64]
        self.enc2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # -> [B, 128, 2, 32, 32]

        # Bottleneck: [B, 128, 2, 32, 32] -> [B, 256, 2, 32, 32]
        self.bottleneck = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )

        # Decoder Up 1
        self.up1 = nn.ConvTranspose3d(
            256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )
        # Dopo concatenazione con e2 (128 + 128 = 256 canali)
        self.dec1 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )

        # Decoder Up 2
        self.up2 = nn.ConvTranspose3d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )
        # Dopo concatenazione con e1 (64 + 64 = 128 canali)
        self.dec2 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        # Final Layer
        self.out_conv = nn.Sequential(
            nn.ConvTranspose3d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def _crop_and_cat(self, upsampled, encoder_feat):
        """Allinea le dimensioni spaziali e di profondità prima del concat."""
        if upsampled.shape != encoder_feat.shape:
            diffs = [e - u for e, u in zip(encoder_feat.shape[2:], upsampled.shape[2:])]
            paddings = []
            for d in reversed(diffs):
                paddings.append(d // 2)
                paddings.append(d - d // 2)
            upsampled = F.pad(upsampled, tuple(paddings))
        return torch.cat([upsampled, encoder_feat], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)               # [B, 64, 10, 128, 128]
        p1 = self.pool1(e1)             # [B, 64, 5, 64, 64]

        e2 = self.enc2(p1)              # [B, 128, 5, 64, 64]
        p2 = self.pool2(e2)             # [B, 128, 2, 32, 32]

        b = self.bottleneck(p2)         # [B, 256, 2, 32, 32]

        u1 = self.up1(b)
        d1 = self.dec1(self._crop_and_cat(u1, e2))

        u2 = self.up2(d1)
        d2 = self.dec2(self._crop_and_cat(u2, e1))

        out = self.out_conv(d2)         # [B, 1, 10, 128, 128]
        return out


class CNN3DDiscriminator(nn.Module):
    """
    Discriminatore CNN 3D.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, vol):
        x = F.interpolate(vol, size=(32, 32, 32), mode="trilinear", align_corners=False)
        feat = self.features(x)
        prob = self.classifier(feat)
        return prob
