"""RED-CNN model used as a convolutional denoising baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class REDCNN(nn.Module):
    """Ten-layer residual encoder-decoder CNN for image denoising."""

    def __init__(self, feature_channels: int = 96, final_relu: bool = False) -> None:
        super().__init__()
        c = feature_channels

        self.conv1 = nn.Conv2d(1, c, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(c, c, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(c, c, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(c, c, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(c, c, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(c, c, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(c, c, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(c, c, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(c, c, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(c, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)
        self.final_relu = final_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out

        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out

        out = self.relu(self.conv5(out))
        out = self.tconv1(out)
        out = out + residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out = out + residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out = out + residual_1

        if self.final_relu:
            out = self.relu(out)
        return out


def build_redcnn(feature_channels: int = 96, final_relu: bool = False) -> REDCNN:
    """Build RED-CNN with article-compatible defaults."""

    return REDCNN(feature_channels=feature_channels, final_relu=final_relu)

