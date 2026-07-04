"""Restormer integration helpers.

The Restormer architecture is imported from the official upstream checkout so
this repository does not vendor or silently fork the original implementation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn


RESTORMER_REPO_URL = "https://github.com/swz30/Restormer.git"
RESTORMER_REAL_DENOISING_WEIGHTS_URL = (
    "https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth"
)


def load_restormer_module(restormer_root: Path) -> ModuleType:
    """Load the official Restormer architecture module from a local checkout."""

    arch_path = Path(restormer_root) / "basicsr" / "models" / "archs" / "restormer_arch.py"
    if not arch_path.exists():
        raise FileNotFoundError(
            "Restormer architecture file not found. Expected: "
            f"{arch_path}. Clone {RESTORMER_REPO_URL} first."
        )

    spec = importlib.util.spec_from_file_location("official_restormer_arch", arch_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Restormer architecture from {arch_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_restormer(
    restormer_root: Path,
    inp_channels: int = 3,
    out_channels: int = 3,
    dim: int = 48,
    num_blocks: list[int] | None = None,
    num_refinement_blocks: int = 4,
    heads: list[int] | None = None,
    ffn_expansion_factor: float = 2.66,
    bias: bool = False,
    layer_norm_type: str = "BiasFree",
) -> nn.Module:
    """Build the official Restormer denoising model."""

    module = load_restormer_module(restormer_root)
    return module.Restormer(
        inp_channels=inp_channels,
        out_channels=out_channels,
        dim=dim,
        num_blocks=num_blocks or [4, 6, 6, 8],
        num_refinement_blocks=num_refinement_blocks,
        heads=heads or [1, 2, 4, 8],
        ffn_expansion_factor=ffn_expansion_factor,
        bias=bias,
        LayerNorm_type=layer_norm_type,
    )


def load_restormer_weights(
    model: nn.Module,
    weights_path: Path,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> None:
    """Load official Restormer weights or a compatible state dict."""

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["params"] if isinstance(checkpoint, dict) and "params" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=strict)

