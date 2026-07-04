"""Checkpoint helpers for training and inference."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    model_name: str,
    run_kind: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_name": model_name,
            "run_kind": run_kind,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def load_model_state(path: Path, model: nn.Module, device: torch.device, strict: bool = True) -> dict:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=strict)
        return checkpoint
    if isinstance(checkpoint, dict) and "params" in checkpoint:
        model.load_state_dict(checkpoint["params"], strict=strict)
        return checkpoint
    model.load_state_dict(checkpoint, strict=strict)
    return {"epoch": None, "model_state": checkpoint}


def find_latest_checkpoint(checkpoint_dir: Path, prefix: str) -> Path | None:
    checkpoints = sorted(
        checkpoint_dir.glob(f"{prefix}_ep*.pth"),
        key=lambda p: int(p.stem.replace(f"{prefix}_ep", "")),
    )
    return checkpoints[-1] if checkpoints else None

