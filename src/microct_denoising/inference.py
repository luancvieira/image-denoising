"""Full-volume inference utilities for RED-CNN and Restormer."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import xarray as xr

from microct_denoising.constants import EPS, RAW_MAX, RAW_MIN
from microct_denoising.data import (
    StatsDict,
    inverse_zscore_with_fast_stats,
    zscore_with_fast_stats,
)


def predict_model_slice(
    model: nn.Module,
    slice_norm: np.ndarray,
    device: torch.device,
    channels: int,
    pad_multiple: int | None,
) -> np.ndarray:
    """Predict one normalized 2D slice and return one normalized 2D output."""

    h, w = slice_norm.shape
    model_input = slice_norm.astype(np.float32)

    if pad_multiple is not None:
        pad_h = (pad_multiple - h % pad_multiple) % pad_multiple
        pad_w = (pad_multiple - w % pad_multiple) % pad_multiple
        if pad_h or pad_w:
            mode = "reflect" if h > pad_h and w > pad_w else "edge"
            model_input = np.pad(model_input, ((0, pad_h), (0, pad_w)), mode=mode)

    if channels == 1:
        tensor_np = model_input[np.newaxis, np.newaxis].astype(np.float32)
    else:
        tensor_np = np.stack([model_input] * channels)[np.newaxis].astype(np.float32)

    tensor = torch.from_numpy(tensor_np).to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        out = model(tensor)

    out_np = out.squeeze(0).detach().cpu().numpy().astype(np.float32)
    if out_np.ndim == 3:
        pred = out_np[0] if out_np.shape[0] == 1 else out_np.mean(axis=0)
    else:
        pred = out_np

    return pred[:h, :w].astype(np.float32)


def predict_slice_tiled(
    model: nn.Module,
    slice_norm: np.ndarray,
    device: torch.device,
    channels: int,
    tile_size: int,
    overlap: int,
    pad_multiple: int | None,
) -> np.ndarray:
    """Predict one slice with overlapping tiles to reduce memory usage."""

    h, w = slice_norm.shape
    pred = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    step = max(1, tile_size - overlap)

    ys = list(range(0, max(1, h - tile_size + 1), step))
    xs = list(range(0, max(1, w - tile_size + 1), step))
    if ys[-1] != max(0, h - tile_size):
        ys.append(max(0, h - tile_size))
    if xs[-1] != max(0, w - tile_size):
        xs.append(max(0, w - tile_size))

    for y in ys:
        for x in xs:
            patch = slice_norm[y : y + tile_size, x : x + tile_size]
            ph = tile_size - patch.shape[0]
            pw = tile_size - patch.shape[1]
            if ph or pw:
                mode = "reflect" if patch.shape[0] > ph and patch.shape[1] > pw else "edge"
                patch = np.pad(patch, ((0, ph), (0, pw)), mode=mode)

            out_patch = predict_model_slice(
                model=model,
                slice_norm=patch,
                device=device,
                channels=channels,
                pad_multiple=pad_multiple,
            )
            out_patch = out_patch[: tile_size - ph, : tile_size - pw]

            yy = slice(y, y + out_patch.shape[0])
            xx = slice(x, x + out_patch.shape[1])
            pred[yy, xx] += out_patch
            weight[yy, xx] += 1.0

    return pred / np.maximum(weight, EPS)


def predict_volume(
    model: nn.Module,
    fast_path: Path,
    var_name: str,
    stats: StatsDict,
    device: torch.device,
    channels: int,
    pad_multiple: int | None = None,
    tile_size: int = 0,
    overlap: int = 32,
    raw_min: float = RAW_MIN,
    raw_max: float = RAW_MAX,
    progress: bool = True,
) -> np.ndarray:
    """Run full-volume inference for a FAST NetCDF volume."""

    model.eval()
    with xr.open_dataset(fast_path) as ds:
        da = ds[var_name]
        pred_vol = np.empty(da.shape, dtype=np.float32)

        for z_idx in range(da.shape[0]):
            raw_slice = da.isel(z=z_idx).values.astype(np.float32)
            slice_norm = zscore_with_fast_stats(raw_slice, fast_path, stats)
            if tile_size > 0:
                pred_norm = predict_slice_tiled(
                    model=model,
                    slice_norm=slice_norm,
                    device=device,
                    channels=channels,
                    tile_size=tile_size,
                    overlap=overlap,
                    pad_multiple=pad_multiple,
                )
            else:
                pred_norm = predict_model_slice(
                    model=model,
                    slice_norm=slice_norm,
                    device=device,
                    channels=channels,
                    pad_multiple=pad_multiple,
                )

            pred_raw = inverse_zscore_with_fast_stats(pred_norm, fast_path, stats)
            pred_vol[z_idx] = np.clip(pred_raw, raw_min, raw_max).astype(np.float32)

            if progress and ((z_idx + 1) % 10 == 0 or (z_idx + 1) == da.shape[0]):
                print(f"Predicted slice {z_idx + 1}/{da.shape[0]}", flush=True)

    model.train()
    return pred_vol


def save_prediction_volume(
    pred_vol: np.ndarray,
    fast_path: Path,
    var_name: str,
    stats: StatsDict,
    output_path: Path,
    model_name: str,
) -> None:
    """Save a raw-intensity prediction volume as NetCDF."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(fast_path) as ds:
        da = ds[var_name]
        out_da = xr.DataArray(
            pred_vol.astype(np.float32),
            coords=da.coords,
            dims=da.dims,
            attrs={
                **da.attrs,
                "model": model_name,
                "input_file": str(fast_path),
                "normalization": "zscore_using_fast_mean_std",
                "saved_space": "raw_intensity_after_inverse_fast_zscore",
                "raw_output_min": RAW_MIN,
                "raw_output_max": RAW_MAX,
                "raw_output_clipped_to_valid_range": "True",
            },
        )
        out_ds = xr.Dataset({var_name: out_da}, attrs=ds.attrs)
        tmp_path = output_path.with_suffix(f".{os.getpid()}.tmp.nc")
        out_ds.to_netcdf(tmp_path)
        os.replace(tmp_path, output_path)


def predict_and_save_volume(
    model: nn.Module,
    fast_path: Path,
    var_name: str,
    stats: StatsDict,
    output_path: Path,
    model_name: str,
    device: torch.device,
    channels: int,
    pad_multiple: int | None = None,
    tile_size: int = 0,
    overlap: int = 32,
) -> np.ndarray:
    """Run prediction and save the result to NetCDF."""

    pred_vol = predict_volume(
        model=model,
        fast_path=fast_path,
        var_name=var_name,
        stats=stats,
        device=device,
        channels=channels,
        pad_multiple=pad_multiple,
        tile_size=tile_size,
        overlap=overlap,
    )
    save_prediction_volume(
        pred_vol=pred_vol,
        fast_path=fast_path,
        var_name=var_name,
        stats=stats,
        output_path=output_path,
        model_name=model_name,
    )
    return pred_vol


def save_middle_slice_preview(pred_vol: np.ndarray, output_path: Path, title: str) -> None:
    """Save a simple PNG preview for the middle slice of a prediction volume."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    z_idx = pred_vol.shape[0] // 2
    image = pred_vol[z_idx]
    vmin = float(np.nanpercentile(image, 1))
    vmax = float(np.nanpercentile(image, 99))

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

