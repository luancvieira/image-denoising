"""Validation metrics and CSV helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import xarray as xr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from microct_denoising.constants import DATA_RANGE


def load_prediction_volume(path: Path, var_name: str) -> np.ndarray:
    with xr.open_dataset(path) as ds:
        return ds[var_name].values.astype(np.float32)


def compute_full_volume_psnr_ssim(
    pred_vol: np.ndarray,
    ref_vol: np.ndarray,
    data_range: float = DATA_RANGE,
) -> tuple[float, float]:
    """Compute full-volume PSNR and SSIM in raw intensity space."""

    pred_vol = pred_vol.astype(np.float32)
    ref_vol = ref_vol.astype(np.float32)
    psnr = peak_signal_noise_ratio(ref_vol, pred_vol, data_range=data_range)
    ssim = structural_similarity(ref_vol, pred_vol, data_range=data_range)
    return float(psnr), float(ssim)


def upsert_metrics_row(csv_path: Path, row: dict[str, object], key_fields: tuple[str, ...]) -> None:
    """Insert or replace a metrics row identified by key_fields."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    fieldnames = list(row.keys())

    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = reader.fieldnames
            rows = list(reader)

    new_row = {key: str(value) for key, value in row.items()}
    for key in new_row:
        if key not in fieldnames:
            fieldnames.append(key)
    key = tuple(new_row[field] for field in key_fields)

    replaced = False
    updated_rows: list[dict[str, str]] = []
    for existing in rows:
        existing_key = tuple(existing.get(field, "") for field in key_fields)
        if existing_key == key:
            updated_rows.append(new_row)
            replaced = True
        else:
            updated_rows.append(existing)
    if not replaced:
        updated_rows.append(new_row)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
