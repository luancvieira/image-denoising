#!/usr/bin/env python3
"""Export the central views shown by the sample viewer (docs/) from the volumes of each sample.

Input: a CSV with one row per sample and the columns
    sample, voxel_um, reference, fast, intermediate, redcnn, restormer_finetuned, restormer_scratch, manual_seg
(paths to .nc volumes; intermediate and the model columns may be empty). Use aluminium-standardized volumes.

Output (default docs/data/):
    manifest.json
    sampleNN/{volume}_{plane}.png                  gray values, 8-bit, one pixel per voxel
    sampleNN/err_{volume}_{plane}.png              volume - reference, blue-white-red, +-ERR_LIM gray levels
    sampleNN/seg_{pipeline}_{volume}_{plane}.png   automatic segmentation vs manual: black both, red added,
                                                   blue missed, light gray solid
Volume keys: ref60, fast, int8, redcnn, finetuned, scratch, manual. Planes: central axial (z), coronal (y) and
sagittal (x) slices; all images are flipped vertically (origin at the lower left, as in the paper figures).
Gray window of each sample: 0.1-99.9 percentiles of the central views of the fast, intermediate and reference
volumes. PSNR/SSIM (3D, L = 65535), porosity and Dice (3D, against the manual segmentation) are computed here.

usage: python scripts/export_viewer.py --samples samples.csv --output docs/data [--no-metrics]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image
from scipy import ndimage as ndi

from microct_denoising.metrics import compute_full_volume_psnr_ssim, pore_mask_from_segmentation
from microct_denoising.segmentation import PIPELINES, segment

COLUMNS = {"ref60": "reference", "fast": "fast", "int8": "intermediate", "redcnn": "redcnn",
           "finetuned": "restormer_finetuned", "scratch": "restormer_scratch"}
LABELS = {"ref60": "Reference (60 min)", "fast": "Fast (2 min 35 s)", "int8": "Intermediate (8 min)",
          "redcnn": "RED-CNN", "finetuned": "Restormer fine-tuned", "scratch": "Restormer random init.",
          "manual": "Manual segmentation (60 min)"}
PLANES = {"axial": 0, "coronal": 2, "sagittal": 1}  # axis removed from (z, x, y)
ERR_LIM, ERR_LEVELS = 2500.0, 33
SEG_PALETTE = [0, 0, 0, 214, 39, 40, 31, 119, 180, 242, 242, 242]


def load(path: str | Path, var: str) -> np.ndarray:
    with xr.open_dataset(path, decode_times=False) as ds:
        da = ds[var] if var in ds else next(v for v in ds.data_vars.values() if v.ndim == 3)
        if set(da.dims) == {"z", "x", "y"}:
            da = da.transpose("z", "x", "y")
        return da.values


def central(vol: np.ndarray, plane: str) -> np.ndarray:
    ax = PLANES[plane]
    return np.take(vol, vol.shape[ax] // 2, axis=ax)


def save_png(arr: np.ndarray, path: Path, palette: list[int] | None = None) -> list[int]:
    im = Image.fromarray(np.flipud(arr).astype(np.uint8), mode="P" if palette else "L")
    if palette:
        im.putpalette(palette)
    im.save(path, optimize=True)
    return list(im.size)


def err_palette() -> list[int]:
    pal = []
    for k in range(ERR_LEVELS):
        t = k / (ERR_LEVELS - 1) * 2 - 1
        pal += [round(255 * (1 + t)), round(255 * (1 + t)), 255] if t < 0 else [255, round(255 * (1 - t)), round(255 * (1 - t))]
    return pal


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("docs/data"))
    ap.add_argument("--var-name", default="microtom")
    ap.add_argument("--seg-var-name", default=None)
    ap.add_argument("--no-metrics", action="store_true", help="Skip PSNR/SSIM (segmentation metrics are always computed).")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.samples.open(encoding="utf-8")))
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    epal = err_palette()
    man = {"max_px": None, "planes": list(PLANES), "err_lim": ERR_LIM, "has_extra": True,
           "volumes": [{"key": k, "label": v} for k, v in LABELS.items()], "samples": []}
    for row in rows:
        n = int(row["sample"])
        d = out / f"sample{n:02d}"
        d.mkdir(exist_ok=True)
        ref = load(row["reference"], args.var_name).astype(np.float32)
        manual = pore_mask_from_segmentation(load(row["manual_seg"], args.seg_var_name or args.var_name), ref)
        keys = [k for k, c in COLUMNS.items() if row.get(c)]
        vols = {k: (ref if k == "ref60" else load(row[COLUMNS[k]], args.var_name).astype(np.float32)) for k in keys}
        views = {k: {p: central(v, p) for p in PLANES} for k, v in vols.items()}
        win = [k for k in ("fast", "int8", "ref60") if k in views]
        vals = np.concatenate([views[k][p].ravel() for k in win for p in PLANES])
        lo, hi = (float(x) for x in np.percentile(vals, [0.1, 99.9]))
        sizes = {}
        for k in keys:
            for p in PLANES:
                g = np.clip((views[k][p] - lo) / (hi - lo) * 255.0, 0, 255)
                sizes[p] = save_png(g, d / f"{k}_{p}.png")
                if k != "ref60":
                    r = views[k][p] - views["ref60"][p]
                    q = np.clip(np.rint((r / ERR_LIM + 1) / 2 * (ERR_LEVELS - 1)), 0, ERR_LEVELS - 1)
                    save_png(q, d / f"err_{k}_{p}.png", epal)
        for p in PLANES:
            save_png(np.where(central(manual, p), 0, 242), d / f"manual_{p}.png")
        dice, porosity, metrics = {}, {}, {}
        for k, v in vols.items():
            med = ndi.median_filter(v, size=3)
            for pipe in PIPELINES:
                mask, info = segment(v, pipe, med)
                dice.setdefault(pipe, {})[k] = round(2 * np.count_nonzero(mask & manual) /
                                                     (np.count_nonzero(mask) + np.count_nonzero(manual)), 3)
                porosity.setdefault(pipe, {})[k] = round(info["porosity"], 3)
                for p in PLANES:
                    a, m_ = central(mask, p), central(manual, p)
                    code = np.full(a.shape, 3, np.uint8)
                    code[a & m_], code[a & ~m_], code[~a & m_] = 0, 1, 2
                    save_png(code, d / f"seg_{pipe}_{k}_{p}.png", SEG_PALETTE)
            if k != "ref60" and not args.no_metrics:
                psnr, ssim = compute_full_volume_psnr_ssim(v, ref)
                metrics[k] = {"psnr": round(psnr, 2), "ssim": round(ssim, 4)}
            print(f"sample {n}: {k} done", flush=True)
        z, x, y = ref.shape
        man["samples"].append(dict(
            id=n, label=f"Sample {n}", dir=d.name, voxel_um=float(row["voxel_um"]), shape_zxy=[z, x, y],
            window=[round(lo), round(hi)], image_px=sizes, metrics=metrics, dice=dice, porosity=porosity,
            porosity_manual=round(float(manual.mean()), 3)))
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"Saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
