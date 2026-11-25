"""Prespecified automatic segmentation and segmentation metrics of the article.

Pipelines (threshold computed independently on each volume, never tuned on the reference):
  otsu2_raw  : two-class Otsu of the unfiltered volume, darker class = pore
  otsu2_med3 : 3 x 3 x 3 median filter + two-class Otsu
  otsu3_raw  : three-class multi-Otsu of the unfiltered volume, darkest class = pore
  otsu3_med3 : 3 x 3 x 3 median filter + three-class multi-Otsu
Thresholds come from a 1024-bin histogram over the 0.01-99.99 percentile range of the volume, and pore components
smaller than 27 voxels (6-connectivity) are removed.

Metrics of each automatic mask against the manual segmentation of the reference (columns *_man) and against the
reference segmented by the same pipeline: porosity and porosity error, pore Dice and IoU, boundary F1 with a
1-voxel tolerance (3 x 3 x 3 neighbourhood), and percolation along z (a 6-connected pore cluster touching the first
and the last axial slice).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu, threshold_otsu

from microct_denoising.metrics import parse_named, pore_mask_from_segmentation, write_rows

NBINS = 1024
MIN_SIZE = 27
PIPELINES = ["otsu2_raw", "otsu2_med3", "otsu3_raw", "otsu3_med3"]
S6 = ndi.generate_binary_structure(3, 1)
CUBE = np.ones((3, 3, 3), bool)


def threshold(vol: np.ndarray, pipeline: str) -> float:
    lo, hi = np.percentile(vol[::7, ::7, ::7], [0.01, 99.99])
    h, edges = np.histogram(vol, bins=NBINS, range=(lo, hi))
    c = 0.5 * (edges[:-1] + edges[1:])
    if pipeline.startswith("otsu2"):
        return float(threshold_otsu(hist=(h, c)))
    return float(threshold_multiotsu(hist=(h, c), classes=3)[0])


def clean_and_connectivity(mask: np.ndarray, min_size: int = MIN_SIZE) -> tuple[np.ndarray, dict]:
    """Remove pore components smaller than min_size voxels (6-connectivity); percolation along axis 0."""
    lab, _ = ndi.label(mask, S6)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_size
    keep[0] = False
    clean = keep[lab]
    kept = sizes * keep
    total = kept.sum()
    top = set(np.unique(lab[0])) & set(np.unique(lab[-1]))
    perc_ids = [i for i in top if i != 0 and keep[i]]
    conn = dict(
        n_components=int(keep.sum()),
        largest_component_fraction=float(kept.max() / total) if total else np.nan,
        z_percolating_fraction=float(kept[perc_ids].sum() / total) if total else np.nan,
        percolates_z=bool(perc_ids),
    )
    return clean, conn


def segment(vol: np.ndarray, pipeline: str, median: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Pore mask of one pipeline. Pass a precomputed 3 x 3 x 3 median of vol to reuse it."""
    if pipeline.endswith("med3"):
        vol = median if median is not None else ndi.median_filter(vol, size=3)
    t = threshold(vol, pipeline)
    raw_mask = vol < t
    mask, conn = clean_and_connectivity(raw_mask)
    return mask, dict(threshold=t, porosity_uncleaned=float(raw_mask.mean()), porosity=float(mask.mean()), **conn)


def boundary(m: np.ndarray) -> np.ndarray:
    return m & ~ndi.binary_erosion(m, S6)


def compare(m: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Pore Dice, IoU and boundary F1 (1-voxel tolerance) of mask m against mask ref."""
    inter = np.count_nonzero(m & ref)
    sm, sr = np.count_nonzero(m), np.count_nonzero(ref)
    bm, br = boundary(m), boundary(ref)
    nbm, nbr = np.count_nonzero(bm), np.count_nonzero(br)
    prec = np.count_nonzero(bm & ndi.binary_dilation(br, CUBE)) / nbm if nbm else np.nan
    rec = np.count_nonzero(br & ndi.binary_dilation(bm, CUBE)) / nbr if nbr else np.nan
    return dict(
        dice=2 * inter / (sm + sr) if (sm + sr) else np.nan,
        iou=inter / (sm + sr - inter) if (sm + sr - inter) else np.nan,
        boundary_precision=prec,
        boundary_recall=rec,
        boundary_f1=2 * prec * rec / (prec + rec) if (prec + rec) else np.nan,
    )


def evaluate_segmentations(
    reference: np.ndarray,
    manual: np.ndarray,
    volumes: dict[str, np.ndarray],
    pipelines: list[str] = PIPELINES,
) -> list[dict]:
    """Rows of metrics for the reference (automatic) and every other volume, for every pipeline."""
    rows = []
    _, man_conn = clean_and_connectivity(manual)
    rows.append(dict(volume="manual", pipeline="manual", porosity=float(manual.mean()), **man_conn))
    ref_masks = {}
    med_ref = ndi.median_filter(reference, size=3) if any(p.endswith("med3") for p in pipelines) else None
    for p in pipelines:
        mask, info = segment(reference, p, med_ref)
        ref_masks[p] = (mask, info["percolates_z"])
        row = dict(volume="reference", pipeline=p, **info)
        row.update({f"{k}_man": v for k, v in compare(mask, manual).items()})
        row.update(porosity_error_man=info["porosity"] - float(manual.mean()),
                   percolates_z_agree_man=info["percolates_z"] == man_conn["percolates_z"])
        rows.append(row)
    del med_ref
    for name, vol in volumes.items():
        med = ndi.median_filter(vol, size=3) if any(p.endswith("med3") for p in pipelines) else None
        for p in pipelines:
            mask, info = segment(vol, p, med)
            ref_mask, ref_perc = ref_masks[p]
            row = dict(volume=name, pipeline=p, **info)
            row.update({f"{k}_man": v for k, v in compare(mask, manual).items()})
            row.update(compare(mask, ref_mask))
            row.update(porosity_error_man=info["porosity"] - float(manual.mean()),
                       porosity_error=info["porosity"] - float(ref_mask.mean()),
                       percolates_z_agree_man=info["percolates_z"] == man_conn["percolates_z"],
                       percolates_z_agree_ref=info["percolates_z"] == ref_perc)
            rows.append(row)
            print(f"{name}/{p}: porosity {info['porosity']:.4f} | Dice vs manual {row['dice_man']:.3f}", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True, help="Long (60 min) reference volume.")
    parser.add_argument("--manual-seg", type=Path, required=True, help="Labelled manual segmentation of the reference.")
    parser.add_argument("--volume", action="append", default=[], metavar="NAME=PATH",
                        help="Volume to segment (repeat), e.g. fast=fast.nc restormer=pred.nc.")
    parser.add_argument("--pipelines", nargs="+", default=PIPELINES, choices=PIPELINES)
    parser.add_argument("--output", type=Path, required=True, help="Output CSV.")
    parser.add_argument("--var-name", default="microtom")
    parser.add_argument("--seg-var-name", default=None)
    args = parser.parse_args()

    def load(path, var):
        with xr.open_dataset(path) as ds:
            return ds[var].values

    ref = load(args.reference, args.var_name).astype(np.float32)
    manual = pore_mask_from_segmentation(load(args.manual_seg, args.seg_var_name or args.var_name), ref)
    vols = {n: load(p, args.var_name).astype(np.float32) for n, p in parse_named(args.volume).items()}
    rows = evaluate_segmentations(ref, manual, vols, args.pipelines)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    write_rows(args.output, [{k: r.get(k, "") for k in keys} for r in rows])
    print(f"Saved {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
