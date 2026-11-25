"""Image metrics of the article and CSV helpers.

compute_full_volume_psnr_ssim : 3D PSNR and SSIM with scikit-image (SSIM: 7 x 7 x 7 uniform window, L = 65535).

The `microct-evaluate` CLI (main) computes, for each restored or acquired volume against the long reference:
  - PSNR and 3D SSIM with L = 65535 and with L_ref = p99.9 - p0.1 of the reference (sensitivity to the data range);
  - 2D SSIM with 7 x 7 windows in the axial plane and in the two longitudinal planes;
  - MAE, mean signed residual and RMSE in three regions defined from the manual segmentation of the reference:
      pore interior, solid interior, and pore-solid boundary (voxels with a voxel of the other phase in their
      3 x 3 x 3 neighbourhood);
  - consistency: mean absolute difference between neighbouring voxels along each axis, for the volume and for its
    residual (volume - reference).

SSIM is computed in z-chunks with a 3-slice halo; the windowed statistics of every kept voxel are exact and the
borders that scikit-image crops (3 voxels on every axis spanned by the window) are excluded, so the 3D value
equals skimage.metrics.structural_similarity(ref, vol, data_range=L).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage as ndi
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from microct_denoising.constants import DATA_RANGE

K1, K2, WIN, PAD = 0.01, 0.03, 7, 3
WINDOWS = {"3d": (WIN, WIN, WIN), "axial": (1, WIN, WIN), "long_x": (WIN, WIN, 1), "long_y": (WIN, 1, WIN)}
REGIONS = {0: "pore_interior", 1: "solid_interior", 2: "boundary"}


def load_prediction_volume(path: Path, var_name: str) -> np.ndarray:
    with xr.open_dataset(path) as ds:
        return ds[var_name].values.astype(np.float32)


def compute_full_volume_psnr_ssim(
    pred_vol: np.ndarray,
    ref_vol: np.ndarray,
    data_range: float = DATA_RANGE,
) -> tuple[float, float]:
    """Full-volume 3D PSNR and SSIM in gray-value space (L = 65535 by default)."""

    pred_vol = pred_vol.astype(np.float32)
    ref_vol = ref_vol.astype(np.float32)
    psnr = peak_signal_noise_ratio(ref_vol, pred_vol, data_range=data_range)
    ssim = structural_similarity(ref_vol, pred_vol, data_range=data_range)
    return float(psnr), float(ssim)


def reference_range(ref: np.ndarray, lo: float = 0.1, hi: float = 99.9) -> tuple[float, float]:
    """Percentiles of the reference used for L_ref = p99.9 - p0.1 (histogram of rounded gray values)."""
    h = np.bincount(np.clip(np.rint(ref), 0, 65535).astype(np.uint16).ravel(), minlength=65536)
    c = np.cumsum(h) / h.sum()
    return float(np.searchsorted(c, lo / 100)), float(np.searchsorted(c, hi / 100))


def pore_mask_from_segmentation(seg: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Pore phase of a labelled segmentation: the label with the darkest mean gray value in the reference."""
    labels = np.unique(seg)
    means = {int(lab): float(ref[seg == lab].mean()) for lab in labels}
    return seg == min(means, key=means.get)


def region_map(pore: np.ndarray) -> np.ndarray:
    """0 = pore interior, 1 = solid interior, 2 = pore-solid boundary (3 x 3 x 3 neighbourhood)."""
    cube = np.ones((3, 3, 3), bool)
    solid = ~pore
    bnd = (pore & ndi.binary_dilation(solid, cube)) | (solid & ndi.binary_dilation(pore, cube))
    region = np.zeros(pore.shape, np.uint8)
    region[solid] = 1
    region[bnd] = 2
    return region


def _ssim_map(ux, uy, uxx, uyy, uxy, npix, L):
    cn = npix / (npix - 1.0)
    vx, vy, vxy = cn * (uxx - ux * ux), cn * (uyy - uy * uy), cn * (uxy - ux * uy)
    c1, c2 = (K1 * L) ** 2, (K2 * L) ** 2
    return ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))


def evaluate_volumes(
    ref: np.ndarray,
    volumes: dict[str, np.ndarray],
    pore: np.ndarray | None = None,
    chunk: int = 100,
) -> dict[str, list[dict]]:
    """All metrics of the article for several volumes against one reference. Returns lists of CSV rows."""
    Z, H, W = ref.shape
    p001, p999 = reference_range(ref)
    Ls = {"L65535": DATA_RANGE, "Lref": p999 - p001}
    region = region_map(pore) if pore is not None else None
    acc: dict = {}
    reg: dict = {}
    sq = {m: 0.0 for m in volumes}
    for z0 in range(0, Z, chunk):
        z1 = min(Z, z0 + chunk)
        e0, e1 = max(0, z0 - PAD), min(Z, z1 + PAD)
        off, nz = z0 - e0, z1 - z0
        gz = np.arange(z0, z1)
        Yc = ref[e0:e1].astype(np.float64)
        ystats = {w: (ndi.uniform_filter(Yc, s), ndi.uniform_filter(Yc * Yc, s)) for w, s in WINDOWS.items()}
        Ycore = Yc[off:off + nz]
        for m, vol in volumes.items():
            Xc = vol[e0:e1].astype(np.float64)
            e = Xc[off:off + nz] - Ycore
            sq[m] += float((e * e).sum())
            if region is not None:
                rc = region[z0:z1]
                for r in REGIONS:
                    er = e[rc == r]
                    t = reg.setdefault((m, r), [0, 0.0, 0.0, 0.0])
                    t[0] += er.size; t[1] += np.abs(er).sum(); t[2] += er.sum(); t[3] += (er * er).sum()
            for w, s in WINDOWS.items():
                ux, uxx, uxy = ndi.uniform_filter(Xc, s), ndi.uniform_filter(Xc * Xc, s), ndi.uniform_filter(Xc * Yc, s)
                uy, uyy = ystats[w]
                sl = [slice(off, off + nz), slice(None), slice(None)]
                zmask = (gz >= PAD) & (gz < Z - PAD) if s[0] > 1 else np.ones(nz, bool)
                if s[1] > 1:
                    sl[1] = slice(PAD, H - PAD)
                if s[2] > 1:
                    sl[2] = slice(PAD, W - PAD)
                sl = tuple(sl)
                for ln, L in Ls.items():
                    S = _ssim_map(ux[sl], uy[sl], uxx[sl], uyy[sl], uxy[sl], float(np.prod(s)), L)[zmask]
                    t = acc.setdefault((m, w, ln), [0.0, 0])
                    t[0] += S.sum(); t[1] += S.size

    n = float(ref.size)
    images = []
    for m in volumes:
        row = dict(volume=m, ref_p0_1=p001, ref_p99_9=p999, L_ref=Ls["Lref"])
        for ln, L in Ls.items():
            row[f"psnr_{ln}"] = 10 * np.log10(L * L / (sq[m] / n))
            for w in WINDOWS:
                s, c = acc[(m, w, ln)]
                row[f"ssim_{w}_{ln}"] = s / c
        images.append(row)
    regions = []
    if region is not None:
        for (m, r), (cnt, sa, se, s2) in reg.items():
            regions.append(dict(volume=m, region=REGIONS[r], fraction=cnt / n, mae=sa / cnt, bias=se / cnt,
                                rmse=float(np.sqrt(s2 / cnt))))
    consistency = []
    for m, vol in {"reference": ref, **volumes}.items():
        d = [float(np.mean(np.abs(np.diff(vol, axis=a)), dtype=np.float64)) for a in range(3)]
        row = dict(volume=m, diff_axis0=d[0], diff_axis1=d[1], diff_axis2=d[2])
        if m != "reference":
            res = vol.astype(np.float32) - ref.astype(np.float32)
            de = [float(np.mean(np.abs(np.diff(res, axis=a)), dtype=np.float64)) for a in range(3)]
            row.update(res_diff_axis0=de[0], res_diff_axis1=de[1], res_diff_axis2=de[2])
        consistency.append(row)
    return dict(images=images, regions=regions, consistency=consistency)


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def upsert_metrics_row(csv_path: Path, row: dict[str, object], key_fields: tuple[str, ...]) -> None:
    """Insert or replace a metrics row identified by key_fields."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    fieldnames = list(row.keys())

    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            rows = list(reader)

    new_row = {key: str(value) for key, value in row.items()}
    for key in new_row:
        if key not in fieldnames:
            fieldnames.append(key)
    key = tuple(new_row[field] for field in key_fields)

    updated_rows = [new_row if tuple(r.get(f, "") for f in key_fields) == key else r for r in rows]
    if key not in [tuple(r.get(f, "") for f in key_fields) for r in rows]:
        updated_rows.append(new_row)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)


def parse_named(items: list[str]) -> dict[str, Path]:
    out = {}
    for item in items:
        name, _, path = item.partition("=")
        if not path:
            raise ValueError(f"Use name=path, got {item!r}")
        out[name] = Path(path)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True, help="Long (60 min) reference volume.")
    parser.add_argument("--volume", action="append", required=True, metavar="NAME=PATH",
                        help="Volume to evaluate (repeat), e.g. fast=fast.nc restormer=pred.nc.")
    parser.add_argument("--manual-seg", type=Path, default=None,
                        help="Labelled segmentation of the reference (enables the region-wise errors).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--var-name", default="microtom")
    parser.add_argument("--seg-var-name", default=None, help="Variable of --manual-seg (default: --var-name).")
    args = parser.parse_args()

    ref = load_prediction_volume(args.reference, args.var_name)
    vols = {name: load_prediction_volume(p, args.var_name) for name, p in parse_named(args.volume).items()}
    pore = None
    if args.manual_seg is not None:
        with xr.open_dataset(args.manual_seg) as ds:
            seg = ds[args.seg_var_name or args.var_name].values
        pore = pore_mask_from_segmentation(seg, ref)
    res = evaluate_volumes(ref, vols, pore)
    for key, rows in res.items():
        write_rows(args.output_dir / f"{key}.csv", rows)
    for row in res["images"]:
        print(f"{row['volume']}: PSNR {row['psnr_L65535']:.3f} dB | SSIM {row['ssim_3d_L65535']:.4f} "
              f"| SSIM(L_ref) {row['ssim_3d_Lref']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
