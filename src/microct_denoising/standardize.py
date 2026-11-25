"""Aluminium standardization of micro-CT volumes (article, Methods).

Each scan is shifted so that the aluminium standard scanned with the plug reads the same value T in every scan:

    X_al = X + (T - A),        T = 6500

A is the median gray value of the aluminium ring, measured on the uncropped reconstruction over every slice of the
z-range of the rock crop. The ring is given per slice by two circles (inner and outer radius of the aluminium
tube); only voxels between 1.01 x the inner radius and 0.99 x the outer radius are used, so the edges of the ring
are excluded. The shift is applied to the rock crop without clipping (float32 output). The fast, intermediate and
long scans of a plug are standardized independently, each with its own A; a new plug needs only its own scan.

Ring geometry, either
  --rings rings.csv        one row per slice and circle: slice, region (1 = inner, 2 = outer), center_x, center_y,
                           radii (center_y indexes the first in-plane axis, center_x the second); or
  --center-x/--center-y/--r-inner/--r-outer   one circle pair for every slice.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from microct_denoising.constants import ALUMINIUM_TARGET

INNER_FACTOR = 1.01
OUTER_FACTOR = 0.99


def read_rings(path: Path) -> dict[tuple[int, int], tuple[float, float, float]]:
    """{(slice, region): (center_y, center_x, radius)} from a ring CSV (comma or semicolon separated)."""
    text = Path(path).read_text(encoding="utf-8")
    dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",;")
    rings = {}
    for row in csv.DictReader(text.splitlines(), dialect=dialect):
        key = (int(float(row["slice"])), int(float(row["region"])))
        rings[key] = (float(row["center_y"]), float(row["center_x"]), float(row["radii"]))
    return rings


def aluminium_value(
    reconstruction: Path,
    z_range: tuple[int, int],
    rings: dict | None = None,
    circle: tuple[float, float, float, float] | None = None,
    var_name: str = "microtom",
    block: int = 100,
) -> dict[str, float]:
    """Median (and mean) gray value of the aluminium ring between z_range[0] (inclusive) and z_range[1] (exclusive).

    circle = (center_y, center_x, r_inner, r_outer) when the ring does not move between slices.
    """
    if (rings is None) == (circle is None):
        raise ValueError("Give either rings or circle.")
    z0, z1 = z_range
    hist = np.zeros(65536, np.int64)
    with xr.open_dataset(reconstruction) as ds:
        da = ds[var_name]
        zdim = da.dims[0]
        yy = xx = None
        for b0 in range(z0, z1, block):
            blk = da.isel({zdim: slice(b0, min(z1, b0 + block))}).values
            if yy is None:
                yy, xx = np.indices(blk.shape[1:])
            for i, img in enumerate(blk):
                z = b0 + i
                if rings is not None:
                    cy, cx, ri = rings[(z, 1)]
                    ro = rings[(z, 2)][2]
                else:
                    cy, cx, ri, ro = circle
                d = np.hypot(yy - cy, xx - cx)
                vals = img[(d > ri * INNER_FACTOR) & (d < ro * OUTER_FACTOR)]
                vals = np.clip(np.rint(vals), 0, 65535).astype(np.int64)
                hist += np.bincount(vals, minlength=65536)
    n = int(hist.sum())
    if n == 0:
        raise RuntimeError("No aluminium voxels found; check the ring geometry.")
    cdf = np.cumsum(hist) / n
    g = np.arange(65536)
    return dict(aluminium_median=float(np.searchsorted(cdf, 0.5)), aluminium_mean=float((hist * g).sum() / n),
                aluminium_voxels=n)


def apply_shift(
    src: Path,
    dst: Path,
    aluminium: float,
    target: float = ALUMINIUM_TARGET,
    var_name: str = "microtom",
) -> float:
    """Write src + (target - aluminium) to dst as float32 (no clipping). Returns the shift."""
    shift = float(target) - float(aluminium)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(src) as ds:
        da = ds[var_name]
        attrs = dict(da.attrs)
        attrs.update(
            aluminium_standardization="X_al = X + (T - A); no clipping",
            aluminium_target_T=float(target),
            aluminium_value_A=float(aluminium),
            shift_applied=shift,
            source_file=str(Path(src).name),
        )
        out = xr.Dataset(
            {var_name: xr.DataArray(da.values.astype(np.float32) + np.float32(shift), dims=da.dims,
                                    coords=da.coords, attrs=attrs)},
            attrs=dict(ds.attrs),
        )
    out.to_netcdf(dst)
    return shift


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Volume (rock crop) to standardize.")
    parser.add_argument("--output", type=Path, required=True, help="Standardized output .nc.")
    parser.add_argument("--aluminium-value", type=float, default=None,
                        help="Known aluminium value A; if omitted it is measured on --reconstruction.")
    parser.add_argument("--reconstruction", type=Path, default=None, help="Uncropped reconstruction with the ring.")
    parser.add_argument("--z-range", type=int, nargs=2, metavar=("Z0", "Z1"), default=None,
                        help="Slices of the reconstruction covered by the crop (Z1 exclusive).")
    parser.add_argument("--rings", type=Path, default=None, help="Per-slice ring circles (CSV).")
    parser.add_argument("--center-x", type=float, default=None)
    parser.add_argument("--center-y", type=float, default=None)
    parser.add_argument("--r-inner", type=float, default=None)
    parser.add_argument("--r-outer", type=float, default=None)
    parser.add_argument("--target", type=float, default=ALUMINIUM_TARGET)
    parser.add_argument("--var-name", default="microtom")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    info: dict[str, object] = {"input": str(args.input), "target": args.target}
    if args.aluminium_value is None:
        if args.reconstruction is None or args.z_range is None:
            raise ValueError("Give --aluminium-value, or --reconstruction and --z-range to measure it.")
        rings = read_rings(args.rings) if args.rings else None
        circle = None
        if rings is None:
            if None in (args.center_x, args.center_y, args.r_inner, args.r_outer):
                raise ValueError("Give --rings or all of --center-x, --center-y, --r-inner, --r-outer.")
            circle = (args.center_y, args.center_x, args.r_inner, args.r_outer)
        al = aluminium_value(args.reconstruction, tuple(args.z_range), rings, circle, args.var_name)
        info.update(al, reconstruction=str(args.reconstruction), z_range=list(args.z_range))
        aluminium = al["aluminium_median"]
    else:
        aluminium = args.aluminium_value
        info["aluminium_median"] = aluminium
    info["shift"] = apply_shift(args.input, args.output, aluminium, args.target, args.var_name)
    args.output.with_suffix(".aluminium.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"A = {aluminium:.0f}, shift = {info['shift']:+.0f} -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
