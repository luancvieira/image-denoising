#!/usr/bin/env python3
"""Clone the official Restormer code at the commit used in the article and download the real-denoising checkpoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path


RESTORMER_REPO_URL = "https://github.com/swz30/Restormer.git"
# Commit of the official repository used for all Restormer experiments of the article.
RESTORMER_COMMIT = "68dc6ac"
REAL_DENOISING_WEIGHTS_URL = (
    "https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path("external/Restormer"))
    parser.add_argument("--commit", default=RESTORMER_COMMIT, help="Restormer commit to check out.")
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--download-real-denoising", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.skip_clone:
        clone_restormer(args.repo_dir, args.commit)

    if args.download_real_denoising:
        weights_path = args.repo_dir / "Denoising" / "pretrained_models" / "real_denoising.pth"
        download_file(REAL_DENOISING_WEIGHTS_URL, weights_path)
        print(f"Restormer real-denoising weights: {weights_path}", flush=True)

    return 0


def clone_restormer(repo_dir: Path, commit: str) -> None:
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", RESTORMER_REPO_URL, str(repo_dir)], check=True)
    else:
        print(f"Restormer checkout already exists: {repo_dir}", flush=True)
    subprocess.run(["git", "-C", str(repo_dir), "checkout", "--quiet", commit], check=True)
    head = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    print(f"Restormer checked out at commit {head}", flush=True)


def download_file(url: str, output_path: Path) -> None:
    if output_path.exists():
        print(f"File already exists: {output_path}", flush=True)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}", flush=True)
    urllib.request.urlretrieve(url, output_path)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
