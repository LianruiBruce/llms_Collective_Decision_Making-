#!/usr/bin/env python3
"""Quick inspector for NumPy output files (.npy/.npz)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def _preview(arr: np.ndarray, max_items: int = 6) -> str:
    if arr.size == 0:
        return "[]"
    flat = arr.reshape(-1)
    head = flat[:max_items]
    return np.array2string(head, threshold=max_items, separator=", ")


def inspect_npy(path: Path, sample_rows: int, sample_cols: int) -> None:
    arr = np.load(path, allow_pickle=True)
    print(f"\n=== {path.name} (.npy) ===")
    print(f"shape: {arr.shape}")
    print(f"dtype: {arr.dtype}")
    if arr.ndim >= 2:
        r = min(sample_rows, arr.shape[0])
        c = min(sample_cols, arr.shape[1])
        print(f"sample[{r}x{c}]:")
        print(arr[:r, :c])
    else:
        print(f"head: {_preview(arr)}")


def inspect_npz(path: Path, sample_rows: int, sample_cols: int) -> None:
    data = np.load(path, allow_pickle=True)
    print(f"\n=== {path.name} (.npz) ===")
    print(f"keys: {list(data.files)}")
    for k in data.files:
        arr = data[k]
        print(f"  - {k}: shape={arr.shape}, dtype={arr.dtype}")
        if arr.ndim >= 2:
            r = min(sample_rows, arr.shape[0])
            c = min(sample_cols, arr.shape[1])
            print(f"    sample[{r}x{c}]:")
            print(arr[:r, :c])
        else:
            print(f"    head: {_preview(arr)}")


def iter_target_files(output_dir: Path) -> Iterable[Path]:
    for ext in ("*.npy", "*.npz"):
        for p in sorted(output_dir.glob(ext)):
            yield p


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect .npy/.npz outputs quickly.")
    ap.add_argument(
        "--dir",
        type=Path,
        default=Path("output"),
        help="Directory containing .npy/.npz files",
    )
    ap.add_argument(
        "--rows",
        type=int,
        default=3,
        help="Sample rows to print for 2D arrays",
    )
    ap.add_argument(
        "--cols",
        type=int,
        default=6,
        help="Sample cols to print for 2D arrays",
    )
    args = ap.parse_args()

    if not args.dir.exists():
        raise FileNotFoundError(f"Directory not found: {args.dir}")

    files = list(iter_target_files(args.dir))
    if not files:
        print(f"No .npy/.npz files under {args.dir}")
        return

    print(f"Inspecting {len(files)} files in: {args.dir}")
    for p in files:
        if p.suffix == ".npy":
            inspect_npy(p, args.rows, args.cols)
        elif p.suffix == ".npz":
            inspect_npz(p, args.rows, args.cols)


if __name__ == "__main__":
    main()
