#!/usr/bin/env python3
"""Task 3 — Build voter similarity graph for GNN-based preference imputation.

Corresponds to Paper §3 Imputation module and §4 Structured preference heterogeneity.

Edge rule (within same neighborhood):
    OR(|age_i - age_j| <= threshold, sex_i == sex_j)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from task2_infer_voter_preferences import parse_votes


# ---------------------------------------------------------------------------
# Demographic encoding
# ---------------------------------------------------------------------------

def encode_voter_demographics(
    voter_records: List[Dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract age (float, NaN for missing), sex (0=M, 1=F, NaN=missing),
    and neighborhood strings from voter records."""
    n = len(voter_records)
    ages = np.full(n, np.nan, dtype=np.float32)
    sex_enc = np.full(n, np.nan, dtype=np.float32)
    neighborhoods: List[str] = []

    for i, rec in enumerate(voter_records):
        age_str = (rec.get("age") or "").strip()
        if age_str:
            try:
                ages[i] = float(age_str)
            except ValueError:
                pass

        sex_str = (rec.get("sex") or "").strip().upper()
        if sex_str == "M":
            sex_enc[i] = 0.0
        elif sex_str == "F":
            sex_enc[i] = 1.0

        neighborhoods.append((rec.get("neighborhood") or "").strip())

    return ages, sex_enc, neighborhoods


# ---------------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------------

def build_edges_within_neighborhoods(
    ages: np.ndarray,
    sex_enc: np.ndarray,
    neighborhoods: List[str],
    age_threshold: float = 5.0,
) -> np.ndarray:
    """Build undirected edge_index [2, 2E] using OR(age_close, same_sex)
    within each neighborhood.  Voters without a neighborhood are isolated."""
    nb_groups: Dict[str, List[int]] = defaultdict(list)
    for i, nb in enumerate(neighborhoods):
        if nb:
            nb_groups[nb].append(i)

    all_src: List[np.ndarray] = []
    all_dst: List[np.ndarray] = []

    print(f"  Neighborhoods: {len(nb_groups)}")
    for nb_name, indices in sorted(nb_groups.items(), key=lambda x: -len(x[1])):
        idx = np.array(indices, dtype=np.int64)
        n = len(idx)
        if n < 2:
            continue

        local_ages = ages[idx]
        local_sex = sex_enc[idx]

        # |age_i - age_j| <= threshold   (NaN diff → NaN → False)
        age_close = np.abs(local_ages[:, None] - local_ages[None, :]) <= age_threshold

        # same sex  (NaN == NaN → False in numpy, which is what we want)
        same_sex = local_sex[:, None] == local_sex[None, :]

        mask = np.triu(age_close | same_sex, k=1)
        rows, cols = np.nonzero(mask)

        n_edges = len(rows)
        max_edges = n * (n - 1) // 2
        density = n_edges / max(1, max_edges)
        print(f"    {nb_name:25s}  voters={n:6d}  "
              f"edges={n_edges:>10,}  density={density:.3f}")

        all_src.append(idx[rows])
        all_dst.append(idx[cols])

    if all_src:
        src = np.concatenate(all_src)
        dst = np.concatenate(all_dst)
        edge_index = np.stack(
            [np.concatenate([src, dst]), np.concatenate([dst, src])],
            axis=0,
        ).astype(np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    print(f"  Total directed entries in edge_index: {edge_index.shape[1]:,}")
    return edge_index


# ---------------------------------------------------------------------------
# Node features
# ---------------------------------------------------------------------------

def build_node_features(
    theta: np.ndarray,
    ages: np.ndarray,
    sex_enc: np.ndarray,
) -> np.ndarray:
    """Node feature X = [θ_i | age_norm | sex].  Shape (n, m+2)."""
    median_age = float(np.nanmedian(ages))
    age_filled = np.where(np.isnan(ages), median_age, ages)
    a_min, a_max = float(age_filled.min()), float(age_filled.max())
    age_norm = (age_filled - a_min) / max(a_max - a_min, 1e-8)

    sex_filled = np.where(np.isnan(sex_enc), 0.5, sex_enc)

    X = np.column_stack([theta, age_norm.astype(np.float64), sex_filled.astype(np.float64)])
    return X


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Task 3: Build voter similarity graph.",
    )
    ap.add_argument("--input", type=Path,
                    default=Path("raw/Poland_Warszawa_2024.pb"))
    ap.add_argument("--theta", type=Path,
                    default=Path("output/task2_preferences.npy"))
    ap.add_argument("--output-prefix", type=Path,
                    default=Path("output/task3_graph"))
    ap.add_argument("--age-threshold", type=float, default=5.0,
                    help="Age difference threshold for edge creation")
    args = ap.parse_args()

    # ---- Load θ from Task 2 ----
    Theta = np.load(args.theta)
    print(f"Loaded Θ: {Theta.shape}")

    # ---- Parse voter records ----
    voter_ids, _, voter_records = parse_votes(args.input)
    print(f"Voters: {len(voter_ids)}")

    # ---- Demographics ----
    ages, sex_enc, neighborhoods = encode_voter_demographics(voter_records)
    n_age = int((~np.isnan(ages)).sum())
    n_sex = int((~np.isnan(sex_enc)).sum())
    n_nb = sum(1 for nb in neighborhoods if nb)
    print(f"Valid age: {n_age}/{len(ages)}, "
          f"valid sex: {n_sex}/{len(sex_enc)}, "
          f"has neighborhood: {n_nb}/{len(neighborhoods)}")

    # ---- Edges ----
    print("Building edges (within-neighborhood, OR criterion) ...")
    edge_index = build_edges_within_neighborhoods(
        ages, sex_enc, neighborhoods, args.age_threshold,
    )

    # ---- Node features ----
    X = build_node_features(Theta, ages, sex_enc)
    print(f"Node features X: {X.shape}  "
          f"(= {Theta.shape[1]} θ-dims + age_norm + sex)")

    # ---- Save ----
    out = args.output_prefix
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out.with_suffix(".npz"),
        x=X,
        edge_index=edge_index,
        theta=Theta,
        ages=ages,
        sex_enc=sex_enc,
        voter_ids=np.array(voter_ids, dtype=object),
        neighborhoods=np.array(neighborhoods, dtype=object),
    )

    print(f"\nSaved: {out.with_suffix('.npz')}")
    print(f"  x          : {X.shape}")
    print(f"  edge_index : {edge_index.shape}")
    print(f"\nTo load as PyG Data:")
    print("  import torch; from torch_geometric.data import Data")
    print(f"  d = np.load('{out.with_suffix('.npz')}', allow_pickle=True)")
    print("  data = Data(x=torch.from_numpy(d['x']).float(),")
    print("              edge_index=torch.from_numpy(d['edge_index']))")


if __name__ == "__main__":
    main()
