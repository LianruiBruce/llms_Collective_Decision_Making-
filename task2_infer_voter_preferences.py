#!/usr/bin/env python3
"""Task 2 — Infer continuous latent voter preferences θ_i from approval votes.

Corresponds to Paper Section 2 — Population State / Individual Utility:
    u_i(x) = θ_i^T f(x)

Two methods:
  ridge    — closed-form vectorized ridge regression (fast, default)
  logistic — per-voter sklearn LogisticRegression (slower, principled for binary)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from task1_build_project_features import build_feature_matrix, parse_pb_file


# ---------------------------------------------------------------------------
# VOTES parsing
# ---------------------------------------------------------------------------

def parse_votes(
    pb_path: Path,
) -> Tuple[List[str], List[List[str]], List[Dict[str, str]]]:
    """Parse the VOTES block of a PB file.

    Returns
    -------
    voter_ids : list[str]
    vote_lists : list[list[str]]  — each inner list contains voted project IDs
    voter_records : list[dict]    — raw rows (voter_id, vote, age, sex, neighborhood)
    """
    text = pb_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    votes_idx = lines.index("VOTES")
    vote_lines = lines[votes_idx + 1 :]
    reader = csv.DictReader(vote_lines, delimiter=";")

    voter_ids: List[str] = []
    vote_lists: List[List[str]] = []
    voter_records: List[Dict[str, str]] = []

    for row in reader:
        voter_ids.append(row["voter_id"])
        raw = (row.get("vote") or "").strip()
        pids = [p.strip() for p in raw.split(",") if p.strip()] if raw else []
        vote_lists.append(pids)
        voter_records.append(row)

    return voter_ids, vote_lists, voter_records


# ---------------------------------------------------------------------------
# Vote matrix
# ---------------------------------------------------------------------------

def build_vote_matrix(
    vote_lists: List[List[str]],
    project_ids: List[str],
) -> np.ndarray:
    """Build dense binary vote matrix Y of shape [num_voters, num_projects]."""
    pid_to_col = {pid: idx for idx, pid in enumerate(project_ids)}
    n_voters = len(vote_lists)
    n_projects = len(project_ids)
    Y = np.zeros((n_voters, n_projects), dtype=np.float64)

    for i, pids in enumerate(vote_lists):
        for pid in pids:
            col = pid_to_col.get(pid)
            if col is not None:
                Y[i, col] = 1.0

    return Y


# ---------------------------------------------------------------------------
# Preference inference — Ridge (vectorized, fast)
# ---------------------------------------------------------------------------

def infer_preferences_ridge(
    Y: np.ndarray,
    F: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Closed-form ridge regression, vectorized over all voters.

    For each voter i:
        θ_i = (F^T F + α I)^{-1} F^T y_i

    Vectorized as:
        P = solve(F^T F + α I, F^T)       shape (m, n_projects)
        Θ = Y @ P^T                        shape (n_voters, m)
    """
    m = F.shape[1]
    FtF_reg = F.T @ F + alpha * np.eye(m)
    P = np.linalg.solve(FtF_reg, F.T)  # (m, n_projects)
    Theta = Y @ P.T  # (n_voters, m)
    return Theta


# ---------------------------------------------------------------------------
# Preference inference — Logistic Regression (per-voter, sklearn)
# ---------------------------------------------------------------------------

def infer_preferences_logistic(
    Y: np.ndarray,
    F: np.ndarray,
    C: float = 1.0,
) -> np.ndarray:
    """Per-voter logistic regression with fit_intercept=False.

    C is the inverse regularization strength (larger C → less regularization).
    """
    from sklearn.linear_model import LogisticRegression

    n_voters = Y.shape[0]
    m = F.shape[1]
    Theta = np.zeros((n_voters, m), dtype=np.float64)

    report_every = max(1, n_voters // 20)
    for i in range(n_voters):
        y_i = Y[i]
        n_pos = int(y_i.sum())
        if n_pos == 0 or n_pos == F.shape[0]:
            continue
        lr = LogisticRegression(
            fit_intercept=False, C=C, solver="lbfgs",
            max_iter=200, penalty="l2",
        )
        lr.fit(F, y_i)
        Theta[i] = lr.coef_[0]

        if (i + 1) % report_every == 0:
            print(f"  logistic: {i + 1}/{n_voters} "
                  f"({100 * (i + 1) / n_voters:.0f}%)", flush=True)

    return Theta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Task 2: Infer voter preference vectors θ_i from approval votes.",
    )
    ap.add_argument("--input", type=Path,
                    default=Path("raw/Poland_Warszawa_2024.pb"))
    ap.add_argument("--output-prefix", type=Path,
                    default=Path("output/task2_preferences"))
    ap.add_argument("--method", choices=["ridge", "logistic"], default="ridge",
                    help="ridge (fast closed-form) or logistic (per-voter sklearn)")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Ridge regularization (only for ridge)")
    ap.add_argument("--C", type=float, default=1.0, dest="C_param",
                    help="Logistic inverse regularization (only for logistic)")
    args = ap.parse_args()

    # ---- Task 1 artefacts: parse file & build feature matrix ----
    meta, projects = parse_pb_file(args.input)
    budget = float(meta["budget"])
    F, feature_names, project_ids = build_feature_matrix(projects, budget)
    print(f"Feature matrix F : {F.shape}  (m={F.shape[1]})")

    # ---- Parse votes ----
    voter_ids, vote_lists, voter_records = parse_votes(args.input)
    print(f"Total voters     : {len(voter_ids)}")

    Y = build_vote_matrix(vote_lists, project_ids)
    nnz = int(Y.sum())
    print(f"Vote matrix Y    : {Y.shape}  "
          f"(nnz={nnz}, density={Y.mean():.4f})")

    # ---- Infer Θ ----
    if args.method == "ridge":
        print(f"Running ridge regression (alpha={args.alpha}) ...")
        Theta = infer_preferences_ridge(Y, F, alpha=args.alpha)
    else:
        print(f"Running per-voter logistic regression (C={args.C_param}) ...")
        Theta = infer_preferences_logistic(Y, F, C=args.C_param)

    print(f"Preference Θ     : {Theta.shape}")

    # ---- Save ----
    out = args.output_prefix
    out.parent.mkdir(parents=True, exist_ok=True)

    np.save(out.with_suffix(".npy"), Theta)
    np.savez_compressed(
        out.with_suffix(".npz"),
        theta=Theta,
        feature_names=np.array(feature_names, dtype=object),
        voter_ids=np.array(voter_ids, dtype=object),
        project_ids=np.array(project_ids, dtype=object),
    )

    # ---- Summary stats ----
    print(f"\nΘ statistics (shape {Theta.shape}):")
    print(f"  mean : {Theta.mean():.6f}")
    print(f"  std  : {Theta.std():.6f}")
    print(f"  min  : {Theta.min():.6f}")
    print(f"  max  : {Theta.max():.6f}")

    print(f"\nPer-feature population mean θ̄:")
    for j, name in enumerate(feature_names):
        print(f"  {name:40s}  {Theta[:, j].mean():+.6f}")

    print(f"\nSaved: {out.with_suffix('.npy')}")
    print(f"Saved: {out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
