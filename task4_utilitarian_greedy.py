#!/usr/bin/env python3
"""Task 4 — Utilitarian Greedy (UG) baseline allocation.

Corresponds to Paper §2.2.5 — Utilitarian Greedy selection rule.

For each project j:
    aggregate_utility(j) = Σ_i  θ_i^T f(x_j)
    cost_effectiveness(j) = aggregate_utility(j) / cost(j)

Projects are greedily added by descending cost-effectiveness
until the budget is exhausted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from task1_build_project_features import build_feature_matrix, parse_pb_file


def utilitarian_greedy(
    utility_per_project: np.ndarray,
    costs: np.ndarray,
    budget: float,
) -> List[int]:
    """Return list of selected project indices (0-based)."""
    cost_eff = utility_per_project / np.maximum(costs, 1e-12)
    order = np.argsort(-cost_eff)

    selected: List[int] = []
    remaining = budget
    for j in order:
        c = costs[int(j)]
        if c <= remaining:
            selected.append(int(j))
            remaining -= c
    return selected


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Task 4: Utilitarian Greedy baseline allocation.",
    )
    ap.add_argument("--input", type=Path,
                    default=Path("raw/Poland_Warszawa_2024.pb"))
    ap.add_argument("--theta", type=Path,
                    default=Path("output/task2_preferences.npy"))
    ap.add_argument("--output", type=Path,
                    default=Path("output/task4_ug_baseline.json"))
    args = ap.parse_args()

    # ---- Parse PB file ----
    meta, projects = parse_pb_file(args.input)
    budget = float(meta["budget"])
    F, feature_names, project_ids = build_feature_matrix(projects, budget)
    costs = np.array([float(p["cost"]) for p in projects], dtype=np.float64)
    names = [p["name"] for p in projects]
    actual_selected = [int(p["selected"]) for p in projects]

    print(f"Budget β         : {budget:,.0f} PLN")
    print(f"Projects         : {len(projects)}")
    print(f"Feature matrix F : {F.shape}")

    # ---- Load Θ ----
    Theta = np.load(args.theta)
    print(f"Preference Θ     : {Theta.shape}")

    # ---- Aggregate utility per project ----
    # u_i(x_j) = θ_i^T f(x_j)  →  Σ_i = (Σ_i θ_i)^T f(x_j)
    theta_sum = Theta.sum(axis=0)                 # (m,)
    utility = F @ theta_sum                       # (n_projects,)
    cost_eff = utility / np.maximum(costs, 1e-12) # (n_projects,)

    # ---- Greedy selection ----
    ug_indices = utilitarian_greedy(utility, costs, budget)
    ug_cost = costs[ug_indices].sum()
    ug_welfare = utility[ug_indices].sum()

    print(f"\n{'='*70}")
    print(f"Utilitarian Greedy Baseline")
    print(f"{'='*70}")
    print(f"Selected projects : {len(ug_indices)}")
    print(f"Total cost        : {ug_cost:>14,.0f} / {budget:,.0f} PLN "
          f"({100 * ug_cost / budget:.1f}%)")
    print(f"Social Welfare    : {ug_welfare:>14,.2f}")
    print()

    print(f"{'Rank':<5} {'ID':>6} {'Cost':>12} {'Utility':>12} "
          f"{'Eff':>10} {'Actual':>7}  Name")
    print("-" * 100)
    for rank, j in enumerate(ug_indices, 1):
        pid = project_ids[j]
        act_flag = "✓" if actual_selected[j] else ""
        print(f"{rank:<5} {pid:>6} {costs[j]:>12,.0f} {utility[j]:>12,.2f} "
              f"{cost_eff[j]:>10.4f} {act_flag:>7}  {names[j][:60]}")

    # ---- Compare with actual outcome ----
    actual_indices = [j for j, s in enumerate(actual_selected) if s == 1]
    actual_cost = costs[actual_indices].sum()
    actual_welfare = utility[actual_indices].sum()
    overlap = set(ug_indices) & set(actual_indices)

    print(f"\n{'='*70}")
    print(f"Comparison with actual PB outcome")
    print(f"{'='*70}")
    print(f"Actual selected   : {len(actual_indices)} projects, "
          f"cost={actual_cost:,.0f} PLN, welfare={actual_welfare:,.2f}")
    print(f"UG selected       : {len(ug_indices)} projects, "
          f"cost={ug_cost:,.0f} PLN, welfare={ug_welfare:,.2f}")
    print(f"Overlap           : {len(overlap)} projects")
    print(f"Welfare gain (UG) : {ug_welfare - actual_welfare:+,.2f} "
          f"({100 * (ug_welfare - actual_welfare) / max(abs(actual_welfare), 1e-12):+.2f}%)")

    # ---- Save ----
    result = {
        "method": "utilitarian_greedy",
        "budget": budget,
        "selected_project_ids": [project_ids[j] for j in ug_indices],
        "selected_indices": ug_indices,
        "total_cost": float(ug_cost),
        "social_welfare": float(ug_welfare),
        "num_selected": len(ug_indices),
        "actual_selected_ids": [project_ids[j] for j in actual_indices],
        "actual_welfare": float(actual_welfare),
        "overlap_ids": sorted(project_ids[j] for j in overlap),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
