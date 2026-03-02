#!/usr/bin/env python3
"""Task 1 — Build a multi-dimensional project feature matrix f(x) from a .pb file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _split_tags(raw_value: str) -> List[str]:
    """Split comma-separated tags and normalize whitespace."""
    if raw_value is None:
        return []
    value = raw_value.strip()
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _minmax(values: np.ndarray) -> np.ndarray:
    """Min-Max normalize 1D array to [0, 1]."""
    if values.size == 0:
        return values
    v_min = float(values.min())
    v_max = float(values.max())
    if np.isclose(v_min, v_max):
        return np.zeros_like(values, dtype=np.float64)
    return (values - v_min) / (v_max - v_min)


def parse_pb_file(pb_path: Path) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """Parse META and PROJECTS blocks from a PB file."""
    text = pb_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    if not lines or lines[0].strip() != "META":
        raise ValueError("Invalid PB format: missing META block header.")

    # Locate section boundaries.
    try:
        projects_start = lines.index("PROJECTS")
        votes_start = lines.index("VOTES")
    except ValueError as exc:
        raise ValueError("Invalid PB format: missing PROJECTS or VOTES block.") from exc

    # Parse META.
    meta_rows = lines[1:projects_start]
    meta_reader = csv.DictReader(meta_rows, delimiter=";")
    meta: Dict[str, str] = {}
    for row in meta_reader:
        key = (row.get("key") or "").strip()
        value = (row.get("value") or "").strip()
        if key:
            meta[key] = value

    # Parse PROJECTS.
    project_rows = lines[projects_start + 1 : votes_start]
    project_reader = csv.DictReader(project_rows, delimiter=";")
    projects = [row for row in project_reader]

    return meta, projects


def build_feature_matrix(
    projects: List[Dict[str, str]],
    budget: float,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build feature matrix f(x) = [category one-hot, target one-hot, cost feature]."""
    if budget <= 0:
        raise ValueError("Budget must be positive.")

    all_categories = sorted(
        {tag for p in projects for tag in _split_tags(p.get("category", ""))}
    )
    all_targets = sorted({tag for p in projects for tag in _split_tags(p.get("target", ""))})

    n_projects = len(projects)
    n_features = len(all_categories) + len(all_targets) + 1
    matrix = np.zeros((n_projects, n_features), dtype=np.float64)

    category_index = {name: idx for idx, name in enumerate(all_categories)}
    target_offset = len(all_categories)
    target_index = {name: target_offset + idx for idx, name in enumerate(all_targets)}

    costs = np.array([float(p["cost"]) for p in projects], dtype=np.float64)
    cost_ratio = costs / budget
    cost_ratio_minmax = _minmax(cost_ratio)

    for i, project in enumerate(projects):
        for tag in _split_tags(project.get("category", "")):
            matrix[i, category_index[tag]] = 1.0

        for tag in _split_tags(project.get("target", "")):
            matrix[i, target_index[tag]] = 1.0

        matrix[i, -1] = cost_ratio_minmax[i]

    feature_names = (
        [f"category::{c}" for c in all_categories]
        + [f"target::{t}" for t in all_targets]
        + ["cost_ratio_minmax"]
    )
    project_ids = [str(p["project_id"]) for p in projects]
    return matrix, feature_names, project_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Task 1: Build [num_projects, m_features] matrix from a PB file "
            "using category/target multi-label encoding and normalized cost."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("raw/Poland_Warszawa_2024.pb"),
        help="Path to input .pb file",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("output/task1_features"),
        help="Output prefix (without extension)",
    )
    args = parser.parse_args()

    meta, projects = parse_pb_file(args.input)
    print("META keys:", sorted(meta.keys())[:30])
    print("PROJECT keys:", sorted(projects[0].keys()) if projects else [])
    print("Projects after filter:", len(projects))

    budget = float(meta["budget"])
    matrix, feature_names, project_ids = build_feature_matrix(projects, budget)

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_prefix.with_suffix(".npy"), matrix)
    np.savez_compressed(
        output_prefix.with_suffix(".npz"),
        matrix=matrix,
        feature_names=np.array(feature_names, dtype=object),
        project_ids=np.array(project_ids, dtype=object),
    )
    output_prefix.with_name(output_prefix.name + "_feature_names.json").write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_prefix.with_name(output_prefix.name + "_project_ids.json").write_text(
        json.dumps(project_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Projects: {matrix.shape[0]}")
    print(f"Features: {matrix.shape[1]}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Saved matrix: {output_prefix.with_suffix('.npy')}")
    print(f"Saved bundle: {output_prefix.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
