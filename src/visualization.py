"""Plots for the SRL benchmark.

Three figures from the guide:
  1. overall F1 across configurations (bar),
  2. per-role F1 across configurations (grouped bar),
  3. role confusion matrix (heatmap).

Plus a dataset role-distribution bar chart used in data exploration. Every
function saves a PNG under results/figures/ and returns the path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless; safe in notebooks and scripts
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "results" / "figures"


def _ensure_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_role_distribution(role_counts: Dict[str, int], filename: str = "role_distribution.png") -> Path:
    _ensure_dir()
    items = sorted(role_counts.items(), key=lambda kv: kv[1], reverse=True)
    roles = [k for k, _ in items]
    counts = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(roles, counts, color="#4C72B0")
    ax.set_title("Distribution of PropBank Semantic Roles")
    ax.set_xlabel("Semantic Role")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_overall_f1(summary_rows: List[dict], filename: str = "overall_f1.png") -> Path:
    _ensure_dir()
    configs = [r["configuration"] for r in summary_rows]
    f1s = [r["f1"] for r in summary_rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(configs, f1s, color="#55A868")
    ax.set_ylim(0, 1)
    ax.set_title("Labeled Argument Span F1 by Configuration")
    ax.set_ylabel("Strict F1")
    ax.tick_params(axis="x", rotation=20)
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_role_f1_comparison(
    per_config_role_f1: Dict[str, Dict[str, float]],
    roles: List[str] | None = None,
    filename: str = "role_f1_comparison.png",
) -> Path:
    """per_config_role_f1: {config_name: {role: f1}}."""
    _ensure_dir()
    if roles is None:
        roles = sorted({r for d in per_config_role_f1.values() for r in d})
    configs = list(per_config_role_f1.keys())
    x = np.arange(len(roles))
    width = 0.8 / max(len(configs), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(roles) * 0.9), 5))
    for i, cfg in enumerate(configs):
        vals = [per_config_role_f1[cfg].get(role, 0.0) for role in roles]
        ax.bar(x + i * width, vals, width, label=cfg)
    ax.set_xticks(x + width * (len(configs) - 1) / 2)
    ax.set_xticklabels(roles, rotation=60)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Strict F1")
    ax.set_title("Per-Role F1 by Configuration")
    ax.legend()
    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrix(
    confusion: Dict[tuple, int],
    filename: str = "role_confusion.png",
    top_k_roles: int = 12,
) -> Path:
    """Heatmap of gold-role (rows) vs predicted-role (cols)."""
    _ensure_dir()
    role_totals: Dict[str, int] = {}
    for (g, p), c in confusion.items():
        role_totals[g] = role_totals.get(g, 0) + c
        role_totals[p] = role_totals.get(p, 0) + c
    roles = [r for r, _ in sorted(role_totals.items(), key=lambda kv: kv[1], reverse=True)][:top_k_roles]
    idx = {r: i for i, r in enumerate(roles)}

    mat = np.zeros((len(roles), len(roles)))
    for (g, p), c in confusion.items():
        if g in idx and p in idx:
            mat[idx[g], idx[p]] += c

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(roles)))
    ax.set_yticks(range(len(roles)))
    ax.set_xticklabels(roles, rotation=60)
    ax.set_yticklabels(roles)
    ax.set_xlabel("Predicted role")
    ax.set_ylabel("Gold role")
    ax.set_title("Role Confusion Matrix")
    for i in range(len(roles)):
        for j in range(len(roles)):
            if mat[i, j]:
                ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
