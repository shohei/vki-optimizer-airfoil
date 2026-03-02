"""
Pareto front with airfoil insets overlaid.

Reads results/pareto_solutions.csv and draws the Pareto front (CD vs CL)
with small airfoil profile insets for evenly-spaced solutions, each
connected to its corresponding point by an annotating line.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import ConnectionPatch

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geometry.naca4 import naca4_coords, naca_label


# ── Layout config ─────────────────────────────────────────────────────────────
N_INSETS   = 6        # number of airfoil insets to show
INSET_W    = 0.10     # inset width  (axes fraction of main figure)
INSET_H    = 0.055    # inset height (axes fraction of main figure)
SAVE_PATH  = "results/pareto_with_airfoils.png"
CSV_PATH   = "results/pareto_solutions.csv"


def _airfoil_color(ld_norm: float) -> tuple:
    return cm.plasma(ld_norm)


def plot_pareto_with_airfoils(csv_path: str = CSV_PATH,
                               save_path: str = SAVE_PATH,
                               n_insets: int = N_INSETS) -> None:
    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    df.sort_values("CL", inplace=True)
    df.reset_index(drop=True, inplace=True)

    CL = df["CL"].values
    CD = df["CD"].values
    LD = df["L_over_D"].values
    camber    = df["camber"].values
    thickness = df["thickness"].values
    labels    = df["NACA_label"].values

    ld_norm = (LD - LD.min()) / (LD.max() - LD.min() + 1e-9)

    # ── Pick evenly-spaced indices along Pareto front ─────────────────────────
    sel_idx = np.linspace(0, len(CL) - 1, n_insets, dtype=int)

    # ── Pre-assign inset anchor positions (figure fraction) ───────────────────
    # Place insets around the main axes so they don't overlap the curve.
    # Positions are (fig_x_left, fig_y_bottom) of each inset box.
    # The main axes occupies roughly [0.10, 0.12] → [0.92, 0.90] in fig coords.
    # We spread insets: alternating above and below the curve.
    ax_l, ax_b, ax_r, ax_t = 0.10, 0.12, 0.90, 0.90   # approx main axes bbox

    # Evenly space inset x-positions across the axes width
    xs = np.linspace(ax_l + 0.01, ax_r - INSET_W - 0.01, n_insets)
    # Alternate above / below
    ys_top    = ax_t - INSET_H - 0.01
    ys_bottom = ax_b + 0.01
    inset_positions = [
        (xs[i], ys_top if i % 2 == 0 else ys_bottom)
        for i in range(n_insets)
    ]

    # ── Main figure ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7))
    ax  = fig.add_axes([ax_l, ax_b, ax_r - ax_l, ax_t - ax_b])

    # Draw all Pareto points
    sc = ax.scatter(CD, CL, c=LD, cmap="plasma", s=55,
                    zorder=3, edgecolors="k", linewidths=0.4)
    ax.plot(CD, CL, "--", color="grey", lw=0.8, zorder=2, alpha=0.5)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("L/D = CL / CD", fontsize=10)

    ax.set_xlabel("Drag coefficient  CD", fontsize=12)
    ax.set_ylabel("Lift coefficient  CL", fontsize=12)
    ax.set_title("Pareto Front with Airfoil Profiles", fontsize=14,
                 fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Highlight selected points
    ax.scatter(CD[sel_idx], CL[sel_idx], s=120, zorder=4,
               facecolors="none", edgecolors="white", linewidths=1.5)

    # ── Add airfoil insets ─────────────────────────────────────────────────────
    for k, idx in enumerate(sel_idx):
        m = camber[idx]
        t = thickness[idx]
        color = _airfoil_color(ld_norm[idx])

        xu, yu, xl, yl = naca4_coords(m, 0.40, t, n_points=120)

        # Create inset axes in figure coordinates
        ix, iy = inset_positions[k]
        ax_in = fig.add_axes([ix, iy, INSET_W, INSET_H])

        ax_in.fill_between(xu, yu, xl, alpha=0.30, color=color)
        ax_in.plot(xu, yu, "-", color=color, lw=1.2)
        ax_in.plot(xl, yl, "-", color=color, lw=1.2)
        ax_in.set_xlim(-0.05, 1.05)
        ax_in.set_ylim(-0.18, 0.28)
        ax_in.set_aspect("equal")
        ax_in.axis("off")

        # Label below/above the inset
        label_str = f"{labels[idx]}\nCL={CL[idx]:.3f}  CD={CD[idx]:.4f}\nL/D={LD[idx]:.1f}"
        v_align = "top" if (k % 2 == 0) else "bottom"
        y_text  = -0.05 if (k % 2 == 0) else 1.05
        ax_in.text(0.5, y_text, label_str,
                   ha="center", va=v_align,
                   fontsize=6.5, transform=ax_in.transAxes,
                   color=color,
                   bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))

        # Draw connecting line from inset to Pareto point
        # Inset corner closest to the Pareto point
        inset_cx = ix + INSET_W / 2          # horizontal centre of inset
        inset_cy = iy if (k % 2 == 0) else iy + INSET_H   # bottom if top, top if bottom

        con = ConnectionPatch(
            xyA=(inset_cx, inset_cy),
            xyB=(CD[idx], CL[idx]),
            coordsA="figure fraction",
            coordsB="data",
            axesA=None,
            axesB=ax,
            color=color,
            lw=0.9,
            arrowstyle="-|>",
            mutation_scale=8,
            zorder=5,
        )
        fig.add_artist(con)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Pareto + airfoils saved → {save_path}")


if __name__ == "__main__":
    plot_pareto_with_airfoils()
