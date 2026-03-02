"""
Post-processing and visualisation for the multi-objective airfoil optimisation.

Generates:
  1. Pareto front plot  (CD vs CL)
  2. Convergence history (hypervolume or IGD per generation)
  3. Airfoil shape overlay for selected Pareto solutions
  4. Design variable distributions on the Pareto front
  5. CSV export of Pareto solutions
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless rendering (no display required)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch

from geometry.naca4 import naca4_coords, naca_label
import config


# ─── Colour palette (accessible) ─────────────────────────────────────────────
_BLUE   = "#1f77b4"
_ORANGE = "#ff7f0e"
_GREEN  = "#2ca02c"
_RED    = "#d62728"


def plot_pareto_front(
    X_pareto: np.ndarray,
    F_pareto: np.ndarray,
    save_path: str = config.PARETO_PLOT_FILE,
) -> None:
    """
    Plot the Pareto front in objective space (CD vs CL).

    Points are coloured by aerodynamic efficiency (L/D = CL/CD).
    Annotates the best CL, best CD, and best L/D solutions.

    Parameters
    ----------
    X_pareto : (N, 3) array – [α, camber, thickness]
    F_pareto : (N, 2) array – [CL, CD]  (physical sign: CL > 0)
    """
    CL = F_pareto[:, 0]
    CD = F_pareto[:, 1]
    LD = CL / np.where(CD > 1e-9, CD, np.nan)

    # Sort by CL for a smooth curve
    order = np.argsort(CL)
    CL, CD, LD = CL[order], CD[order], LD[order]
    X_sorted = X_pareto[order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("VKI Multi-Objective Airfoil Optimisation — Pareto Front",
                 fontsize=14, fontweight="bold")

    # ── Left: Pareto front ────────────────────────────────────────────────────
    ax = axes[0]
    sc = ax.scatter(CD, CL, c=LD, cmap="plasma", s=60, zorder=3,
                    edgecolors="k", linewidths=0.4)
    ax.plot(CD, CL, "--", color="grey", lw=1.0, zorder=2, alpha=0.6)
    cbar = fig.colorbar(sc, ax=ax, label="L/D = CL / CD")

    # Annotate extremes
    def annotate(ax, idx, label, color):
        ax.annotate(
            label,
            xy=(CD[idx], CL[idx]),
            xytext=(CD[idx] + 0.0005, CL[idx] + 0.02),
            fontsize=8,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )

    best_cl  = np.argmax(CL)
    best_cd  = np.argmin(CD)
    best_ld  = np.nanargmax(LD)

    annotate(ax, best_cl, f"Max CL={CL[best_cl]:.3f}", _RED)
    annotate(ax, best_cd, f"Min CD={CD[best_cd]:.4f}", _BLUE)
    annotate(ax, best_ld, f"Max L/D={LD[best_ld]:.1f}", _GREEN)

    ax.set_xlabel("Drag coefficient  CD", fontsize=12)
    ax.set_ylabel("Lift coefficient  CL", fontsize=12)
    ax.set_title("Pareto Front")
    ax.grid(True, alpha=0.3)

    # ── Right: design variables on Pareto front ───────────────────────────────
    ax2 = axes[1]
    alpha_arr = X_sorted[:, 0]
    sc2 = ax2.scatter(CD, CL, c=alpha_arr, cmap="RdYlBu_r",
                      s=60, zorder=3, edgecolors="k", linewidths=0.4)
    cbar2 = fig.colorbar(sc2, ax=ax2, label="Angle of attack  α [°]")
    ax2.set_xlabel("Drag coefficient  CD", fontsize=12)
    ax2.set_ylabel("Lift coefficient  CL", fontsize=12)
    ax2.set_title("Pareto Front (coloured by α)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[viz] Pareto front saved → {save_path}")


def plot_airfoil_gallery(
    X_pareto: np.ndarray,
    F_pareto: np.ndarray,
    n_show: int = 6,
    save_path: str = "results/airfoil_gallery.png",
) -> None:
    """
    Show airfoil profiles for evenly spaced solutions along the Pareto front.

    Profiles are overlaid with colour representing L/D efficiency.
    """
    CL = F_pareto[:, 0]
    CD = F_pareto[:, 1]
    LD = CL / np.where(CD > 1e-9, CD, np.nan)

    order = np.argsort(CL)
    CL, CD, LD = CL[order], CD[order], LD[order]
    X_sorted = X_pareto[order]

    # Pick n_show evenly spaced
    indices = np.linspace(0, len(CL) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(n_show, 1, figsize=(10, n_show * 1.8))
    fig.suptitle("Airfoil Profiles Along the Pareto Front", fontsize=13,
                 fontweight="bold")

    cmap = cm.get_cmap("plasma")
    ld_norm = (LD - np.nanmin(LD)) / (np.nanmax(LD) - np.nanmin(LD) + 1e-9)

    for i, idx in enumerate(indices):
        ax = axes[i]
        m = X_sorted[idx, 1]   # camber
        t = X_sorted[idx, 2]   # thickness
        alpha = X_sorted[idx, 0]

        xu, yu, xl, yl = naca4_coords(m, 0.40, t, n_points=150)
        color = cmap(ld_norm[idx])

        ax.fill_between(xu, yu, xl, alpha=0.25, color=color)
        ax.plot(xu, yu, "-", color=color, lw=1.5)
        ax.plot(xl, yl, "-", color=color, lw=1.5)

        label = (
            f"{naca_label(m, 0.40, t)}  |  "
            f"α={alpha:.1f}°  CL={CL[idx]:.3f}  CD={CD[idx]:.4f}  "
            f"L/D={LD[idx]:.1f}"
        )
        ax.set_title(label, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.15, 0.30)
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[viz] Airfoil gallery saved → {save_path}")


def plot_convergence(
    result,
    save_path: str = config.HISTORY_PLOT_FILE,
) -> None:
    """
    Plot the evolution of minimum CD and maximum CL over generations.
    """
    gens = []
    best_cl = []
    best_cd = []

    for gen, snapshot in enumerate(result.history):
        F_raw = snapshot.pop.get("F")        # shape (pop_size, 2)
        CL_gen = -F_raw[:, 0]
        CD_gen =  F_raw[:, 1]
        gens.append(gen + 1)
        best_cl.append(np.max(CL_gen))
        best_cd.append(np.min(CD_gen))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle("NSGA-II Convergence History", fontsize=13, fontweight="bold")

    ax1.plot(gens, best_cl, "-o", color=_RED, markersize=3, label="Max CL in pop.")
    ax1.set_ylabel("CL")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(gens, best_cd, "-o", color=_BLUE, markersize=3, label="Min CD in pop.")
    ax2.set_ylabel("CD")
    ax2.set_xlabel("Generation")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[viz] Convergence history saved → {save_path}")


def plot_pareto_with_airfoils(
    csv_path: str = "results/pareto_solutions.csv",
    save_path: str = "results/pareto_with_airfoils.png",
    n_insets: int = 6,
) -> None:
    """
    Pareto front with airfoil profile insets overlaid.

    Reads a pareto_solutions.csv and draws the Pareto front (CD vs CL)
    with small airfoil insets for evenly-spaced solutions, each connected
    to its corresponding point by an annotating arrow.

    Parameters
    ----------
    csv_path : str
        Path to the CSV exported by export_csv().
    save_path : str
        Output PNG path.
    n_insets : int
        Number of airfoil insets to show.
    """
    _INSET_W = 0.10
    _INSET_H = 0.055

    df = pd.read_csv(csv_path)
    df.sort_values("CL", inplace=True)
    df.reset_index(drop=True, inplace=True)

    CL        = df["CL"].values
    CD        = df["CD"].values
    LD        = df["L_over_D"].values
    camber    = df["camber"].values
    thickness = df["thickness"].values
    labels    = df["NACA_label"].values

    ld_norm = (LD - LD.min()) / (LD.max() - LD.min() + 1e-9)
    sel_idx = np.linspace(0, len(CL) - 1, n_insets, dtype=int)

    # Inset anchor positions (figure fraction), alternating above/below curve
    ax_l, ax_b, ax_r, ax_t = 0.10, 0.12, 0.90, 0.90
    xs = np.linspace(ax_l + 0.01, ax_r - _INSET_W - 0.01, n_insets)
    inset_positions = [
        (xs[i], ax_t - _INSET_H - 0.01 if i % 2 == 0 else ax_b + 0.01)
        for i in range(n_insets)
    ]

    fig = plt.figure(figsize=(14, 7))
    ax  = fig.add_axes([ax_l, ax_b, ax_r - ax_l, ax_t - ax_b])

    sc = ax.scatter(CD, CL, c=LD, cmap="plasma", s=55,
                    zorder=3, edgecolors="k", linewidths=0.4)
    ax.plot(CD, CL, "--", color="grey", lw=0.8, zorder=2, alpha=0.5)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("L/D = CL / CD", fontsize=10)

    ax.set_xlabel("Drag coefficient  CD", fontsize=12)
    ax.set_ylabel("Lift coefficient  CL", fontsize=12)
    ax.set_title("Pareto Front with Airfoil Profiles", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.scatter(CD[sel_idx], CL[sel_idx], s=120, zorder=4,
               facecolors="none", edgecolors="white", linewidths=1.5)

    for k, idx in enumerate(sel_idx):
        color = cm.plasma(ld_norm[idx])
        xu, yu, xl, yl = naca4_coords(camber[idx], 0.40, thickness[idx], n_points=120)

        ix, iy = inset_positions[k]
        ax_in = fig.add_axes([ix, iy, _INSET_W, _INSET_H])
        ax_in.fill_between(xu, yu, xl, alpha=0.30, color=color)
        ax_in.plot(xu, yu, "-", color=color, lw=1.2)
        ax_in.plot(xl, yl, "-", color=color, lw=1.2)
        ax_in.set_xlim(-0.05, 1.05)
        ax_in.set_ylim(-0.18, 0.28)
        ax_in.set_aspect("equal")
        ax_in.axis("off")

        label_str = f"{labels[idx]}\nCL={CL[idx]:.3f}  CD={CD[idx]:.4f}\nL/D={LD[idx]:.1f}"
        y_text  = -0.05 if (k % 2 == 0) else 1.05
        v_align = "top"  if (k % 2 == 0) else "bottom"
        ax_in.text(0.5, y_text, label_str,
                   ha="center", va=v_align, fontsize=6.5,
                   transform=ax_in.transAxes, color=color,
                   bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))

        inset_cx = ix + _INSET_W / 2
        inset_cy = iy if (k % 2 == 0) else iy + _INSET_H
        fig.add_artist(ConnectionPatch(
            xyA=(inset_cx, inset_cy), xyB=(CD[idx], CL[idx]),
            coordsA="figure fraction", coordsB="data",
            axesA=None, axesB=ax,
            color=color, lw=0.9, arrowstyle="-|>", mutation_scale=8, zorder=5,
        ))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Pareto + airfoils saved → {save_path}")


def plot_surrogate_accuracy(
    surrogate,
    X_test: np.ndarray,
    CL_true: np.ndarray,
    CD_true: np.ndarray,
    save_path: str = "results/surrogate_accuracy.png",
) -> None:
    """
    2×2 panel plot for ANN surrogate accuracy assessment.

    - Top-left:  CL parity plot (true vs predicted) with R²
    - Top-right: CD parity plot (true vs predicted) with R²
    - Bottom-left: CL residuals vs angle of attack α
    - Bottom-right: DoE sample distribution (camber vs thickness, coloured by CL)

    Parameters
    ----------
    surrogate : ANNSurrogate
        A fitted surrogate model.
    X_test : (N, 3) array – [alpha_deg, camber, thickness]
    CL_true : (N,) array
    CD_true : (N,) array
    save_path : str
    """
    from sklearn.metrics import r2_score

    CL_pred, CD_pred = surrogate.predict(X_test)
    r2_cl = r2_score(CL_true, CL_pred)
    r2_cd = r2_score(CD_true, CD_pred)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("ANN Surrogate Accuracy", fontsize=14, fontweight="bold")

    # ── Top-left: CL parity ───────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.scatter(CL_true, CL_pred, s=25, alpha=0.7, color=_BLUE, edgecolors="k",
               linewidths=0.3)
    lims = [min(CL_true.min(), CL_pred.min()), max(CL_true.max(), CL_pred.max())]
    ax.plot(lims, lims, "k--", lw=1.0, label="Perfect fit")
    ax.set_xlabel("CL  (CFD)")
    ax.set_ylabel("CL  (ANN)")
    ax.set_title(f"CL Parity  (R² = {r2_cl:.4f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Top-right: CD parity ──────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.scatter(CD_true, CD_pred, s=25, alpha=0.7, color=_ORANGE, edgecolors="k",
               linewidths=0.3)
    lims = [min(CD_true.min(), CD_pred.min()), max(CD_true.max(), CD_pred.max())]
    ax.plot(lims, lims, "k--", lw=1.0, label="Perfect fit")
    ax.set_xlabel("CD  (CFD)")
    ax.set_ylabel("CD  (ANN)")
    ax.set_title(f"CD Parity  (R² = {r2_cd:.4f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Bottom-left: CL residuals vs α ────────────────────────────────────────
    ax = axes[1, 0]
    residuals_cl = CL_pred - CL_true
    ax.scatter(X_test[:, 0], residuals_cl, s=25, alpha=0.7, color=_RED,
               edgecolors="k", linewidths=0.3)
    ax.axhline(0, color="k", lw=1.0, linestyle="--")
    ax.set_xlabel("Angle of attack  α [°]")
    ax.set_ylabel("CL residual  (pred − true)")
    ax.set_title("CL Residuals vs α")
    ax.grid(True, alpha=0.3)

    # ── Bottom-right: DoE sample distribution ─────────────────────────────────
    ax = axes[1, 1]
    sc = ax.scatter(X_test[:, 1], X_test[:, 2], c=CL_true, cmap="plasma",
                    s=40, edgecolors="k", linewidths=0.3)
    fig.colorbar(sc, ax=ax, label="CL (CFD)")
    ax.set_xlabel("Camber (fraction of chord)")
    ax.set_ylabel("Thickness (fraction of chord)")
    ax.set_title("DoE Sample Distribution  (colour = CL)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[viz] Surrogate accuracy plot saved → {save_path}")


def export_csv(
    X_pareto: np.ndarray,
    F_pareto: np.ndarray,
    save_path: str = config.RESULTS_CSV,
) -> None:
    """Export Pareto solutions to CSV for further analysis."""
    CL = F_pareto[:, 0]
    CD = F_pareto[:, 1]
    LD = CL / np.where(CD > 1e-9, CD, np.nan)

    df = pd.DataFrame({
        "alpha_deg":  X_pareto[:, 0],
        "camber":     X_pareto[:, 1],
        "thickness":  X_pareto[:, 2],
        "CL":         CL,
        "CD":         CD,
        "L_over_D":   LD,
        "NACA_label": [
            naca_label(m, 0.40, t)
            for m, t in zip(X_pareto[:, 1], X_pareto[:, 2])
        ],
    })
    df.sort_values("L_over_D", ascending=False, inplace=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False, float_format="%.6f")
    print(f"[viz] Pareto solutions exported → {save_path}")
    print(df.head(10).to_string(index=False))
