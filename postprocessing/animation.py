"""
Workflow animation for the DoE → ANN → NSGA-II optimisation pipeline.

Produces a 4-panel animated GIF:
  Top-left     Phase 1 · DoE LHS samples appearing in design space
  Top-right    Phase 2 · ANN training loss curves (CL and CD)
  Bottom-left  Phase 3 · NSGA-II Pareto front evolving generation by generation
  Bottom-right Phase 3 · Convergence history (max CL, min CD per generation)

When surrogate data is unavailable only Phase 3 is shown (2-panel layout).
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

import config


# ─── Dark colour palette ──────────────────────────────────────────────────────
_BG_FIG  = "#0f0f1a"
_BG_AX   = "#16213e"
_SPINE   = "#2a2a4a"
_WHITE   = "#e8e8f0"
_MUTED   = "#8888aa"
_ACCENT1 = "#ff6b6b"   # CL / warm
_ACCENT2 = "#4ecdc4"   # CD / cool
_GOLD    = "#ffd166"   # Pareto points


def _style_ax(ax) -> None:
    ax.set_facecolor(_BG_AX)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(_SPINE)


def animate_workflow(
    nsga_result,
    doe_csv: str | None = None,
    surrogate=None,
    save_path: str = "results/workflow_animation.gif",
    fps: int = 15,
) -> None:
    """
    Build and save the workflow animation.

    Parameters
    ----------
    nsga_result : pymoo Result
        Must have been created with ``save_history=True``.
    doe_csv : str | None
        Path to DoE CSV (Phase 1 panel).  Skipped when None.
    surrogate : ANNSurrogate | None
        Trained surrogate with ``loss_curve_CL_`` / ``loss_curve_CD_``
        attributes (Phase 2 panel).  Skipped when None.
    save_path : str
        Output GIF path.
    fps : int
        Frames per second.
    """
    has_doe = doe_csv is not None and os.path.exists(doe_csv)
    has_ann = surrogate is not None and hasattr(surrogate, "loss_curve_CL_")

    # ── Collect DoE data ──────────────────────────────────────────────────────
    if has_doe:
        df_doe     = pd.read_csv(doe_csv)
        doe_camber = df_doe["camber"].values
        doe_thick  = df_doe["thickness"].values
        doe_CL     = df_doe["CL"].values
        n_doe      = len(df_doe)
        cl_vmin, cl_vmax = doe_CL.min(), doe_CL.max()
    else:
        n_doe = 0

    # ── Collect ANN loss data ─────────────────────────────────────────────────
    if has_ann:
        _MAX_ANN_FRAMES = 80
        loss_cl_raw = np.array(surrogate.loss_curve_CL_)
        loss_cd_raw = np.array(surrogate.loss_curve_CD_)
        step = max(1, len(loss_cl_raw) // _MAX_ANN_FRAMES)
        loss_cl = loss_cl_raw[::step]
        loss_cd = loss_cd_raw[::step]
        n_ann   = len(loss_cl)
        r2_cl   = getattr(surrogate, "_r2_cl", None)
    else:
        n_ann = 0

    # ── Collect NSGA-II history ───────────────────────────────────────────────
    gen_pareto: list[tuple[np.ndarray, np.ndarray]] = []
    gen_best_cl: list[float] = []
    gen_best_cd: list[float] = []

    for snap in (nsga_result.history or []):
        F_raw = snap.pop.get("F")                       # (pop, 2): [-CL, CD]
        gen_best_cl.append(float(-np.min(F_raw[:, 0])))
        gen_best_cd.append(float(np.min(F_raw[:, 1])))

        opt = snap.opt
        if opt is not None and len(opt) > 0:
            F_opt = opt.get("F")
            gen_pareto.append((-F_opt[:, 0], F_opt[:, 1]))
        else:
            gen_pareto.append((np.array([]), np.array([])))

    n_gen = len(gen_pareto)

    # ── Frame budget ──────────────────────────────────────────────────────────
    # DoE: show 2 samples per frame to keep runtime short
    _DOE_PER_FRAME = 2
    n_doe_frames = (n_doe + _DOE_PER_FRAME - 1) // _DOE_PER_FRAME if has_doe else 0
    n_ann_frames = n_ann
    n_gen_frames = n_gen
    total        = n_doe_frames + n_ann_frames + n_gen_frames

    # Frame boundary indices
    _P1_END = n_doe_frames
    _P2_END = _P1_END + n_ann_frames

    # ── Figure / axes setup ───────────────────────────────────────────────────
    if has_doe or has_ann:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        ax_doe, ax_ann   = axes[0, 0], axes[0, 1]
        ax_par, ax_conv  = axes[1, 0], axes[1, 1]
        _style_ax(ax_doe); _style_ax(ax_ann)
    else:
        fig, axes_1d = plt.subplots(1, 2, figsize=(13, 5))
        ax_par, ax_conv = axes_1d[0], axes_1d[1]
        ax_doe = ax_ann = None

    fig.patch.set_facecolor(_BG_FIG)
    _style_ax(ax_par); _style_ax(ax_conv)

    phase_lbl = fig.text(
        0.5, 0.97, "", ha="center", va="top",
        fontsize=12, color=_WHITE, fontweight="bold",
    )

    # ── Panel: DoE ────────────────────────────────────────────────────────────
    if ax_doe is not None and has_doe:
        ax_doe.set_title("Phase 1 · DoE Sampling (LHS)",
                         color=_WHITE, fontsize=10, fontweight="bold")
        ax_doe.set_xlabel("Camber", color=_MUTED, fontsize=8)
        ax_doe.set_ylabel("Thickness", color=_MUTED, fontsize=8)
        ax_doe.set_xlim(config.CAMBER_MIN - 0.002, config.CAMBER_MAX + 0.002)
        ax_doe.set_ylim(config.THICKNESS_MIN - 0.005, config.THICKNESS_MAX + 0.005)
        ax_doe.grid(True, alpha=0.12, color=_WHITE)

        # Initialise scatter with full data (size=0) so colorbar can be set up
        scat_doe = ax_doe.scatter(
            doe_camber, doe_thick, c=doe_CL,
            cmap="plasma", vmin=cl_vmin, vmax=cl_vmax,
            s=np.zeros(n_doe), edgecolors="none",
        )
        cbar = fig.colorbar(scat_doe, ax=ax_doe, fraction=0.04, pad=0.02)
        cbar.set_label("CL", color=_MUTED, fontsize=8)
        cbar.ax.yaxis.set_tick_params(color=_MUTED, labelcolor=_MUTED, labelsize=7)

        doe_count = ax_doe.text(
            0.97, 0.03, "", transform=ax_doe.transAxes,
            ha="right", va="bottom", fontsize=8, color=_ACCENT2,
        )
    else:
        scat_doe = doe_count = None

    if ax_doe is not None and not has_doe:
        ax_doe.set_visible(False)

    # ── Panel: ANN training loss ──────────────────────────────────────────────
    if ax_ann is not None and has_ann:
        ax_ann.set_title("Phase 2 · ANN Training Loss",
                         color=_WHITE, fontsize=10, fontweight="bold")
        ax_ann.set_xlabel("Epoch", color=_MUTED, fontsize=8)
        ax_ann.set_ylabel("Loss (log)", color=_MUTED, fontsize=8)
        ax_ann.set_yscale("log")
        ax_ann.set_xlim(0, n_ann)
        _all_loss = np.concatenate([loss_cl, loss_cd])
        ax_ann.set_ylim(_all_loss.min() * 0.4, _all_loss.max() * 2.5)
        ax_ann.grid(True, alpha=0.12, color=_WHITE)

        line_cl, = ax_ann.plot([], [], color=_ACCENT1, lw=1.8, label="CL model")
        line_cd, = ax_ann.plot([], [], color=_ACCENT2, lw=1.8, label="CD model")
        ax_ann.legend(fontsize=8, facecolor=_BG_AX, edgecolor=_SPINE,
                      labelcolor=_WHITE, loc="upper right")
        ann_lbl = ax_ann.text(
            0.97, 0.95, "", transform=ax_ann.transAxes,
            ha="right", va="top", fontsize=8, color=_ACCENT2,
        )
    else:
        line_cl = line_cd = ann_lbl = None

    if ax_ann is not None and not has_ann:
        ax_ann.set_visible(False)

    # ── Panel: Pareto front ───────────────────────────────────────────────────
    ax_par.set_title("Phase 3 · NSGA-II Pareto Front",
                     color=_WHITE, fontsize=10, fontweight="bold")
    ax_par.set_xlabel("CD", color=_MUTED, fontsize=8)
    ax_par.set_ylabel("CL", color=_MUTED, fontsize=8)
    ax_par.grid(True, alpha=0.12, color=_WHITE)

    _all_cl_flat = np.concatenate(
        [cl for cl, _ in gen_pareto if len(cl) > 0] or [np.array([0.3, 2.0])]
    )
    _all_cd_flat = np.concatenate(
        [cd for _, cd in gen_pareto if len(cd) > 0] or [np.array([0.005, 0.06])]
    )
    _cd_pad = (_all_cd_flat.max() - _all_cd_flat.min()) * 0.15 + 1e-4
    _cl_pad = (_all_cl_flat.max() - _all_cl_flat.min()) * 0.15 + 1e-3
    ax_par.set_xlim(_all_cd_flat.min() - _cd_pad, _all_cd_flat.max() + _cd_pad)
    ax_par.set_ylim(_all_cl_flat.min() - _cl_pad, _all_cl_flat.max() + _cl_pad)

    par_line, = ax_par.plot([], [], "--", color="#555577", lw=0.8, zorder=2)
    par_scat  = ax_par.scatter([], [], c=[], cmap="plasma", s=45, zorder=3,
                                edgecolors=_WHITE, linewidths=0.3,
                                vmin=0, vmax=max(gen_best_cl) if gen_best_cl else 2)
    gen_lbl = ax_par.text(
        0.03, 0.97, "", transform=ax_par.transAxes,
        ha="left", va="top", fontsize=9, color=_GOLD, fontweight="bold",
    )

    # ── Panel: Convergence history ────────────────────────────────────────────
    ax_conv.set_title("Phase 3 · Convergence History",
                      color=_WHITE, fontsize=10, fontweight="bold")
    ax_conv.set_xlabel("Generation", color=_MUTED, fontsize=8)
    ax_conv.grid(True, alpha=0.12, color=_WHITE)
    ax_conv2 = ax_conv.twinx()
    ax_conv2.set_facecolor(_BG_AX)
    _style_ax(ax_conv2)

    conv_cl, = ax_conv.plot([], [], "-o", color=_ACCENT1, ms=2.5, lw=1.5,
                             markerfacecolor=_ACCENT1, label="Max CL")
    conv_cd, = ax_conv2.plot([], [], "-o", color=_ACCENT2, ms=2.5, lw=1.5,
                              markerfacecolor=_ACCENT2, label="Min CD")

    ax_conv.set_ylabel("Max CL",  color=_ACCENT1, fontsize=8)
    ax_conv2.set_ylabel("Min CD", color=_ACCENT2, fontsize=8)
    ax_conv.tick_params(axis="y",  colors=_ACCENT1, labelsize=7)
    ax_conv2.tick_params(axis="y", colors=_ACCENT2, labelsize=7)
    ax_conv.set_xlim(1, n_gen)

    if gen_best_cl:
        _cl_lo = min(gen_best_cl) * 0.92
        _cl_hi = max(gen_best_cl) * 1.08
        _cd_lo = min(gen_best_cd) * 0.85
        _cd_hi = max(gen_best_cd) * 1.15
        ax_conv.set_ylim(_cl_lo, _cl_hi)
        ax_conv2.set_ylim(_cd_lo, _cd_hi)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    # ── Update function ───────────────────────────────────────────────────────
    def update(frame: int):
        artists = [phase_lbl]

        if frame < _P1_END:
            # ─ Phase 1: DoE ─────────────────────────────────────────────────
            n_shown = min((frame + 1) * _DOE_PER_FRAME, n_doe)
            phase_lbl.set_text(
                f"Phase 1  ·  DoE Sampling  ({n_shown} / {n_doe} samples)"
            )
            if scat_doe is not None:
                sizes = np.zeros(n_doe)
                sizes[:n_shown] = 35
                scat_doe.set_sizes(sizes)
                scat_doe.set_edgecolors(
                    np.where(sizes > 0, _WHITE, "none").reshape(-1)
                    if False else _WHITE   # keep it simple
                )
                doe_count.set_text(f"{n_shown} / {n_doe} pts")
                artists += [scat_doe, doe_count]

        elif frame < _P2_END:
            # ─ Phase 2: ANN training ─────────────────────────────────────────
            k = frame - _P1_END + 1
            epoch_shown = k * (len(surrogate.loss_curve_CL_) // n_ann
                               if n_ann > 0 else 1)
            phase_lbl.set_text(
                f"Phase 2  ·  ANN Training  (epoch {epoch_shown})"
            )
            xs = np.arange(k)
            if line_cl is not None:
                line_cl.set_data(xs, loss_cl[:k])
                line_cd.set_data(xs, loss_cd[:k])
                ann_lbl.set_text(f"epoch {epoch_shown}")
                artists += [line_cl, line_cd, ann_lbl]

        else:
            # ─ Phase 3: NSGA-II ──────────────────────────────────────────────
            g = frame - _P2_END
            phase_lbl.set_text(
                f"Phase 3  ·  NSGA-II  (generation {g + 1} / {n_gen})"
            )
            cl_arr, cd_arr = gen_pareto[g]
            if len(cl_arr) > 0:
                order = np.argsort(cl_arr)
                par_line.set_data(cd_arr[order], cl_arr[order])
                par_scat.set_offsets(
                    np.column_stack([cd_arr[order], cl_arr[order]])
                )
                par_scat.set_array(cl_arr[order])
            gen_lbl.set_text(f"Gen {g + 1} / {n_gen}  |  {len(cl_arr)} pts")

            gens = np.arange(1, g + 2)
            conv_cl.set_data(gens, gen_best_cl[:g + 1])
            conv_cd.set_data(gens, gen_best_cd[:g + 1])
            artists += [par_line, par_scat, gen_lbl, conv_cl, conv_cd]

        return artists

    # ── Render and save ───────────────────────────────────────────────────────
    anim = FuncAnimation(
        fig, update,
        frames=total,
        interval=1000 // fps,
        blit=False,
    )

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True)
    print(f"[viz] Rendering animation ({total} frames @ {fps} fps) …")
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=110)
    plt.close(fig)
    print(f"[viz] Workflow animation saved → {save_path}")
