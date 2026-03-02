"""
Workflow animation for the DoE → ANN → NSGA-II → Infill pipeline.

4-panel dark-theme animated GIF across up to 6 phases:

  Phase 1  (top-left)   DoE LHS samples appearing in design space
  Phase 2  (top-right)  ANN training loss curves
  Phase 3  (bottom)     NSGA-II Pareto front evolving + convergence history
  Phase 4a (bottom-left) Infill: top Pareto candidates highlighted
  Phase 4b (top-left + bottom-left) Infill: ANN prediction vs CFD result
  Phase 4c (top-right)  Infill: retrained ANN loss curves
  Phase 4d (bottom)     Refined NSGA-II Pareto front + convergence

Phases 4a-4d are only rendered when nsga_result_infill and infill_data
are provided.
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
_BG_FIG   = "#0f0f1a"
_BG_AX    = "#16213e"
_SPINE    = "#2a2a4a"
_WHITE    = "#e8e8f0"
_MUTED    = "#8888aa"
_ACCENT1  = "#ff6b6b"   # CL / warm
_ACCENT2  = "#4ecdc4"   # CD / cool
_GOLD     = "#ffd166"   # Pareto points
_INFILL_C = "#ff3399"   # infill highlight colour
_RETRAIN  = "#aaffaa"   # retrained ANN colour


def _style_ax(ax) -> None:
    ax.set_facecolor(_BG_AX)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(_SPINE)


def _subsample(arr: np.ndarray, max_frames: int) -> tuple[np.ndarray, int]:
    """Return subsampled array and the step size used."""
    step = max(1, len(arr) // max_frames)
    return arr[::step], step


def animate_workflow(
    nsga_result,
    doe_csv: str | None = None,
    surrogate=None,
    infill_data: dict | None = None,
    nsga_result_infill=None,
    save_path: str = "results/workflow_animation.gif",
    fps: int = 15,
) -> None:
    """
    Build and save the workflow animation.

    Parameters
    ----------
    nsga_result : pymoo Result
        Phase 3 NSGA-II result (save_history=True).
    doe_csv : str | None
        Path to DoE CSV for Phase 1 panel.
    surrogate : ANNSurrogate | None
        Fitted surrogate; loss_curve_CL_/CD_ used for Phase 2
        (unless overridden by infill_data["loss_cl_phase2"]).
    infill_data : dict | None
        Dict returned by run_infill(); keys: X_cand, CL_ann, CD_ann,
        CL_cfd, CD_cfd, loss_cl_phase2, loss_cd_phase2,
        loss_cl_retrain, loss_cd_retrain.
    nsga_result_infill : pymoo Result | None
        Phase 4d refined NSGA-II result (save_history=True).
    save_path : str
    fps : int
    """

    # ── Flags ─────────────────────────────────────────────────────────────────
    has_doe    = doe_csv is not None and os.path.exists(doe_csv)
    has_infill = infill_data is not None and nsga_result_infill is not None

    # ── Phase 1: DoE data ─────────────────────────────────────────────────────
    if has_doe:
        df_doe     = pd.read_csv(doe_csv)
        doe_camber = df_doe["camber"].values
        doe_thick  = df_doe["thickness"].values
        doe_CL     = df_doe["CL"].values
        n_doe      = len(df_doe)
        cl_vmin, cl_vmax = doe_CL.min(), doe_CL.max()
    else:
        n_doe = 0

    # ── Phase 2: ANN loss curves ──────────────────────────────────────────────
    _MAX_ANN = 80
    # Use pre-infill loss curves if available; fall back to surrogate attribute
    if has_infill and "loss_cl_phase2" in infill_data:
        _raw_cl_p2 = np.array(infill_data["loss_cl_phase2"])
        _raw_cd_p2 = np.array(infill_data["loss_cd_phase2"])
    elif surrogate is not None and hasattr(surrogate, "loss_curve_CL_"):
        _raw_cl_p2 = np.array(surrogate.loss_curve_CL_)
        _raw_cd_p2 = np.array(surrogate.loss_curve_CD_)
    else:
        _raw_cl_p2 = np.array([])
        _raw_cd_p2 = np.array([])

    has_ann = len(_raw_cl_p2) > 0
    if has_ann:
        # Truncate to the shorter curve so CL and CD always have equal length
        # (the two MLPRegressors may converge at slightly different iterations)
        _min_p2 = min(len(_raw_cl_p2), len(_raw_cd_p2))
        p2_loss_cl, _p2_step = _subsample(_raw_cl_p2[:_min_p2], _MAX_ANN)
        p2_loss_cd           = _raw_cd_p2[:_min_p2][::_p2_step]
        n_p2 = len(p2_loss_cl)
    else:
        n_p2 = 0

    # ── Phase 3: NSGA-II history ──────────────────────────────────────────────
    gen3_pareto: list[tuple[np.ndarray, np.ndarray]] = []
    gen3_best_cl: list[float] = []
    gen3_best_cd: list[float] = []
    for snap in (nsga_result.history or []):
        F_raw = snap.pop.get("F")
        gen3_best_cl.append(float(-np.min(F_raw[:, 0])))
        gen3_best_cd.append(float(np.min(F_raw[:, 1])))
        opt = snap.opt
        if opt is not None and len(opt) > 0:
            F_opt = opt.get("F")
            gen3_pareto.append((-F_opt[:, 0], F_opt[:, 1]))
        else:
            gen3_pareto.append((np.array([]), np.array([])))
    n_gen3 = len(gen3_pareto)

    # ── Phase 4: Infill data ──────────────────────────────────────────────────
    if has_infill:
        X_cand  = infill_data["X_cand"]
        CL_ann  = infill_data["CL_ann"]
        CD_ann  = infill_data["CD_ann"]
        CL_cfd  = infill_data["CL_cfd"]
        CD_cfd  = infill_data["CD_cfd"]
        n_cand  = len(X_cand)

        _raw_cl_r4 = np.array(infill_data.get("loss_cl_retrain", []))
        _raw_cd_r4 = np.array(infill_data.get("loss_cd_retrain", []))
        has_retrain = len(_raw_cl_r4) > 0
        if has_retrain:
            _min_r4 = min(len(_raw_cl_r4), len(_raw_cd_r4))
            p4c_loss_cl, _p4c_step = _subsample(_raw_cl_r4[:_min_r4], _MAX_ANN)
            p4c_loss_cd            = _raw_cd_r4[:_min_r4][::_p4c_step]
            n_p4c = len(p4c_loss_cl)
        else:
            n_p4c = 0

        gen4_pareto: list[tuple[np.ndarray, np.ndarray]] = []
        gen4_best_cl: list[float] = []
        gen4_best_cd: list[float] = []
        for snap in (nsga_result_infill.history or []):
            F_raw = snap.pop.get("F")
            gen4_best_cl.append(float(-np.min(F_raw[:, 0])))
            gen4_best_cd.append(float(np.min(F_raw[:, 1])))
            opt = snap.opt
            if opt is not None and len(opt) > 0:
                F_opt = opt.get("F")
                gen4_pareto.append((-F_opt[:, 0], F_opt[:, 1]))
            else:
                gen4_pareto.append((np.array([]), np.array([])))
        n_gen4 = len(gen4_pareto)
    else:
        n_cand = n_p4c = n_gen4 = 0
        has_retrain = False

    # ── Frame boundaries ──────────────────────────────────────────────────────
    _DOE_PER_FRAME = 2
    n_doe_frames = (n_doe + _DOE_PER_FRAME - 1) // _DOE_PER_FRAME if has_doe else 0

    _P1_END  = n_doe_frames
    _P2_END  = _P1_END  + n_p2
    _P3_END  = _P2_END  + n_gen3
    _P4a_END = _P3_END  + n_cand          # candidates highlighted one-by-one
    _P4b_END = _P4a_END + n_cand          # CFD evaluation one-by-one
    _P4c_END = _P4b_END + n_p4c           # ANN retraining
    total    = _P4c_END + n_gen4          # refined NSGA-II

    # ── Axis bounds helpers ───────────────────────────────────────────────────
    def _pareto_bounds(gen_list):
        all_cl = np.concatenate([cl for cl, _ in gen_list if len(cl)] or [np.array([0.3, 2.0])])
        all_cd = np.concatenate([cd for _, cd in gen_list if len(cd)] or [np.array([0.005, 0.06])])
        cd_pad = (all_cd.max() - all_cd.min()) * 0.18 + 1e-4
        cl_pad = (all_cl.max() - all_cl.min()) * 0.18 + 1e-3
        return (all_cd.min() - cd_pad, all_cd.max() + cd_pad,
                all_cl.min() - cl_pad, all_cl.max() + cl_pad)

    cd_lo3, cd_hi3, cl_lo3, cl_hi3 = _pareto_bounds(gen3_pareto)
    if has_infill:
        # Combine both runs so axes stay stable across Phase 3 and 4d
        cd_lo4, cd_hi4, cl_lo4, cl_hi4 = _pareto_bounds(gen4_pareto)
        cd_lo_par = min(cd_lo3, cd_lo4)
        cd_hi_par = max(cd_hi3, cd_hi4)
        cl_lo_par = min(cl_lo3, cl_lo4)
        cl_hi_par = max(cl_hi3, cl_hi4)
    else:
        cd_lo_par, cd_hi_par, cl_lo_par, cl_hi_par = cd_lo3, cd_hi3, cl_lo3, cl_hi3

    # ── Figure / axes ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor(_BG_FIG)
    ax_doe, ax_ann = axes[0, 0], axes[0, 1]
    ax_par, ax_conv = axes[1, 0], axes[1, 1]
    for ax in (ax_doe, ax_ann, ax_par, ax_conv):
        _style_ax(ax)

    phase_lbl = fig.text(0.5, 0.97, "", ha="center", va="top",
                         fontsize=12, color=_WHITE, fontweight="bold")

    # ── Panel: DoE (top-left) ─────────────────────────────────────────────────
    ax_doe.set_title("Phase 1 · DoE Sampling (LHS)", color=_WHITE,
                     fontsize=10, fontweight="bold")
    ax_doe.set_xlabel("Camber",    color=_MUTED, fontsize=8)
    ax_doe.set_ylabel("Thickness", color=_MUTED, fontsize=8)
    ax_doe.set_xlim(config.CAMBER_MIN    - 0.002, config.CAMBER_MAX    + 0.002)
    ax_doe.set_ylim(config.THICKNESS_MIN - 0.005, config.THICKNESS_MAX + 0.005)
    ax_doe.grid(True, alpha=0.12, color=_WHITE)

    if has_doe:
        # Initialise scatter with full extent so colorbar works; sizes start at 0
        scat_doe = ax_doe.scatter(
            doe_camber, doe_thick, c=doe_CL,
            cmap="plasma", vmin=cl_vmin, vmax=cl_vmax,
            s=np.zeros(n_doe), edgecolors="none", zorder=3,
        )
        cbar_doe = fig.colorbar(scat_doe, ax=ax_doe, fraction=0.04, pad=0.02)
        cbar_doe.set_label("CL", color=_MUTED, fontsize=8)
        cbar_doe.ax.yaxis.set_tick_params(color=_MUTED, labelcolor=_MUTED, labelsize=7)
    else:
        scat_doe = None

    doe_count_lbl = ax_doe.text(0.97, 0.03, "", transform=ax_doe.transAxes,
                                ha="right", va="bottom", fontsize=8, color=_ACCENT2)

    # Infill candidates on DoE panel (Phase 4b) – stars, initially invisible
    scat_infill_doe = ax_doe.scatter(
        [], [], marker="*", s=180, c=[], cmap="autumn",
        vmin=0.5, vmax=2.0, zorder=5, edgecolors=_WHITE, linewidths=0.5,
    )

    # ── Panel: ANN training (top-right) ───────────────────────────────────────
    ax_ann.set_title("Phase 2 · ANN Training Loss", color=_WHITE,
                     fontsize=10, fontweight="bold")
    ax_ann.set_xlabel("Epoch", color=_MUTED, fontsize=8)
    ax_ann.set_ylabel("Loss (log)", color=_MUTED, fontsize=8)
    ax_ann.set_yscale("log")
    ax_ann.grid(True, alpha=0.12, color=_WHITE)

    if has_ann:
        _all_p2 = np.concatenate([p2_loss_cl, p2_loss_cd])
        ax_ann.set_xlim(0, n_p2)
        ax_ann.set_ylim(_all_p2.min() * 0.4, _all_p2.max() * 2.5)

    line_p2_cl, = ax_ann.plot([], [], color=_ACCENT1, lw=1.8, label="CL  (init)")
    line_p2_cd, = ax_ann.plot([], [], color=_ACCENT2, lw=1.8, label="CD  (init)")
    # Retrained loss curves (Phase 4c) – different style, initially empty
    line_r4_cl, = ax_ann.plot([], [], color=_RETRAIN, lw=1.8, ls="--",
                               alpha=0.0, label="CL  (retrained)")
    line_r4_cd, = ax_ann.plot([], [], color="#ffdd88", lw=1.8, ls="--",
                               alpha=0.0, label="CD  (retrained)")
    ann_epoch_lbl = ax_ann.text(0.97, 0.95, "", transform=ax_ann.transAxes,
                                ha="right", va="top", fontsize=8, color=_ACCENT2)
    leg_ann = ax_ann.legend(fontsize=7, facecolor=_BG_AX, edgecolor=_SPINE,
                             labelcolor=_WHITE, loc="upper right")

    # ── Panel: Pareto front (bottom-left) ─────────────────────────────────────
    ax_par.set_title("Phase 3 · NSGA-II Pareto Front", color=_WHITE,
                     fontsize=10, fontweight="bold")
    ax_par.set_xlabel("CD", color=_MUTED, fontsize=8)
    ax_par.set_ylabel("CL", color=_MUTED, fontsize=8)
    ax_par.set_xlim(cd_lo_par, cd_hi_par)
    ax_par.set_ylim(cl_lo_par, cl_hi_par)
    ax_par.grid(True, alpha=0.12, color=_WHITE)

    cl_vmax_par = max(gen3_best_cl + (gen4_best_cl if has_infill else [])) if gen3_best_cl else 2.0

    # Pre-compute max Pareto sizes for NaN-padding (avoids shape-mismatch in FuncAnimation)
    _max_par3 = max((len(cl) for cl, _ in gen3_pareto if len(cl) > 0), default=1)
    _max_par4 = max((len(cl) for cl, _ in gen4_pareto if len(cl) > 0), default=1) if has_infill else 1

    # Phase 3 Pareto artists – initialised with NaN so size is fixed
    par3_line, = ax_par.plot([], [], "--", color="#555577", lw=0.8, zorder=2)
    _par3_xy0  = np.full((_max_par3, 2), np.nan)
    par3_scat  = ax_par.scatter(_par3_xy0[:, 0], _par3_xy0[:, 1],
                                 c=np.zeros(_max_par3), cmap="plasma", s=45, zorder=3,
                                 edgecolors=_WHITE, linewidths=0.3,
                                 vmin=0, vmax=cl_vmax_par)
    gen3_lbl = ax_par.text(0.03, 0.97, "", transform=ax_par.transAxes,
                            ha="left", va="top", fontsize=9,
                            color=_GOLD, fontweight="bold")

    # Phase 4 infill artists (initially invisible)
    # Hollow diamond = ANN prediction, filled star = CFD result
    scat_cand_ann = ax_par.scatter([], [], marker="D", s=100, zorder=6,
                                    facecolors="none", edgecolors=_INFILL_C,
                                    linewidths=1.8)
    scat_cand_cfd = ax_par.scatter([], [], marker="*", s=160, zorder=7,
                                    c=[], cmap="autumn", vmin=0.5, vmax=cl_vmax_par,
                                    edgecolors=_WHITE, linewidths=0.4)

    # Phase 4d Pareto artists (overlay refined run)
    par4_line, = ax_par.plot([], [], "--", color="#ff9966", lw=0.8, zorder=4,
                              alpha=0.0)
    _par4_xy0  = np.full((_max_par4, 2), np.nan)
    par4_scat  = ax_par.scatter(_par4_xy0[:, 0], _par4_xy0[:, 1],
                                 c=np.zeros(_max_par4), cmap="inferno", s=45, zorder=5,
                                 edgecolors=_WHITE, linewidths=0.3, alpha=0.0,
                                 vmin=0, vmax=cl_vmax_par)
    gen4_lbl = ax_par.text(0.03, 0.85, "", transform=ax_par.transAxes,
                            ha="left", va="top", fontsize=9,
                            color=_INFILL_C, fontweight="bold")

    # ── Panel: Convergence history (bottom-right) ─────────────────────────────
    ax_conv.set_title("Phase 3 · Convergence History", color=_WHITE,
                      fontsize=10, fontweight="bold")
    ax_conv.set_xlabel("Generation", color=_MUTED, fontsize=8)
    ax_conv.grid(True, alpha=0.12, color=_WHITE)

    ax_conv2 = ax_conv.twinx()
    ax_conv2.set_facecolor(_BG_AX)
    _style_ax(ax_conv2)

    conv3_cl, = ax_conv.plot([], [], "-o", color=_ACCENT1, ms=2.5, lw=1.5)
    conv3_cd, = ax_conv2.plot([], [], "-o", color=_ACCENT2, ms=2.5, lw=1.5)
    ax_conv.set_ylabel("Max CL",  color=_ACCENT1, fontsize=8)
    ax_conv2.set_ylabel("Min CD", color=_ACCENT2, fontsize=8)
    ax_conv.tick_params(axis="y",  colors=_ACCENT1, labelsize=7)
    ax_conv2.tick_params(axis="y", colors=_ACCENT2, labelsize=7)

    _all_cl_conv = gen3_best_cl + (gen4_best_cl if has_infill else [])
    _all_cd_conv = gen3_best_cd + (gen4_best_cd if has_infill else [])
    ax_conv.set_xlim(1, n_gen3 + (n_gen4 if has_infill else 0) + 1)
    if _all_cl_conv:
        ax_conv.set_ylim(min(_all_cl_conv) * 0.92, max(_all_cl_conv) * 1.08)
        ax_conv2.set_ylim(min(_all_cd_conv) * 0.85, max(_all_cd_conv) * 1.15)

    # Phase 4d refined convergence lines (dashed overlay)
    conv4_cl, = ax_conv.plot([], [], "--s", color=_INFILL_C, ms=2.5, lw=1.2,
                              alpha=0.0, label="Infill NSGA-II")
    conv4_cd, = ax_conv2.plot([], [], "--s", color="#ffaa44", ms=2.5, lw=1.2,
                               alpha=0.0)

    # Divider line on convergence chart (shown at start of Phase 4d)
    _infill_vline = ax_conv.axvline(x=0, color=_INFILL_C, lw=1.0, ls=":",
                                     alpha=0.0)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    # ── update() ─────────────────────────────────────────────────────────────
    def update(frame: int):  # noqa: C901

        # ── Phase 1: DoE ─────────────────────────────────────────────────────
        if frame < _P1_END:
            n_shown = min((frame + 1) * _DOE_PER_FRAME, n_doe)
            phase_lbl.set_text(
                f"Phase 1  ·  DoE Sampling  ({n_shown} / {n_doe} samples)"
            )
            if scat_doe is not None:
                sizes = np.zeros(n_doe)
                sizes[:n_shown] = 35
                scat_doe.set_sizes(sizes)
                doe_count_lbl.set_text(f"{n_shown} / {n_doe} pts")

        # ── Phase 2: ANN training ─────────────────────────────────────────────
        elif frame < _P2_END:
            k = frame - _P1_END + 1
            epoch = k * _p2_step
            phase_lbl.set_text(f"Phase 2  ·  ANN Training  (epoch {epoch})")
            xs = np.arange(k)
            line_p2_cl.set_data(xs, p2_loss_cl[:k])
            line_p2_cd.set_data(xs, p2_loss_cd[:k])
            ann_epoch_lbl.set_text(f"epoch {epoch}")

        # ── Phase 3: NSGA-II ──────────────────────────────────────────────────
        elif frame < _P3_END:
            g = frame - _P2_END
            phase_lbl.set_text(
                f"Phase 3  ·  NSGA-II  (generation {g+1} / {n_gen3})"
            )
            cl_arr, cd_arr = gen3_pareto[g]
            if len(cl_arr):
                order = np.argsort(cl_arr)
                n3 = len(cl_arr)
                xy3 = np.full((_max_par3, 2), np.nan)
                xy3[:n3] = np.c_[cd_arr[order], cl_arr[order]]
                c3 = np.zeros(_max_par3)
                c3[:n3] = cl_arr[order]
                par3_line.set_data(cd_arr[order], cl_arr[order])
                par3_scat.set_offsets(xy3)
                par3_scat.set_array(c3)
            gen3_lbl.set_text(f"Gen {g+1}/{n_gen3}  |  {len(cl_arr)} pts")

            gens = np.arange(1, g + 2)
            conv3_cl.set_data(gens, gen3_best_cl[:g+1])
            conv3_cd.set_data(gens, gen3_best_cd[:g+1])

        # ── Phase 4a: Infill – highlight candidates on Pareto front ───────────
        elif has_infill and frame < _P4a_END:
            k = frame - _P3_END + 1        # 1 … n_cand
            phase_lbl.set_text(
                f"Phase 4a  ·  Infill: Selecting top {k} / {n_cand} candidates"
            )
            # Fade Phase 3 Pareto and show selected candidates (ANN positions)
            par3_scat.set_alpha(0.25)
            par3_line.set_alpha(0.2)
            scat_cand_ann.set_offsets(np.c_[CD_ann[:k], CL_ann[:k]])

        # ── Phase 4b: Infill – CFD re-evaluation ─────────────────────────────
        elif has_infill and frame < _P4b_END:
            k = frame - _P4a_END + 1        # 1 … n_cand
            phase_lbl.set_text(
                f"Phase 4b  ·  Infill: CFD re-evaluation  ({k} / {n_cand})"
            )
            # Add infill star to DoE panel
            scat_infill_doe.set_offsets(
                np.c_[X_cand[:k, 1], X_cand[:k, 2]]   # camber, thickness
            )
            scat_infill_doe.set_array(CL_cfd[:k])

            # Show CFD results (filled stars) alongside ANN diamonds
            scat_cand_cfd.set_offsets(np.c_[CD_cfd[:k], CL_cfd[:k]])
            scat_cand_cfd.set_array(CL_cfd[:k])

        # ── Phase 4c: Infill – ANN retraining ────────────────────────────────
        elif has_infill and has_retrain and frame < _P4c_END:
            k = frame - _P4b_END + 1
            epoch = k * _p4c_step
            phase_lbl.set_text(
                f"Phase 4c  ·  Infill: ANN Retraining  (epoch {epoch})"
            )
            xs = np.arange(k)
            # Expand x-axis to fit retrained epochs alongside original
            ax_ann.set_xlim(0, max(n_p2, n_p4c))
            line_r4_cl.set_data(xs, p4c_loss_cl[:k])
            line_r4_cd.set_data(xs, p4c_loss_cd[:k])
            line_r4_cl.set_alpha(0.95)
            line_r4_cd.set_alpha(0.95)
            ann_epoch_lbl.set_text(f"retrain epoch {epoch}")
            ax_ann.set_title("Phase 2 & 4c · ANN Training Loss",
                             color=_WHITE, fontsize=10, fontweight="bold")

        # ── Phase 4d: Refined NSGA-II ─────────────────────────────────────────
        elif has_infill:
            g = frame - _P4c_END
            phase_lbl.set_text(
                f"Phase 4d  ·  Infill NSGA-II  (generation {g+1} / {n_gen4})"
            )
            cl_arr4, cd_arr4 = gen4_pareto[g]
            if len(cl_arr4):
                order4 = np.argsort(cl_arr4)
                n4 = len(cl_arr4)
                xy4 = np.full((_max_par4, 2), np.nan)
                xy4[:n4] = np.c_[cd_arr4[order4], cl_arr4[order4]]
                c4 = np.zeros(_max_par4)
                c4[:n4] = cl_arr4[order4]
                par4_line.set_data(cd_arr4[order4], cl_arr4[order4])
                par4_scat.set_offsets(xy4)
                par4_scat.set_array(c4)
                par4_line.set_alpha(0.85)
                par4_scat.set_alpha(0.85)
            gen4_lbl.set_text(f"Infill Gen {g+1}/{n_gen4}  |  {len(cl_arr4)} pts")
            ax_par.set_title("Phase 4d · Refined Pareto Front", color=_WHITE,
                             fontsize=10, fontweight="bold")

            # Append refined convergence to chart; offset x by n_gen3
            gens4 = np.arange(n_gen3 + 1, n_gen3 + g + 2)
            conv4_cl.set_data(gens4, gen4_best_cl[:g+1])
            conv4_cd.set_data(gens4, gen4_best_cd[:g+1])
            conv4_cl.set_alpha(0.9)
            conv4_cd.set_alpha(0.9)
            if g == 0:
                _infill_vline.set_xdata([n_gen3 + 0.5, n_gen3 + 0.5])
                _infill_vline.set_alpha(0.6)
            ax_conv.set_title("Phase 4d · Convergence (Phase 3 + Infill)",
                              color=_WHITE, fontsize=10, fontweight="bold")

    # ── Render ────────────────────────────────────────────────────────────────
    anim = FuncAnimation(fig, update, frames=total, interval=1000 // fps, blit=False)
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True)
    print(f"[viz] Rendering animation ({total} frames @ {fps} fps) …")
    anim.save(save_path, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    print(f"[viz] Workflow animation saved → {save_path}")
