"""
VKI-style Multi-Objective CFD Optimisation
==========================================

Problem
-------
NACA 4-digit airfoil design to simultaneously:
  (1) maximise lift coefficient CL
  (2) minimise drag coefficient CD

This gives a trade-off (Pareto) front between high-lift and low-drag designs.

Algorithm
---------
NSGA-II (Non-dominated Sorting Genetic Algorithm II)
Implemented via the pymoo library.

CFD Solver
----------
  - Mock mode (USE_MOCK_CFD = True in config.py):
      Physics-based surrogate using thin airfoil theory + drag polar model.
      No external software required.  Runs in seconds.

  - Fluent mode (USE_MOCK_CFD = False):
      Full RANS simulation via Ansys Fluent + PyFluent.
      Requires Ansys licence and gmsh for mesh generation.
      Each evaluation takes ~5-15 minutes depending on hardware.

  - Surrogate mode (--surrogate):
      Phase 1: DoE sampling (LHS/Sobol) → CFD evaluation
      Phase 2: ANN training (MLPRegressor approximates CL/CD)
      Phase 3: NSGA-II using ANN predictions (very fast)
      Phase 4: Infill — Pareto solutions re-evaluated with real CFD (optional)

Usage
-----
  python main.py               # uses settings in config.py
  python main.py --mock        # force mock CFD
  python main.py --fluent      # force real Fluent CFD
  python main.py --quick       # reduced generations for a fast demo
  python main.py --surrogate --mock          # DoE+ANN surrogate mode
  python main.py --surrogate --mock --infill # include infill phase
"""

import argparse
import os

import numpy as np

import config
from optimization.runner import run_nsga2, extract_pareto
from postprocessing.visualization import (
    export_csv,
    plot_airfoil_gallery,
    plot_convergence,
    plot_pareto_front,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VKI Multi-Objective CFD Optimisation (NSGA-II + NACA airfoil)"
    )
    cfd_group = parser.add_mutually_exclusive_group()
    cfd_group.add_argument("--mock",   action="store_true",
                           help="Use physics-based surrogate (no Fluent)")
    cfd_group.add_argument("--fluent", action="store_true",
                           help="Use real Ansys Fluent via PyFluent")
    parser.add_argument("--surrogate", action="store_true",
                        help="DoE + ANN surrogate mode (Phase 1-3, optionally Phase 4)")
    parser.add_argument("--doe-samples", type=int, default=None,
                        help="Number of DoE samples (overrides config.DOE_N_SAMPLES)")
    parser.add_argument("--infill", action="store_true",
                        help="Run infill phase: re-evaluate Pareto solutions with real CFD")
    parser.add_argument("--quick", action="store_true",
                        help="Run with reduced populations/generations for demo")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots (useful for CI/CD)")
    parser.add_argument("--restart", action="store_true",
                        help="Ignore checkpoint and restart optimisation from scratch")
    return parser.parse_args()


def print_top_solutions(X: np.ndarray, F: np.ndarray, n: int = 5) -> None:
    """Print the top Pareto solutions sorted by L/D."""
    CL = F[:, 0]
    CD = F[:, 1]
    LD = CL / np.where(CD > 1e-9, CD, np.nan)
    order = np.nanargmax(LD) if n == 1 else np.argsort(-LD)[:n]
    if isinstance(order, (int, np.integer)):
        order = [order]

    print("\nTop solutions by aerodynamic efficiency (L/D):")
    print(f"  {'#':>3}  {'α [°]':>7}  {'camber':>8}  {'thick.':>8}  "
          f"{'CL':>8}  {'CD':>8}  {'L/D':>8}")
    print("  " + "-" * 65)
    for rank, idx in enumerate(order, 1):
        print(
            f"  {rank:3d}  {X[idx, 0]:7.2f}  {X[idx, 1]:8.4f}  "
            f"{X[idx, 2]:8.4f}  {CL[idx]:8.4f}  {CD[idx]:8.5f}  "
            f"{LD[idx]:8.2f}"
        )


def run_infill(result, surrogate, use_mock: bool, verbose: bool):
    """
    Infill phase: pick the top INFILL_N_CANDIDATES Pareto solutions,
    re-evaluate them with real CFD, append the new data to the surrogate's
    training set, optionally retrain, and re-run NSGA-II.

    Parameters
    ----------
    result : pymoo Result
        Pareto front from the ANN-based NSGA-II run.
    surrogate : ANNSurrogate
        The trained surrogate model (will be updated in place if retrain=True).
    use_mock : bool
        True → re-evaluate with mock CFD; False → real Fluent.
    verbose : bool
        Print progress.

    Returns
    -------
    pymoo Result (from the refined NSGA-II run)
    """
    X_pareto, F_pareto = extract_pareto(result)
    n_cand = min(config.INFILL_N_CANDIDATES, len(X_pareto))

    # Select top candidates by L/D
    CL_p = F_pareto[:, 0]
    CD_p = F_pareto[:, 1]
    LD_p = CL_p / np.where(CD_p > 1e-9, CD_p, np.nan)
    idx_top = np.argsort(-np.nan_to_num(LD_p))[:n_cand]
    X_cand = X_pareto[idx_top]

    print(f"\n[infill] Re-evaluating {n_cand} Pareto candidates with real CFD …")

    if use_mock:
        from cfd.mock_evaluator import evaluate
    else:
        from cfd.fluent_runner import evaluate

    CL_new = np.zeros(n_cand)
    CD_new = np.zeros(n_cand)
    for i, x in enumerate(X_cand):
        res = evaluate(alpha_deg=float(x[0]), camber=float(x[1]), thickness=float(x[2]))
        CL_new[i] = res["CL"]
        CD_new[i] = res["CD"]
        if verbose:
            print(f"[infill]   {i+1}/{n_cand}  α={x[0]:.2f}°  "
                  f"CL={CL_new[i]:.4f}  CD={CD_new[i]:.5f}")

    if config.INFILL_RETRAIN:
        # Retrieve the original DoE data stored in the CSV and append infill points
        import pandas as pd
        if os.path.exists(config.DOE_CSV):
            df_doe = pd.read_csv(config.DOE_CSV)
            X_old = df_doe[["alpha_deg", "camber", "thickness"]].values
            CL_old = df_doe["CL"].values
            CD_old = df_doe["CD"].values
        else:
            X_old = np.empty((0, 3))
            CL_old = np.empty(0)
            CD_old = np.empty(0)

        X_aug  = np.vstack([X_old, X_cand])
        CL_aug = np.concatenate([CL_old, CL_new])
        CD_aug = np.concatenate([CD_old, CD_new])

        print(f"[infill] Retraining surrogate on {len(X_aug)} samples …")
        surrogate.fit(X_aug, CL_aug, CD_aug)
        surrogate.save(config.ANN_SAVE_PATH)

    print("[infill] Re-running NSGA-II with refined surrogate …")
    result_refined = run_nsga2(
        use_mock=use_mock,
        surrogate_model=surrogate,
        verbose=verbose,
        restart=True,
    )
    return result_refined


def main() -> None:
    args = parse_args()

    # ── Determine CFD mode ────────────────────────────────────────────────────
    if args.fluent:
        use_mock = False
    elif args.mock or args.surrogate:
        use_mock = True
    else:
        use_mock = config.USE_MOCK_CFD

    # ── Quick demo mode ───────────────────────────────────────────────────────
    if args.quick:
        config.POPULATION_SIZE = 20
        config.N_GENERATIONS   = 15
        config.N_OFFSPRING     = 10
        if not args.surrogate:
            config.DOE_N_SAMPLES = 30
        print("[main] Quick mode: pop=20, gen=15")

    # ── Create output directory ───────────────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # ── Surrogate mode (DoE → ANN → NSGA-II [→ Infill]) ──────────────────────
    if args.surrogate:
        from doe.sampler import run_doe
        from surrogate.ann_model import ANNSurrogate
        from postprocessing.visualization import plot_surrogate_accuracy

        n_doe = args.doe_samples if args.doe_samples is not None else config.DOE_N_SAMPLES

        # Phase 1: DoE
        X_doe, CL_doe, CD_doe = run_doe(
            use_mock=use_mock,
            n_samples=n_doe,
            sampler=config.DOE_SAMPLER,
        )

        # Phase 2: ANN training
        surrogate = ANNSurrogate(
            hidden_layers=config.ANN_HIDDEN_LAYERS,
            activation=config.ANN_ACTIVATION,
            max_iter=config.ANN_MAX_ITER,
            random_state=config.ANN_RANDOM_STATE,
        )
        metrics = surrogate.fit(X_doe, CL_doe, CD_doe)
        print(f"  ANN R²: CL={metrics['r2_CL']:.4f}, CD={metrics['r2_CD']:.4f}")

        if not args.no_plots:
            plot_surrogate_accuracy(surrogate, X_doe, CL_doe, CD_doe)

        surrogate.save(config.ANN_SAVE_PATH)

        # Phase 3: NSGA-II with ANN evaluations
        result = run_nsga2(
            use_mock=use_mock,
            surrogate_model=surrogate,
            verbose=True,
            restart=args.restart,
        )

        # Phase 4: Infill (optional)
        if args.infill:
            result = run_infill(result, surrogate, use_mock=use_mock, verbose=True)

    else:
        # ── Standard mode (direct CFD) ────────────────────────────────────────
        result = run_nsga2(use_mock=use_mock, verbose=True, restart=args.restart)

    # ── Extract Pareto front ──────────────────────────────────────────────────
    X_pareto, F_pareto = extract_pareto(result)

    print_top_solutions(X_pareto, F_pareto, n=5)

    # ── Post-processing ───────────────────────────────────────────────────────
    if not args.no_plots:
        print("\n[main] Generating plots …")
        plot_pareto_front(X_pareto, F_pareto)
        plot_convergence(result)
        plot_airfoil_gallery(X_pareto, F_pareto, n_show=6)
        export_csv(X_pareto, F_pareto)
    else:
        export_csv(X_pareto, F_pareto)

    print(f"\nAll results written to: {os.path.abspath(config.RESULTS_DIR)}/")


if __name__ == "__main__":
    main()
