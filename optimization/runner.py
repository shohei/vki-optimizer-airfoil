"""
NSGA-II runner for multi-objective airfoil optimisation.

Uses pymoo's NSGA-II (Non-dominated Sorting Genetic Algorithm II)
with Simulated Binary Crossover (SBX) and Polynomial Mutation (PM).

Reference:
  Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
  "A fast and elitist multiobjective genetic algorithm: NSGA-II".
  IEEE Transactions on Evolutionary Computation, 6(2), 182–197.
"""

from __future__ import annotations

import os
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination

import config
from optimization.problem import AirfoilProblem
from postprocessing.visualization import plot_pareto_front


# ── Checkpoint helpers ────────────────────────────────────────────────────────

_CHECKPOINT_FILE = os.path.join(config.RESULTS_DIR, "nsga2_checkpoint.pkl")


def _save_checkpoint(data: dict) -> None:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    tmp = _CHECKPOINT_FILE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, _CHECKPOINT_FILE)   # atomic replace to avoid corruption
    print(f"[nsga2] Checkpoint saved at generation {data['n_gen']} → {_CHECKPOINT_FILE}")


def _load_checkpoint() -> dict | None:
    if not os.path.exists(_CHECKPOINT_FILE):
        return None
    try:
        with open(_CHECKPOINT_FILE, "rb") as f:
            data = pickle.load(f)
        print(f"[nsga2] Checkpoint loaded: generation {data['n_gen']} / {config.N_GENERATIONS}")
        return data
    except Exception as e:
        print(f"[nsga2] Warning: could not load checkpoint ({e}), starting fresh.")
        return None


# ── Per-generation callback ───────────────────────────────────────────────────

class _GenerationPlotCallback(Callback):
    """
    After each generation:
      - regenerate pareto_front.png and convergence_history.png
      - save a checkpoint to results/nsga2_checkpoint.pkl
    On resume, generation numbers continue from the checkpoint.
    """

    def __init__(self, problem: AirfoilProblem, checkpoint: dict | None = None):
        super().__init__()
        self._problem = problem
        # Generation offset so plots show absolute generation numbers after resume
        self._gen_offset: int = 0
        self._gens: list[int] = []
        self._best_cl: list[float] = []
        self._best_cd: list[float] = []
        if checkpoint:
            self._gen_offset = checkpoint["n_gen"]
            self._gens    = list(checkpoint["history_gens"])
            self._best_cl = list(checkpoint["history_best_cl"])
            self._best_cd = list(checkpoint["history_best_cd"])

    def notify(self, algorithm) -> None:
        actual_gen = self._gen_offset + algorithm.n_gen

        # Track per-generation best values
        F_raw  = algorithm.pop.get("F")   # shape (pop_size, 2): [-CL, CD]
        CL_pop = -F_raw[:, 0]
        CD_pop =  F_raw[:, 1]
        self._gens.append(actual_gen)
        self._best_cl.append(float(np.max(CL_pop)))
        self._best_cd.append(float(np.min(CD_pop)))

        # Update Pareto front plot from current optimal set
        opt = algorithm.opt
        if opt is not None and len(opt) > 0:
            X_opt = opt.get("X")
            F_opt = opt.get("F")
            F_phys = F_opt.copy()
            F_phys[:, 0] = -F_opt[:, 0]
            try:
                plot_pareto_front(X_opt, F_phys)
            except Exception as e:
                print(f"[viz] Pareto plot skipped at gen {actual_gen}: {e}")

        # Update convergence history plot
        try:
            self._save_convergence()
        except Exception as e:
            print(f"[viz] Convergence plot skipped at gen {actual_gen}: {e}")

        # Save checkpoint
        try:
            _save_checkpoint({
                "n_gen":           actual_gen,
                "pop_X":           algorithm.pop.get("X"),
                "pop_F":           algorithm.pop.get("F"),
                "pop_G":           algorithm.pop.get("G"),
                "eval_cache":      self._problem._eval_cache,
                "history_gens":    self._gens,
                "history_best_cl": self._best_cl,
                "history_best_cd": self._best_cd,
            })
        except Exception as e:
            print(f"[nsga2] Warning: checkpoint save failed at gen {actual_gen}: {e}")

    def _save_convergence(self) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fig.suptitle("NSGA-II Convergence History", fontsize=13, fontweight="bold")

        ax1.plot(self._gens, self._best_cl, "-o", color="#d62728", markersize=3,
                 label="Max CL in pop.")
        ax1.set_ylabel("CL")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.plot(self._gens, self._best_cd, "-o", color="#1f77b4", markersize=3,
                 label="Min CD in pop.")
        ax2.set_ylabel("CD")
        ax2.set_xlabel("Generation")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(config.HISTORY_PLOT_FILE, dpi=150)
        plt.close(fig)
        print(f"[viz] Convergence history saved → {config.HISTORY_PLOT_FILE}")


# ── Main runner ───────────────────────────────────────────────────────────────

def run_nsga2(
    use_mock: bool = True,
    verbose: bool = True,
    restart: bool = False,
    surrogate_model=None,
) -> object:
    """
    Run NSGA-II optimisation and return the pymoo Result object.

    Automatically resumes from results/nsga2_checkpoint.pkl if it exists.

    Parameters
    ----------
    use_mock : bool
        True  → physics surrogate (fast, no Fluent required)
        False → full CFD via PyFluent (slow, requires Fluent licence)
    verbose : bool
        Print progress during optimisation.
    restart : bool
        Ignore any existing checkpoint and start from scratch.
    surrogate_model : ANNSurrogate | None
        If provided, use ANN predictions instead of CFD evaluations.
        Initial population is generated via LHS for better space coverage.

    Returns
    -------
    pymoo.core.result.Result
    """
    # ── Load checkpoint (if any) ──────────────────────────────────────────────
    if restart:
        print("[nsga2] --restart: ignoring checkpoint, starting from scratch.")
    checkpoint = None if restart else _load_checkpoint()

    if checkpoint and checkpoint["n_gen"] >= config.N_GENERATIONS:
        print(f"[nsga2] Checkpoint shows all {config.N_GENERATIONS} generations already "
              "complete. Delete results/nsga2_checkpoint.pkl to restart.")
        checkpoint = None

    # ── Build problem (with cached evaluations from checkpoint) ──────────────
    eval_cache = checkpoint["eval_cache"] if checkpoint else {}
    problem = AirfoilProblem(
        use_mock=use_mock if surrogate_model is None else True,
        verbose=verbose,
        eval_cache=eval_cache,
        surrogate_model=surrogate_model,
    )

    # ── Set up sampling and termination ───────────────────────────────────────
    if checkpoint:
        n_done = checkpoint["n_gen"]
        remaining = config.N_GENERATIONS - n_done
        # Restore population; cache ensures re-evaluation is O(1) not O(Fluent)
        pop_init = Population.new(
            X=checkpoint["pop_X"],
            F=checkpoint["pop_F"],
            G=checkpoint["pop_G"],
        )
        sampling     = pop_init
        termination  = DefaultMultiObjectiveTermination(n_max_gen=remaining)
        print(f"[nsga2] Resuming: {n_done} generations done, {remaining} remaining.")
    elif surrogate_model is not None:
        # LHS initial population for better design-space coverage with ANN
        from doe.sampler import lhs_population
        X_init = lhs_population(config.POPULATION_SIZE)
        sampling    = Population.new(X=X_init)
        termination = DefaultMultiObjectiveTermination(n_max_gen=config.N_GENERATIONS)
    else:
        sampling     = FloatRandomSampling()
        termination  = DefaultMultiObjectiveTermination(n_max_gen=config.N_GENERATIONS)

    algorithm = NSGA2(
        pop_size=config.POPULATION_SIZE,
        n_offsprings=config.N_OFFSPRING,
        sampling=sampling,
        crossover=SBX(prob=config.SBX_PROB, eta=config.SBX_ETA),
        mutation=PM(eta=config.PM_ETA),
        eliminate_duplicates=True,
    )

    print("=" * 60)
    print("VKI-style Multi-Objective CFD Optimisation (NSGA-II)")
    print("=" * 60)
    print(f"  Problem   : 2-D NACA airfoil (CL↑, CD↓)")
    print(f"  Variables : α ∈ [{config.ALPHA_MIN}, {config.ALPHA_MAX}]°  "
          f"  m ∈ [{config.CAMBER_MIN}, {config.CAMBER_MAX}]  "
          f"  t ∈ [{config.THICKNESS_MIN}, {config.THICKNESS_MAX}]")
    print(f"  Algorithm : NSGA-II,  pop={config.POPULATION_SIZE},  "
          f"gen={config.N_GENERATIONS}")
    if surrogate_model is not None:
        print(f"  CFD mode  : ANN Surrogate (DoE: {config.DOE_SAMPLER.upper()} {config.DOE_N_SAMPLES} pts)")
    else:
        print(f"  CFD mode  : {'Mock (physics surrogate)' if use_mock else 'PyFluent (RANS)'}")
    print("-" * 60)

    t0 = time.time()
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=config.RANDOM_SEED,
        save_history=True,
        verbose=verbose,
        callback=_GenerationPlotCallback(problem, checkpoint),
    )
    elapsed = time.time() - t0

    print("-" * 60)
    print(f"Optimisation completed in {elapsed:.1f} s")
    print(f"  Pareto-front solutions : {len(result.X)}")
    print(f"  Total evaluations      : {problem._eval_count}")
    print("=" * 60)

    # Shut down Fluent if used
    if not use_mock:
        from cfd.fluent_runner import shutdown
        shutdown()

    return result


def extract_pareto(result) -> tuple[np.ndarray, np.ndarray]:
    """
    Return design variables and physical objectives from a pymoo Result.

    Returns
    -------
    X_pareto : np.ndarray, shape (N, 3)
        [α_deg, camber, thickness] for each Pareto-optimal solution.
    F_pareto : np.ndarray, shape (N, 2)
        [CL, CD] in physical sign convention (CL positive = lift).
    """
    X = result.X                      # shape (N, 3)
    F_raw = result.F                  # shape (N, 2): [-CL, CD]
    F_physical = F_raw.copy()
    F_physical[:, 0] = -F_raw[:, 0]  # restore CL sign
    return X, F_physical
