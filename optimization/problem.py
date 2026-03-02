"""
Multi-objective airfoil optimization problem for pymoo.

Design variables
────────────────
  x[0] : angle of attack α  [degrees]   ALPHA_MIN … ALPHA_MAX
  x[1] : max camber         [fraction]  CAMBER_MIN … CAMBER_MAX
  x[2] : max thickness      [fraction]  THICKNESS_MIN … THICKNESS_MAX

Objectives  (pymoo always minimises)
─────────────────────────────────────
  f[0] = –CL   →  maximise lift
  f[1] =  CD   →  minimise drag

Constraints
────────────
  g[0] = –CL + 0.3 ≤ 0   →  CL ≥ 0.3  (meaningful lift)
  g[1] =  CD – 0.05 ≤ 0  →  CD ≤ 0.05 (no extremely blunt shapes)
"""

from __future__ import annotations

import numpy as np
from pymoo.core.problem import ElementwiseProblem

import config


class AirfoilProblem(ElementwiseProblem):
    """
    Two-objective, two-constraint airfoil optimisation problem.

    Parameters
    ----------
    use_mock : bool
        If True, use the physics surrogate instead of Fluent.
    verbose : bool
        Print each evaluation result.
    """

    def __init__(self, use_mock: bool = True, verbose: bool = True,
                 eval_cache: dict | None = None, surrogate_model=None):
        self.use_mock = use_mock
        self.verbose = verbose
        self._eval_count = 0
        # Cache maps tuple(x) → (F_list, G_list); enables instant re-evaluation
        # of checkpoint population on resume without calling Fluent again.
        self._eval_cache: dict = eval_cache if eval_cache is not None else {}

        # Store function reference (not module) so deepcopy works correctly
        if surrogate_model is not None:
            self._evaluate_fn = surrogate_model.evaluate
        elif use_mock:
            from cfd.mock_evaluator import evaluate
            self._evaluate_fn = evaluate
        else:
            from cfd.fluent_runner import evaluate
            self._evaluate_fn = evaluate

        super().__init__(
            n_var=3,
            n_obj=2,
            n_ieq_constr=2,
            xl=np.array([
                config.ALPHA_MIN,
                config.CAMBER_MIN,
                config.THICKNESS_MIN,
            ]),
            xu=np.array([
                config.ALPHA_MAX,
                config.CAMBER_MAX,
                config.THICKNESS_MAX,
            ]),
        )

    # ─── pymoo interface ─────────────────────────────────────────────────────

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        # Return instantly for already-evaluated designs (used on checkpoint resume)
        cache_key = tuple(x.tolist())
        if cache_key in self._eval_cache:
            F_cached, G_cached = self._eval_cache[cache_key]
            out["F"] = list(F_cached)
            out["G"] = list(G_cached)
            return

        alpha_deg = float(x[0])
        camber    = float(x[1])
        thickness = float(x[2])

        result = self._evaluate_fn(
            alpha_deg=alpha_deg,
            camber=camber,
            thickness=thickness,
        )

        CL = result["CL"]
        CD = result["CD"]

        self._eval_count += 1
        if self.verbose and self._eval_count % 10 == 0:
            print(
                f"  eval #{self._eval_count:4d}:  α={alpha_deg:6.2f}°  "
                f"m={camber:.3f}  t={thickness:.3f}  "
                f"CL={CL:.4f}  CD={CD:.5f}  L/D={CL/CD if CD>0 else 0:.1f}"
            )

        # ── Objectives ───────────────────────────────────────────────────────
        out["F"] = [-CL, CD]   # negate CL because pymoo minimises

        # ── Inequality constraints  g ≤ 0 ────────────────────────────────────
        out["G"] = [
            -CL + 0.3,   # CL ≥ 0.3
             CD - 0.05,  # CD ≤ 0.05
        ]

        # Store in cache for potential checkpoint resume
        self._eval_cache[cache_key] = (list(out["F"]), list(out["G"]))
