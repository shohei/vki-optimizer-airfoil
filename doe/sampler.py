"""
Design of Experiments (DoE) sampler for airfoil optimization.

Generates space-filling samples using Latin Hypercube Sampling (LHS)
or Sobol sequences, then evaluates each sample with CFD (mock or Fluent).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube, Sobol

import config


def run_doe(
    use_mock: bool = True,
    n_samples: int = 100,
    sampler: str = "lhs",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate DoE samples, evaluate with CFD, and save to CSV.

    Parameters
    ----------
    use_mock : bool
        True → physics surrogate, False → full Fluent CFD.
    n_samples : int
        Number of design points to sample.
    sampler : str
        "lhs" for Latin Hypercube Sampling, "sobol" for Sobol sequence.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 3)
        Design variables [alpha_deg, camber, thickness].
    CL : np.ndarray, shape (n_samples,)
        Lift coefficients from CFD.
    CD : np.ndarray, shape (n_samples,)
        Drag coefficients from CFD.
    """
    print(f"[doe] Generating {n_samples} {sampler.upper()} samples …")

    # ── Sample in [0, 1]^3 ───────────────────────────────────────────────────
    if sampler == "sobol":
        engine = Sobol(d=3, scramble=True, seed=seed)
        unit_samples = engine.random(n_samples)
    else:
        engine = LatinHypercube(d=3, seed=seed)
        unit_samples = engine.random(n_samples)

    # ── Scale to design bounds ────────────────────────────────────────────────
    bounds_low = np.array([config.ALPHA_MIN, config.CAMBER_MIN, config.THICKNESS_MIN])
    bounds_high = np.array([config.ALPHA_MAX, config.CAMBER_MAX, config.THICKNESS_MAX])
    X = bounds_low + unit_samples * (bounds_high - bounds_low)

    # ── Evaluate each sample with CFD ─────────────────────────────────────────
    if use_mock:
        from cfd.mock_evaluator import evaluate
    else:
        from cfd.fluent_runner import evaluate

    CL = np.zeros(n_samples)
    CD = np.zeros(n_samples)

    for i, x in enumerate(X):
        result = evaluate(
            alpha_deg=float(x[0]),
            camber=float(x[1]),
            thickness=float(x[2]),
        )
        CL[i] = result["CL"]
        CD[i] = result["CD"]
        if (i + 1) % 10 == 0 or (i + 1) == n_samples:
            print(f"[doe]   {i + 1}/{n_samples} evaluated …")

    # ── Save to CSV ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(config.DOE_CSV), exist_ok=True)
    df = pd.DataFrame({
        "alpha_deg": X[:, 0],
        "camber":    X[:, 1],
        "thickness": X[:, 2],
        "CL":        CL,
        "CD":        CD,
    })
    df.to_csv(config.DOE_CSV, index=False, float_format="%.6f")
    print(f"[doe] DoE samples saved → {config.DOE_CSV}")

    return X, CL, CD


def lhs_population(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Generate an LHS-sampled initial population for NSGA-II.

    Returns an (n_samples, 3) array of design variables scaled to the
    design bounds, suitable for use as a pymoo Population initial sampling.

    Parameters
    ----------
    n_samples : int
        Number of individuals in the initial population.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (n_samples, 3)
        [alpha_deg, camber, thickness] for each individual.
    """
    engine = LatinHypercube(d=3, seed=seed)
    unit_samples = engine.random(n_samples)

    bounds_low = np.array([config.ALPHA_MIN, config.CAMBER_MIN, config.THICKNESS_MIN])
    bounds_high = np.array([config.ALPHA_MAX, config.CAMBER_MAX, config.THICKNESS_MAX])
    return bounds_low + unit_samples * (bounds_high - bounds_low)
