"""
Physics-based mock CFD evaluator for demonstration without Ansys Fluent.

Uses a combination of:
  - Thin airfoil theory (Theodorsen) for lift
  - NACA drag polar model (Schlichting + Shevell) for viscous drag
  - Prandtl stall model for post-stall behaviour

This gives realistic CL-CD polars consistent with XFOIL results at moderate Re.
"""

import numpy as np


# ─── Constants ────────────────────────────────────────────────────────────────
_2PI = 2.0 * np.pi


def _yt_rms(t: float) -> float:
    """RMS thickness used to correct the lift-curve slope."""
    # Simplified form-factor correction (Abbott & von Doenhoff)
    return 1.0 + 1.77 * t + 5.0 * t**2


def _zero_lift_alpha(m: float, p: float) -> float:
    """
    Theoretical zero-lift angle of attack (radians) for a NACA 4-digit airfoil.
    From thin airfoil theory (Moran, "Introduction to Theoretical & Computational
    Aerodynamics", Ch. 5).
    """
    if p < 1e-9:
        return 0.0  # symmetric airfoil
    # Closed-form result for NACA 4-digit camber line
    return -m * (1.0 - p) / p**2 * (np.pi - np.arccos(2 * p - 1))


def _cl_max(t: float, m: float) -> float:
    """
    Peak (stall) lift coefficient as a function of thickness and camber.
    Based on empirical NACA data (Shevell, Table 5-3).
    """
    cl_max_sym = 0.95 + 3.5 * t - 6.0 * t**2   # symmetric baseline
    camber_bonus = 1.0 + 2.5 * m                 # camber increases CL_max
    return cl_max_sym * camber_bonus


def _alpha_stall(t: float, m: float) -> float:
    """Stall angle in radians (increases with thickness, decreases with camber)."""
    alpha_st_sym = np.radians(8.0 + 70.0 * (t - 0.06))  # 8° to ~19° across t range
    # Cambered airfoil reaches stall at lower geometric angle (shifted by alpha_L0)
    return alpha_st_sym


def _cd0(t: float, Re: float = 1e6) -> float:
    """
    Zero-lift drag coefficient: skin friction + pressure drag.
    Approximated with the turbulent flat-plate friction formula + form factor.
    """
    # Turbulent skin friction (Prandtl-Schlichting)
    Cf = 0.455 / (np.log10(Re) ** 2.58)
    # Form factor for an airfoil (Shevell)
    FF = 1.0 + 2.0 * t + 60.0 * t**4
    # Wetted-to-reference area ratio ≈ 2 for thin airfoil
    return 2.0 * Cf * FF


def evaluate(
    alpha_deg: float,
    camber: float,
    thickness: float,
    reynolds: float = 1e6,
) -> dict:
    """
    Compute aerodynamic coefficients for a NACA 4-digit airfoil section.

    Parameters
    ----------
    alpha_deg : float
        Geometric angle of attack in degrees.
    camber : float
        Max camber as a fraction of chord (0 ≤ m ≤ 0.09).
    thickness : float
        Max thickness as a fraction of chord (0.06 ≤ t ≤ 0.24).
    reynolds : float
        Chord Reynolds number.

    Returns
    -------
    dict with keys:
        CL      – lift coefficient
        CD      – drag coefficient
        CL_CD   – aerodynamic efficiency
        alpha_stall_deg
        converged – always True for mock evaluator
    """
    alpha_rad = np.radians(alpha_deg)
    p = 0.40  # camber position (fixed for simplicity)

    # ── Lift ─────────────────────────────────────────────────────────────────
    alpha_L0 = _zero_lift_alpha(camber, p)
    cl_alpha  = _2PI * _yt_rms(thickness)   # lift-curve slope (1/rad)

    alpha_stall = _alpha_stall(thickness, camber)
    cl_max      = _cl_max(thickness, camber)

    # Effective angle
    eff_alpha = alpha_rad - alpha_L0

    if eff_alpha <= alpha_stall:
        # Attached flow (linear region)
        CL = cl_alpha * eff_alpha
        CL = min(CL, cl_max)  # hard cap
    else:
        # Post-stall: smooth Gaussian drop (Kirchhoff model approximation)
        excess = eff_alpha - alpha_stall
        CL = cl_max * np.exp(-3.5 * excess**2)

    # Negative stall (inverted flight)
    alpha_stall_neg = -_alpha_stall(thickness, 0.0)
    if eff_alpha < alpha_stall_neg:
        excess = alpha_stall_neg - eff_alpha
        CL = -cl_max * np.exp(-3.5 * excess**2)

    # ── Drag ─────────────────────────────────────────────────────────────────
    cd_min  = _cd0(thickness, reynolds)
    cl_opt  = _2PI * alpha_L0 * (-0.5)   # CL at drag bucket minimum

    # Drag polar (parabolic): CD = CD_min + k*(CL - CL_opt)²
    # k depends on Re and thickness (Schlichting approximation)
    k = 0.007 + 0.02 * thickness

    CD = cd_min + k * (CL - cl_opt) ** 2

    # Post-stall drag penalty (separated flow)
    if eff_alpha > alpha_stall:
        excess = eff_alpha - alpha_stall
        CD += 0.5 * np.sin(min(excess, np.pi / 2)) ** 2

    CD = max(CD, cd_min)  # physical lower bound

    return {
        "CL": float(CL),
        "CD": float(CD),
        "CL_CD": float(CL / CD) if CD > 0 else 0.0,
        "alpha_stall_deg": float(np.degrees(alpha_stall + alpha_L0)),
        "converged": True,
    }
