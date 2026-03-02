"""
NACA 4-digit airfoil parameterization.

A NACA 4-digit airfoil is defined by three parameters:
  m  – maximum camber as fraction of chord  (1st digit / 100)
  p  – position of max camber (2nd digit / 10)
  t  – maximum thickness as fraction of chord (last 2 digits / 100)

Example: NACA 2412  →  m=0.02, p=0.40, t=0.12
"""

import numpy as np


def naca4_coords(m: float, p: float, t: float, n_points: int = 200) -> tuple:
    """
    Compute upper and lower surface coordinates for a NACA 4-digit airfoil.

    Parameters
    ----------
    m : float
        Maximum camber as a fraction of chord (0 ≤ m ≤ 0.09).
    p : float
        Chord-wise position of maximum camber (0.1 ≤ p ≤ 0.9).
    t : float
        Maximum thickness as a fraction of chord (0.06 ≤ t ≤ 0.24).
    n_points : int
        Number of points on each surface (total 2*n_points - 1).

    Returns
    -------
    x_upper, y_upper, x_lower, y_lower : np.ndarray
        Coordinates of upper and lower surfaces.
    """
    # Cosine spacing for better resolution near leading/trailing edges
    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(beta))

    # Thickness distribution (NACA 4-digit formula)
    yt = (t / 0.2) * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    # Camber line and gradient
    yc = np.where(
        x < p,
        (m / p**2) * (2 * p * x - x**2),
        (m / (1 - p)**2) * (1 - 2 * p + 2 * p * x - x**2),
    )

    dyc_dx = np.where(
        x < p,
        (2 * m / p**2) * (p - x),
        (2 * m / (1 - p)**2) * (p - x),
    )

    theta = np.arctan(dyc_dx)

    x_upper = x - yt * np.sin(theta)
    y_upper = yc + yt * np.cos(theta)

    x_lower = x + yt * np.sin(theta)
    y_lower = yc - yt * np.cos(theta)

    return x_upper, y_upper, x_lower, y_lower


def airfoil_to_array(m: float, p: float, t: float, n_points: int = 200) -> np.ndarray:
    """
    Return airfoil as a closed polygon ordered counter-clockwise
    (upper surface TE→LE, lower surface LE→TE).

    Returns
    -------
    coords : np.ndarray, shape (N, 2)
        [x, y] coordinates going around the airfoil.
    """
    xu, yu, xl, yl = naca4_coords(m, p, t, n_points)
    # Upper: trailing edge → leading edge
    # Lower: leading edge → trailing edge
    x_all = np.concatenate([xu[::-1], xl[1:]])
    y_all = np.concatenate([yu[::-1], yl[1:]])
    return np.column_stack([x_all, y_all])


def save_profile(m: float, p: float, t: float,
                 filepath: str, n_points: int = 200) -> None:
    """Save airfoil coordinates to a plain-text file (x y per line)."""
    coords = airfoil_to_array(m, p, t, n_points)
    np.savetxt(filepath, coords, header="x y", comments="")
    print(f"[geometry] Saved airfoil profile → {filepath}")


def naca_label(m: float, p: float, t: float) -> str:
    """Return a human-readable NACA label (e.g. 'NACA 2412')."""
    d1 = round(m * 100)
    d2 = round(p * 10)
    d34 = round(t * 100)
    return f"NACA {d1}{d2}{d34:02d}"
