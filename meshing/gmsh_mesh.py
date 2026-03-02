"""
Automated C-type mesh generation for a NACA airfoil using gmsh.

Domain topology (2D, C-type):
  - Upstream boundary : semicircle (two quarter-circle arcs) → "inlet"
  - Top/bottom walls  : horizontal lines connecting semicircle to outlet → "farfield_wall"
  - Outlet            : vertical right boundary → "outlet"
  - Airfoil surface   : no-slip wall → "airfoil"

Coordinate conventions
──────────────────────
  LE = (0, 0),  TE = (chord, 0)  (airfoil at origin)
  Semicircle center at LE, radius R = domain_radius * chord

       p1(0,R)  --- l_top --------------  p4(L,R)
      /                                        |
  arc1(p1->p2)                           l_outlet
      |                                        |
      p2(-R,0)    [airfoil]              p5(L,-R)
      |                                        |
  arc2(p2->p3)                           l_bot
        |                                       |
       p3(0,-R) --- l_bot --------------  p5(L,-R)
"""

from __future__ import annotations

import os

import numpy as np


def generate_airfoil_mesh(
    m: float,
    p: float,
    t: float,
    output_path: str,
    chord: float = 1.0,
    domain_radius: float = 15.0,
    wake_length: float = 20.0,
    n_airfoil: int = 200,
    n_boundary_layer: int = 20,
    first_layer_height: float = 1e-5,
) -> str:
    """
    Generate a 2-D unstructured mesh with boundary layer around a NACA 4-digit airfoil.

    Parameters
    ----------
    m, p, t : float
        NACA 4-digit camber, camber-position, thickness fractions.
    output_path : str
        Path for the output .msh file (gmsh format 2 or 4, auto-detected by extension).
    chord : float
        Chord length [m].
    domain_radius : float
        Far-field radius in chord lengths.
    wake_length : float
        Downstream wake extent in chord lengths (from TE).
    n_airfoil : int
        Approximate number of points on the airfoil surface.
    n_boundary_layer : int
        Number of layers in the prismatic boundary layer.
    first_layer_height : float
        Height of first boundary-layer cell [m].

    Returns
    -------
    str
        Absolute path to the saved mesh file.
    """
    try:
        import gmsh
    except ImportError:
        raise ImportError(
            "gmsh is required for mesh generation.\n"
            "Install it with:  pip install gmsh"
        )

    from geometry.naca4 import naca4_coords

    gmsh.initialize()
    gmsh.model.add("airfoil_mesh")
    gmsh.option.setNumber("General.Terminal", 0)    # suppress console output
    gmsh.option.setNumber("Mesh.Algorithm", 6)      # Frontal-Delaunay (good for BL)

    R   = domain_radius * chord
    L   = wake_length   * chord
    lc_ff = R / 10.0                           # far-field mesh size
    lc_af = chord / max(n_airfoil, 50)         # airfoil mesh size

    # ── Airfoil coordinates ───────────────────────────────────────────────────
    # naca4_coords returns x ∈ [0,1] → scale by chord
    # xu, yu : upper surface, LE (index 0) → TE (index -1)
    # xl, yl : lower surface, LE (index 0) → TE (index -1)
    n_pts = max(n_airfoil // 2 + 1, 51)
    xu, yu, xl, yl = naca4_coords(m, p, t, n_points=n_pts)
    xu *= chord;  yu *= chord
    xl *= chord;  yl *= chord

    # Close the trailing edge (NACA formula leaves a tiny open gap at TE)
    te_x = float(0.5 * (xu[-1] + xl[-1]))   # ≈ chord
    te_y = float(0.5 * (yu[-1] + yl[-1]))   # ≈ 0

    # ── Far-field points ──────────────────────────────────────────────────────
    # Semicircle is centred at the origin (LE).
    p0 = gmsh.model.geo.addPoint(0.0,   0.0,  0.0, lc_ff)  # centre of arcs
    p1 = gmsh.model.geo.addPoint(0.0,   R,    0.0, lc_ff)  # (0, +R)  – top
    p2 = gmsh.model.geo.addPoint(-R,    0.0,  0.0, lc_ff)  # (-R, 0)  – left
    p3 = gmsh.model.geo.addPoint(0.0,  -R,    0.0, lc_ff)  # (0, -R)  – bottom
    p4 = gmsh.model.geo.addPoint(te_x + L,  R,  0.0, lc_ff)  # outlet top
    p5 = gmsh.model.geo.addPoint(te_x + L, -R,  0.0, lc_ff)  # outlet bottom

    # ── Far-field curves ──────────────────────────────────────────────────────
    # gmsh.model.geo.addCircleArc(startTag, centerTag, endTag)
    # Each arc must subtend less than π.  Split into two quarter-circles.
    arc1 = gmsh.model.geo.addCircleArc(p1, p0, p2)   # top   → left  (upper arc)
    arc2 = gmsh.model.geo.addCircleArc(p2, p0, p3)   # left  → bot   (lower arc)

    l_top    = gmsh.model.geo.addLine(p1, p4)   # (0,+R)   → outlet top
    l_outlet = gmsh.model.geo.addLine(p4, p5)   # outlet top → outlet bot
    l_bot    = gmsh.model.geo.addLine(p5, p3)   # outlet bot → (0,-R)

    # ── Airfoil surface points ────────────────────────────────────────────────
    # Upper surface: LE → TE  (exclude last point; TE added separately)
    upper_tags: list[int] = []
    for xi, yi in zip(xu[:-1], yu[:-1]):
        upper_tags.append(
            gmsh.model.geo.addPoint(float(xi), float(yi), 0.0, lc_af)
        )

    # Single shared TE point (closes the gap)
    p_te = gmsh.model.geo.addPoint(te_x, te_y, 0.0, lc_af)
    upper_tags.append(p_te)

    # Lower surface: LE → TE  (LE shared with upper; TE shared as p_te)
    lower_tags: list[int] = [upper_tags[0]]    # share LE
    for xi, yi in zip(xl[1:-1], yl[1:-1]):
        lower_tags.append(
            gmsh.model.geo.addPoint(float(xi), float(yi), 0.0, lc_af)
        )
    lower_tags.append(p_te)                   # share TE

    # ── Airfoil splines ───────────────────────────────────────────────────────
    spl_upper = gmsh.model.geo.addSpline(upper_tags)   # LE → TE (upper)
    spl_lower = gmsh.model.geo.addSpline(lower_tags)   # LE → TE (lower)

    # ── Curve loops ───────────────────────────────────────────────────────────
    # Outer boundary (counter-clockwise when viewed from +z):
    #   p1 → p4  (l_top)
    #   p4 → p5  (l_outlet)
    #   p5 → p3  (l_bot)
    #   p3 → p2  (-arc2, because arc2 goes p2 → p3)
    #   p2 → p1  (-arc1, because arc1 goes p1 → p2)
    cl_outer   = gmsh.model.geo.addCurveLoop(
        [l_top, l_outlet, l_bot, -arc2, -arc1]
    )

    # Airfoil hole (forms a closed loop LE → TE → LE):
    #   spl_upper : LE → TE
    #  -spl_lower : TE → LE  (reverse of LE → TE lower)
    cl_airfoil = gmsh.model.geo.addCurveLoop(
        [spl_upper, -spl_lower]
    )

    # Plane surface: outer loop minus airfoil hole
    surface = gmsh.model.geo.addPlaneSurface([cl_outer, cl_airfoil])

    gmsh.model.geo.synchronize()

    # ── Physical groups (Fluent boundary-condition names) ─────────────────────
    gmsh.model.addPhysicalGroup(1, [arc1, arc2],         name="inlet")
    gmsh.model.addPhysicalGroup(1, [l_outlet],           name="outlet")
    gmsh.model.addPhysicalGroup(1, [l_top, l_bot],       name="farfield_wall")
    gmsh.model.addPhysicalGroup(1, [spl_upper, spl_lower], name="airfoil")
    gmsh.model.addPhysicalGroup(2, [surface],            name="fluid")

    # ── Boundary-layer field ──────────────────────────────────────────────────
    f   = gmsh.model.mesh.field
    bl  = f.add("BoundaryLayer")
    f.setNumbers(bl, "CurvesList",  [spl_upper, spl_lower])
    f.setNumber( bl, "Size",         first_layer_height)
    f.setNumber( bl, "Ratio",        1.2)
    f.setNumber( bl, "NbLayers",     n_boundary_layer)
    f.setNumber( bl, "Quads",        1)
    gmsh.model.mesh.field.setAsBoundaryLayer(bl)

    # ── Generate mesh ─────────────────────────────────────────────────────────
    gmsh.model.mesh.generate(2)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    gmsh.write(output_path)
    gmsh.finalize()

    print(f"[meshing] Mesh saved → {output_path}")
    return os.path.abspath(output_path)
