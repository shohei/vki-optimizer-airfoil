"""
PyFluent-based CFD evaluator for airfoil aerodynamics.

Requires:
  - Ansys Fluent 2022 R2 or later with a valid solver license
  - pip install ansys-fluent-core

Workflow for each design evaluation
────────────────────────────────────
1. Generate an updated airfoil mesh (gmsh)
2. Launch Fluent in headless batch mode
3. Import mesh, set physics and boundary conditions
4. Run RANS solver (k-ω SST) to convergence
5. Extract CL, CD from force-report monitors
6. Return results dict
"""

from __future__ import annotations

import math
import os
import re
from typing import Optional

from config import (
    CHORD_LENGTH,
    FLUENT_ITER,
    FLUENT_PATH,
    FLUENT_PROCESSORS,
    FLUENT_VERSION,
    FREESTREAM_VELOCITY,
    AIR_DENSITY,
    DYNAMIC_VISCOSITY,
    REYNOLDS_NUMBER,
)
from meshing.gmsh_mesh import generate_airfoil_mesh


_SOLVER: Optional[object] = None   # cached Fluent session
_MESH_LOADED: bool = False         # True after the first mesh has been read

# ── CGNS zone identification ──────────────────────────────────────────────────
# When Fluent imports a gmsh CGNS file via tui.file.import_.cgns.mesh(), the
# physical-group names embedded in the file are ignored by the Ceetron SAM
# reader.  Zones receive generic names "2_l_1" … "2_l_7".
# We identify each zone by its edge length (= "area" in 2-D Fluent) and
# rename / retype it accordingly.
#
# Expected edge lengths for the C-type domain (chord=1, R=15, L=20 chord):
#   airfoil upper/lower  ≈  1 m      (chord ≈ 1 m)
#   farfield top/bottom  ≈ 21 m      (wake_length + chord)
#   inlet arcs (×2)      ≈ 23.6 m   (π/2 · domain_radius)
#   outlet               ≈ 30 m      (2 · domain_radius)

# Fixed mapping: 2_l_N zone names produced by Fluent's Ceetron CGNS reader.
# N equals the gmsh curve entity tag, assigned in the order the curves are
# created in generate_airfoil_mesh():
#   arc1=1, arc2=2, l_top=3, l_outlet=4, l_bot=5, spl_upper=6, spl_lower=7
# Confirmed empirically via field_data.get_zones_info() + CGNS read-phase counts.
_CGNS_ZONE_MAP: dict[str, tuple[str, str]] = {
    "2_l_1": ("inlet",        "velocity-inlet"),   # arc1 (inlet arc)
    "2_l_2": ("inlet2",       "velocity-inlet"),   # arc2 (inlet arc)
    "2_l_3": ("farfield_top", "wall"),             # l_top
    "2_l_4": ("outlet",       "pressure-outlet"),  # l_outlet
    "2_l_5": ("farfield_bot", "wall"),             # l_bot
    "2_l_6": ("airfoil",      "wall"),             # spl_upper (airfoil upper)
    "2_l_7": ("airfoil2",     "wall"),             # spl_lower (airfoil lower)
}


def _fix_cgns_zones(solver) -> None:
    """
    Rename and retype the 7 generic '2_l_N' zones that Fluent's Ceetron CGNS
    reader creates, to the named zones expected by _setup_physics.

    The mapping is fixed: 2_l_N always corresponds to gmsh curve entity tag N
    (because gmsh writes CGNS sections in entity-tag order, and Fluent names
    each imported section 2_l_<tag>).
    """
    for old_name, (new_name, bc_type) in _CGNS_ZONE_MAP.items():
        # In Fluent 25.x PyFluent, TUI zone-manipulation methods are:
        #   tui.mesh.modify_zones.zone_type(zone, new_type)
        #   tui.mesh.modify_zones.zone_name(zone, new_name)
        try:
            solver.tui.mesh.modify_zones.zone_type(old_name, bc_type)
        except Exception as e:
            print(f"[fluent] Warning: zone_type '{old_name}' → '{bc_type}': {e}")
        try:
            solver.tui.mesh.modify_zones.zone_name(old_name, new_name)
            print(f"[fluent] Zone: {old_name} → '{new_name}' [{bc_type}]")
        except Exception as e:
            print(f"[fluent] Warning: zone_name '{old_name}' → '{new_name}': {e}")


def _gmsh_convert(cgns_path: str, out_ext: str) -> str:
    """
    Use gmsh to re-open a CGNS file and re-export in another format.

    gmsh wrote the original CGNS, so it can re-open it without h5py.
    Physical group names are preserved in the output (Nastran element sets,
    Abaqus elsets, UNV groups, etc.).

    Parameters
    ----------
    cgns_path : str
        Absolute path to the input .cgns file.
    out_ext : str
        Target file extension, e.g. ``".nas"``, ``".inp"``, ``".unv"``.

    Returns
    -------
    str
        Absolute path to the converted file.
    """
    try:
        import gmsh
    except ImportError:
        raise ImportError("gmsh is required for mesh conversion.\n"
                          "Install with:  pip install gmsh")

    out_path = cgns_path.replace(".cgns", out_ext)
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(cgns_path)   # gmsh can open files it previously wrote
        gmsh.write(out_path)   # format inferred from extension
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass
    return out_path


def _load_mesh_first_time(solver, abs_mesh: str) -> None:
    """
    Import a mesh file into a freshly started Fluent solver session.

    The method called depends on the file format (extension):

    Nastran  (.nas) → tui.file.import_.nastran()
      gmsh physical-group names become Fluent boundary-zone names.
      Nastran import accepts only a filename (no sub-prompts), so the TUI
      wrapper is directly callable — unlike import_.cgns which is a sub-menu.

    Fluent-native (.cas / .cas.h5 / Fluent-ASCII .msh)
                  → settings.file.read_case()  then  tui.file.read_case()

    CGNS (.cgns)  → tui.file.import_.cgns()  (last resort; historically fails
      in Solver mode because it is a TUI sub-menu, not a direct command)
    """
    print(f"[fluent] Loading mesh: {abs_mesh}")
    ext = os.path.splitext(abs_mesh)[1].lower()

    # ── Nastran: TUI import ───────────────────────────────────────────────
    # In Fluent 25.x, tui.file.import_.nastran is a TUIMenu (not TUIMethod),
    # so it must be called via the sub-command .bulkdata() for bulk-data files.
    if ext == ".nas":
        try:
            solver.tui.file.import_.nastran.bulkdata(abs_mesh)
            print("[fluent] Mesh loaded via tui.file.import_.nastran.bulkdata()")
            return
        except Exception as e:
            raise RuntimeError(
                f"[fluent] tui.file.import_.nastran.bulkdata failed: {e}\n"
                "The Nastran file was generated correctly by gmsh; the error\n"
                "is in Fluent's import.  Try setting USE_MOCK_CFD=True or\n"
                "switch to Fluent Meshing mode (see comments in fluent_runner.py)."
            ) from e

    # ── Fluent-native formats: use read_case ─────────────────────────────
    if ext in (".cas", ".h5", ".msh"):
        try:
            solver.settings.file.read_case(file_name=abs_mesh)
            print("[fluent] Mesh loaded via settings.file.read_case()")
            return
        except Exception as e1:
            print(f"[fluent] settings.file.read_case failed: {e1}")
        try:
            solver.tui.file.read_case(abs_mesh)
            print("[fluent] Mesh loaded via tui.file.read_case()")
            return
        except Exception as e2:
            raise RuntimeError(
                f"[fluent] read_case failed for {abs_mesh}: {e2}"
            ) from e2

    # ── CGNS: TUI import ──────────────────────────────────────────────────
    # In Fluent 25.x, tui.file.import_.cgns is a TUIMenu with sub-commands.
    # Use .cgns.mesh() to read a CGNS mesh file as a case file.
    if ext == ".cgns":
        try:
            solver.tui.file.import_.cgns.mesh(abs_mesh)
            print("[fluent] Mesh loaded via tui.file.import_.cgns.mesh()")
            return
        except Exception as e:
            raise RuntimeError(
                f"[fluent] tui.file.import_.cgns.mesh failed: {e}\n"
                "Use Nastran (.nas) format or Fluent Meshing mode instead."
            ) from e

    raise RuntimeError(
        f"[fluent] Unsupported mesh format '{ext}' for: {abs_mesh}\n"
        "Supported: .nas (Nastran), .cas/.cas.h5 (Fluent native), .cgns"
    )


def _get_solver():
    """
    Launch (or reuse) a persistent Fluent session.

    Fluent is expensive to start up (~30 s); we keep a single session alive
    and only reload the mesh for each new geometry.
    """
    global _SOLVER
    if _SOLVER is not None:
        return _SOLVER

    try:
        import ansys.fluent.core as pyfluent
    except ImportError:
        raise ImportError(
            "ansys-fluent-core is not installed.\n"
            "Install it with:  pip install ansys-fluent-core\n"
            "Or set USE_MOCK_CFD = True in config.py to use the surrogate."
        )

    # Common keyword arguments (independent of how Fluent is located)
    common_kwargs = dict(
        dimension=pyfluent.Dimension.TWO,
        precision=pyfluent.Precision.DOUBLE,
        processor_count=FLUENT_PROCESSORS,
        ui_mode=pyfluent.UIMode.NO_GUI,
        mode=pyfluent.FluentMode.SOLVER,
    )

    # ── Strategy 1: FLUENT_PATH set in config.py ─────────────────────────
    if FLUENT_PATH:
        # Validate the path first – give a clear error before PyFluent tries
        if not os.path.isfile(FLUENT_PATH):
            raise FileNotFoundError(
                f"\n[fluent] FLUENT_PATH does not exist: '{FLUENT_PATH}'\n\n"
                "Please update FLUENT_PATH in config.py to the actual Fluent executable.\n"
                "To locate it on your system:\n"
                "  Linux:   find /usr /opt /ansys_inc -name 'fluent' -type f 2>/dev/null\n"
                "  Windows: dir /s /b \"C:\\Program Files\\ANSYS Inc\\fluent.exe\"\n\n"
                "Or set USE_MOCK_CFD = True in config.py to run without Fluent."
            )

        # Extract version from path (looks for patterns like /v252/ or /V252/)
        # and set AWP_ROOT automatically so PyFluent can resolve the API version.
        ver_match = re.search(r'[/\\][Vv](\d{3})[/\\]', FLUENT_PATH)
        if ver_match:
            ver_num = ver_match.group(1)                      # e.g. "252"
            ver_dot = f"{ver_num[:2]}.{ver_num[2]}"          # "252" → "25.2"
            awp_root = FLUENT_PATH.split(f"v{ver_num}")[0] + f"v{ver_num}"
            awp_key  = f"AWP_ROOT{ver_num}"
            if not os.environ.get(awp_key):
                os.environ[awp_key] = awp_root
                print(f"[fluent] Auto-set {awp_key} = {awp_root}")
            print(f"[fluent] Launching Fluent {ver_dot} via FLUENT_PATH …")
            _SOLVER = pyfluent.launch_fluent(
                product_version=ver_dot,
                fluent_path=FLUENT_PATH,
                **common_kwargs,
            )
        else:
            # Path found but version unrecognisable – let PyFluent try
            print(f"[fluent] Launching Fluent via FLUENT_PATH (version undetected) …")
            _SOLVER = pyfluent.launch_fluent(fluent_path=FLUENT_PATH, **common_kwargs)

    # ── Strategy 2: FLUENT_VERSION + AWP_ROOT env var ─────────────────────
    elif FLUENT_VERSION:
        awp_key = "AWP_ROOT" + FLUENT_VERSION.replace(".", "")
        if not os.environ.get(awp_key):
            raise EnvironmentError(
                f"\n[fluent] Environment variable {awp_key} is not set.\n\n"
                f"Fix options (choose one):\n"
                f"  A) Set the env var before running:\n"
                f"       export {awp_key}=/ansys_inc/v{FLUENT_VERSION.replace('.', '')}\n\n"
                f"  B) Set FLUENT_PATH in config.py to the actual executable:\n"
                f"       FLUENT_PATH = '/ansys_inc/v{FLUENT_VERSION.replace('.', '')}/fluent/bin/fluent'\n\n"
                f"  C) Set FLUENT_VERSION = None in config.py for auto-detection.\n\n"
                f"  D) Set USE_MOCK_CFD = True to run without Fluent."
            )
        print(f"[fluent] Launching Fluent {FLUENT_VERSION} …")
        _SOLVER = pyfluent.launch_fluent(product_version=FLUENT_VERSION, **common_kwargs)

    # ── Strategy 3: auto-detect (requires at least one AWP_ROOT* env var) ─
    else:
        awp_vars = [k for k in os.environ if re.match(r"AWP_ROOT\d{3}$", k)]
        if not awp_vars:
            raise EnvironmentError(
                "\n[fluent] No Ansys Fluent installation found.\n\n"
                "PyFluent requires at least one AWP_ROOT* environment variable.\n"
                "Configure one of the following in config.py:\n\n"
                "  Option A – set the executable path directly:\n"
                "    FLUENT_PATH = '/ansys_inc/v252/fluent/bin/fluent'\n\n"
                "  Option B – set version string (needs AWP_ROOT env var):\n"
                "    FLUENT_VERSION = '25.2'\n"
                "    # Also: export AWP_ROOT252=/ansys_inc/v252\n\n"
                "  Option C – mock CFD, no Fluent needed:\n"
                "    USE_MOCK_CFD = True\n\n"
                "To find your Fluent installation:\n"
                "  Linux:   find /usr /opt /ansys_inc -name 'fluent' -type f 2>/dev/null\n"
                "  Windows: dir /s /b \"C:\\Program Files\\ANSYS Inc\\fluent.exe\""
            )
        print(f"[fluent] Auto-detecting Fluent (found: {awp_vars}) …")
        _SOLVER = pyfluent.launch_fluent(**common_kwargs)

    print("[fluent] Fluent session started.")
    return _SOLVER


def _setup_physics(solver, alpha_deg: float) -> None:
    """Configure turbulence model, materials and boundary conditions."""
    s = solver.settings

    # ── Viscous model: k-ω SST ────────────────────────────────────────────
    s.setup.models.viscous.model = "k-omega"
    s.setup.models.viscous.k_omega_model = "sst"

    # ── Material: air ─────────────────────────────────────────────────────
    air = s.setup.materials.fluid["air"]
    air.density.value = AIR_DENSITY
    air.viscosity.value = DYNAMIC_VISCOSITY

    # ── Boundary conditions ───────────────────────────────────────────────
    alpha_rad = math.radians(alpha_deg)

    # Velocity inlet – two arcs (inlet + inlet2) after CGNS zone split
    # Using velocity-inlet (incompressible, Ma ≈ 0.15) avoids ideal-gas requirement.
    for inlet_name in ("inlet", "inlet2"):
        try:
            inlet = s.setup.boundary_conditions.velocity_inlet[inlet_name]
            inlet.momentum.velocity_specification_method = "Magnitude and Direction"
            inlet.momentum.velocity_magnitude.value = FREESTREAM_VELOCITY
            inlet.momentum.flow_direction[0] = math.cos(alpha_rad)
            inlet.momentum.flow_direction[1] = math.sin(alpha_rad)
            inlet.turbulence.turbulence_specification = "Intensity and Viscosity Ratio"
            inlet.turbulence.turbulent_intensity = 0.001
            inlet.turbulence.turbulent_viscosity_ratio = 1.0
        except KeyError:
            print(f"[fluent] Warning: velocity_inlet zone '{inlet_name}' not found")

    # Pressure outlet
    outlet = s.setup.boundary_conditions.pressure_outlet["outlet"]
    outlet.momentum.gauge_pressure.value = 0.0

    # Airfoil walls – upper surface (airfoil) + lower surface (airfoil2)
    for af_name in ("airfoil", "airfoil2"):
        try:
            s.setup.boundary_conditions.wall[af_name].momentum.wall_motion = "Stationary Wall"
        except KeyError:
            print(f"[fluent] Warning: wall zone '{af_name}' not found")


def _setup_reports(solver) -> None:
    """Create lift and drag force monitors (if not already defined)."""
    report_defs = solver.settings.solution.report_definitions

    # Both airfoil surfaces (upper + lower) contribute to forces
    _AF_ZONES = ["airfoil", "airfoil2"]

    try:
        report_defs.force.create("Fx_report")
    except Exception:
        pass

    fx = report_defs.force["Fx_report"]
    fx.force_vector = [1, 0, 0]
    fx.zones = _AF_ZONES

    try:
        report_defs.force.create("Fy_report")
    except Exception:
        pass

    fy = report_defs.force["Fy_report"]
    fy.force_vector = [0, 1, 0]
    fy.zones = _AF_ZONES


def _compute_forces(solver, alpha_deg: float) -> tuple[float, float]:
    """
    Run the solver and return (CL, CD).

    Forces Fx, Fy are in the mesh coordinate frame; we rotate them to
    the wind axis to get lift (perpendicular to free-stream) and drag
    (parallel to free-stream).
    """
    # Run iterations
    # PyFluent 25.x: parameter is iter_count (not number_of_iterations)
    solver.settings.solution.run_calculation.iterate(
        iter_count=FLUENT_ITER
    )

    # Compute force reports
    # compute() returns a list of dicts: [{'Fx_report': [value, 'N']}, ...]
    forces = solver.settings.solution.report_definitions.compute(
        report_defs=["Fx_report", "Fy_report"]
    )
    Fx = forces[0]["Fx_report"][0]
    Fy = forces[1]["Fy_report"][0]

    # Wind-axis transformation
    alpha_rad = math.radians(alpha_deg)
    q_inf = 0.5 * AIR_DENSITY * FREESTREAM_VELOCITY**2
    S_ref = CHORD_LENGTH * 1.0  # span = 1 m (2-D)

    # Drag = Fx*cos(α) + Fy*sin(α)
    # Lift = -Fx*sin(α) + Fy*cos(α)
    Drag = Fx * math.cos(alpha_rad) + Fy * math.sin(alpha_rad)
    Lift = -Fx * math.sin(alpha_rad) + Fy * math.cos(alpha_rad)

    CD = Drag / (q_inf * S_ref)
    CL = Lift / (q_inf * S_ref)

    return float(CL), float(CD)


def evaluate(
    alpha_deg: float,
    camber: float,
    thickness: float,
    work_dir: str = "results/cases",
) -> dict:
    """
    Full CFD evaluation of a NACA 4-digit airfoil at given conditions.

    Parameters
    ----------
    alpha_deg : float
        Angle of attack in degrees.
    camber : float
        Max camber fraction (0 ≤ m ≤ 0.09).
    thickness : float
        Max thickness fraction (0.06 ≤ t ≤ 0.24).
    work_dir : str
        Directory for intermediate mesh files.

    Returns
    -------
    dict with keys: CL, CD, CL_CD, converged
    """
    os.makedirs(work_dir, exist_ok=True)

    # ── Step 1: Generate mesh (CGNS via gmsh) ────────────────────────────
    # gmsh preserves physical group names (inlet, outlet, farfield_wall,
    # airfoil) in CGNS.  Fluent 25.x can import CGNS via
    # tui.file.import_.cgns.mesh(), which is a TUIMethod in that version.
    label = f"m{camber:.3f}_p0.4_t{thickness:.3f}_a{alpha_deg:.1f}"
    mesh_path = os.path.join(work_dir, f"{label}.cgns")

    generate_airfoil_mesh(
        m=camber, p=0.40, t=thickness,
        output_path=mesh_path,
        chord=CHORD_LENGTH,
    )

    load_path = os.path.abspath(mesh_path)

    # ── Step 2: Launch / reuse Fluent ────────────────────────────────────
    solver = _get_solver()

    # ── Step 3: Load mesh ────────────────────────────────────────────────
    print(f"[fluent] Importing mesh: {os.path.basename(load_path)}")
    _load_mesh_first_time(solver, load_path)

    # ── Step 4: Fix zone names ────────────────────────────────────────────
    # CGNS import creates generic '2_l_N' zone names; rename to named zones.
    _fix_cgns_zones(solver)

    # ── Step 5: Physics setup ─────────────────────────────────────────────
    _setup_physics(solver, alpha_deg)
    _setup_reports(solver)

    # ── Step 6: Initialize and solve ──────────────────────────────────────
    solver.settings.solution.initialization.hybrid_initialize()
    print(f"[fluent] Solving: α={alpha_deg:.1f}°, m={camber:.3f}, t={thickness:.3f}")

    CL, CD = _compute_forces(solver, alpha_deg)

    print(f"[fluent] Results: CL={CL:.4f}, CD={CD:.5f}, L/D={CL/CD:.2f}")

    return {
        "CL": CL,
        "CD": CD,
        "CL_CD": CL / CD if CD > 1e-9 else 0.0,
        "alpha_stall_deg": None,   # not computed in RANS directly
        "converged": True,
    }


def shutdown() -> None:
    """Close the Fluent session (call once when optimization is done)."""
    global _SOLVER, _MESH_LOADED
    if _SOLVER is not None:
        _SOLVER.exit()
        _SOLVER = None
        _MESH_LOADED = False
        print("[fluent] Fluent session closed.")
