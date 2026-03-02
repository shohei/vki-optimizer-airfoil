"""
CFD Multi-Objective Optimization Configuration
VKI-style NSGA-II optimization of NACA 4-digit airfoils

Example: Maximize CL, Minimize CD
"""

# ─── CFD Backend ──────────────────────────────────────────────────────────────
USE_MOCK_CFD = True       # True: use physics-based surrogate (no Fluent needed)
                           # False: launch real Ansys Fluent via PyFluent

# ─── Fluent Settings (used only when USE_MOCK_CFD = False) ───────────────────
# FLUENT_VERSION is used to locate the installation via the AWP_ROOT{ver} env var.
# PyFluent maps "25.2" → AWP_ROOT252, "24.2" → AWP_ROOT242, etc.
# Set to None to let PyFluent auto-detect the newest installed version.
FLUENT_VERSION = None      # e.g. "25.2", "24.2", or None for auto-detect

# Alternative: set FLUENT_PATH to the full path of the Fluent executable.
# This takes precedence over FLUENT_VERSION and bypasses AWP_ROOT entirely.
# Example (Linux):  "/ansys_inc/v252/fluent/bin/fluent"
# Example (Windows): r"C:\Program Files\ANSYS Inc\v252\fluent\ntbin\win64\fluent.exe"
FLUENT_PATH = "/ansys_inc/v252/fluent/bin/fluent"

FLUENT_PROCESSORS = 64    # number of parallel MPI processes
FLUENT_CASE_TEMPLATE = "results/cases/airfoil_template.cas.h5"
FLUENT_ITER = 300          # solver iterations per evaluation

# ─── Flow Conditions ─────────────────────────────────────────────────────────
CHORD_LENGTH = 1.0         # m
FREESTREAM_VELOCITY = 50.0 # m/s
AIR_DENSITY = 1.225        # kg/m³
DYNAMIC_VISCOSITY = 1.789e-5  # Pa·s
REYNOLDS_NUMBER = AIR_DENSITY * FREESTREAM_VELOCITY * CHORD_LENGTH / DYNAMIC_VISCOSITY

# ─── Design Space ────────────────────────────────────────────────────────────
# Variable 0: angle of attack α [degrees]
ALPHA_MIN = -4.0
ALPHA_MAX = 16.0

# Variable 1: max camber as fraction of chord (NACA 4-digit 1st digit / 100)
CAMBER_MIN = 0.00   # symmetric (NACA 00xx)
CAMBER_MAX = 0.09   # NACA 9xxx

# Variable 2: max thickness as fraction of chord (NACA 4-digit last 2 digits / 100)
THICKNESS_MIN = 0.06  # NACA xx06
THICKNESS_MAX = 0.24  # NACA xx24

# Camber position (kept fixed at 40% chord for NACA 4-digit simplicity)
CAMBER_POS = 0.40

# ─── NSGA-II Settings ────────────────────────────────────────────────────────
POPULATION_SIZE = 40
N_GENERATIONS = 60
N_OFFSPRING = 20

# Simulated Binary Crossover (SBX) parameters
SBX_PROB = 0.9
SBX_ETA = 15.0

# Polynomial Mutation (PM) parameters
PM_ETA = 20.0

RANDOM_SEED = 42

# ─── Output ──────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
PARETO_PLOT_FILE = "results/pareto_front.png"
HISTORY_PLOT_FILE = "results/convergence_history.png"
RESULTS_CSV = "results/pareto_solutions.csv"

# ─── DoE Settings ─────────────────────────────────────────────────────────────
DOE_N_SAMPLES = 100          # LHS サンプル数
DOE_SAMPLER   = "lhs"        # "lhs" | "sobol"
DOE_CSV       = "results/doe_samples.csv"

# ─── ANN Surrogate Settings ───────────────────────────────────────────────────
ANN_HIDDEN_LAYERS  = (64, 64, 32)
ANN_ACTIVATION     = "relu"
ANN_MAX_ITER       = 2000
ANN_RANDOM_STATE   = 42
ANN_SAVE_PATH      = "results/surrogate_model.pkl"

# ─── Infill Settings ──────────────────────────────────────────────────────────
INFILL_N_CANDIDATES = 10     # パレート解の中で実CFDで検証する点数
INFILL_RETRAIN      = True   # 検証後にサロゲートを再学習するか
