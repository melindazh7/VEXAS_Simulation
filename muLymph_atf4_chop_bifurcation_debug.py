"""
ATF4/CHOP Bifurcation Analysis — Erguler 2013 UPR Model
=========================================================
Model: BIOMD0000000446  |  BMC Syst Biol 7:16 (PMID: 23433609)

Figures produced:
  fig1_bifurcation.png  — simplified bifurcation diagram (ratio + apoptotic signal vs stress)
  fig2_hopf.png         — eigenvalue tracking + oscillatory time courses
  fig3_2param.png       — 2-parameter fate map (kphos × trcCHOP)
  fig4_phase.png        — phase portraits in ATF4–CHOP projection
  fig5_vexas.png        — VEXAS lineage bifurcation (myeloid vs lymphoid ks3p)

Run:
  python atf4_chop_bifurcation.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
import warnings

from claude.Cornell_Code import ATF4_muMyemax_value, ATF4_muLymphmax_value, CHOP_muLymphmax_value
from claude.atf4_chop_bifurcation import ATF4_IC_MAX

warnings.filterwarnings("ignore")

import os
import urllib.request
import tellurium as te
import roadrunner
roadrunner.Logger.setLevel(roadrunner.Logger.LOG_ERROR)
import random
import numpy as np
import Cornell_Code
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BIOMD_ID = "BIOMD0000000446"

# Continuation
KPHOS_MIN, KPHOS_MAX = 0.1, 6.863157894736842
N_CONTINUATION      = 200       # points per direction; reduce to 100 if slow
SS_CONVERGENCE_TOL  = 1.0       # steadyState() norm above this = non-convergent
SS_FALLBACK_T       = 400       # fallback simulation end-time
SS_FALLBACK_N       = 4000

# Oscillation detection
OSC_SIM_T  = 600
OSC_SIM_N  = 6000
OSC_STD_THRESHOLD = 0.05        # std(BAXmBCL2 tail) above this = oscillatory

# Bifurcation classification
HOPF_IM_MIN      = 1e-3         # min |Im(λ)| to flag a zero-crossing as Hopf
REAL_ZERO_TOL    = 0.05         # |Re(λ)| below this = "near zero"
BAX_THRESHOLD    = 33.4       # BAXmBCL2 above this → apoptosis

# 2-parameter map
TRCHOP_MIN, TRCHOP_MAX = 0.1, 5.0
N_2PARAM_KPHOS   = 50
N_2PARAM_TRCHOP  = 50

# Phase portraits
N_IC_TRAJ        = 20
PHASE_SIM_T      = 300
PHASE_SIM_N      = 3000
ATF4_IC_MAX = Cornell_Code.ATF4_muLymphmax_value
CHOP_IC_MAX = Cornell_Code.CHOP_muLymphmax_value

# VEXAS ks3p sweep
N_KS3P           = 150
KS3P_MIN, KS3P_MAX = 0.05, 1.0
KS3P_MYELOID     = 0.15         # lower → myeloid (apoptosis-resistant)
KS3P_LYMPHOID    = 0.60         # higher → lymphoid (apoptosis-sensitive)
N_KS3P_KPHOS     = 100          # kphos points for VEXAS overlay

# Colors
C_SURVIVE   = "#3A86FF"
C_APOPTOSIS = "#FF4040"
C_STABLE    = "#2196F3"
C_UNSTABLE  = "#FF8F00"
C_HOPF      = "#9C27B0"
C_FOLD      = "#F44336"
C_OSCILLATE = "#FF6F00"
C_ATF4      = "#2196F3"
C_CHOP      = "#E53935"
C_BAX       = "#7B1FA2"
C_MYELOID   = "#4CAF50"
C_LYMPHOID  = "#F44336"

SELECTIONS_FULL = ["time", "ATF4", "CHOP", "BAXmBCL2", "eIF2a", "GADD34"]

np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MODEL MANAGEMENT & HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    print(f"Loading {BIOMD_ID} …")
    local_path = f"{BIOMD_ID}.xml"
    if not os.path.exists(local_path):
        url = (f"https://www.ebi.ac.uk/biomodels/model/download/{BIOMD_ID}"
               f"?filename={BIOMD_ID}_url.xml")
        print(f"  Downloading from BioModels …")
        urllib.request.urlretrieve(url, local_path)
    r = te.loadSBMLModel(local_path)
    r.integrator.absolute_tolerance = 1e-8   # default ~1e-12 causes t+h=t warnings at stiff t≈0 transients
    r.integrator.relative_tolerance = 1e-6
    r.integrator.setValue("maximum_num_steps", 50000)
    print("Model loaded.")
    return r


def _apply_params(r, param_dict: dict):
    """Set parameters on r without calling resetAll()."""
    for k, v in param_dict.items():
        try:
            r[k] = float(v)
        except Exception:
            pass

def _set_species(r, ss: dict):
    """Warm-start: set all species to values in ss dict."""
    for sp, val in ss.items():
        try:
            r[sp] = float(val)
        except Exception:
            pass


def _get_species_dict(r) -> dict:
    return {sp: float(r[sp]) for sp in r.getFloatingSpeciesIds()}


def _sim_to_ss(r, param_dict: dict) -> dict | None:
    """Simulate to SS with progressively looser tolerances until CVODE converges."""
    species = ["time"] + r.getFloatingSpeciesIds()
    for atol in (1e-8, 1e-6, 1e-4, 1e-2):
        try:
            r.integrator.absolute_tolerance = atol
            res = r.simulate(0, SS_FALLBACK_T, SS_FALLBACK_N, selections=species)
            return {sp: float(res[sp][-1]) for sp in r.getFloatingSpeciesIds()}
        except Exception:
            pass
    return None


def try_steady_state(r, param_dict: dict) -> tuple[dict | None, str]:
    """
    Attempt r.steadyState(). On failure or non-convergence, fall through to
    a long simulation from the current (warm-started) state.
    Returns (ss_dict, method) where method is 'ss_solver'|'sim_fallback'|'failed'.
    """
    try:
        norm = r.steadyState()
        if norm < SS_CONVERGENCE_TOL:
            return _get_species_dict(r), "ss_solver"
    except Exception:
        pass

    # Fallback: integrate from current warm state with escalating tolerances
    ss = _sim_to_ss(r, param_dict)
    if ss is not None:
        return ss, "sim_fallback"

    # Last resort: full reset + re-apply params + simulate
    r.resetAll()
    _apply_params(r, param_dict)
    ss = _sim_to_ss(r, param_dict)
    if ss is not None:
        return ss, "sim_fallback"
    return None, "failed"


def get_eigenvalues(r) -> np.ndarray:
    """Jacobian eigenvalues at current state. Returns empty array on failure."""
    try:
        J = r.getFullJacobian()
        return np.linalg.eigvals(np.array(J, dtype=complex))
    except Exception:
        return np.array([], dtype=complex)


def get_leading_eigenvalue(eigs: np.ndarray) -> complex:
    """Eigenvalue with the largest real part."""
    if len(eigs) == 0:
        return complex(np.nan, np.nan)
    return eigs[np.argmax(np.real(eigs))]


def classify_ss(eigs: np.ndarray) -> str:
    if len(eigs) == 0:
        return "unknown"
    re = np.real(eigs)
    im = np.imag(eigs)
    lead = get_leading_eigenvalue(eigs)
    if abs(np.real(lead)) < REAL_ZERO_TOL and abs(np.imag(lead)) > HOPF_IM_MIN:
        return "hopf_cand"
    if np.all(re < -REAL_ZERO_TOL):
        return "stable"
    if np.any(re > REAL_ZERO_TOL):
        return "unstable"
    return "near_bifurc"


def _is_oscillatory_sim(bax_trace: np.ndarray) -> bool:
    tail = bax_trace[int(len(bax_trace) * 0.8):]
    return float(np.std(tail)) > OSC_STD_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONTINUATION DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BifPoint:
    kphos: float
    ss: dict
    eigs: np.ndarray
    classification: str
    method: str
    leading_eig: complex

    def get(self, species: str, default=np.nan) -> float:
        return float(self.ss.get(species, default)) if self.ss else default

    @property
    def ratio(self) -> float:
        atf4 = self.get("ATF4")
        chop = self.get("CHOP")
        return atf4 / chop if (not np.isnan(chop) and chop > 1e-9) else np.nan

    @property
    def bax(self) -> float:
        return self.get("BAXmBCL2")

    @property
    def fate(self) -> str:
        b = self.bax
        return "apoptosis" if (not np.isnan(b) and b > BAX_THRESHOLD) else "survival"


@dataclass
class ContinuationResult:
    forward:     list = field(default_factory=list)
    backward:    list = field(default_factory=list)
    fold_points: list = field(default_factory=list)
    hopf_points: list = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — NUMERICAL CONTINUATION
# ══════════════════════════════════════════════════════════════════════════════

def _detect_bifurcations(points: list[BifPoint]) -> tuple[list, list]:
    """
    Scan for sign changes in Re(λ_lead). Returns (fold_kphos, hopf_kphos).
    Uses linear interpolation to estimate exact crossing.
    """
    folds, hopfs = [], []
    for i in range(1, len(points)):
        a, b = points[i-1], points[i]
        if np.isnan(a.leading_eig.real) or np.isnan(b.leading_eig.real):
            continue
        re_a, re_b = a.leading_eig.real, b.leading_eig.real
        if re_a * re_b < 0:
            alpha = abs(re_a) / (abs(re_a) + abs(re_b) + 1e-12)
            kphos_cross = a.kphos + alpha * (b.kphos - a.kphos)
            im_avg = (abs(a.leading_eig.imag) + abs(b.leading_eig.imag)) / 2
            if im_avg > HOPF_IM_MIN:
                hopfs.append(kphos_cross)
            else:
                folds.append(kphos_cross)
    return folds, hopfs


def _run_single_sweep(r, kphos_values: np.ndarray,
                      init_reset: bool = True) -> list[BifPoint]:
    """Run continuation along kphos_values; warm-start between steps."""
    points = []
    prev_ss = None


    for i, kp in enumerate(kphos_values):
        param_dict = {"kphos": kp}

        if prev_ss is None or init_reset and i == 0:
            r.resetAll()
            _apply_params(r, param_dict)
        else:
            # Warm-start: set species to previous SS, set new kphos
            _set_species(r, prev_ss)
            _apply_params(r, param_dict)

        ss, method = try_steady_state(r, param_dict)

        if ss is not None:
            # Get eigenvalues immediately after SS, before any state change
            eigs = get_eigenvalues(r)
            lead = get_leading_eigenvalue(eigs)
            clf  = classify_ss(eigs)

            # Override classification if simulation shows oscillation
            if method == "sim_fallback":
                try:
                    res = r.simulate(0, OSC_SIM_T // 3, OSC_SIM_N // 3,
                                     selections=["BAXmBCL2"])
                    if _is_oscillatory_sim(res["BAXmBCL2"]):
                        clf = "oscillatory"
                except Exception:
                    pass
        else:
            eigs = np.array([], dtype=complex)
            lead = complex(np.nan, np.nan)
            clf  = "failed"

        pt = BifPoint(kphos=kp, ss=ss or {}, eigs=eigs,
                      classification=clf, method=method, leading_eig=lead)
        points.append(pt)
        prev_ss = ss

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(kphos_values)} done")

    return points


def run_continuation(r, n_points: int = N_CONTINUATION) -> ContinuationResult:
    """
    Forward + backward parameter continuation along kphos.
    Returns ContinuationResult with both sweeps and detected bifurcation points.
    """
    result = ContinuationResult()
    kphos_fwd = np.linspace(KPHOS_MIN, KPHOS_MAX, n_points)
    kphos_bwd = np.linspace(KPHOS_MAX, KPHOS_MIN, n_points)

    print("  Forward sweep …")
    result.forward = _run_single_sweep(r, kphos_fwd, init_reset=True)

    print("  Backward sweep …")
    result.backward = _run_single_sweep(r, kphos_bwd, init_reset=True)

    result.fold_points, result.hopf_points = _detect_bifurcations(result.forward)
    print(f"  Fold points:  {[f'{v:.2f}' for v in result.fold_points]}")
    print(f"  Hopf points:  {[f'{v:.2f}' for v in result.hopf_points]}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FIGURE 1: SIMPLIFIED BIFURCATION DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════

def _pts_to_arrays(points: list[BifPoint], attr: str) -> tuple[np.ndarray, np.ndarray]:
    kp  = np.array([p.kphos for p in points])
    val = np.array([getattr(p, attr)() if callable(getattr(p, attr))
                    else getattr(p, attr) for p in points])
    return kp, val


def plot_bifurcation_diagram(result: ContinuationResult,
                             save_path: str = "fig1_bifurcation.png"):
    """
    Two-panel simplified bifurcation diagram.
    Panel A: ATF4/CHOP ratio vs kphos — colored blue (survival) / red (apoptosis).
    Panel B: BAXmBCL2 vs kphos — forward (solid) + backward (dashed).
    """
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(10, 8),
                                      sharex=True, gridspec_kw={"hspace": 0.12})

    fwd = result.forward
    bwd = result.backward
    all_bif = sorted(result.fold_points + result.hopf_points)

    # ── determine shading boundaries ─────────────────────────────────────────
    # Find where the forward sweep first crosses BAX_THRESHOLD
    bif_kphos = all_bif[0] if all_bif else None

    # Fallback: use last survival → first apoptosis midpoint from forward sweep
    if bif_kphos is None:
        surv_idx = [i for i, p in enumerate(fwd) if p.fate == "survival"]
        apo_idx  = [i for i, p in enumerate(fwd) if p.fate == "apoptosis"]
        if surv_idx and apo_idx:
            bif_kphos = (fwd[max(surv_idx)].kphos + fwd[min(apo_idx)].kphos) / 2
    # ── Panel A: ATF4/CHOP ratio ──────────────────────────────────────────────
    kp_fwd  = np.array([p.kphos for p in fwd])
    rat_fwd = np.array([p.ratio for p in fwd])


    # Color each segment by whether ratio > 1 (survival) or < 1 (apoptosis)

    rat_x = 0
    rat_y = 0
    for i in range(len(kp_fwd) - 1):
        if np.isnan(rat_fwd[i]) or np.isnan(rat_fwd[i+1]):
            continue
        avg_ratio = (rat_fwd[i] + rat_fwd[i+1]) / 2

        color = C_SURVIVE if avg_ratio >= 1.0 else C_APOPTOSIS
        ls    = "-" if fwd[i].classification in ("stable", "unknown") else "--"
        ax_a.plot(kp_fwd[i:i+2], rat_fwd[i:i+2], color=color, lw=2.5,
                  ls=ls, alpha=0.85, solid_capstyle="round")
        # Create variables that represent the x and y coordinates for when the avg_ratio is close to 1
        if abs(1.0 - avg_ratio) <= 0.02:
            rat_x = kp_fwd[i+1]
            rat_y = rat_fwd[i+1]


    ax_a.axhline(1.0, color="grey", ls="--", lw=1.0, alpha=0.7, label="ATF4 = CHOP")
    ax_a.set_ylabel("ATF4 / CHOP ratio\n(steady state)", fontsize=11)
    ax_a.set_ylim(bottom=0)

    # Background shading
    if bif_kphos:
        ax_a.axvspan(KPHOS_MIN, rat_x, color=C_APOPTOSIS, alpha=0.07)
        ax_a.axvspan(rat_x, KPHOS_MAX, color=C_SURVIVE, alpha=0.07)
        ax_a.text(bif_kphos * 0.45, ax_a.get_ylim()[1] * 0.88, "Apoptosis\n(CHOP dominant)", color=C_APOPTOSIS, fontsize=10, fontweight="bold", ha="center")
        ax_a.text((bif_kphos + KPHOS_MAX) / 2, ax_a.get_ylim()[1] * 0.88, "Survival\n(ATF4 dominant)", color = C_SURVIVE, fontsize = 10, fontweight = "bold", ha = "center")

    # Mark bifurcation point(s)
    for kp in result.hopf_points:
        ax_a.axvline(kp, color=C_HOPF, ls=":", lw=1.0, alpha=0.8)
        #ax_a.annotate("Hopf bifurcation", xy=(kp, ax_a.get_ylim()[1] * 0.5),
                      #xytext=(kp + 0.6, ax_a.get_ylim()[1] * 0.6),
                      #fontsize=8, color=C_HOPF,
                      #arrowprops=dict(arrowstyle="->", color=C_HOPF, lw=1.0))
    for kp in result.fold_points:
        ax_a.axvline(kp, color=C_FOLD, ls=":", lw=1.5, alpha=0.8)
        ax_a.annotate("Fold bifurcation", xy=(kp, ax_a.get_ylim()[1] * 0.3),
                      xytext=(kp + 0.6, ax_a.get_ylim()[1] * 0.4),
                      fontsize=8, color=C_FOLD,
                      arrowprops=dict(arrowstyle="->", color=C_FOLD, lw=1.0))

    legend_a = [
        mlines.Line2D([], [], color=C_SURVIVE,   lw=2.5, label="Survival branch"),
        mlines.Line2D([], [], color=C_APOPTOSIS, lw=2.5, label="Apoptosis branch"),
        mlines.Line2D([], [], color="grey", ls="--", lw=1.0, label="ATF4 = CHOP"),
        mlines.Line2D([], [], color="k", ls="-",  lw=2, label="Stable SS"),
        mlines.Line2D([], [], color="k", ls="--", lw=1.5, alpha=0.5, label="Unstable SS"),
        mlines.Line2D([], [], color=C_HOPF, ls="--", lw=1.5, alpha=0.5, label="HOPF Bifurcation"),
    ]
    ax_a.legend(handles=legend_a, fontsize=8, loc="upper right", framealpha=0.9)
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.set_title("A  ATF4/CHOP ratio along the stress axis",
                   fontsize=12, loc="left", fontweight="bold")

    # ── Panel B: BAXmBCL2 ────────────────────────────────────────────────────
    kp_bwd  = np.array([p.kphos for p in bwd])
    bax_fwd = np.array([p.bax  for p in fwd])
    bax_bwd = np.array([p.bax  for p in bwd])

    ax_b.plot(kp_fwd, bax_fwd, color=C_BAX, lw=2.5, label="Forward sweep")
    ax_b.plot(kp_bwd, bax_bwd, color=C_BAX, lw=1.5, ls="--", alpha=0.55,
              label="Backward sweep (hysteresis check)")
    ax_b.axhline(BAX_THRESHOLD, color="red", ls=":", lw=1.2,
                 label=f"Death threshold ({BAX_THRESHOLD})")
    ax_b.fill_between(kp_fwd, bax_fwd, BAX_THRESHOLD,
                      where=(bax_fwd > BAX_THRESHOLD),
                      color=C_APOPTOSIS, alpha=0.15, label="Apoptotic zone")

    if bif_kphos:
        ax_b.axvspan(KPHOS_MIN, rat_x, color=C_APOPTOSIS,   alpha=0.07)
        ax_b.axvspan(rat_x, KPHOS_MAX, color=C_SURVIVE, alpha=0.07)

    for kp in result.hopf_points + result.fold_points:
        ax_b.axvline(kp, color="grey", ls=":", lw=1.0, alpha=0.6)

    ax_b.set_ylabel("BAXmBCL2\n(apoptotic signal, a.u.)", fontsize=11)
    ax_b.set_xlabel("ER Stress Intensity  (kphos — PERK→eIF2α phosphorylation rate)",
                    fontsize=11)
    ax_b.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.set_title("B  Apoptotic signal along the stress axis",
                   fontsize=12, loc="left", fontweight="bold")
    ax_b.set_xlim(KPHOS_MIN, KPHOS_MAX)

    fig.suptitle("Bifurcation Diagram: How ER Stress Intensity Decides Cell Fate\n"
                 "Model: Erguler et al. 2013  |  BIOMD0000000446",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.show()


import numpy as np
from scipy.optimize import brentq

def compute_ratio(r, kphos_value: float, trchop_value: float = None,
                   t_end: float = 300, n_points: int = 1500) -> float:
    r.resetAll()
    r["kphos"] = float(kphos_value)
    if trchop_value is not None:
        r["trcCHOP"] = float(trchop_value)
    res = r.simulate(0, t_end, n_points, selections=["ATF4", "CHOP"])
    atf4 = float(res["ATF4"][-1])
    chop = float(res["CHOP"][-1])
    if chop == 0:
        return np.nan
    return atf4 / chop

def find_kphos_range_by_scan(r, kphos_min: float, kphos_max: float,
                              ratio_min: float, ratio_max: float,
                              n_points: int = 200,
                              trchop_value: float = None) -> dict:
    kphos_ax = np.linspace(kphos_min, kphos_max, n_points)
    ratios = np.array([
        compute_ratio(r, kp, trchop_value=trchop_value)
        for kp in kphos_ax
    ])

    valid_mask = (ratios >= ratio_min) & (ratios <= ratio_max)
    valid_kphos = kphos_ax[valid_mask]

    if valid_kphos.size == 0:
        return {"kphos_ax": kphos_ax, "ratios": ratios,
                "valid_range": None, "message": "No kphos values in range found."}

    return {
        "kphos_ax": kphos_ax,
        "ratios": ratios,
        "valid_range": (float(valid_kphos.min()), float(valid_kphos.max())),
        "valid_mask": valid_mask,
    }

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FIGURE 2: HOPF DETECTION + OSCILLATION TIME COURSES
# ══════════════════════════════════════════════════════════════════════════════

def _choose_representative_kphos(result: ContinuationResult) -> list[float]:
    all_bif = sorted(result.hopf_points + result.fold_points)
    if all_bif:
        first = all_bif[0]
        return [max(KPHOS_MIN + 0.3, first * 0.45),
                first,
                min(KPHOS_MAX - 0.3, first * 2.0)]
    return [2.0, 6.0, 12.0]


def run_oscillation_timecourses(r, kphos_values: list) -> dict:
    results = {}
    for kp in kphos_values:
        r.resetAll()
        try:
            r["kphos"] = float(kp)
            res = r.simulate(0, OSC_SIM_T, OSC_SIM_N, selections=SELECTIONS_FULL)
            results[kp] = {k: res[k] for k in SELECTIONS_FULL}
        except Exception as e:
            print(f"  [warn] timecourse failed at kphos={kp}: {e}")
    return results


def plot_hopf_analysis(result: ContinuationResult, timecourses: dict,
                       rep_kphos: list,
                       save_path: str = "fig2_hopf.png"):
    fig = plt.figure(figsize=(16, 8))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    fwd = result.forward
    kp_arr  = np.array([p.kphos         for p in fwd])
    re_arr  = np.array([p.leading_eig.real if not np.isnan(p.leading_eig.real)
                        else np.nan for p in fwd])
    im_arr  = np.array([abs(p.leading_eig.imag) if not np.isnan(p.leading_eig.imag)
                        else np.nan for p in fwd])

    # ── Eigenvalue panel (top, spans all 3 cols) ──────────────────────────────
    ax_eig = fig.add_subplot(gs[0, :])
    ax_im  = ax_eig.twinx()

    # Color Re(λ) by sign
    for i in range(len(kp_arr) - 1):
        if np.isnan(re_arr[i]) or np.isnan(re_arr[i + 1]):
            continue

        if re_arr[i] * re_arr[i + 1] < 0:
            # Sign change within this segment — find where it crosses zero
            frac = re_arr[i] / (re_arr[i] - re_arr[i + 1])
            kp_cross = kp_arr[i] + frac * (kp_arr[i + 1] - kp_arr[i])

            color_left = C_APOPTOSIS if re_arr[i] > 0 else C_SURVIVE
            color_right = C_APOPTOSIS if re_arr[i + 1] > 0 else C_SURVIVE

            ax_eig.plot([kp_arr[i], kp_cross], [re_arr[i], 0], color=color_left, lw=2.0)
            ax_eig.plot([kp_cross, kp_arr[i + 1]], [0, re_arr[i + 1]], color=color_right, lw=2.0)
        else:
            color = C_APOPTOSIS if re_arr[i] > 0 else C_SURVIVE
            ax_eig.plot(kp_arr[i:i + 2], re_arr[i:i + 2], color=color, lw=2.0)

    ax_im.plot(kp_arr, im_arr, color="grey", lw=1.2, ls="--", alpha=0.7,
               label="|Im(λ_lead)|")
    ax_eig.axhline(0, color="k", lw=0.8, ls="-")

    for kp in result.hopf_points:
        ax_eig.axvline(kp, color=C_HOPF, ls=":", lw=1.5)
        ax_eig.text(kp + 0.15, ax_eig.get_ylim()[1] * 0.85 if not np.all(np.isnan(re_arr))
                    else 0.1, f"Hopf\n{kp:.1f}", color=C_HOPF, fontsize=8)
    for kp in result.fold_points:
        ax_eig.axvline(kp, color=C_FOLD, ls=":", lw=1.5)
        ax_eig.text(kp + 0.15, 0.02, f"Fold\n{kp:.1f}", color=C_FOLD, fontsize=8)

    ax_eig.set_ylabel("Re(λ_max)  —  stability indicator", fontsize=10)
    ax_im.set_ylabel("|Im(λ)|  —  oscillation frequency", fontsize=10, color="grey")
    ax_im.tick_params(axis="y", labelcolor="grey")
    ax_eig.set_xlabel("kphos", fontsize=10)
    ax_eig.set_title("A  Leading eigenvalue along the stress axis\n"
                     "Blue = stable (Re < 0) | Red = unstable (Re > 0)",
                     fontsize=11, loc="left")
    ax_eig.spines[["top"]].set_visible(False)

    stable_patch   = mpatches.Patch(color=C_STABLE,   label="Stable (Re < 0)")
    unstable_patch = mpatches.Patch(color=C_APOPTOSIS, label="Unstable (Re > 0)")
    ax_eig.legend(handles=[stable_patch, unstable_patch], fontsize=9, loc="lower right")

    # ── Time course panels (bottom row) ──────────────────────────────────────
    labels = ["B  Pre-bifurcation", "C  Near bifurcation", "D  Post-bifurcation"]
    for col, (kp, label) in enumerate(zip(rep_kphos, labels)):
        ax = fig.add_subplot(gs[1, col])
        ax2 = ax.twinx()

        tc = timecourses.get(kp)
        if tc is not None:
            t = tc["time"]
            ax.plot(t, tc["ATF4"], color=C_ATF4, lw=1.8, label="ATF4")
            ax.plot(t, tc["CHOP"], color=C_CHOP, lw=1.8, label="CHOP")
            ax2.plot(t, tc["BAXmBCL2"], color=C_BAX, lw=1.2, ls="--",
                     alpha=0.7, label="BAXmBCL2")
            ax2.axhline(BAX_THRESHOLD, color=C_BAX, lw=0.8, ls=":", alpha=0.5)

        ax.set_xlabel("Time (a.u.)", fontsize=9)
        ax.set_ylabel("ATF4 / CHOP (a.u.)", fontsize=9)
        ax2.set_ylabel("BAXmBCL2", fontsize=9, color=C_BAX)
        ax2.tick_params(axis="y", labelcolor=C_BAX)
        ax.set_title(f"{label}\nkphos = {kp:.1f}", fontsize=10)
        ax.spines[["top"]].set_visible(False)

        if col == 0:
            lines  = [mlines.Line2D([], [], color=C_ATF4, lw=1.8, label="ATF4"),
                      mlines.Line2D([], [], color=C_CHOP, lw=1.8, label="CHOP"),
                      mlines.Line2D([], [], color=C_BAX,  lw=1.2, ls="--", label="BAXmBCL2")]
            ax.legend(handles=lines, fontsize=8)

    fig.suptitle("Hopf Bifurcation Analysis: Emergence of Oscillations\n"
                 "Model: Erguler et al. 2013  |  BIOMD0000000446",
                 fontsize=12, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FIGURE 3: 2-PARAMETER MAP (kphos × trcCHOP)
# ══════════════════════════════════════════════════════════════════════════════

def run_2param_map(r, n_kphos: int = N_2PARAM_KPHOS,
                   n_trc: int = N_2PARAM_TRCHOP) -> dict:
    kphos_ax  = np.linspace(KPHOS_MIN, KPHOS_MAX, n_kphos)
    trchop_ax = np.linspace(TRCHOP_MIN, TRCHOP_MAX, n_trc)
    fate_grid = np.full((n_trc, n_kphos), np.nan)

    total = n_kphos * n_trc
    done  = 0
    for i, tc in enumerate(trchop_ax):
        for j, kp in enumerate(kphos_ax):
            r.resetAll()
            try:
                r["kphos"]   = float(kp)
                r["trcCHOP"] = float(tc)
                res = r.simulate(0, 300, 1500, selections=["BAXmBCL2"])
                bax = res["BAXmBCL2"]
                osc = float(np.std(bax[int(len(bax)*0.8):]))
                if osc > OSC_STD_THRESHOLD:
                    fate_grid[i, j] = 2.0   # oscillatory
                elif float(bax[-1]) > BAX_THRESHOLD:
                    fate_grid[i, j] = 1.0   # apoptosis
                else:
                    fate_grid[i, j] = 0.0   # survival
            except Exception:
                fate_grid[i, j] = np.nan
            done += 1
            if done % 500 == 0:
                print(f"    {done}/{total} grid points done")

    return {"kphos_ax": kphos_ax, "trchop_ax": trchop_ax, "fate_grid": fate_grid}


def plot_2param_map(map_result: dict,
                    fold_points: list, hopf_points: list,
                    save_path: str = "fig3_2param.png"):
    plt.ion()
    kphos_ax  = map_result["kphos_ax"]
    trchop_ax = map_result["trchop_ax"]
    fate_grid = map_result["fate_grid"]

    cmap = mcolors.ListedColormap([C_SURVIVE, C_APOPTOSIS, C_OSCILLATE])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.pcolormesh(kphos_ax, trchop_ax, fate_grid,
                       cmap=cmap, norm=norm, shading="auto", alpha=0.85)

    for kp in hopf_points:
        ax.axvline(kp, color="white", ls="--", lw=1.5, alpha=0.8)
        ax.text(kp + 0.15, TRCHOP_MAX * 0.95, f"Hopf\n{kp:.1f}",
                color="white", fontsize=8, va="top")
    for kp in fold_points:
        ax.axvline(kp, color="white", ls=":", lw=1.5, alpha=0.8)

    # VEXAS lineage trcCHOP reference lines
    for trc_val, label, color in [(1.0, "Myeloid (trcCHOP ≈ 1)", C_MYELOID),
                                   (3.5, "Lymphoid (trcCHOP ≈ 3.5)", C_LYMPHOID)]:
        if TRCHOP_MIN <= trc_val <= TRCHOP_MAX:
            ax.axhline(trc_val, color=color, ls="--", lw=1.5,
                       alpha=0.9, label=label)

    # Region annotations
    ax.text(KPHOS_MAX * 0.75, TRCHOP_MAX * 0.12,
            "Adaptation\n(Survival)", color="white", fontsize=10,
            fontweight="bold", ha="center")
    ax.text(KPHOS_MAX * 0.15, TRCHOP_MAX * 0.12,
            "Apoptosis\nInitiation", color="white", fontsize=10,
            fontweight="bold", ha="center")
    ax.text(KPHOS_MAX * 0.5, TRCHOP_MAX * 0.65,
            "Oscillatory\nRegime", color="white", fontsize=9,
            fontweight="bold", ha="center")

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["Survival / Adaptation", "Apoptosis", "Oscillatory"], fontsize=9)

    ax.set_xlabel("ER Stress Intensity (kphos)", fontsize=11)
    ax.set_ylabel("CHOP Transcription Capacity (trcCHOP)", fontsize=11)
    ax.set_title("2-Parameter Fate Map: Stress × CHOP Transcription Rate\n"
                 "Dashed white = bifurcation boundary  |  "
                 "Dashed colored = VEXAS lineage reference",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — FIGURE 4: PHASE PORTRAITS
# ══════════════════════════════════════════════════════════════════════════════

def _find_fixed_points(r, kphos: float, n_grid: int = 3) -> list[dict]:
    fps = []
    for a0 in np.linspace(0.1, ATF4_IC_MAX, n_grid):
        for c0 in np.linspace(0.1, CHOP_IC_MAX, n_grid):
            r.resetAll()
            try:
                r["kphos"] = float(kphos)
                r["ATF4"] = float(a0)
                r["CHOP"] = float(c0)
                norm = r.steadyState()
                if norm < SS_CONVERGENCE_TOL:
                    eigs = get_eigenvalues(r)
                    fps.append({"atf4": float(r["ATF4"]),
                                "chop": float(r["CHOP"]),
                                "bax":  float(r["BAXmBCL2"]),
                                "classification": classify_ss(eigs)})
            except Exception:
                pass

    # Deduplicate within tolerance
    unique = []
    for fp in fps:
        is_dup = any(abs(fp["atf4"] - u["atf4"]) < 0.2
                     and abs(fp["chop"] - u["chop"]) < 0.2
                     for u in unique)
        if not is_dup:
            unique.append(fp)
    return unique


def run_phase_portraits(r, kphos_values: list,
                        n_ic: int = N_IC_TRAJ) -> dict:
    portrait_data = {}
    for kp in kphos_values:
        trajectories = []
        for _ in range(n_ic):
            r.resetAll()
            try:
                r["kphos"] = float(kp)
                barcode, ATF4 = random.choice(list(Cornell_Code.muLymph_dict.items()))
                r.ATF4 = ATF4.get('ATF4')
                barcode, CHOP = random.choice(list(Cornell_Code.muLymph_dict.items()))
                r.CHOP = CHOP.get('DDIT3')
                res = r.simulate(0, PHASE_SIM_T, PHASE_SIM_N,
                                 selections=["time", "ATF4", "CHOP", "BAXmBCL2"])
                fate = ("apoptosis" if float(res["BAXmBCL2"][-1]) > BAX_THRESHOLD
                        else "survival")
                trajectories.append({"atf4": res["ATF4"],
                                     "chop": res["CHOP"],
                                     "fate": fate})
            except Exception:
                pass

        fps = _find_fixed_points(r, kp)
        portrait_data[kp] = {"trajectories": trajectories, "fixed_points": fps}

    return portrait_data


def plot_phase_portraits(portrait_data: dict, kphos_values: list,
                         save_path: str = "fig4_phase.png"):
    labels = ["A  Sub-threshold", "B  Near bifurcation", "C  Supra-threshold"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                              gridspec_kw={"wspace": 0.35})

    for ax, kp, label in zip(axes, kphos_values, labels):
        data = portrait_data.get(kp, {})

        for traj in data.get("trajectories", []):
            color = C_SURVIVE if traj["fate"] == "survival" else C_APOPTOSIS
            ax.plot(traj["atf4"], traj["chop"], color=color,
                    lw=0.9, alpha=0.45)
            # Arrow at 60% of trajectory
            n = len(traj["atf4"])
            mid = int(n * 0.6)
            if mid + 1 < n:
                ax.annotate("", xy=(traj["atf4"][mid+1], traj["chop"][mid+1]),
                            xytext=(traj["atf4"][mid], traj["chop"][mid]),
                            arrowprops=dict(arrowstyle="-|>", color=color,
                                            lw=0.8, mutation_scale=10))

        for fp in data.get("fixed_points", []):
            clf = fp["classification"]
            if clf == "stable":
                marker, mfc, ms = "o", C_STABLE, 12
            elif clf in ("unstable", "saddle"):
                marker, mfc, ms = "o", "none", 12
            elif clf == "hopf_cand":
                marker, mfc, ms = "D", C_HOPF, 10
            else:
                marker, mfc, ms = "x", "grey", 10
            ax.scatter(fp["atf4"], fp["chop"], marker=marker,
                       s=ms**2, color=C_STABLE if "stable" in clf else C_UNSTABLE,
                       facecolors=mfc, linewidths=1.5, zorder=5)

        diag = np.linspace(0, max(ATF4_IC_MAX, CHOP_IC_MAX), 50)
        ax.plot(diag, diag, "k--", lw=0.8, alpha=0.35, label="ATF4 = CHOP")
        ax.set_xlim(0, ATF4_IC_MAX)
        ax.set_ylim(0, CHOP_IC_MAX)
        ax.set_xlabel("ATF4 (a.u.)", fontsize=10)
        ax.set_ylabel("CHOP (a.u.)", fontsize=10)
        ax.set_title(f"{label}\nkphos = {kp:.1f}", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=C_SURVIVE,   label="Survival trajectory"),
        mpatches.Patch(color=C_APOPTOSIS, label="Apoptosis trajectory"),
        mlines.Line2D([0],[0], marker="o", color=C_STABLE,   lw=0,
                      markersize=10, label="Stable fixed point"),
        mlines.Line2D([0],[0], marker="o", color=C_UNSTABLE, lw=0,
                      markersize=10, fillstyle="none", label="Unstable fixed point"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Phase Portraits: ATF4 vs CHOP Concentration Space\n"
                 "Trajectories from random initial conditions | "
                 "Arrows show direction of flow",
                 fontsize=12, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FIGURE 5: VEXAS LINEAGE BIFURCATION (ks3p sweep)
# ══════════════════════════════════════════════════════════════════════════════

def run_ks3p_sweep(r, n_points: int = N_KS3P) -> dict:
    ks3p_vals = np.linspace(KS3P_MIN, KS3P_MAX, n_points)
    bax_arr   = np.full(n_points, np.nan)
    atf4_arr  = np.full(n_points, np.nan)
    chop_arr  = np.full(n_points, np.nan)
    clf_list  = ["unknown"] * n_points
    default_kphos = 5.0   # model default

    prev_ss = None
    for i, ks in enumerate(ks3p_vals):
        r.resetAll()
        try:
            r["ks3p"]  = float(ks)
            r["kphos"] = default_kphos
            if prev_ss:
                _set_species(r, prev_ss)
                r["ks3p"]  = float(ks)
                r["kphos"] = default_kphos
            param_dict = {"ks3p": ks, "kphos": default_kphos}
            ss, method = try_steady_state(r, param_dict)
            if ss:
                bax_arr[i]  = ss.get("BAXmBCL2", np.nan)
                atf4_arr[i] = ss.get("ATF4", np.nan)
                chop_arr[i] = ss.get("CHOP", np.nan)
                eigs = get_eigenvalues(r)
                clf_list[i] = classify_ss(eigs)
                prev_ss = ss
        except Exception:
            pass

    return {"ks3p": ks3p_vals, "bax": bax_arr,
            "atf4": atf4_arr, "chop": chop_arr,
            "classification": clf_list}


def run_ks3p_kphos_overlays(r, ks3p_values: list,
                              n_kphos: int = N_KS3P_KPHOS) -> dict:
    kphos_vals = np.linspace(KPHOS_MIN, KPHOS_MAX, n_kphos)
    results = {}
    for ks in ks3p_values:
        bax_arr = np.full(n_kphos, np.nan)
        prev_ss = None
        for j, kp in enumerate(kphos_vals):
            r.resetAll()
            try:
                r["ks3p"]  = float(ks)
                r["kphos"] = float(kp)
                if prev_ss:
                    _set_species(r, prev_ss)
                    r["ks3p"]  = float(ks)
                    r["kphos"] = float(kp)
                param_dict = {"ks3p": ks, "kphos": kp}
                ss, _ = try_steady_state(r, param_dict)
                if ss:
                    bax_arr[j] = ss.get("BAXmBCL2", np.nan)
                    prev_ss = ss
            except Exception:
                pass
        results[ks] = {"kphos": kphos_vals, "bax": bax_arr}
    return results


def plot_vexas_bifurcation(sweep_result: dict, overlay_result: dict,
                            save_path: str = "fig5_vexas.png"):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6),
                                      gridspec_kw={"wspace": 0.38})

    # ── Panel A: BAXmBCL2 vs ks3p at default kphos ───────────────────────────
    ks  = sweep_result["ks3p"]
    bax = sweep_result["bax"]
    clf = sweep_result["classification"]

    for i in range(len(ks) - 1):
        if np.isnan(bax[i]) or np.isnan(bax[i+1]):
            continue
        c = C_STABLE if clf[i] in ("stable", "unknown") else C_UNSTABLE
        ax_a.plot(ks[i:i+2], bax[i:i+2], color=c, lw=2.5)

    ax_a.text(ks[len(ks)-1], bax[len(ks)-1], "BAXmBCL2", color= C_UNSTABLE)
    ax_a.axhline(BAX_THRESHOLD, color="red", ls=":", lw=1.2,
                 label=f"Death threshold ({BAX_THRESHOLD})")
    ax_a.axvline(KS3P_MYELOID,  color=C_MYELOID,  ls="--", lw=1.8,
                 label=f"Myeloid  ks3p ≈ {KS3P_MYELOID}")
    ax_a.axvline(KS3P_LYMPHOID, color=C_LYMPHOID, ls="--", lw=1.8,
                 label=f"Lymphoid ks3p ≈ {KS3P_LYMPHOID}")
    ax_a.text(KS3P_MYELOID,  ax_a.get_ylim()[1] if ax_a.get_ylim()[1] > 0
              else 1.5, "Myeloid",  color=C_MYELOID,  fontsize=9,
              rotation=90, va="bottom", ha="right")
    ax_a.text(KS3P_LYMPHOID, ax_a.get_ylim()[1] if ax_a.get_ylim()[1] > 0
              else 1.5, "Lymphoid", color=C_LYMPHOID, fontsize=9,
              rotation=90, va="bottom", ha="right")

    ax_a.set_xlabel("CHOP→BH3 Apoptotic Coupling (ks3p)", fontsize=11)
    ax_a.set_ylabel("BAXmBCL2 steady state (apoptotic signal)", fontsize=11)
    ax_a.set_title("A  Apoptotic sensitivity across lineages\n"
                   "(at default ER stress kphos = 5)", fontsize=11)
    ax_a.legend(fontsize=8, loc="upper left")
    ax_a.spines[["top", "right"]].set_visible(False)

    # ── Panel B: BAXmBCL2 vs kphos for 3 ks3p values ─────────────────────────
    ks3p_vals = sorted(overlay_result.keys())
    colors_overlay = {ks3p_vals[0]: C_MYELOID,
                      ks3p_vals[1]: "grey",
                      ks3p_vals[2]: C_LYMPHOID}
    labels_overlay = {ks3p_vals[0]: f"Myeloid  (ks3p = {ks3p_vals[0]:.2f})",
                      ks3p_vals[1]: f"Intermediate (ks3p = {ks3p_vals[1]:.2f})",
                      ks3p_vals[2]: f"Lymphoid (ks3p = {ks3p_vals[2]:.2f})"}

    bax_curves = {}
    for ks, data in overlay_result.items():
        kp  = data["kphos"]
        bax = data["bax"]
        ax_b.plot(kp, bax, color=colors_overlay[ks], lw=2.2,
                  label=labels_overlay[ks])
        bax_curves[ks] = bax

    # Shaded VEXAS susceptibility window between myeloid and lymphoid
    ks_my = ks3p_vals[0]
    ks_ly = ks3p_vals[2]
    if ks_my in bax_curves and ks_ly in bax_curves:
        kp_ref = overlay_result[ks_my]["kphos"]
        bax_my = bax_curves[ks_my]
        bax_ly = bax_curves[ks_ly]
        valid  = ~(np.isnan(bax_my) | np.isnan(bax_ly))
        ax_b.fill_between(kp_ref[valid], bax_my[valid], bax_ly[valid],
                          where=(bax_ly[valid] > bax_my[valid]),
                          color="orange", alpha=0.18,
                          label="VEXAS susceptibility window")

    ax_b.axhline(BAX_THRESHOLD, color="red", ls=":", lw=1.2,
                 label=f"Death threshold ({BAX_THRESHOLD})")
    ax_b.set_xlabel("ER Stress Intensity (kphos)", fontsize=11)
    ax_b.set_ylabel("BAXmBCL2 steady state (apoptotic signal)", fontsize=11)
    ax_b.set_title("B  Same stress → different fate by lineage\n"
                   "Orange shading = stress range where lineages diverge",
                   fontsize=11)
    ax_b.legend(fontsize=8, loc="upper left")
    ax_b.spines[["top", "right"]].set_visible(False)

    fig.suptitle("VEXAS Lineage-Specific Bifurcation: Why mt_UBA1 Kills Lymphoid but Not Myeloid\n"
                 "Model: Erguler et al. 2013  |  ks3p = CHOP→BH3 apoptotic coupling",
                 fontsize=11, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

r = load_model()

# ── 1. Continuation ──────────────────────────────────────────────────────
print("\n[1/5] Numerical continuation (kphos sweep) …")
scan_result = find_kphos_range_by_scan(r, kphos_min=KPHOS_MIN, kphos_max=KPHOS_MAX, ratio_min=0.0, ratio_max= Cornell_Code.ATF4_muLymphmax_value/Cornell_Code.CHOP_muLymphmax_value, n_points=400)
scan_result["valid_range"]
cont = run_continuation(r, n_points=N_CONTINUATION)
#set parameters
r.ATF4 = Cornell_Code.ATF4_muLymphavg
r.CHOP = Cornell_Code.CHOP_muLymphavg
r.BAX = Cornell_Code.BAX_muLymphavg

plot_bifurcation_diagram(cont)

# ── 2. Hopf characterization ─────────────────────────────────────────────
print("\n[2/5] Hopf bifurcation & oscillation time courses …")
rep_kphos = _choose_representative_kphos(cont)
print(f"  Representative kphos: {[f'{v:.2f}' for v in rep_kphos]}")
timecourses = run_oscillation_timecourses(r, rep_kphos)
plot_hopf_analysis(cont, timecourses, rep_kphos)
# ── 3. 2-parameter map ───────────────────────────────────────────────────
print(f"\n[3/5] 2-parameter map ({N_2PARAM_KPHOS}×{N_2PARAM_TRCHOP} grid) …")
map_result = run_2param_map(r)
plot_2param_map(map_result, cont.fold_points, cont.hopf_points)
# ── 4. Phase portraits ───────────────────────────────────────────────────
print(f"\n[4/5] Phase portraits at kphos = {[f'{v:.1f}' for v in rep_kphos]} …")
portrait_data = run_phase_portraits(r, rep_kphos, n_ic=N_IC_TRAJ)
plot_phase_portraits(portrait_data, rep_kphos)
# ── 5. VEXAS ks3p sweep ──────────────────────────────────────────────────
print("\n[5/5] VEXAS lineage bifurcation (ks3p sweep) …")
ks3p_mid = (KS3P_MYELOID + KS3P_LYMPHOID) / 2
sweep_result  = run_ks3p_sweep(r)
overlay_result = run_ks3p_kphos_overlays(r, [KS3P_MYELOID, ks3p_mid, KS3P_LYMPHOID])
plot_vexas_bifurcation(sweep_result, overlay_result)
print("\n✓ Done. Figures: fig1_bifurcation.png  fig2_hopf.png fig3_2param.png  fig4_phase.png  fig5_vexas.png")

