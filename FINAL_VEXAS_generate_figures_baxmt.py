"""
This file is the updated version to FINAL_VEXAS_generate_figures.py.
It incorporates the BAXmt data as well as an updated Figure 5.

============================================================================
STATUS SUMMARY
============================================================================
BAXmT (total mitochondrial-bound BAX) is used as the apoptotic readout
throughout, per Erguler et al.'s own Figure S10 -- NOT BAXmBCL2. Switching
to BAXmT resolved the wrong-direction problem for 3/4 groups instantly,
using existing parameters, no new fitting.

CONFIRMED DYNAMICS (via hysteresis-based bistability detection -- NOT an
arbitrary threshold; the fold point is a genuine property of the system,
confirmed by forward/backward sweep divergence):
  mutant_myeloid:    GENUINE BISTABILITY, fold ~kphos 3.83-4.0
  wildtype_myeloid:  GENUINE BISTABILITY, fold ~kphos 3.9
  wildtype_lymphoid: GENUINE BISTABILITY, fold ~kphos 8.7
  mutant_lymphoid:   FLAT, NO RESPONSE -- fate NOT determinable from this
                       model (BAXmT stays ~33.36 regardless of stress;
                       three independent mechanistic hypotheses -- BAXT
                       supply, kbx loss, kasx association -- all ruled out;
                       likely rooted in this group's own weak/non-
                       identifiable CHOP induction, trcCHOP=0.05, smallest
                       of all four groups)

CAVEAT ON "APOPTOSIS" LABELING: crossing into the high-BAXmT (ON) branch
means the model predicts a large, stable pool of mitochondrial-bound BAX
-- a necessary precursor to MOMP and downstream apoptosis in the real
biology, and what Erguler et al. use this readout to represent. This
model does NOT simulate MOMP or cell death directly. Labels below say
"heading towards apoptosis" per request, but this should be understood as
the model's proxy for apoptotic commitment, not literal apoptosis.

IDENTIFIABILITY (unchanged from prior analysis -- see project notes for
full profile-likelihood/multi-seed evidence):
  mutant_myeloid:    trcCHOP/ktrATF4/trcGADD34/kfbc HIGH confidence
  wildtype_myeloid:  trcCHOP/trcGADD34/kfbc HIGH; ktrATF4 one-sided bound
  mutant_lymphoid:   ktrATF4 HIGH only; trcCHOP/kfbc/kmbc/trcGADD34 all
                       CONFIRMED NON-IDENTIFIABLE
  wildtype_lymphoid: kfbc HIGH only; trcCHOP/ktrATF4/trcGADD34 all
                       CONFIRMED NON-IDENTIFIABLE (tested 3 independent ways)

FIGURE 5: unchanged from the original GADD34-vs-stress replacement --
uses ONLY trcCHOP/ktrATF4/trcGADD34, completely independent of the BAX
pathway. Uses the corrected try_steady_state (proper steady-state
solving) to avoid a jagged fixed-time-window artifact seen earlier.
============================================================================
"""

import matplotlib
matplotlib.use('Agg')   # MUST be set before pyplot is imported anywhere, including
                          # transitively -- prevents plt.show() from opening a
                          # blocking window. Figures are saved to disk instead;
                          # see open_image() at the bottom to view them automatically.

import numpy as np
import json
import os
import sys
import subprocess
import urllib.request
import tellurium as te
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field
from scipy.stats import spearmanr


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 -- CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
BIOMD_ID = "BIOMD0000000446"
KPHOS_MIN, KPHOS_MAX = 0.1, 20.0
TRCHOP_MIN, TRCHOP_MAX = 0.01, 5.0
SS_CONVERGENCE_TOL = 1e-6
SS_FALLBACK_T, SS_FALLBACK_N = 500, 2000   # global default -- fast, used by Figures 1/2/3/4
                                              # (continuation/bifurcation work), where this
                                              # resolution was already adequate.
# Figure 5 (GADD34) specifically showed the fallback undershooting by 10-20% in the
# transition region at T=500 -- steadyState() itself fails for ~90% of GADD34
# evaluations, making the fallback the PRIMARY path there, not a rare backup.
# Only Figure 5's sweep uses these slower, more thorough settings.
SS_FALLBACK_T_GADD34, SS_FALLBACK_N_GADD34 = 2000, 3000
REAL_ZERO_TOL, HOPF_IM_MIN = 1e-6, 1e-4
HYSTERESIS_GAP_THRESHOLD = 2.0
OSC_STD_THRESHOLD = 1.0
OSC_SIM_T, OSC_SIM_N = 600, 2000
ATF4_IC_MAX, CHOP_IC_MAX = 5.0, 5.0
N_IC_TRAJ_FAST = 6

# FAST_MODE: cuts resolution across the board for a quick full pipeline
# check. FINAL/PRODUCTION RUN: set to False for publication-resolution
# figures -- confirmed working end-to-end in FAST_MODE first (all 4 groups,
# all 6 figures, no errors) before committing to the slower full run.
FAST_MODE = False
N_CONTINUATION = 60 if FAST_MODE else 200
BRANCH_SWEEP_POINTS = 20 if FAST_MODE else 60
N_2PARAM_KPHOS, N_2PARAM_TRCHOP = (8, 8) if FAST_MODE else (50, 50)
N_IC_TRAJ = N_IC_TRAJ_FAST if FAST_MODE else 12
PHASE_SIM_T, PHASE_SIM_N = 300, (300 if FAST_MODE else 1000)
N_KPHOS_GADD34 = 30 if FAST_MODE else 80   # bumped up from 60 for final -- smoother
                                              # curves through the transition region,
                                              # given SS_FALLBACK_T_GADD34 already
                                              # confirmed accurate there

C_OFF, C_ON = "#4C72B0", "#C44E52"      # OFF branch (survival-leaning) / ON branch (apoptosis-leaning)
C_HOPF, C_FOLD = "purple", "orange"
C_ATF4, C_CHOP = "#4C72B0", "#C44E52"
C_STABLE, C_UNSTABLE = "black", "grey"

CONFIRMED_DYNAMICS = {
    "mutant_myeloid":    {"type": "bistable", "threshold_kphos": 3.83},
    "wildtype_myeloid":  {"type": "bistable", "threshold_kphos": 3.9},
    "wildtype_lymphoid": {"type": "bistable", "threshold_kphos": 8.7},
    "mutant_lymphoid":   {"type": "flat", "threshold_kphos": None},
}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 -- MODEL LOADING + CORE HELPERS (bug-fixed versions)
# ═══════════════════════════════════════════════════════════════════════════
def load_model():
    print(f"Loading {BIOMD_ID} …")
    local_path = f"{BIOMD_ID}.xml"
    if not os.path.exists(local_path):
        url = (f"https://www.ebi.ac.uk/biomodels/model/download/{BIOMD_ID}"
               f"?filename={BIOMD_ID}_url.xml")
        urllib.request.urlretrieve(url, local_path)
    r = te.loadSBMLModel(local_path)
    r.integrator.absolute_tolerance = 1e-8
    r.integrator.relative_tolerance = 1e-6
    r.integrator.setValue("maximum_num_steps", 50000)
    print("Model loaded.")
    return r


def _apply_params(r, param_dict: dict):
    for k, v in param_dict.items():
        try: r[k] = float(v)
        except Exception: pass


def _apply_extra(r, extra_fixed_params: dict = None) -> None:
    """FIX: called in every fallback branch of try_steady_state -- prevents
    fitted parameters from silently reverting to SBML defaults whenever
    the primary steady-state solver fails to converge."""
    if extra_fixed_params:
        for name, value in extra_fixed_params.items():
            r[name] = float(value)


def _set_species(r, ss: dict):
    for sp, val in ss.items():
        try: r[sp] = float(val)
        except Exception: pass


def _get_species_dict(r) -> dict:
    return {sp: float(r[sp]) for sp in r.getFloatingSpeciesIds()}


def _sim_to_ss(r, param_dict: dict, extra_fixed_params: dict = None,
                 fallback_t=None, fallback_n=None):
    species = ["time"] + r.getFloatingSpeciesIds()
    t_end = fallback_t if fallback_t is not None else SS_FALLBACK_T
    n_pts = fallback_n if fallback_n is not None else SS_FALLBACK_N
    for atol in (1e-8, 1e-6, 1e-4, 1e-2):
        try:
            r.integrator.absolute_tolerance = atol
            res = r.simulate(0, t_end, n_pts, selections=species)
            return {sp: float(res[sp][-1]) for sp in r.getFloatingSpeciesIds()}
        except Exception:
            pass
    return None


def try_steady_state(r, param_dict: dict, extra_fixed_params: dict = None,
                        fallback_t=None, fallback_n=None):
    """FIXED VERSION -- extra_fixed_params reapplied in every fallback branch,
    AND resetAll() called immediately after a steadyState() failure. Accepts
    optional fallback_t/fallback_n to override the global fast defaults for
    specific call sites (e.g. Figure 5's GADD34 sweep, where steadyState()
    fails ~90% of the time and the fallback needs a longer window to avoid
    undershooting in the transition region)."""
    try:
        norm = r.steadyState()
        if norm < SS_CONVERGENCE_TOL:
            return _get_species_dict(r), "ss_solver"
    except Exception:
        pass
    r.resetAll()
    _apply_extra(r, extra_fixed_params)
    _apply_params(r, param_dict)
    ss = _sim_to_ss(r, param_dict, extra_fixed_params, fallback_t, fallback_n)
    if ss is not None:
        return ss, "sim_fallback"
    r.resetAll()
    _apply_extra(r, extra_fixed_params)
    _apply_params(r, param_dict)
    ss = _sim_to_ss(r, param_dict, extra_fixed_params, fallback_t, fallback_n)
    return (ss, "sim_fallback") if ss is not None else (None, "failed")


def get_eigenvalues(r) -> np.ndarray:
    try:
        J = r.getFullJacobian()
        return np.linalg.eigvals(np.array(J, dtype=complex))
    except Exception:
        return np.array([], dtype=complex)


def get_leading_eigenvalue(eigs: np.ndarray) -> complex:
    if len(eigs) == 0:
        return complex(np.nan, np.nan)
    return eigs[np.argmax(np.real(eigs))]


def classify_ss(eigs: np.ndarray) -> str:
    if len(eigs) == 0:
        return "unknown"
    re = np.real(eigs)
    lead = get_leading_eigenvalue(eigs)
    if abs(np.real(lead)) < REAL_ZERO_TOL and abs(np.imag(lead)) > HOPF_IM_MIN:
        return "hopf_cand"
    if np.all(re < -REAL_ZERO_TOL):
        return "stable"
    if np.any(re > REAL_ZERO_TOL):
        return "unstable"
    return "near_bifurc"


def open_image(path, delay=0.3):
    """Opens a saved image with the OS default viewer. Waits briefly to
    ensure the write has fully completed, and is spaced out (see
    AUTO_OPEN_FIGURES usage in __main__) to avoid overwhelming some
    default viewers (e.g. Windows Photos) when many images are opened in
    quick succession -- confirmed causing intermittent slow-loading/
    stuck-window behavior during batch runs."""
    import time
    time.sleep(delay)
    if not os.path.exists(path):
        print(f"  [warn] {path} not found yet, skipping auto-open")
        return
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception as e:
        print(f"  [note] could not auto-open {path}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 -- DATA STRUCTURES (BAXmT-tracking version)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class BifPoint:
    kphos: float
    ss: dict
    eigs: np.ndarray
    classification: str
    leading_eig: complex

    def get(self, species, default=np.nan):
        return float(self.ss.get(species, default)) if self.ss else default

    @property
    def chop(self): return self.get("CHOP")
    @property
    def atf4(self): return self.get("ATF4")
    @property
    def baxmt(self): return self.get("BAXmT")   # apoptotic readout, per Erguler et al. S10


@dataclass
class ContinuationResult:
    forward: list = field(default_factory=list)
    backward: list = field(default_factory=list)
    fold_points: list = field(default_factory=list)
    hopf_points: list = field(default_factory=list)


def _detect_bifurcations(points):
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
            (hopfs if im_avg > HOPF_IM_MIN else folds).append(kphos_cross)
    return folds, hopfs


def _run_single_sweep(r, kphos_values, extra_fixed_params=None):
    points, prev_ss = [], None
    for i, kp in enumerate(kphos_values):
        param_dict = {"kphos": kp}
        if prev_ss is None:
            r.resetAll()
            _apply_extra(r, extra_fixed_params)
            _apply_params(r, param_dict)
        else:
            _set_species(r, prev_ss)
            _apply_extra(r, extra_fixed_params)
            _apply_params(r, param_dict)

        ss, method = try_steady_state(r, param_dict, extra_fixed_params)
        if ss is not None:
            eigs = get_eigenvalues(r)
            lead = get_leading_eigenvalue(eigs)
            clf = classify_ss(eigs)
        else:
            eigs, lead, clf = np.array([], dtype=complex), complex(np.nan, np.nan), "failed"

        points.append(BifPoint(kphos=kp, ss=ss or {}, eigs=eigs,
                                  classification=clf, leading_eig=lead))
        prev_ss = ss
        if (i + 1) % max(1, len(kphos_values) // 2) == 0:
            print(f"    {i+1}/{len(kphos_values)} done")
    return points


def run_continuation(r, n_points=N_CONTINUATION, extra_fixed_params=None):
    result = ContinuationResult()
    kphos_fwd = np.linspace(KPHOS_MIN, KPHOS_MAX, n_points)
    kphos_bwd = np.linspace(KPHOS_MAX, KPHOS_MIN, n_points)
    print("  Forward sweep …")
    result.forward = _run_single_sweep(r, kphos_fwd, extra_fixed_params)
    print("  Backward sweep …")
    result.backward = _run_single_sweep(r, kphos_bwd, extra_fixed_params)
    result.fold_points, result.hopf_points = _detect_bifurcations(result.forward)
    print(f"  Fold points: {[f'{v:.2f}' for v in result.fold_points]}")
    print(f"  Hopf points: {[f'{v:.2f}' for v in result.hopf_points]}")
    return result


def _choose_representative_kphos(result: ContinuationResult, dynamics_threshold=None) -> list:
    all_bif = sorted(result.hopf_points + result.fold_points)
    if dynamics_threshold is not None:
        first = dynamics_threshold
        return [max(KPHOS_MIN + 0.3, first * 0.45), first, min(KPHOS_MAX - 0.3, first * 2.0)]
    if all_bif:
        first = all_bif[0]
        return [max(KPHOS_MIN + 0.3, first * 0.45), first, min(KPHOS_MAX - 0.3, first * 2.0)]
    return [2.0, 6.0, 12.0]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 -- FIGURE 1: ATF4/CHOP + BAXmT, with fold/Hopf labels AND
#              apoptosis-direction shading labels
# ═══════════════════════════════════════════════════════════════════════════
def plot_figure1(result: ContinuationResult, group_name: str, save_path: str = None):
    dynamics = CONFIRMED_DYNAMICS[group_name]
    fig, (ax_top, ax_bax) = plt.subplots(2, 1, figsize=(10, 9), sharex=True,
                                            gridspec_kw={"hspace": 0.15})
    fwd = result.forward
    kp_fwd = np.array([p.kphos for p in fwd])
    chop_fwd = np.array([p.chop for p in fwd])
    atf4_fwd = np.array([p.atf4 for p in fwd])
    baxmt_fwd = np.array([p.baxmt for p in fwd])
    kp_bwd = np.array([p.kphos for p in result.backward])
    baxmt_bwd = np.array([p.baxmt for p in result.backward])

    # --- Shading + labels: OFF (lower BAXmT) vs. ON (heading towards apoptosis) ---
    if dynamics["type"] == "bistable":
        t = dynamics["threshold_kphos"]
        ax_top.axvspan(kp_fwd.min(), t, color=C_OFF, alpha=0.07,
                         label="OFF branch (lower BAXmT)")
        ax_top.axvspan(t, kp_fwd.max(), color=C_ON, alpha=0.07,
                         label="ON branch (heading towards apoptosis)")
        ax_bax.axvspan(kp_fwd.min(), t, color=C_OFF, alpha=0.07)
        ax_bax.axvspan(t, kp_fwd.max(), color=C_ON, alpha=0.07)

    # --- Top panel: ATF4/CHOP ---
    for i in range(len(kp_fwd) - 1):
        if np.isnan(chop_fwd[i]) or np.isnan(chop_fwd[i+1]):
            continue
        ax_top.plot(kp_fwd[i:i+2], atf4_fwd[i:i+2], color=C_OFF, lw=2.5)
        ax_top.plot(kp_fwd[i:i+2], chop_fwd[i:i+2], color=C_ON, lw=2.5)
    ax_top.set_ylabel("ATF4 / CHOP (steady state)")
    ax_top.set_title(f"A  ATF4/CHOP response -- '{group_name}'", loc="left")

    # --- Fold/Hopf point labels + arrows (RESTORED, top panel) ---
    y_max_top = max(np.nanmax(atf4_fwd), np.nanmax(chop_fwd))
    for kp in result.fold_points:
        ax_top.axvline(kp, color=C_FOLD, ls=":", lw=1.5)
        ax_top.annotate("Fold point", xy=(kp, y_max_top * 0.5), xytext=(kp + 1.2, y_max_top * 0.68),
                          arrowprops=dict(arrowstyle="->", color=C_FOLD, lw=1.5),
                          fontsize=9, color=C_FOLD, fontweight="bold")
    for kp in result.hopf_points:
        ax_top.axvline(kp, color=C_HOPF, ls=":", lw=1.5)
        ax_top.annotate("Hopf point", xy=(kp, y_max_top * 0.35), xytext=(kp + 1.2, y_max_top * 0.18),
                          arrowprops=dict(arrowstyle="->", color=C_HOPF, lw=1.5),
                          fontsize=9, color=C_HOPF, fontweight="bold")

    ax_top.legend(handles=[
        mlines.Line2D([], [], color=C_OFF, lw=2.5, label="ATF4"),
        mlines.Line2D([], [], color=C_ON, lw=2.5, label="CHOP"),
        mlines.Line2D([], [], color=C_FOLD, ls=":", lw=1.5, label="Fold point"),
        mlines.Line2D([], [], color=C_HOPF, ls=":", lw=1.5, label="Hopf point"),
        mpatches.Patch(color=C_OFF, alpha=0.07, label="OFF branch (lower BAXmT)"),
        mpatches.Patch(color=C_ON, alpha=0.07, label="ON branch (heading towards apoptosis)"),
    ], fontsize=8, loc="lower right")   # moved from upper left -- was colliding with the
                                          # fold-point annotation text in that corner

    # --- Bottom panel: BAXmT ---
    ax_bax.plot(kp_fwd, baxmt_fwd, color="purple", lw=2.5, label="Forward sweep")
    ax_bax.plot(kp_bwd, baxmt_bwd, color="purple", lw=1.5, ls="--", alpha=0.55, label="Backward sweep")

    if dynamics["type"] == "bistable":
        ax_bax.axvline(dynamics["threshold_kphos"], color=C_FOLD, ls=":", lw=2,
                          label=f"Fold point (kphos={dynamics['threshold_kphos']:.2f})\n"
                                f"[confirmed via hysteresis]")
    elif dynamics["type"] == "flat":
        ax_bax.text(0.5, 0.5, "FLAT -- no bistability, no meaningful fate threshold",
                      transform=ax_bax.transAxes, ha="center", va="center",
                      fontsize=11, color="darkred", style="italic")

    ax_bax.set_xlabel("ER Stress Intensity (kphos)")
    ax_bax.set_ylabel("BAXmT (total mitochondrial-bound BAX)")
    ax_bax.set_title("B  BAXmT vs. stress (apoptotic readout, per Erguler et al. S10)", loc="left")
    ax_bax.legend(fontsize=9, loc="best")

    fig.suptitle(f"Figure 1 -- {group_name}", fontsize=12, fontweight="bold")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4b -- FIGURE 2: Hopf/eigenvalue analysis + ATF4/CHOP/BAXmT time courses
# ═══════════════════════════════════════════════════════════════════════════
def run_oscillation_timecourses(r, kphos_values: list, extra_fixed_params=None) -> dict:
    results = {}
    for kp in kphos_values:
        r.resetAll()
        _apply_extra(r, extra_fixed_params)
        try:
            r["kphos"] = float(kp)
            res = r.simulate(0, OSC_SIM_T, OSC_SIM_N, selections=["time", "ATF4", "CHOP", "BAXmT"])
            results[kp] = {k: res[k] for k in ["time", "ATF4", "CHOP", "BAXmT"]}
        except Exception as e:
            print(f"  [warn] timecourse failed at kphos={kp}: {e}")
    return results


def plot_figure2(result: ContinuationResult, timecourses: dict, rep_kphos: list,
                    group_name: str, save_path: str = None):
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    fwd = result.forward
    kp_arr = np.array([p.kphos for p in fwd])
    re_arr = np.array([p.leading_eig.real if not np.isnan(p.leading_eig.real) else np.nan for p in fwd])
    im_arr = np.array([abs(p.leading_eig.imag) if not np.isnan(p.leading_eig.imag) else np.nan for p in fwd])

    # --- Top: leading eigenvalue (full-system Jacobian) ---
    ax_eig = fig.add_subplot(gs[0, :])
    ax_im = ax_eig.twinx()
    for i in range(len(kp_arr) - 1):
        if np.isnan(re_arr[i]) or np.isnan(re_arr[i+1]):
            continue
        if re_arr[i] * re_arr[i+1] < 0:
            frac = re_arr[i] / (re_arr[i] - re_arr[i+1])
            kp_cross = kp_arr[i] + frac * (kp_arr[i+1] - kp_arr[i])
            color_left = C_ON if re_arr[i] > 0 else C_OFF
            color_right = C_ON if re_arr[i+1] > 0 else C_OFF
            ax_eig.plot([kp_arr[i], kp_cross], [re_arr[i], 0], color=color_left, lw=2.0)
            ax_eig.plot([kp_cross, kp_arr[i+1]], [0, re_arr[i+1]], color=color_right, lw=2.0)
        else:
            color = C_ON if re_arr[i] > 0 else C_OFF
            ax_eig.plot(kp_arr[i:i+2], re_arr[i:i+2], color=color, lw=2.0)

    ax_im.plot(kp_arr, im_arr, color="grey", lw=1.2, ls="--", alpha=0.7, label="|Im(λ_lead)|")
    ax_eig.axhline(0, color="k", lw=0.8)
    for kp in result.hopf_points:
        ax_eig.axvline(kp, color=C_HOPF, ls=":", lw=1.5)
        ax_eig.annotate("Hopf point", xy=(kp, 0), xytext=(kp + 1.0, ax_eig.get_ylim()[1]*0.5),
                          arrowprops=dict(arrowstyle="->", color=C_HOPF, lw=1.5),
                          fontsize=9, color=C_HOPF, fontweight="bold")
    for kp in result.fold_points:
        ax_eig.axvline(kp, color=C_FOLD, ls=":", lw=1.5)
        ax_eig.annotate("Fold point", xy=(kp, 0), xytext=(kp + 1.0, ax_eig.get_ylim()[0]*0.5),
                          arrowprops=dict(arrowstyle="->", color=C_FOLD, lw=1.5),
                          fontsize=9, color=C_FOLD, fontweight="bold")

    ax_eig.set_ylabel("Re(λ_max)")
    ax_im.set_ylabel("|Im(λ)|", color="grey")
    ax_eig.set_xlabel("ER Stress Intensity (kphos)")
    ax_eig.set_title(f"A  Leading eigenvalue -- '{group_name}'", loc="left")
    ax_eig.legend(handles=[
        mpatches.Patch(color=C_STABLE, label="Stable"),
        mpatches.Patch(color=C_ON, label="Unstable"),
        mlines.Line2D([], [], color=C_FOLD, ls=":", lw=1.5, label="Fold point"),
        mlines.Line2D([], [], color=C_HOPF, ls=":", lw=1.5, label="Hopf point"),
    ], fontsize=8, loc="lower right")

    # --- Bottom row: ATF4/CHOP + BAXmT time courses ---
    labels = ["B  Pre-bifurcation (OFF)", "C  Near bifurcation",
                "D  Post-bifurcation (ON -- heading towards apoptosis)"]
    for col, (kp, label) in enumerate(zip(rep_kphos, labels)):
        ax = fig.add_subplot(gs[1, col])
        ax2 = ax.twinx()
        tc = timecourses.get(kp)
        if tc is not None:
            t = tc["time"]
            ax.plot(t, tc["ATF4"], color=C_ATF4, lw=1.8, label="ATF4")
            ax.plot(t, tc["CHOP"], color=C_CHOP, lw=1.8, label="CHOP")
            ax2.plot(t, tc["BAXmT"], color="purple", lw=1.2, ls="--", alpha=0.7, label="BAXmT")
        ax.set_xlabel("Time (a.u.)")
        ax.set_ylabel("ATF4 / CHOP")
        ax2.set_ylabel("BAXmT", color="purple")
        ax.set_title(f"{label}\nkphos = {kp:.1f}", fontsize=9.5)
        if col == 0:
            ax.legend(handles=[
                mlines.Line2D([], [], color=C_ATF4, lw=1.8, label="ATF4"),
                mlines.Line2D([], [], color=C_CHOP, lw=1.8, label="CHOP"),
                mlines.Line2D([], [], color="purple", lw=1.2, ls="--", label="BAXmT"),
            ], fontsize=7.5)

    fig.suptitle(f"Figure 2 -- Hopf Bifurcation Analysis -- {group_name}", fontweight="bold")
    if save_path:
        # Lower DPI than other figures -- this one draws ~2x N_CONTINUATION
        # individual line segments in the eigenvalue panel (colored per
        # stable/unstable region) plus 3 twinx() subplots, producing a much
        # heavier file at dpi=150 that was observed to load slowly in some
        # image viewers (e.g. Windows Photos) despite generating correctly.
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    plt.close(fig)



def get_group_branch_values(r, group_name, upstream_params,
                               kphos_range=(KPHOS_MIN, KPHOS_MAX, BRANCH_SWEEP_POINTS)):
    kphos_vals = np.linspace(*kphos_range)
    r.resetAll()
    _apply_extra(r, upstream_params)

    baxmt_vals = []
    prev_ss = None
    n_points = len(kphos_vals)
    for idx, kp in enumerate(kphos_vals):
        if prev_ss is not None:
            _set_species(r, prev_ss)
            _apply_extra(r, upstream_params)
        r["kphos"] = float(kp)
        try:
            res = r.simulate(0, 300, 300, selections=["BAXmT"] + r.getFloatingSpeciesIds())
            baxmt_vals.append(float(res["BAXmT"][-1]))
            prev_ss = {sp: float(res[sp][-1]) for sp in r.getFloatingSpeciesIds()}
        except Exception:
            baxmt_vals.append(np.nan)
            prev_ss = None
        if (idx + 1) % max(1, n_points // 4) == 0:
            print(f"      branch sweep: {idx+1}/{n_points} done")

    baxmt_vals = np.array(baxmt_vals)
    n = len(baxmt_vals)
    off_value = float(np.nanmean(baxmt_vals[:max(1, n // 10)]))
    on_value = float(np.nanmean(baxmt_vals[-max(1, n // 10):]))
    print(f"    {group_name}: OFF branch≈{off_value:.2f}, ON branch≈{on_value:.2f}")
    return off_value, on_value


def classify_nearest_branch(baxmt_value, off_value, on_value):
    if np.isnan(baxmt_value):
        return np.nan
    return 1.0 if abs(baxmt_value - on_value) < abs(baxmt_value - off_value) else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 -- FIGURE 3: 2-parameter map (BAXmT nearest-branch classification)
# ═══════════════════════════════════════════════════════════════════════════
def run_2param_map(r, extra_fixed_params, off_value, on_value,
                      n_kphos=N_2PARAM_KPHOS, n_trc=N_2PARAM_TRCHOP):
    kphos_ax = np.linspace(KPHOS_MIN, KPHOS_MAX, n_kphos)
    trchop_ax = np.linspace(TRCHOP_MIN, TRCHOP_MAX, n_trc)
    fate_grid = np.full((n_trc, n_kphos), np.nan)
    total = n_trc * n_kphos
    print(f"    Figure 3 grid: {n_trc}x{n_kphos} = {total} simulations...")
    count = 0
    for i, tc in enumerate(trchop_ax):
        for j, kp in enumerate(kphos_ax):
            r.resetAll()
            _apply_extra(r, extra_fixed_params)
            try:
                r["kphos"], r["trcCHOP"] = float(kp), float(tc)
                res = r.simulate(0, 300, 300, selections=["BAXmT"])
                baxmt = float(res["BAXmT"][-1])
                fate_grid[i, j] = classify_nearest_branch(baxmt, off_value, on_value)
            except Exception:
                fate_grid[i, j] = np.nan
            count += 1
            if count % max(1, total // 4) == 0:
                print(f"      {count}/{total} done")
    return {"kphos_ax": kphos_ax, "trchop_ax": trchop_ax, "fate_grid": fate_grid}


def plot_2param_map(map_result, group_name, threshold_kphos, dynamics_type, save_path=None):
    kphos_ax, trchop_ax, fate_grid = map_result["kphos_ax"], map_result["trchop_ax"], map_result["fate_grid"]

    if dynamics_type == "flat":
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.text(0.5, 0.5, f"{group_name}: FLAT dynamics\n(no OFF/ON branch distinction --\n"
                             f"fate not determinable from this model)",
                  transform=ax.transAxes, ha="center", va="center", fontsize=13, color="darkred")
        ax.set_xlabel("kphos"); ax.set_ylabel("trcCHOP")
        ax.set_title(f"Figure 3 -- {group_name}  [NOT CLASSIFIABLE]", color="darkred")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved -> {save_path}")
        plt.close(fig)
        return

    cmap = mcolors.ListedColormap([C_OFF, C_ON])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.pcolormesh(kphos_ax, trchop_ax, fate_grid, cmap=cmap, norm=norm, shading="auto", alpha=0.75)
    if threshold_kphos is not None:
        ax.axvline(threshold_kphos, color="white", ls=":", lw=2,
                     label=f"Reference fold (kphos={threshold_kphos:.2f})")
        ax.legend(fontsize=8, loc="upper right")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["OFF branch", "ON branch (heading towards apoptosis)"])
    ax.set_xlabel("kphos"); ax.set_ylabel("trcCHOP")
    ax.set_title(f"Figure 3 -- {group_name}  [BAXmT, nearest-branch classification]")
    fig.text(0.5, -0.03,
               "CAVEAT: OFF/ON branch values computed once at this group's fitted trcCHOP; "
               "classification across the grid assumes similar branch values hold elsewhere.",
               ha="center", fontsize=7.5, style="italic", color="dimgray", wrap=True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 -- FIGURE 4: phase portraits (BAXmT nearest-branch coloring)
# ═══════════════════════════════════════════════════════════════════════════
def run_phase_portraits(r, kphos_values, off_value, on_value, group_name,
                           n_ic=N_IC_TRAJ, extra_fixed_params=None):
    portrait_data = {}
    rng = np.random.default_rng(0)
    is_flat = CONFIRMED_DYNAMICS[group_name]["type"] == "flat"
    for kp in kphos_values:
        trajectories = []
        for _ in range(n_ic):
            r.resetAll()
            _apply_extra(r, extra_fixed_params)
            try:
                r["kphos"] = float(kp)
                r["ATF4"] = float(rng.uniform(0.01, ATF4_IC_MAX))
                r["CHOP"] = float(rng.uniform(0.01, CHOP_IC_MAX))
                res = r.simulate(0, PHASE_SIM_T, PHASE_SIM_N,
                                    selections=["time", "ATF4", "CHOP", "BAXmT"])
                final_baxmt = float(res["BAXmT"][-1])
                fate = np.nan if is_flat else classify_nearest_branch(final_baxmt, off_value, on_value)
                trajectories.append({"atf4": res["ATF4"], "chop": res["CHOP"], "fate": fate})
            except Exception:
                pass
        portrait_data[kp] = {"trajectories": trajectories}
    return portrait_data


def plot_phase_portraits(portrait_data, kphos_values, group_name, save_path=None):
    labels = ["A  Sub-threshold", "B  Near bifurcation", "C  Supra-threshold"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"wspace": 0.35})
    is_flat = CONFIRMED_DYNAMICS[group_name]["type"] == "flat"

    for ax, kp, label in zip(axes, kphos_values, labels):
        data = portrait_data.get(kp, {})
        for traj in data.get("trajectories", []):
            if is_flat or np.isnan(traj["fate"]):
                color = "#888888"
            else:
                color = C_ON if traj["fate"] == 1.0 else C_OFF
            ax.plot(traj["atf4"], traj["chop"], color=color, lw=0.9, alpha=0.5)
        diag = np.linspace(0, max(ATF4_IC_MAX, CHOP_IC_MAX), 50)
        ax.plot(diag, diag, "k--", lw=0.8, alpha=0.35)
        ax.set_xlim(0, ATF4_IC_MAX); ax.set_ylim(0, CHOP_IC_MAX)
        ax.set_xlabel("ATF4"); ax.set_ylabel("CHOP")
        ax.set_title(f"{label}\nkphos = {kp:.1f}", fontsize=11)

    if is_flat:
        legend_handles = [mlines.Line2D([0], [0], color="#888888", lw=1.5,
                                           label="Trajectory (no meaningful fate -- flat dynamics)")]
    else:
        legend_handles = [
            mlines.Line2D([0], [0], color=C_ON, lw=1.5, label="Trajectory -> ON (heading towards apoptosis)"),
            mlines.Line2D([0], [0], color=C_OFF, lw=1.5, label="Trajectory -> OFF branch"),
        ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Figure 4 -- Phase Portraits -- {group_name}", fontweight="bold")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 -- FIGURE 5 (unchanged from original replacement): GADD34 vs. stress
# ═══════════════════════════════════════════════════════════════════════════
GROUP_CONFIDENCE = {
    "mutant_myeloid": "HIGH",
    "wildtype_myeloid": "HIGH",
    "mutant_lymphoid": "trcGADD34 NON-IDENTIFIABLE",
    "wildtype_lymphoid": "trcGADD34 NON-IDENTIFIABLE",
}
GROUP_COLORS = {
    "mutant_myeloid": "#C44E52",
    "mutant_lymphoid": "#8172B2",
    "wildtype_myeloid": "#4C72B0",
    "wildtype_lymphoid": "#55A868",
}


def run_gadd34_kphos_sweep(r, group_name: str, params: dict,
                              kphos_range=(KPHOS_MIN, KPHOS_MAX, N_KPHOS_GADD34)) -> dict:
    """Uses ONLY trcCHOP/ktrATF4/trcGADD34 -- independent of the BAX
    pathway. Uses try_steady_state with a LONGER fallback window
    (SS_FALLBACK_T_GADD34) than the pipeline default -- diagnostic
    confirmed r.steadyState() fails for ~90% of GADD34 evaluations across
    all groups, and the default 500-unit fallback undershoots by 10-20%
    in the transition region (kphos~6-10). This slower setting is used
    ONLY here, not globally, to avoid slowing down Figures 1-4."""
    r.resetAll()
    upstream_only = {k: v for k, v in params.items()
                        if k in ("trcCHOP", "ktrATF4", "trcGADD34")}
    _apply_extra(r, upstream_only)

    kphos_vals = np.linspace(*kphos_range)
    gadd34_vals, mgadd34_vals = [], []
    for kp in kphos_vals:
        # FIX: kphos MUST be set directly on r BEFORE calling try_steady_state.
        # try_steady_state only applies param_dict's kphos value INSIDE its
        # fallback branch, AFTER its first r.steadyState() attempt -- so if
        # that first attempt succeeds using whatever kphos was left over
        # from the PREVIOUS iteration (or the SBML default on the first
        # call), the result is silently computed at the WRONG kphos. This
        # was confirmed as the cause of every group's flat-line output --
        # e.g. mutant_myeloid got stuck at kphos~0.5 for the entire sweep.
        r["kphos"] = float(kp)
        ss, method = try_steady_state(r, {"kphos": kp}, extra_fixed_params=upstream_only,
                                          fallback_t=SS_FALLBACK_T_GADD34,
                                          fallback_n=SS_FALLBACK_N_GADD34)
        if ss is not None:
            gadd34_vals.append(ss.get("GADD34", np.nan))
            mgadd34_vals.append(ss.get("mGADD34", np.nan))
        else:
            gadd34_vals.append(np.nan)
            mgadd34_vals.append(np.nan)

    return {"kphos": kphos_vals, "GADD34": np.array(gadd34_vals), "mGADD34": np.array(mgadd34_vals)}


def plot_gadd34_vs_stress(all_group_results: dict, save_path: str = None):
    fig, (ax_protein, ax_mrna) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"wspace": 0.3})
    for group_name, data in all_group_results.items():
        color = GROUP_COLORS[group_name]
        confidence = GROUP_CONFIDENCE[group_name]
        linestyle = "-" if confidence == "HIGH" else "--"
        label = f"{group_name} ({confidence})"
        ax_protein.plot(data["kphos"], data["GADD34"], color=color, lw=2.2, ls=linestyle, label=label)
        ax_mrna.plot(data["kphos"], data["mGADD34"], color=color, lw=2.2, ls=linestyle, label=label)

    ax_protein.set_xlabel("ER Stress Intensity (kphos)")
    ax_protein.set_ylabel("GADD34 (protein, steady state)")
    ax_protein.set_title("A  GADD34 protein vs. ER stress, by group")
    ax_protein.legend(fontsize=8, loc="best")

    ax_mrna.set_xlabel("ER Stress Intensity (kphos)")
    ax_mrna.set_ylabel("mGADD34 (mRNA, steady state)")
    ax_mrna.set_title("B  GADD34 mRNA vs. ER stress, by group")
    ax_mrna.legend(fontsize=8, loc="best")

    fig.suptitle("Figure 5 -- GADD34 Induction Across ER Stress, by Lineage/Genotype\n"
                   "(solid = HIGH confidence params; dashed = non-identifiable trcGADD34)",
                   fontweight="bold")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 -- LOAD FITTED PARAMS
# ═══════════════════════════════════════════════════════════════════════════
def load_fitted_params(filepath="fitted_params.json") -> dict:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"'{filepath}' not found -- run fit_parameters.py first.")
    with open(filepath, "r") as f:
        loaded = json.load(f)
    METADATA_KEYS = {"ks3p_by_lineage", "_ks3p_by_lineage", "confidence_notes",
                       "_confidence_notes", "_notes"}
    group_params = {k: v for k, v in loaded.items() if k not in METADATA_KEYS}
    metadata = {k: v for k, v in loaded.items() if k in METADATA_KEYS}
    print(f"Loaded fitted parameters for {len(group_params)} groups:")
    for group_name, params in group_params.items():
        print(f"  {group_name}: {params}")
    if metadata:
        print(f"\n(Excluded {len(metadata)} metadata key(s): {list(metadata.keys())})")
    return group_params


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # False for full batch runs (all 4 groups x 6 figures = 24 files) --
    # rapid-fire auto-opening was confirmed to cause slow/stuck-loading
    # windows in some viewers. Set True only when testing a single
    # group/figure in isolation. Browse the output folder once this
    # finishes instead.
    AUTO_OPEN_FIGURES = False

    print(f"\n{'='*70}")
    print(f"PRODUCTION RUN -- FAST_MODE={FAST_MODE}")
    if not FAST_MODE:
        print("Full resolution: N_CONTINUATION=200, 50x50 Figure 3 grid, "
              "80-point Figure 5 sweep. This will take substantially longer "
              "than a FAST_MODE run -- expect this to run for a while.")
    print(f"{'='*70}\n")

    r = load_model()
    combined_params_by_group = load_fitted_params("fitted_params.json")

    # Tag every filename with the resolution mode, so FAST_MODE test runs
    # and full-resolution production runs never collide or get confused --
    # confirmed necessary after multiple iterations of this file produced
    # ambiguous same-named outputs.
    run_tag = "FAST" if FAST_MODE else "FINAL"
    print(f"Run tag: {run_tag} -- all filenames will include this suffix.\n")

    all_gadd34_results = {}

    for group_name, params in combined_params_by_group.items():
        print(f"\n{'#'*70}\n# {group_name}\n{'#'*70}")
        upstream_params = {k: v for k, v in params.items()
                              if k in ("trcCHOP", "ktrATF4", "trcGADD34")}
        dynamics = CONFIRMED_DYNAMICS[group_name]

        # --- Figure 1 ---
        print("  [1/6] Figure 1 (ATF4/CHOP + BAXmT)...")
        r.resetAll()
        cont_result = run_continuation(r, extra_fixed_params=upstream_params)
        path1 = f"fig1_{group_name}_{run_tag}.png"
        plot_figure1(cont_result, group_name, save_path=path1)
        if AUTO_OPEN_FIGURES: open_image(path1)
        print(f"  [1/6] DONE")

        # --- Figure 2 ---
        print("  [2/6] Figure 2 (Hopf analysis + time courses)...")
        rep_kphos_fig2 = _choose_representative_kphos(cont_result, dynamics["threshold_kphos"])
        timecourses = run_oscillation_timecourses(r, rep_kphos_fig2, extra_fixed_params=upstream_params)
        path2 = f"fig2_{group_name}_{run_tag}.png"
        plot_figure2(cont_result, timecourses, rep_kphos_fig2, group_name, save_path=path2)
        if AUTO_OPEN_FIGURES: open_image(path2)
        print(f"  [2/6] DONE")

        # --- Branch values (needed for Figures 3 & 4) ---
        print("  [3/6] Computing OFF/ON branch values...")
        off_val, on_val = get_group_branch_values(r, group_name, upstream_params)
        print(f"  [3/6] DONE")

        # --- Figure 3 ---
        print("  [4/6] Figure 3 (2-param map)...")
        extra_for_2param = {k: v for k, v in params.items() if k != "trcCHOP"}
        map_result = run_2param_map(r, extra_for_2param, off_val, on_val)
        path3 = f"fig3_{group_name}_{run_tag}.png"
        plot_2param_map(map_result, group_name, dynamics["threshold_kphos"],
                          dynamics["type"], save_path=path3)
        if AUTO_OPEN_FIGURES: open_image(path3)
        print(f"  [4/6] DONE")

        # --- Figure 4 ---
        print("  [5/6] Figure 4 (phase portraits)...")
        rep_kphos = _choose_representative_kphos(cont_result, dynamics["threshold_kphos"])
        portrait_data = run_phase_portraits(r, rep_kphos, off_val, on_val, group_name,
                                               extra_fixed_params=params)
        path4 = f"fig4_{group_name}_{run_tag}.png"
        plot_phase_portraits(portrait_data, rep_kphos, group_name, save_path=path4)
        if AUTO_OPEN_FIGURES: open_image(path4)
        print(f"  [5/6] DONE")

        # --- Figure 5 data collection (comparison plot drawn once, after all groups) ---
        print(f"  [6/6] Figure 5 data (confidence: {GROUP_CONFIDENCE[group_name]})...")
        all_gadd34_results[group_name] = run_gadd34_kphos_sweep(r, group_name, params)
        print(f"  [6/6] DONE")

        print(f"\n>>> ALL FIGURES COMPLETE FOR {group_name} <<<")

    # --- Figure 5: cross-group comparison ---
    print(f"\n{'='*70}\nFigure 5: GADD34 vs. ER stress -- all groups\n{'='*70}")
    path5 = f"fig5_gadd34_vs_stress_{run_tag}.png"
    plot_gadd34_vs_stress(all_gadd34_results, save_path=path5)
    if AUTO_OPEN_FIGURES: open_image(path5)

    print(f"\nDone [{run_tag}]. Figures 1, 2, 3, 4 generated per group; "
          f"Figure 5 comparison generated once.")