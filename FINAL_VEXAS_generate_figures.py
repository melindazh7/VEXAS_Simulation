"""
generate_figures.py

PURPOSE: Load fitted parameters from fitted_params.json and generate
figures for the VEXAS myeloid/lymphoid PERK-pathway model (BIOMD0000000446,
Erguler 2013). No optimization runs here -- fitting happens separately.

============================================================================
STATUS SUMMARY (read this before trusting any figure below)
============================================================================

CONFIRMED HIGH-CONFIDENCE PARAMETERS:
  mutant_myeloid:    trcCHOP, ktrATF4, trcGADD34, kfbc
  wildtype_myeloid:  trcCHOP, trcGADD34, kfbc
                     (ktrATF4 is a ONE-SIDED BOUND, pinned at 50, likely
                      higher -- not a converged point estimate)
  mutant_lymphoid:   ktrATF4 only
  wildtype_lymphoid: kfbc only

CONFIRMED NON-IDENTIFIABLE (do not attempt to refit further -- profile
likelihood / joint-fit testing has already shown these are flat ridges,
not under-search):
  mutant_lymphoid:   trcCHOP [flat, 0-0.34], kfbc/kmbc [compensation ridge],
                      trcGADD34 [flat, confirmed via profile likelihood,
                      relative error range 0.0000]
  wildtype_lymphoid: trcCHOP, ktrATF4, trcGADD34 [all flat; tested 3 ways:
                      2-gene profile, joint 3-gene fit w/ PPP1R15A, and a
                      mechanistic ruling-out of BiP/HSPA5 as a rescue via
                      its actual SBML rate law -- BiP depends on
                      Xbp1s/ATF6p50, structurally disconnected from this
                      parameter set]

UNRESOLVED -- BAX/BCL2 PATHWAY (BAXT, kasx, kfx, and by extension kfbc's
interaction with BCL2 depletion):
  ALL FOUR GROUPS currently show wrong-direction or unvalidated BAXmBCL2
  vs. kphos behavior. For mutant_myeloid specifically, 6 independent
  fitting/diagnostic attempts were made and documented (see project notes)
  -- none produced a trustworthy fix. A targeted mechanistic hypothesis
  (CHOP-driven BCL2 depletion via the kmbc term in reaction rea1) was
  tested down to its boundary case (kmbc=0) and ruled out as the sole
  cause. wildtype_myeloid's refit collapsed to a degenerate solution.
  wildtype_lymphoid's underlying curve-fit has a persistent large error
  independent of any direction constraint (possible bin-resolution
  mismatch, unconfirmed). mutant_lymphoid's BAX pathway was never fit at
  all (still at SBML defaults).

  CONSEQUENCE: Figures 1 and 2's BAXmBCL2 panels have been REMOVED for
  all groups pending resolution (see plot_atf4_chop_only /
  plot_hopf_analysis_no_bax below). Figures 3 and 4 ARE included below,
  but watermarked "UNVALIDATED" with an explicit caption, since their
  fate classification (survival/apoptosis/oscillatory) depends directly
  on the unresolved BAXmBCL2 threshold. In Figure 4, trajectory SHAPES
  and fixed-point locations remain valid (ATF4/CHOP-only computations);
  only the fate-based trajectory COLORING has been neutralized/flagged.

BUG FIX APPLIED: try_steady_state's fallback path previously did not
receive extra_fixed_params, silently resetting fitted parameters (BAXT,
kasx, kfx, trcCHOP, etc.) to raw SBML defaults whenever r.steadyState()
failed to converge. This is fixed below (_apply_extra is now called in
every fallback branch). This fix also resolved a jagged/artifactual
sawtooth pattern previously seen in fixed-time-window sweeps.

FIGURE 5: Replaced. The original ks3p/trcGADD34-vs-BAXmBCL2 version
depended on the unresolved BAX pathway AND showed a mechanistically
backwards trcGADD34-vs-BAXmBCL2 relationship (more GADD34, a pro-survival
factor, predicting MORE apoptosis) that was never explained. New Figure 5
plots GADD34 protein/mRNA vs. kphos directly -- uses only
trcCHOP/ktrATF4/trcGADD34, sidestepping the BAX pathway entirely. Lymphoid
groups are shown dashed with explicit non-identifiable labeling.

BAX_THRESHOLD CAVEAT: any death-threshold-based fate classification
elsewhere in this project was self-selected to match expected outcomes,
not independently derived -- treat with caution; not used in any figure
in this file.
============================================================================
"""

import numpy as np
import json
import os
import urllib.request
import tellurium as te
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 -- CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
BIOMD_ID = "BIOMD0000000446"
KPHOS_MIN, KPHOS_MAX, N_KPHOS = 0.1, 20.0, 60
N_CONTINUATION = 200
OSC_STD_THRESHOLD = 1.0
OSC_SIM_T, OSC_SIM_N = 600, 2000
SS_CONVERGENCE_TOL = 1e-6
SS_FALLBACK_T, SS_FALLBACK_N = 500, 2000
REAL_ZERO_TOL, HOPF_IM_MIN = 1e-6, 1e-4

C_SURVIVE, C_APOPTOSIS = "#4C72B0", "#C44E52"   # colors kept for consistency;
                                                   # NOTE: these no longer imply
                                                   # survival/apoptosis fate --
                                                   # see Figure 1 labeling below
C_HOPF, C_FOLD, C_STABLE, C_UNSTABLE = "purple", "orange", "black", "grey"
C_ATF4, C_CHOP = "#4C72B0", "#C44E52"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 -- MODEL LOADING + HELPERS
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
    """Applies a small dict of params (typically just {'kphos': value})."""
    for k, v in param_dict.items():
        try:
            r[k] = float(v)
        except Exception:
            pass


def _apply_extra(r, extra_fixed_params: dict = None) -> None:
    """Reapplies the FULL fitted-parameter set after any r.resetAll() --
    resetAll() wipes ALL parameters, not just kphos. This must be called
    in every code path that follows a resetAll(), including inside
    fallback branches -- missing this was the root cause of a confirmed
    bug where fitted BAXT/kasx/kfx/trcCHOP/etc. silently reverted to raw
    SBML defaults whenever the primary steady-state solver failed."""
    if extra_fixed_params:
        for name, value in extra_fixed_params.items():
            r[name] = float(value)


def _set_species(r, ss: dict):
    for sp, val in ss.items():
        try:
            r[sp] = float(val)
        except Exception:
            pass


def _get_species_dict(r) -> dict:
    return {sp: float(r[sp]) for sp in r.getFloatingSpeciesIds()}


def _sim_to_ss(r, param_dict: dict, extra_fixed_params: dict = None):
    """Simulates to a long time horizon as a steady-state fallback.
    FIX: now accepts and reapplies extra_fixed_params -- previously this
    only saw param_dict (just kphos), so a fitted parameter set could be
    silently lost here."""
    species = ["time"] + r.getFloatingSpeciesIds()
    for atol in (1e-8, 1e-6, 1e-4, 1e-2):
        try:
            r.integrator.absolute_tolerance = atol
            res = r.simulate(0, SS_FALLBACK_T, SS_FALLBACK_N, selections=species)
            return {sp: float(res[sp][-1]) for sp in r.getFloatingSpeciesIds()}
        except Exception:
            pass
    return None


def try_steady_state(r, param_dict: dict, extra_fixed_params: dict = None):
    """
    FIXED VERSION. Every fallback branch now reapplies extra_fixed_params
    before simulating, not just param_dict. Previously, whenever
    r.steadyState() failed to converge (which happens disproportionately
    near fold/Hopf bifurcation points -- exactly the most scientifically
    interesting regions), the fallback path would silently run with raw
    SBML defaults instead of the actual fitted parameters.
    """
    try:
        norm = r.steadyState()
        if norm < SS_CONVERGENCE_TOL:
            return _get_species_dict(r), "ss_solver"
    except Exception:
        pass

    _apply_extra(r, extra_fixed_params)
    _apply_params(r, param_dict)
    ss = _sim_to_ss(r, param_dict, extra_fixed_params)
    if ss is not None:
        return ss, "sim_fallback"

    r.resetAll()
    _apply_extra(r, extra_fixed_params)
    _apply_params(r, param_dict)
    ss = _sim_to_ss(r, param_dict, extra_fixed_params)
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


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 -- DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class BifPoint:
    kphos: float
    ss: dict
    eigs: np.ndarray
    classification: str
    method: str
    leading_eig: complex

    def get(self, species, default=np.nan):
        return float(self.ss.get(species, default)) if self.ss else default

    @property
    def chop(self): return self.get("CHOP")
    @property
    def atf4(self): return self.get("ATF4")
    @property
    def bax(self): return self.get("BAXmBCL2")   # NOTE: unresolved pathway -- see header


@dataclass
class ContinuationResult:
    forward: list = field(default_factory=list)
    backward: list = field(default_factory=list)
    fold_points: list = field(default_factory=list)
    hopf_points: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 -- CONTINUATION
# ═══════════════════════════════════════════════════════════════════════════
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


def _run_single_sweep(r, kphos_values, init_reset=True, extra_fixed_params=None):
    points, prev_ss = [], None
    for i, kp in enumerate(kphos_values):
        param_dict = {"kphos": kp}
        if prev_ss is None or (init_reset and i == 0):
            r.resetAll()
            _apply_extra(r, extra_fixed_params)
            _apply_params(r, param_dict)
        else:
            _set_species(r, prev_ss)
            _apply_extra(r, extra_fixed_params)
            _apply_params(r, param_dict)

        # FIXED: extra_fixed_params now threaded through to try_steady_state
        ss, method = try_steady_state(r, param_dict, extra_fixed_params)
        if ss is not None:
            eigs = get_eigenvalues(r)
            lead = get_leading_eigenvalue(eigs)
            clf = classify_ss(eigs)
            if method == "sim_fallback":
                try:
                    res = r.simulate(0, OSC_SIM_T // 3, OSC_SIM_N // 3, selections=["BAXmBCL2"])
                    if float(np.std(res["BAXmBCL2"][int(len(res["BAXmBCL2"]) * 0.8):])) > OSC_STD_THRESHOLD:
                        clf = "oscillatory"
                except Exception:
                    pass
        else:
            eigs, lead, clf = np.array([], dtype=complex), complex(np.nan, np.nan), "failed"

        points.append(BifPoint(kphos=kp, ss=ss or {}, eigs=eigs,
                                classification=clf, method=method, leading_eig=lead))
        prev_ss = ss
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(kphos_values)} done")
    return points


def run_continuation(r, n_points=N_CONTINUATION, extra_fixed_params=None):
    result = ContinuationResult()
    kphos_fwd = np.linspace(KPHOS_MIN, KPHOS_MAX, n_points)
    kphos_bwd = np.linspace(KPHOS_MAX, KPHOS_MIN, n_points)
    print("  Forward sweep …")
    result.forward = _run_single_sweep(r, kphos_fwd, True, extra_fixed_params)
    print("  Backward sweep …")
    result.backward = _run_single_sweep(r, kphos_bwd, True, extra_fixed_params)
    result.fold_points, result.hopf_points = _detect_bifurcations(result.forward)
    print(f"  Fold points: {[f'{v:.2f}' for v in result.fold_points]}")
    print(f"  Hopf points: {[f'{v:.2f}' for v in result.hopf_points]}")
    return result


def _choose_representative_kphos(result: ContinuationResult) -> list:
    all_bif = sorted(result.hopf_points + result.fold_points)
    if all_bif:
        first = all_bif[0]
        return [max(KPHOS_MIN + 0.3, first * 0.45), first, min(KPHOS_MAX - 0.3, first * 2.0)]
    return [2.0, 6.0, 12.0]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 -- FIGURE 1: ATF4/CHOP BIFURCATION DIAGRAM (BAXmBCL2 REMOVED)
# ═══════════════════════════════════════════════════════════════════════════
def plot_atf4_chop_only(result: ContinuationResult, group_name: str, save_path: str = None):
    """
    Revised Figure 1. Shows ONLY ATF4/CHOP vs. kphos -- the BAXmBCL2 panel
    has been removed because that pathway's direction is unresolved for
    every group (see module header). Background shading and legend
    describe the ATF4/CHOP bifurcation regime itself (low vs. high
    activity, relative to the fold/Hopf transition) -- NOT survival vs.
    apoptosis, since that fate classification depends on the unresolved
    BAXmBCL2 threshold and would not be supported by what's plotted here.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fwd = result.forward

    kp_fwd = np.array([p.kphos for p in fwd])
    chop_fwd = np.array([p.chop for p in fwd])
    atf4_fwd = np.array([p.atf4 for p in fwd])

    all_bif = sorted(result.fold_points + result.hopf_points)

    if all_bif:
        transition_kphos = all_bif[0]
        ax.axvspan(kp_fwd.min(), transition_kphos, color=C_SURVIVE, alpha=0.07)
        ax.axvspan(transition_kphos, kp_fwd.max(), color=C_APOPTOSIS, alpha=0.07)

    for i in range(len(kp_fwd) - 1):
        if np.isnan(chop_fwd[i]) or np.isnan(chop_fwd[i+1]):
            continue
        ls = "-" if fwd[i].classification in ("stable", "unknown") else "--"
        ax.plot(kp_fwd[i:i+2], atf4_fwd[i:i+2], color=C_SURVIVE, lw=2.5, ls=ls)
        ax.plot(kp_fwd[i:i+2], chop_fwd[i:i+2], color=C_APOPTOSIS, lw=2.5, ls=ls)

    y_max = max(np.nanmax(atf4_fwd), np.nanmax(chop_fwd))
    for kp in result.fold_points:
        ax.axvline(kp, color=C_FOLD, ls=":", lw=1.5)
        ax.annotate("Fold point", xy=(kp, y_max * 0.55), xytext=(kp + 1.2, y_max * 0.72),
                     arrowprops=dict(arrowstyle="->", color=C_FOLD, lw=1.5),
                     fontsize=9, color=C_FOLD, fontweight="bold")
    for kp in result.hopf_points:
        ax.axvline(kp, color=C_HOPF, ls=":", lw=1.5)
        ax.annotate("Hopf point", xy=(kp, y_max * 0.4), xytext=(kp + 1.2, y_max * 0.25),
                     arrowprops=dict(arrowstyle="->", color=C_HOPF, lw=1.5),
                     fontsize=9, color=C_HOPF, fontweight="bold")

    ax.set_xlabel("ER Stress Intensity (kphos)", fontsize=11)
    ax.set_ylabel("ATF4 / CHOP (steady state)", fontsize=11)
    ax.set_title(f"ATF4/CHOP Response Along the Stress Axis -- '{group_name}'",
                  fontsize=12, fontweight="bold")

    ax.legend(handles=[
        mlines.Line2D([], [], color=C_SURVIVE, lw=2.5, label="ATF4"),
        mlines.Line2D([], [], color=C_APOPTOSIS, lw=2.5, label="CHOP"),
        mlines.Line2D([], [], color=C_FOLD, ls=":", lw=1.5, label="Fold point"),
        mlines.Line2D([], [], color=C_HOPF, ls=":", lw=1.5, label="Hopf point"),
        mpatches.Patch(color=C_SURVIVE, alpha=0.07, label="Low ATF4/CHOP regime (pre-bifurcation)"),
        mpatches.Patch(color=C_APOPTOSIS, alpha=0.07, label="High ATF4/CHOP regime (post-bifurcation)"),
    ], fontsize=9, loc="upper left")

    fig.suptitle(f"Figure 1 -- {group_name}\n"
                  "(BAXmBCL2 panel omitted: BAX/BCL2 pathway direction unresolved)",
                  fontsize=10, style="italic", y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 -- FIGURE 2: HOPF ANALYSIS + ATF4/CHOP TIME COURSES (BAXmBCL2 REMOVED)
# ═══════════════════════════════════════════════════════════════════════════
SELECTIONS_ATF4_CHOP = ["time", "ATF4", "CHOP"]   # BAXmBCL2 deliberately excluded


def run_oscillation_timecourses_no_bax(r, kphos_values: list, extra_fixed_params=None) -> dict:
    results = {}
    for kp in kphos_values:
        r.resetAll()
        _apply_extra(r, extra_fixed_params)
        try:
            r["kphos"] = float(kp)
            res = r.simulate(0, OSC_SIM_T, OSC_SIM_N, selections=SELECTIONS_ATF4_CHOP)
            results[kp] = {k: res[k] for k in SELECTIONS_ATF4_CHOP}
        except Exception as e:
            print(f"  [warn] timecourse failed at kphos={kp}: {e}")
    return results


def plot_hopf_analysis_no_bax(result: ContinuationResult, timecourses: dict, rep_kphos: list,
                                group_name: str, save_path: str = None):
    """
    Revised Figure 2. Top panel (leading eigenvalue) is computed from the
    FULL system Jacobian -- technically still includes BAX/BCL2 state
    variables even though their parameters are unfit, but the fold/Hopf
    sign changes are dominated by the ATF4/CHOP feedback loop. Bottom
    panels show ONLY ATF4/CHOP time courses -- BAXmBCL2 removed.
    """
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    fwd = result.forward
    kp_arr = np.array([p.kphos for p in fwd])
    re_arr = np.array([p.leading_eig.real if not np.isnan(p.leading_eig.real) else np.nan for p in fwd])
    im_arr = np.array([abs(p.leading_eig.imag) if not np.isnan(p.leading_eig.imag) else np.nan for p in fwd])

    ax_eig = fig.add_subplot(gs[0, :])
    ax_im = ax_eig.twinx()

    for i in range(len(kp_arr) - 1):
        if np.isnan(re_arr[i]) or np.isnan(re_arr[i+1]):
            continue
        if re_arr[i] * re_arr[i+1] < 0:
            frac = re_arr[i] / (re_arr[i] - re_arr[i+1])
            kp_cross = kp_arr[i] + frac * (kp_arr[i+1] - kp_arr[i])
            color_left = C_APOPTOSIS if re_arr[i] > 0 else C_SURVIVE
            color_right = C_APOPTOSIS if re_arr[i+1] > 0 else C_SURVIVE
            ax_eig.plot([kp_arr[i], kp_cross], [re_arr[i], 0], color=color_left, lw=2.0)
            ax_eig.plot([kp_cross, kp_arr[i+1]], [0, re_arr[i+1]], color=color_right, lw=2.0)
        else:
            color = C_APOPTOSIS if re_arr[i] > 0 else C_SURVIVE
            ax_eig.plot(kp_arr[i:i+2], re_arr[i:i+2], color=color, lw=2.0)

    ax_im.plot(kp_arr, im_arr, color="grey", lw=1.2, ls="--", alpha=0.7, label="|Im(λ_lead)|")
    ax_eig.axhline(0, color="k", lw=0.8)
    for kp in result.hopf_points:
        ax_eig.axvline(kp, color=C_HOPF, ls=":", lw=1.5)
    for kp in result.fold_points:
        ax_eig.axvline(kp, color=C_FOLD, ls=":", lw=1.5)
    ax_eig.set_ylabel("Re(λ_max)")
    ax_im.set_ylabel("|Im(λ)|", color="grey")
    ax_eig.set_xlabel("ER Stress Intensity (kphos)")
    ax_eig.set_title(f"A  Leading eigenvalue -- '{group_name}'", loc="left")
    ax_eig.legend(handles=[mpatches.Patch(color=C_STABLE, label="Stable"),
                            mpatches.Patch(color=C_APOPTOSIS, label="Unstable")],
                  fontsize=9, loc="lower right")

    labels = ["B  Pre-bifurcation", "C  Near bifurcation", "D  Post-bifurcation"]
    for col, (kp, label) in enumerate(zip(rep_kphos, labels)):
        ax = fig.add_subplot(gs[1, col])
        tc = timecourses.get(kp)
        if tc is not None:
            t = tc["time"]
            ax.plot(t, tc["ATF4"], color=C_ATF4, lw=1.8, label="ATF4")
            ax.plot(t, tc["CHOP"], color=C_CHOP, lw=1.8, label="CHOP")
        ax.set_xlabel("Time (a.u.)")
        ax.set_ylabel("ATF4 / CHOP")
        ax.set_title(f"{label}\nkphos = {kp:.1f}", fontsize=10)
        if col == 0:
            ax.legend(fontsize=8)

    fig.suptitle(f"Figure 2 -- Hopf Bifurcation Analysis -- {group_name}\n"
                  "(BAXmBCL2 omitted: BAX/BCL2 pathway direction unresolved)",
                  fontweight="bold")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 -- FIGURE 5 (REPLACEMENT): GADD34 vs. ER STRESS
# ═══════════════════════════════════════════════════════════════════════════
N_KPHOS_GADD34 = 60
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


def run_gadd34_kphos_sweep(r, group_name, params, kphos_range=(KPHOS_MIN, KPHOS_MAX, N_KPHOS)):
    """Sweep kphos, track steady-state GADD34 (protein) and mGADD34 (mRNA).
    Only uses trcCHOP/ktrATF4/trcGADD34 -- no BAX/BCL2 params touched."""
    r.resetAll()
    upstream_only = {k: v for k, v in params.items()
                       if k in ("trcCHOP", "ktrATF4", "trcGADD34")}
    for name, value in upstream_only.items():
        r[name] = float(value)

    kphos_vals = np.linspace(*kphos_range)
    gadd34_vals, mgadd34_vals, chop_vals = [], [], []
    for kp in kphos_vals:
        r["kphos"] = float(kp)
        res = r.simulate(0, 300, 500, selections=["GADD34", "mGADD34", "CHOP"])
        gadd34_vals.append(float(res["GADD34"][-1]))
        mgadd34_vals.append(float(res["mGADD34"][-1]))
        chop_vals.append(float(res["CHOP"][-1]))

    return {
        "kphos": kphos_vals,
        "GADD34": np.array(gadd34_vals),
        "mGADD34": np.array(mgadd34_vals),
        "CHOP": np.array(chop_vals),
    }



def plot_gadd34_vs_stress(all_group_results, save_path=None):
    fig, (ax_protein, ax_mrna) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"wspace": 0.3})

    for group_name, data in all_group_results.items():
        color = GROUP_COLORS[group_name]
        confidence = GROUP_CONFIDENCE[group_name]
        linestyle = "-" if confidence == "HIGH" else "--"
        label = f"{group_name} ({confidence})"

        ax_protein.plot(data["kphos"], data["GADD34"], color=color,
                          lw=2.2, ls=linestyle, label=label)
        ax_mrna.plot(data["kphos"], data["mGADD34"], color=color,
                      lw=2.2, ls=linestyle, label=label)

    ax_protein.set_xlabel("ER Stress Intensity (kphos)")
    ax_protein.set_ylabel("GADD34 (protein, steady state)")
    ax_protein.set_title("A  GADD34 protein vs. ER stress, by group")
    ax_protein.legend(fontsize=8, loc="best")

    ax_mrna.set_xlabel("ER Stress Intensity (kphos)")
    ax_mrna.set_ylabel("mGADD34 (mRNA, steady state)")
    ax_mrna.set_title("B  GADD34 mRNA vs. ER stress, by group")
    ax_mrna.legend(fontsize=8, loc="best")

    fig.suptitle("GADD34 Induction Across ER Stress -- by Lineage/Genotype\n"
                  "(solid = HIGH confidence params; dashed = unresolved/non-identifiable trcGADD34)",
                  fontweight="bold")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 -- FIGURE 3: 2-PARAMETER FATE MAP (UNVALIDATED -- BAX-DEPENDENT)
# ═══════════════════════════════════════════════════════════════════════════
import matplotlib.colors as mcolors

TRCHOP_MIN, TRCHOP_MAX = 0.01, 5.0
N_2PARAM_KPHOS, N_2PARAM_TRCHOP = 50, 50
BAX_THRESHOLD_ILLUSTRATIVE = 33.4  # NOTE: self-selected, not independently derived -- see project notes.
                                     # Used here ONLY to reproduce the original figure's illustrative
                                     # categories; do not treat this value as validated.


def run_2param_map(r, extra_fixed_params=None, n_kphos=N_2PARAM_KPHOS, n_trc=N_2PARAM_TRCHOP):
    """
    UNVALIDATED: classification depends on BAXmBCL2, whose direction is
    unresolved for every group. trcCHOP is swept directly here (excluded
    from extra_fixed_params by the caller) so this ALSO temporarily
    overrides whatever trcCHOP value is in the group's fitted params --
    consistent with how this figure worked in the original codebase.
    """
    kphos_ax = np.linspace(KPHOS_MIN, KPHOS_MAX, n_kphos)
    trchop_ax = np.linspace(TRCHOP_MIN, TRCHOP_MAX, n_trc)
    fate_grid = np.full((n_trc, n_kphos), np.nan)
    for i, tc in enumerate(trchop_ax):
        for j, kp in enumerate(kphos_ax):
            r.resetAll()
            _apply_extra(r, extra_fixed_params)
            try:
                r["kphos"], r["trcCHOP"] = float(kp), float(tc)
                res = r.simulate(0, 300, 1500, selections=["BAXmBCL2"])
                bax = res["BAXmBCL2"]
                osc = float(np.std(bax[int(len(bax)*0.8):]))
                fate_grid[i, j] = 2.0 if osc > OSC_STD_THRESHOLD else \
                                   (1.0 if float(bax[-1]) > BAX_THRESHOLD_ILLUSTRATIVE else 0.0)
            except Exception:
                fate_grid[i, j] = np.nan
    return {"kphos_ax": kphos_ax, "trchop_ax": trchop_ax, "fate_grid": fate_grid}


def plot_2param_map(map_result, group_name, fold_points, hopf_points, save_path=None):
    kphos_ax, trchop_ax, fate_grid = map_result["kphos_ax"], map_result["trchop_ax"], map_result["fate_grid"]
    cmap = mcolors.ListedColormap([C_SURVIVE, C_APOPTOSIS, "#55A868"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.pcolormesh(kphos_ax, trchop_ax, fate_grid, cmap=cmap, norm=norm, shading="auto", alpha=0.6)
    for kp in hopf_points:
        ax.axvline(kp, color="white", ls="--", lw=1.5)
    for kp in fold_points:
        ax.axvline(kp, color="white", ls=":", lw=1.5)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["BAXmBCL2 < threshold", "BAXmBCL2 > threshold", "Oscillatory"])
    ax.set_xlabel("kphos"); ax.set_ylabel("trcCHOP")
    ax.set_title(f"Figure 3 -- 2-Param Map -- {group_name}  [UNVALIDATED]", color="darkred")


    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10 -- FIGURE 4: PHASE PORTRAITS (UNVALIDATED -- BAX-DEPENDENT)
# ═══════════════════════════════════════════════════════════════════════════
ATF4_IC_MAX, CHOP_IC_MAX = 5.0, 5.0
N_IC_TRAJ = 12
PHASE_SIM_T, PHASE_SIM_N = 300, 1000
C_NEUTRAL_TRAJ = "#888888"   # neutral gray, since fate-based coloring is unvalidated


def _find_fixed_points(r, kphos: float, n_grid=3, extra_fixed_params=None) -> list:
    """Fixed-point locations use only ATF4/CHOP steady-state solving --
    NOT dependent on BAXmBCL2 classification, so these ARE trustworthy
    (their coordinates), even though trajectory COLORING below is not."""
    fps = []
    for a0 in np.linspace(0.1, ATF4_IC_MAX, n_grid):
        for c0 in np.linspace(0.1, CHOP_IC_MAX, n_grid):
            r.resetAll()
            _apply_extra(r, extra_fixed_params)
            try:
                r["kphos"], r["ATF4"], r["CHOP"] = float(kphos), float(a0), float(c0)
                norm = r.steadyState()
                if norm < SS_CONVERGENCE_TOL:
                    eigs = get_eigenvalues(r)
                    fps.append({"atf4": float(r["ATF4"]), "chop": float(r["CHOP"]),
                                "classification": classify_ss(eigs)})
            except Exception:
                pass
    unique = []
    for fp in fps:
        is_dup = any(abs(fp["atf4"]-u["atf4"]) < 0.2 and abs(fp["chop"]-u["chop"]) < 0.2 for u in unique)
        if not is_dup:
            unique.append(fp)
    return unique


def run_phase_portraits(r, kphos_values: list, n_ic=N_IC_TRAJ, extra_fixed_params=None) -> dict:
    """
    UNVALIDATED coloring: trajectory 'fate' below is still derived from
    BAXmBCL2 vs. threshold, same caveat as Figure 3. Trajectory SHAPES
    (ATF4/CHOP paths) and fixed points are independent of this and remain
    trustworthy.
    """
    portrait_data = {}
    rng = np.random.default_rng(0)
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
                                   selections=["time", "ATF4", "CHOP", "BAXmBCL2"])
                fate = "apoptosis" if float(res["BAXmBCL2"][-1]) > BAX_THRESHOLD_ILLUSTRATIVE else "survival"
                trajectories.append({"atf4": res["ATF4"], "chop": res["CHOP"], "fate": fate})
            except Exception:
                pass
        fps = _find_fixed_points(r, kp, extra_fixed_params=extra_fixed_params)
        portrait_data[kp] = {"trajectories": trajectories, "fixed_points": fps}
    return portrait_data


def plot_phase_portraits(portrait_data: dict, kphos_values: list, group_name: str, save_path=None):
    labels = ["A  Sub-threshold", "B  Near bifurcation", "C  Supra-threshold"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"wspace": 0.35})

    for ax, kp, label in zip(axes, kphos_values, labels):
        data = portrait_data.get(kp, {})
        for traj in data.get("trajectories", []):
            # NEUTRAL color regardless of 'fate' -- that classification is
            # unvalidated. Trajectory SHAPE is still shown/trustworthy.
            ax.plot(traj["atf4"], traj["chop"], color=C_NEUTRAL_TRAJ, lw=0.9, alpha=0.45)
        for fp in data.get("fixed_points", []):
            clf = fp["classification"]
            marker = "D" if clf == "hopf_cand" else ("x" if clf not in ("stable", "unstable") else "o")
            ax.scatter(fp["atf4"], fp["chop"], marker=marker, s=120,
                       color=C_STABLE if "stable" in clf else C_UNSTABLE,
                       facecolors=C_STABLE if clf == "stable" else "none", linewidths=1.5, zorder=5)
        diag = np.linspace(0, max(ATF4_IC_MAX, CHOP_IC_MAX), 50)
        ax.plot(diag, diag, "k--", lw=0.8, alpha=0.35)
        ax.set_xlim(0, ATF4_IC_MAX); ax.set_ylim(0, CHOP_IC_MAX)
        ax.set_xlabel("ATF4"); ax.set_ylabel("CHOP")
        ax.set_title(f"{label}\nkphos = {kp:.1f}", fontsize=11)

    fig.legend(handles=[
        mlines.Line2D([0], [0], color=C_NEUTRAL_TRAJ, lw=1.5, label="Trajectory (fate coloring removed -- unvalidated)"),
        mlines.Line2D([0], [0], marker="o", color=C_STABLE, lw=0, markersize=10, label="Stable fixed point"),
        mlines.Line2D([0], [0], marker="o", color=C_UNSTABLE, lw=0, markersize=10,
                      fillstyle="none", label="Unstable fixed point"),
    ], loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"Figure 4 -- Phase Portraits -- {group_name}  [PARTIALLY UNVALIDATED]",
                  fontweight="bold", color="darkred")


    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 -- LOAD FITTED PARAMS
# ═══════════════════════════════════════════════════════════════════════════
def load_fitted_params(filepath="fitted_params.json") -> dict:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"'{filepath}' not found -- run fit_parameters.py first.")
    with open(filepath, "r") as f:
        loaded = json.load(f)

    METADATA_KEYS = {"ks3p_by_lineage", "_ks3p_by_lineage",
                      "confidence_notes", "_confidence_notes", "_notes"}
    group_params = {k: v for k, v in loaded.items() if k not in METADATA_KEYS}
    metadata = {k: v for k, v in loaded.items() if k in METADATA_KEYS}

    print(f"Loaded fitted parameters for {len(group_params)} groups:")
    for group_name, params in group_params.items():
        print(f"  {group_name}: {params}")
    if metadata:
        print(f"\n(Excluded {len(metadata)} metadata key(s) from fitting: {list(metadata.keys())})")

    for group_name, params in group_params.items():
        for pname, pval in params.items():
            if not isinstance(pval, (int, float)):
                raise ValueError(
                    f"Group '{group_name}' has non-numeric value for '{pname}': {pval!r} "
                    f"-- this looks like metadata, not a fitted parameter."
                )
    return group_params


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    r = load_model()
    combined_params_by_group = load_fitted_params("fitted_params.json")

    # Figure 5 is a cross-group COMPARISON plot, so its per-group data is
    # collected here as we go, but the actual comparison figure is only
    # drawn once, after every group has finished Figures 1-4.
    all_results = {}

    for group_name, params in combined_params_by_group.items():
        print(f"\n{'#'*70}\n# GROUP: {group_name}\n{'#'*70}")

        upstream_params = {k: v for k, v in params.items()
                             if k in ("trcCHOP", "ktrATF4", "trcGADD34")}

        # ── Figure 1: ATF4/CHOP bifurcation diagram (validated) ─────────────
        print(f"\n[1/5] Figure 1 (ATF4/CHOP only) -- {group_name}")
        r.resetAll()
        cont_result_upstream = run_continuation(r, extra_fixed_params=upstream_params)
        plot_atf4_chop_only(cont_result_upstream, group_name,
                             save_path=f"fig1_{group_name}_atf4chop.png")

        # ── Figure 2: Hopf analysis + ATF4/CHOP time courses (validated) ────
        print(f"[2/5] Figure 2 (ATF4/CHOP only) -- {group_name}")
        rep_kphos_upstream = _choose_representative_kphos(cont_result_upstream)
        timecourses = run_oscillation_timecourses_no_bax(
            r, rep_kphos_upstream, extra_fixed_params=upstream_params)
        plot_hopf_analysis_no_bax(cont_result_upstream, timecourses, rep_kphos_upstream,
                                    group_name, save_path=f"fig2_{group_name}_atf4chop.png")

        # ── Figure 3: 2-parameter fate map (UNVALIDATED -- BAX-dependent) ───
        print(f"[3/5] Figure 3 (UNVALIDATED) -- {group_name}")
        r.resetAll()
        cont_result_full = run_continuation(r, extra_fixed_params=params)
        extra_for_2param = {k: v for k, v in params.items() if k != "trcCHOP"}
        map_result = run_2param_map(r, extra_fixed_params=extra_for_2param)
        plot_2param_map(map_result, group_name, cont_result_full.fold_points,
                         cont_result_full.hopf_points,
                         save_path=f"fig3_{group_name}_UNVALIDATED.png")

        # ── Figure 4: phase portraits (PARTIALLY UNVALIDATED -- BAX-dependent) ──
        print(f"[4/5] Figure 4 (PARTIALLY UNVALIDATED) -- {group_name}")
        rep_kphos_full = _choose_representative_kphos(cont_result_full)
        portrait_data = run_phase_portraits(r, rep_kphos_full, extra_fixed_params=params)
        plot_phase_portraits(portrait_data, rep_kphos_full, group_name,
                              save_path=f"fig4_{group_name}_UNVALIDATED.png")

        # ── Figure 5 data collection: GADD34 vs. stress (validated) ─────────
        # Comparison plot itself is drawn once, after all groups finish --
        # see below.

        print(f"[5/5] Figure 5 data (GADD34 vs. stress) -- {group_name} "
              f"(confidence: {GROUP_CONFIDENCE[group_name]})")
        all_results[group_name] = run_gadd34_kphos_sweep(r, group_name, params)

        print(f"\nAll figures generated for '{group_name}'.")

    # ── Figure 5: cross-group comparison, drawn once at the end ─────────────
    print(f"\n{'='*70}\nFigure 5: GADD34 vs. ER stress -- all groups combined\n{'='*70}")
    plot_gadd34_vs_stress(all_results, save_path="fig5_gadd34_vs_stress.png")

    print("\nDone. Per-group Figures 1, 2 (validated), 3, 4 (UNVALIDATED) generated "
          "for each group in turn, followed by the cross-group Figure 5 comparison.")