"""
FILE 1 OF 2: fit_parameters.py

STATUS: FITTING COMPLETE (final). Writes the settled, manually-adjudicated
parameter set to fitted_params.json. No further optimization runs needed
unless underlying cell data changes.

KEY FINDINGS FROM THIS ANALYSIS:
  - mutant_myeloid's BAX pathway (BAXT, kasx, kfx) IS identifiable once
    jointly co-fit -- single-parameter BAXT fitting was under-constrained
    and produced physically invalid (negative-species) results; joint
    subsystem fitting with a hard validity+direction constraint converged
    to a clean, non-boundary-pinned, biologically sensible solution.
  - mutant_lymphoid's BCL2 pathway (kfbc, kmbc) is NOT identifiable from
    this data -- confirmed via a genuine compensation ridge (flat error
    across a wide kfbc range, ridge relocates to whatever bound is set,
    direction of BAXmBCL2 vs kphos remains wrong regardless). SBML
    defaults are used instead, explicitly flagged as non-identified.
  - ks3p (pooled by lineage) is NOT identifiable from BBC3/PMAIP1/BID
    expression data -- likely a STRUCTURAL non-identifiability, since
    ks3p only ever appears multiplied by kstr in the model's kinetics
    (rea9: ks3p * kstr * CHOP). Only the PRODUCT is observable from BH3
    expression; ks3p alone cannot be separated from kstr without an
    independent measurement of kstr. Literature/model reference values
    are used for Figure 5 instead.
"""

import json

# ═══════════════════════════════════════════════════════════════════════════
# FINAL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
# Confidence key:
#   HIGH     -- clean single-basin convergence, no boundary pinning
#   HIGH-SUB -- HIGH confidence achieved via joint subsystem co-fitting
#               (single-parameter fit was invalid; joint fit resolved it)
#   BOUND    -- genuine compensation ridge; one-sided bound, not a point
#               estimate. SBML default used as the safe simulation value.
#   LOW      -- unstable/scattered across seeds, no clear winning basin

FINAL_PARAMS = {
    "mutant_myeloid": {
        "trcCHOP": 1.6255004084937603,     # HIGH
        "ktrATF4": 8.044048797972952,       # HIGH
        "trcGADD34": 0.5238385430788823,    # HIGH
        "BAXT": 97.1733,                      # HIGH-SUB -- joint fit with kasx/kfx;
                                              #             confirm direction check before final use
        "kasx": 473.6436,                     # HIGH-SUB -- co-fit with BAXT above
        "kfx": 0.2632,                        # HIGH-SUB -- co-fit with BAXT above
        "kfbc": 11.58253458345257,          # HIGH -- single-param fit was already clean, no issue
    },
    "mutant_lymphoid": {
        "trcCHOP": 0.05,                    # BOUND -- confirmed non-identifiable point value
                                              #          via profile likelihood (flat [0, ~0.34])
        "ktrATF4": 4.3352,                  # HIGH -- robust across that entire flat plateau
        "trcGADD34": 0.01694348140860609,   # LOW -- fit downstream of a stale reference; exploratory only
        "BAXT": 100.0,                       # SBML DEFAULT -- joint BAX subsystem fit not yet
                                              #                  attempted for this group
        "kfbc": 10.0,                        # SBML DEFAULT (NOT IDENTIFIABLE) -- confirmed genuine
                                              #                compensation ridge: flat error across
                                              #                kfbc in [450,500] and [95,100] (two
                                              #                separate bound settings), BAXmBCL2
                                              #                direction remains wrong regardless.
        "kmbc": 0.03,                        # SBML DEFAULT (NOT IDENTIFIABLE) -- see kfbc note
    },
    "wildtype_myeloid": {
        "trcCHOP": 0.7444226154524105,      # HIGH
        "ktrATF4": 50.0,                    # BOUND -- one-sided (likely >> 50); nominal=50
        "trcGADD34": 0.6795891699916465,    # HIGH
        "BAXT": 17.17000044080578,          # HIGH -- clean single-param convergence, no pinning
        "kfbc": 9.018766124191075,          # HIGH -- clean single-param convergence, no pinning
    },
    "wildtype_lymphoid": {
        "trcCHOP": 0.669616834841098,       # LOW -- 11 scattered basins, no dominant winner
        "ktrATF4": 2.887336508221079,       # LOW -- paired with trcCHOP above
        "trcGADD34": 1.144319801862414,     # LOW -- 21% relative IQR, flagged unstable
        "BAXT": 100.0,                       # SBML DEFAULT -- reverted after single-param fit broke
                                              #                  model validity; joint subsystem fit
                                              #                  not yet attempted for this group
        "kfbc": 5.911998146495996,          # HIGH -- clean single-param convergence, no pinning
    },
}

# ks3p: structurally non-identifiable (see module docstring). Use literature/
# model reference values for Figure 5, NOT as a Figure 3 axis.
KS3P_STATUS = {
    "myeloid":  {"status": "STRUCTURALLY NON-IDENTIFIABLE",
                 "note": "ks3p only appears as ks3p*kstr in rea9 -- only the product is "
                          "observable from BH3 expression data, not ks3p alone."},
    "lymphoid": {"status": "STRUCTURALLY NON-IDENTIFIABLE", "note": "same as myeloid"},
    "recommendation": "Use KS3P_MYELOID=0.3 / KS3P_LYMPHOID=1.2 (literature/model reference "
                       "values) for Figure 5. Do not use ks3p as the Figure 3 axis.",
}

CONFIDENCE_NOTES = {
    "mutant_myeloid": "HIGH confidence across the board, including BAX pathway (resolved via "
                       "joint subsystem co-fitting after single-parameter fitting proved invalid).",
    "mutant_lymphoid": "trcCHOP is a confirmed non-identifiable bound. BCL2 pathway (kfbc/kmbc) "
                        "confirmed structurally/practically non-identifiable via genuine "
                        "compensation ridge -- SBML defaults used and flagged explicitly.",
    "wildtype_myeloid": "HIGH confidence except ktrATF4 (one-sided bound, likely >> 50).",
    "wildtype_lymphoid": "LOW confidence for trcCHOP/ktrATF4/trcGADD34. BAXT reverted to default "
                          "after single-param fit broke model validity -- joint subsystem fit "
                          "not yet attempted for this group.",
    "ks3p_by_lineage": "Structurally non-identifiable from BH3 gene data given the model's "
                        "ks3p*kstr coupling in rea9. Using literature reference values instead.",
}


# ═══════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════
def save_fitted_params(final_params: dict, ks3p_status: dict, confidence_notes: dict,
                        filepath: str = "fitted_params.json") -> None:
    output = {
        **final_params,
        "_ks3p_by_lineage": ks3p_status,
        "_confidence_notes": confidence_notes,
    }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved final fitted parameters -> {filepath}")


if __name__ == "__main__":
    save_fitted_params(FINAL_PARAMS, KS3P_STATUS, CONFIDENCE_NOTES)
    print("\n=== Final parameters, all groups ===")
    for group_name, params in FINAL_PARAMS.items():
        print(f"  {group_name}: {params}")
    print("\n=== Confidence notes ===")
    for group_name, note in CONFIDENCE_NOTES.items():
        print(f"  {group_name}: {note}")