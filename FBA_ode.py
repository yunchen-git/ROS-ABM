################################################################################
# File: FBA_ode.py
# Purpose:
#   Intracellular ODE model for the biochemical state variables used by agents.
#   Contains parameter builders (k, x0) and ODE right-hand-sides.
#   Current model includes redox (H2O2 handling and antioxidant systems) and
#   an extended block for Ca handling / NOX activation using Hill-type gates.
#
# Key public API:
#   - get_k(): returns parameter array
#   - get_x0(): returns baseline/initial state array
#   - model(...), model_noref(...): ODE RHS used by solvers
#   - blap_ode_sys(...): alternative/system RHS used in some coupling contexts
#
# How it fits in the codebase:
#   Used by simulationObjects.StemCell (intracellular updates).
#   Also imported by IntC_diff.pyx for coupled transport + reaction updates.
#
# Inputs/Outputs:
#   - No file I/O. Pure numerical model used by SciPy integrators.
#
# Notes / gotchas:
#   - This file is a major “coupling point” for biology assumptions. Future
#     abstraction (e.g., Hill production/consumption surrogates) should be
#     introduced here *without* changing the Simulation/agent plumbing.
################################################################################
import numpy as np

from species_registry import (
        ODE_STATE_CA_E,
        ODE_STATE_CA_ER,
        ODE_STATE_CA_I,
        ODE_STATE_H2O2E,
        ODE_STATE_H2O2I,
)

CHEM_MODE = "full"
VALID_CHEM_MODES = {"full", "hill"}

# Keep these aligned with existing state vector semantics.
H2O2I_INDEX = ODE_STATE_H2O2I
CAI_INDEX = ODE_STATE_CA_I

HILL_EPS = 1e-12
HILL_A0 = 0.0
HILL_A1 = 1.0
HILL_KC = 0.5
HILL_N = 2.0
HILL_BH = 0.2
HILL_C0 = 0.0
HILL_C1 = 1.0
HILL_KH = 0.5
HILL_M = 2.0
HILL_BC = 0.2
HILL_RK4_SUBSTEPS = 32


def _normalize_chem_mode(mode):
        if mode is None:
                mode = "full"
        mode = str(mode).strip().lower()
        if mode not in VALID_CHEM_MODES:
                raise ValueError(f"Invalid chem_mode '{mode}'. Expected one of {sorted(VALID_CHEM_MODES)}.")
        return mode


def set_chem_mode(mode):
        global CHEM_MODE
        CHEM_MODE = _normalize_chem_mode(mode)
        return CHEM_MODE


def get_chem_mode():
        return CHEM_MODE


def _hill_gate(y, k_half, power):
        y = max(float(y), 0.0)
        return (y ** power) / ((k_half ** power) + (y ** power) + HILL_EPS)


def _hill_gate_vectorized(y, k_half, power):
        y = np.maximum(np.asarray(y, dtype=float), 0.0)
        return (y ** power) / ((k_half ** power) + (y ** power) + HILL_EPS)


def hill_derivatives_vectorized(h2o2_i, ca_i):
        h = np.asarray(h2o2_i, dtype=float)
        c = np.asarray(ca_i, dtype=float)

        dhdt = HILL_A0 + HILL_A1 * _hill_gate_vectorized(c, HILL_KC, HILL_N) - HILL_BH * h
        dcadt = HILL_C0 + HILL_C1 * _hill_gate_vectorized(h, HILL_KH, HILL_M) - HILL_BC * c
        return dhdt, dcadt


def model_hill_vectorized(x_matrix):
        x_matrix = np.asarray(x_matrix, dtype=float)
        if x_matrix.ndim != 2:
                raise ValueError("model_hill_vectorized expects a 2D cell-by-state matrix")

        dxdt = np.zeros_like(x_matrix)
        dhdt, dcadt = hill_derivatives_vectorized(
                x_matrix[:, ODE_STATE_H2O2I],
                x_matrix[:, ODE_STATE_CA_I],
        )
        dxdt[:, ODE_STATE_H2O2I] = dhdt
        dxdt[:, ODE_STATE_CA_I] = dcadt
        return dxdt


def hill_step_vectorized(x_matrix, dt, substeps=HILL_RK4_SUBSTEPS):
        xnew = np.array(x_matrix, dtype=float, copy=True)
        if xnew.ndim != 2:
                raise ValueError("hill_step_vectorized expects a 2D cell-by-state matrix")
        if substeps < 1:
                raise ValueError("substeps must be at least 1")

        h = xnew[:, ODE_STATE_H2O2I].copy()
        c = xnew[:, ODE_STATE_CA_I].copy()
        step = float(dt) / int(substeps)

        for _ in range(int(substeps)):
                k1h, k1c = hill_derivatives_vectorized(h, c)
                k2h, k2c = hill_derivatives_vectorized(
                        h + 0.5 * step * k1h,
                        c + 0.5 * step * k1c,
                )
                k3h, k3c = hill_derivatives_vectorized(
                        h + 0.5 * step * k2h,
                        c + 0.5 * step * k2c,
                )
                k4h, k4c = hill_derivatives_vectorized(
                        h + step * k3h,
                        c + step * k3c,
                )

                h = h + (step / 6.0) * (k1h + 2.0 * k2h + 2.0 * k3h + k4h)
                c = c + (step / 6.0) * (k1c + 2.0 * k2c + 2.0 * k3c + k4c)

        xnew[:, ODE_STATE_H2O2I] = h
        xnew[:, ODE_STATE_CA_I] = c
        return xnew


def model_hill(x, t, k, x0, *args, **kwargs):
        dxdt = np.zeros_like(x)

        h = x[ODE_STATE_H2O2I]
        c = x[ODE_STATE_CA_I]

        dxdt[ODE_STATE_H2O2I] = HILL_A0 + HILL_A1 * _hill_gate(c, HILL_KC, HILL_N) - HILL_BH * h
        dxdt[ODE_STATE_CA_I] = HILL_C0 + HILL_C1 * _hill_gate(h, HILL_KH, HILL_M) - HILL_BC * c

        return dxdt


def get_k():
        k = np.zeros(100)
        # --- your original core parameters (kept as you used them) ---
        k[0] = 2e-2     # H2O2 PM permeability (cm/s)
        k[1] = 11.1e-7   # Basal intracellular H2O2 production (M/s)
        k[2] = 2.1e7    # GPx_red + H2O2 (M^-1 s^-1)
        k[3] = 4e4      # GPx_ox + GSH (M^-1 s^-1)
        k[4] = 1e7      # GPx-SSG + GSH (M^-1 s^-1)
        k[5] = 3.4e7    # Catalase * H2O2 (M^-1 s^-1)
        k[6] = 5.7e-5   # Km NADP+ (M)
        k[7] = 1.0e-4   # Prx oxidation by H2O2 (M^-1 s^-1)  (placeholder you tuned)
        k[8] = 4.9e-9   # Prx over-oxidation (M^-1 s^-1)
        k[9] = 3e-3     # Srx repair (s^-1)
        k[10] = 15      # Prx-SOH -> Prx-SS (s^-1)
        k[11] = 2.2e-6  # Trx reduces Prx-SS (M^-1 s^-1)
        k[12] = 7.4e-5  # GSH auto-ox (s^-1)
        k[13] = 26      # Prot-SH + H2O2 (M^-1 s^-1)
        k[14] = 1.2e5   # Prot-SOH + GSH (M^-1 s^-1)
        k[15] = 9.1e4   # GrxSH deglut. Prot-SSG (M^-1 s^-1)
        k[16] = 3.7e4   # GSH deglut. Grx-SSG (M^-1 s^-1)
        k[17] = 62      # Prot-(SH)2 + H2O2 (M^-1 s^-1)
        k[18] = 62      # Prot-SS + Trx (M^-1 s^-1)
        k[19] = 3.2e6   # GR (M^-1 s^-1)
        k[20] = 2e7     # TrxR (M^-1 s^-1)
        k[21] = 3.75e-4 # G6PD NADPH prod (M/s)
        k[22] = 3e-3    # Peroxisome membrane permeability (cm/s)
        k[23] = 4.1e-7  # GSH synthesis (M/s)
        k[24] = 1.2e-8  # GSSG efflux (M/s)
        k[25] = 1.2e-7  # GSH+GSSG efflux (M/s)

        # Superoxide terms left off (29..31 zero) in your latest
        k[28] = 0
        k[29] = 0
        k[30] = 0
        k[31] = 0

        # NOX2 Hill-on-H2O2 gate (used by you)
        k[32] = 5e-7    # V_NOX2 (M/s).       # was 5e-8
        k[33] = 1e-7    # K_act_NOX2 (M)        # was 1e-7

        # ---- Calcium & NOX5 block (as you requested) ----
        # IP3R-driven ER release, SERCA uptake, TRPM2 & SOCE influx
        k[34] = 1e-4    # V_IP3R (M/s)
        k[35] = 1e-7    # K_IP3R_H (M) H2O2 half-act
        k[36] = 4.0     # n_H (H2O2->IP3R)
        k[37] = 5e-7    # K_er (M) ER Ca scale

        k[38] = 9e-7    # V_SERCA (M/s)
        k[39] = 0.46e-6    # K_ser (M), cyt Ca half-sat for SERCA    10.1074/jbc.M114.585604
        k[40] = 1e-5    # (unused here if you skip H2O2 inhibition)
        k[41] = 2.0     # (unused here)

        k[42] = 1e-6    # P_TRPM2 base (1/s)
        k[43] = 5e-7    # K_TRPM2_H (M)
        k[44] = 2.0     # p_TRPM2 Hill

        k[45] = 1e-7    # P_SOCE (1/s)
        k[46] = 1e-5    # K_store (M)
        k[47] = 2.1     # SERCA Hill n

        # NOX5 (Ca-dependent) with NADPH limitation
        k[48] = 8.7e-3    # Vmax_NOX5 (M/s) — you tune                doi.org/10.1016/j.redox.2020.101656
        k[49] = 0.71e-6 # K_Ca_NOX5 (M)                              doi.org/10.1038/s41467-024-48467-y
        k[50] = 4.0     # m_NOX5 (Hill)

        # IP3R high-H2O2 / high-Ca inhibition (optional)
        k[51] = 5e-7    # K_H2O2_inhib (M)
        k[52] = 2.0     # m_H2O2_inhib
        k[53] = 5e-6    # K_Ca_inhib (M)
        k[54] = 2.0     # n_Ca_act

        # ER capacity & overflow
        k[55] = 1.1e-4  # ER Ca max (M)
        k[56] = 1e-2    # overflow rate (1/s)
        k[57] = 5e-7    # K_IP3R_Ca (M) for Ca activation
        k[58] = 1e-8   # Ca extrusion (1/s), saturable with small Km in ODE
        
        # NOX5 NADPH Km
        k[62] = 13.67e-6                                     #  doi.org/10.1038/s41467-024-48467-y

        # Duox1/2 (additional Ca-dependent ROS source)
        k[63] = 1e-8                 # kcat_duox_low_Ca (s-1)  (tune 1e-8 to 1e-7)
        k[64] = 0.87e-6                    # K_Duox_Ca (M) ~ 1 µM half-activation
        k[65] = 4                           # m_duox (Hill exponent for Ca)
        k[66] = 24.31e-6                    # Km_Duox_NADPH_low_Ca [M] (~14 µM)
        k[67] = 11.52e-6                    # Km_Duox_NADPH_high_Ca [M] (~140 µM)

        # TRPM2 Ca influx (you already have H2O2 gate) due to calcium
        k[68] = 16.9e-6     # k_trpm2 ca (M) ca half-activation (~0.5 µM), doi.org/10.1073/pnas.0811725106
        k[69] = 2                         # n (trpm2) ca

        return k

def get_x0():
        # initial conditions
        x0 = np.zeros(32)  # Defining the array, x0, that will house initial concentrations
        # Cytoplasm
        x0[ODE_STATE_H2O2E] = 1e-9  # H2O2media
        x0[ODE_STATE_H2O2I] = 1e-9  # H2O2cytosol

        x0[2] = 5e-5  # GPXr 
        x0[3] = 1e-14  # GPXo 
        x0[4] = 1e-14  # GPX-SG 

        x0[5] = 3.68e-4  # GSH 
        x0[6] = 1.78e-6  # GSSG 

        x0[7] = 9e-7  # Catalase 
        x0[8] = 1.0e-10  # H2O2peroxisome 

        x0[9] = 1.92e-5  # Prx-SH  
        x0[10] = 1e-14  # Prx-SOH 
        x0[11] = 1e-14  # Prx-SOOH 
        x0[12] = x0[9] * (0.5 / 100)  # Prx-SS 

        x0[13] = 4.27e-7  # Trx1-SH 
        x0[14] = 7.54e-8  # Trx1-SS 

        x0[15] = 1.22e-4  # Pr-SH 
        x0[16] = x0[15] * (0.5 / 100)  # Pr-SOH
        x0[17] = x0[15] * (0.5 / 100)  # Pr-SSG 

        x0[18] = 1.20e-6  # Grx-SH 
        x0[19] = x0[18] * (0.5 / 100)  # Grx-SSG 

        x0[20] = 1.09e-3  # Pr-(SH)2  
        x0[21] = x0[20] * (0.5 / 100)  # Pr-SS

        x0[22] = 3.0e-5  # NADPH 
        x0[23] = 3.0e-7  # NADP+

        x0[24] = 0 # superoxide media
        x0[25] = 0#1e-9 # superoxide cytosol

        x0[26] = 0 # superoxide dismutase reduced
        x0[27] = 0 # superoxide dismutase oxidized

        x0[28] = 1e-9 # steady state h2o2 intracellular

        x0[ODE_STATE_CA_I] = 1e-7 # calcium intracellular
        x0[ODE_STATE_CA_E] = 1.5e-3 # calcium extracellular
        x0[ODE_STATE_CA_ER] = 1.0e-4 # calcium ER [M] ~ 0.1 mM


        return x0

# Define the parameterized model (copy your full ODEs here)
def model_noref_for_mode(x, t, k, x_base, chem_mode):
    if _normalize_chem_mode(chem_mode) == "hill":
        return model_hill(x, t, k, x_base)
    return model_noref_full(x, t, k, x_base)


def make_model_rhs(chem_mode, k, x_base):
    mode = _normalize_chem_mode(chem_mode)

    def rhs(t, y):
        return model_noref_for_mode(y, t, k, x_base, mode)

    return rhs


def model_noref(x, t, k, x_base):
    return model_noref_for_mode(x, t, k, x_base, CHEM_MODE)


def model_noref_full(x, t, k, x_base):

    dxdt = np.zeros_like(x)

    # ===== Calcium–ROS crosstalk using k[...] and x[...] only =====
    H   = x[ODE_STATE_H2O2I]     # H2O2 cytosol
    C   = x[ODE_STATE_CA_I]    # Ca cytosol
    Cer = x[ODE_STATE_CA_ER]    # Ca ER pool

    # --- Unpack from k ---
    V_IP3R     = k[34]; K_IP3R_H = k[35]; nH = k[36]; K_er = k[37]
    V_SERCA   = k[38]; K_SERCA_H = k[40]; m_inhib = k[41]
    P_TRPM2   = k[42]; K_TRPM2_H= k[43]; p_TRPM2 = k[44]
    P_SOCE    = k[45]; K_store  = k[46]

    ## --- Helper gates (pure functions of species & k) ---
    Ca_gate_IP3R = (C)**k[54] / (k[57]**k[54] + (C)**k[54]) if C > 0 else 0.0 # Activated by cyt Ca
    Ca_IP3R_inhib = 1.0/(1.0 + (C/k[53])**2) if C > 0 else 0.0 # Inhibited by high Ca
    h2o2_gate_IP3R = (H**nH)/(K_IP3R_H**nH + H**nH) if H > 0 else 0.0 # activation by H2o2
    H2O2_inhibit_IP3R = (k[51]**k[52]) / (k[51]**k[52] + H**k[52]) if H > 0 else 0.0 # inhibition by very high H2O2
    SERCA_inhb = 1.0/(1.0 + (H/K_SERCA_H)**m_inhib) if H > 0 else 1.0
    H2O2_gate_TRPM2 = (H**p_TRPM2)/(K_TRPM2_H**p_TRPM2 + H**p_TRPM2) if H > 0 else 0.0
    Ca_gate_TRPM2 = (C**k[69])/((k[68]**k[69]) + (C**k[69])) if C > 0 else 0.0
    SOCE_drive = K_store/(K_store + Cer)       # ramps as ER depletes

    # --- Fluxes (units consistent with your H2O2 model scaling) ---
    J_IP3R     = V_IP3R   * h2o2_gate_IP3R * Ca_gate_IP3R \
        * Ca_IP3R_inhib * H2O2_inhibit_IP3R * (Cer - C) / (K_er + Cer) if (Cer - C) > 0 else 0.0 # ER -> cyt release via IP3R
    #print(Ca_gate_IP3R,h2o2_gate_IP3R,Ca_IP3R_inhib,H2O2_inhibit_IP3R,J_IP3R)
    J_SERCA   = V_SERCA * (C**k[47]/(k[39]**k[47] + C**k[47])) * SERCA_inhb if C > 0 else 0.0 # cyt -> ER uptake by SERCA
    J_in_TRPM2= P_TRPM2 * H2O2_gate_TRPM2 * Ca_gate_TRPM2 * (x[ODE_STATE_CA_E] - C) if (x[ODE_STATE_CA_E] - C) > 0 else 0.0 # PM influx via TRPM2
    J_in_SOCE = P_SOCE  * SOCE_drive * (x[ODE_STATE_CA_E] - C) if (x[ODE_STATE_CA_E] - C) > 0 else 0.0 # PM influx via SOCE to restore ER

    # --- ER capacity gating for SERCA (prevents overfilling) ---
    cap = 1.0 - (Cer / k[55])         # linear headroom fraction
    cap = 0.0 if cap < 0.0 else cap   # clamp to [0, 1]
    J_SERCA *= cap                    # SERCA stops as ER approaches capacity

    # --- Overflow if ER exceeds the cap (hard push back to cytosol) ---
    J_over = k[56] * cap * Cer if cap < 0.0 else 0.0
    J_out = k[58] * (C)/ (5e-6 + (C))  if C > 0 else 0.0 # cytosolic Ca extrusion (PMCA/NCX lumped)
    # --- NOX5 Ca-dependent ROS production ---
    nox2 = k[32] * (x[1]**6)/(k[33]**6 + x[1]**6) if x[1] > 0 else 0.0 #nox2
    # NOX5 with NADPH limitation
    nox5 = k[48] * (C**k[50])/(k[49]**k[50] + C**k[50]) \
                * (x[22]/(k[62] + x[22])) if C > 0 or x[22] > 0 else 0.0

    km_duox = k[66] - (k[66] - k[67]) * (C**k[65])/(k[64]**k[65] + C**k[65]) if C > 0 else k[67]
    duox = k[63] * x[22]/(km_duox + x[22])

    dxdt[0] = (k[0]*(x[1] - x[0]) * 10e-3 * 5.3e-6 * (1e9)) + nox2 + duox   # extracellular peroxide

    dxdt[1] = (((k[0] * x[0] * 10e-3 * 5.3e-6) / (1.15e-12))
               - ((k[0] * x[1] * 10e-3 * 5.3e-6) / (1.15e-12))
               - 2 * (k[22] * x[1] * 10e-3 * 2.98e-9) / (1.53e-14)              # peroxisomal H2O2     
               - k[2] * x[2] * x[1]                             # GPx reduction of H2O2
               - k[5] * x[7] * x[1]                                # catalase reaction
               - k[7] * x[9] * x[1]                                   # peroxiredoxin oxidation
               - k[8] * x[10] * x[1]                                  # overoxidized peroxiredoxin
               - k[13] * x[15] * x[1]                                 # Protein monothiol oxidation
               - k[17] * x[20] * x[1]                                 # Protein dithiol oxidation
               + k[1]                                                          # peroxide production due to ETC leak
               + nox5)                                                        # nox5 h2o2 production
    
    #GPXr
    dxdt[2] = (- k[2] * x[2] * x[1]
               + k[4] * x[4] * x[5])
    #GPXo
    dxdt[3] = k[2] * x[2] * x[1] - k[3] * x[3] * x[5] 
    #GPX-SG
    dxdt[4] = k[3] * x[3] * x[5] - k[4] * x[4] * x[5]
    #GSH
    dxdt[5] = (- k[3] * x[3] * x[5]
               - k[4] * x[4] * x[5]
               - 2 * k[12] * x[5]
               - k[14] * x[16] * x[5]
               - k[16] * x[16] * x[5]
               + 2 * k[19] * x[6] * x[22]
               + k[23] 
               - k[25])
    #GSSG
    dxdt[6] = (k[4] * x[4]* x[5] 
               + k[12] * x[5] 
               + k[16] * x[19] * x[5] 
               - k[19] * x[6] * x[22] if x[6] > 0 and x[22] > 0 else 0.0
               - k[24] * x[6] / x_base[6] if x_base[6] > 0 else 0.0
               - k[25] * x[6] / x_base[6] if x_base[6] > 0 else 0.0)

    dxdt[7] = 0
    dxdt[8] = ((k[22] * x[1] * 10e-3 * 2.98e-9) / (1.53e-14)            # peroxisomal H2O2
               - k[5] * x[7] * x[8])
    
    #Prx-SH
    dxdt[9] = (- k[7] * x[9] * x[1]                                 # peroxiredoxin oxidation by H2O2, rates = 4e7 M-1s-1
               + k[11] * x[12] * x[13])                             # reduction of peroxiredoxin by thioredoxin
    #Prx-SOH
    dxdt[10] = (k[7] * x[9] * x[1] 
                - k[8] * x[10] * x[1]                           # overoxidized peroxiredoxin by H2O2    
                + k[9] * x[11]                                      # reduction of overoxidized peroxiredoxin by sulfiredoxin
                - k[10] * x[10])                                # self-catalyzed disulfide formation of Prx-SS from Prx-SOH
    #Prx-SOOH
    dxdt[11] = (k[8] * x[10] * x[1]                                 # reduction of peroxiredoxin by thioredoxin
                - k[9] * x[11])                                    # reduction of overoxidized peroxiredoxin by sulfiredoxin
    #Prx-SS
    dxdt[12] = (k[10] * x[10]                                   # self-catalyzed disulfide formation of Prx-SS from Prx-SOH
                - k[11] * x[12] * x[13])
    #Trx1-SH
    dxdt[13] = (- k[11] * x[12] * x[13] 
                - k[18] * x[21] * x[13] 
                + k[20] * x[14] * x[22] 
                - k[26] 
                + k[27])
    #Trx1-SS
    dxdt[14] = (k[11] * x[12] * x[13] 
                + k[18] * x[21] * x[13] 
                - k[20] * x[14] * x[22])

    dxdt[15] = (- k[13] * x[15] * (x[1])
                + k[15] * x[18] * (x[17]))

    dxdt[16] = k[13] * x[15] * x[1] - k[14] * x[16] * x[5]
    dxdt[17] = k[14] * x[16] * x[5] - k[15] * x[18] * x[17]
    dxdt[18] = k[16] * x[19] * x[5] - k[15] * x[18] * x[17]
    
    dxdt[19] = (k[15] * x[18] * x[17] 
                - k[16] * x[19] * x[5])
    
    #Pr-(SH)2
    dxdt[20] = - k[17] * x[20] * x[1] + k[18] * x[21] * x[13]
    dxdt[21] = k[17] * x[20] * x[1] - k[18] * x[21] * x[13]

    #NADPH
    dxdt[22] = (- k[19] * x[6] * x[22] 
                - k[20] * x[14] * x[22] 
                #- k[28] * x[22]                                         # nox generation of superoxide
                - (2 * nox2 if x[22] > 0 else 0.0)
                - (2 * nox5 if x[22] > 0 else 0.0)
                + (k[21] * x[23] / (k[6] + x[23]) if x[23] > 0 or x[22] > 0 else 0.0))
    #NADP+
    dxdt[23] = (k[19] * x[6] * x[22] 
                + k[20] * x[14] * x[22] 
                #+ k[28] * x[22]                                         # nox generation of superoxide and nadp+
                + 2 * nox2
                + 2 * nox5
                - (k[21] * x[23] / (k[6] + x[23]) if x[23] > 0 else 0.0))
    #superoxide media
    dxdt[24] = 0
    #superoxide cytosol
    dxdt[25] = 0
    #SOD reduced
    dxdt[26] = 0
    #SOD oxidized
    dxdt[27] = 0

    # --- Calcium balances --
    dxdt[ODE_STATE_CA_I] = J_IP3R + J_in_TRPM2 + J_in_SOCE + J_over - J_SERCA - J_out            # d[Ca_cyt]/dt
    dxdt[ODE_STATE_CA_E] = 0                                                     # d[Ca_ext]/dt
    dxdt[ODE_STATE_CA_ER] = J_SERCA - J_IP3R - J_over                           # d[Ca_ER]/dt
    #print(dxdt[29], dxdt[31])

    return dxdt

def model_full(x, t, k, x0):
    # if x0 is None:
    #     x0 = get_x0()
    dxdt = np.zeros_like(x)
    nox2 = k[32] * (x[1]**6)/(k[33]**6 + x[1]**6) if x[1] > 0 else 0.0 #nox2
    dxdt[0] = (k[0] * (x[1] - x[0]) * 10e-3 * 1.02e-5 * (1e9)) + nox2
    # --- Copy the rest of your ODEs from model_noref here ---

    dxdt[1] = (((k[0] * x[0] * 10e-3 * 5.3e-6) / (1.15e-12))
               - ((k[0] * x[1] * 10e-3 * 5.3e-6) / (1.15e-12))
               - 2 * (k[22] * x[1] * 10e-3 * 2.98e-9) / (1.53e-14)              # peroxisomal H2O2     
               - k[2] * x[2] * x[1]                             # GPx reduction of H2O2
               - k[5] * x[7] * x[1]                                   # catalase reaction
               - k[7] * x[9] * x[1]                                   # peroxiredoxin oxidation
               - k[8] * x[10] * x[1]                                  # overoxidized peroxiredoxin
               - k[13] * x[15] * x[1]                                 # Protein monothiol oxidation
               - k[17] * x[20] * x[1]                                 # Protein dithiol oxidation
               + k[1]                                                          # intracellular peroxide production
               + k[29] * x[25] * x[27])                                      # superoxide dismutase reduction and oxidation by superoxide

    dxdt[2] = (- k[2] * x[2] * x[1]
               + k[4] * x[4] * x[5])
    
    dxdt[3] = k[2] * x[2] * x[1] - k[3] * x[3] * x[5]
    dxdt[4] = k[3] * x[3] * x[5] - k[4] * x[4] * x[5]
    dxdt[5] = (- k[3] * x[3] * x[5]
               - k[4] * x[4] * x[5]
               - 2 * k[12] * x[5]
               - k[14] * x[16] * x[5]
               - k[16] * x[16] * x[5]
               + 2 * k[19] * x[6] * x[22]
               + k[23] 
               - k[25])
    dxdt[6] = (k[4] * x[4] * x[5] + k[12] * x[5] +
               k[16] * x[19] * x[5] - k[19] * x[6] * x[22] 
               - k[24] * x[6] / x0[6]
               - k[25] * x[6] / x0[6]
               )
    dxdt[7] = 0
    dxdt[8] = (k[22] * x[1] * 10e-3 * 2.98e-9) / (1.53e-14) - k[5] * x[7] * x[8]
    dxdt[9] = - k[7] * x[9] * x[1] + k[11] * x[12] * x[13]
    dxdt[10] = (k[7] * x[9] * x[1] - k[8] * x[10] * x[1] +
                k[9] * x[11] - k[10] * x[10])
    dxdt[11] = k[8] * x[10] * x[1] - k[9] * x[11]
    dxdt[12] = k[10] * x[10] - k[11] * x[12] * x[13]
    dxdt[13] = (- k[11] * x[12] * x[13] 
                - k[18] * x[21] * x[13] 
                + k[20] * x[14] * x[22] 
                - k[26] 
                + k[27])
    dxdt[14] = (k[11] * x[12] * x[13] + k[18] * x[21] * x[13] -
                k[20] * x[14] * x[22])

    dxdt[15] = (- k[13] * x[15] * (x[1])
                + k[15] * x[18] * (x[17]))

    dxdt[16] = k[13] * x[15] * x[1] - k[14] * x[16] * x[5]
    dxdt[17] = k[14] * x[16] * x[5] - k[15] * x[18] * x[17]
    dxdt[18] = k[16] * x[19] * x[5] - k[15] * x[18] * x[17]
    
    dxdt[19] = (k[15] * x[18] * x[17] 
                - k[16] * x[19] * x[5])
    
    dxdt[20] = - k[17] * x[20] * x[1] + k[18] * x[21] * x[13]
    dxdt[21] = k[17] * x[20] * x[1] - k[18] * x[21] * x[13]

    #NADPH
    dxdt[22] = (- k[19] * x[6] * x[22] 
                - k[20] * x[14] * x[22] 
                #- k[28] * x[22]                                         # nox generation of superoxide
                + k[21] * x[23] / (k[6] + x[23]))
    #NADP+
    dxdt[23] = (k[19] * x[6] * x[22] 
                + k[20] * x[14] * x[22] 
                #+ k[28] * x[22]                                         # nox generation of superoxide and nadp+
                - k[21] * x[23] / (k[6] + x[23]))

    dxdt[24] = (k[0] * (x[25] - x[24]) * 10e-3 * 1.02e-5 * (1e9)               # superoxide media
                + k[28] * x[22])
    
    dxdt[25] = (((k[0] * x[24] * 10e-3 * 1.02e-5) / (9.16e-13))                 # superoxide cytosol
               - ((k[0] * x[25] * 10e-3 * 1.02e-5) / (9.16e-13))
               - k[29] * x[25] * x[26]
               - k[29] * x[25] * x[27]
               + k[30])
    
    dxdt[26] = (k[29] * x[25] * x[27] 
                - k[29] * x[25] * x[26])

    dxdt[27] = (k[29] * x[25] * x[26] 
                - k[29] * x[25] * x[27])
    
    dxdt[28] = 0#(((k[0] * x[29] * 10e-3 * 1.02e-5) / (9.16e-13))               # calcium cytosol
               #- ((k[0] * x[28] * 10e-3 * 1.02e-5) / (9.16e-13)))
               #+ (200e-6*(x[28]**2/(0.6e-6**2+x[28]**2))))
    
    #dxdt[29] = ((k[0] * 1e-3 * (x[28] - x[29]) * 10e-3 * 1.02e-5 * (1e9)) + k[32])               # calcium media
    
    return dxdt


def model_for_mode(x, t, k, x0, chem_mode, *args, **kwargs):
    if _normalize_chem_mode(chem_mode) == "hill":
        return model_hill(x, t, k, x0, *args, **kwargs)
    return model_full(x, t, k, x0, *args, **kwargs)


def model(x, t, k, x0, *args, **kwargs):
    return model_for_mode(x, t, k, x0, CHEM_MODE, *args, **kwargs)

def calibrate_k1_for_steady_state(k, x0):
    # geometry
    A_cm2, Vcyt_L = 5.3e-6, 1.15e-12
    J_perm = k[0] * (x0[0] - x0[1]) * (A_cm2 / Vcyt_L) * 1e-3  # membrane flux

    # sinks
    v_cat = k[5]*x0[7]*x0[1]
    v_prx = k[7]*x0[9]*x0[1] + k[8]*x0[10]*x0[1]
    v_gpx = k[2]*x0[2]*x0[1]
    v_prot= k[13]*x0[15]*x0[1] + k[17]*x0[20]*x0[1]

    # sources (at baseline)
    v_nox5 = k[48] * (x0[29]**k[50])/(k[49]**k[50] + x0[29]**k[50])
    v_mito = 0.0
    v_nox2 = 0.0

    sinks  = v_cat + v_prx + v_gpx + v_prot - J_perm
    srcs   = v_nox5 + v_mito + v_nox2

    k[1] = max(sinks - srcs, 0.0)   # M/s
    return print(k)

k = get_k()
x0 = get_x0()
#calibrate_k1_for_steady_state(k, x0)


    
def get_blap_k(protein_abund, blap_conc, extraC, intC, parameter_pctchange=None, species_pctchange=None):
    """
    Model for compartmentalized hydrogen peroxide consumption
    
    Parameters:
    -----------
    endtime : float
        End time for simulation
    step : float
        Maximum step size for solver
    celldens : float
        Cell density
    protein_abund : array-like
        Protein abundance values
    blap_conc : float
        B-lap concentration
    parameter_pctchange : array-like, optional
        Percentage changes for parameters
    species_pctchange : array-like, optional
        Percentage changes for species
    
    Returns:
    --------
    time_series : ndarray
        Array containing time points and state variables
    """
    # Define parameter values
    k = np.zeros(36)
    
    # Rxns (Cytosolic)
    
    # H2O2 Transport
    # Permeability constant (plasma membrane)
    # H2O2cyt -> H2O2out
    k[0] = 2e-4 * protein_abund[29] / 0.0302  # cm/s
    
    # Mitochondrial production of H2O2in
    k[1] = 4
    
    # GPx1_red reacting with H2O2in
    # H2O2cyt + GPX1r -> GPX1o ; EC 1.11.1.9
    # Updated 9/22 with Kassi Stein PLoS Comp Bio paper k2
    k[2] = 60  # uM-1*s-1 (60 uM-1*s-1 Kassi paper)
    
    # GPx1_ox reacting with GSH
    # GPX1o + GSH -> GPX1-SG
    # Updated 9/22 with KS PLoS Comp Bio paper k3
    k[3] = 0.04  # uM-1*s-1
    
    # GPx1-SG reacting with GSH
    # GPX1-SG + GSH -> GPX1r + GSSG
    # Updated 9/22 with KS PLoS Comp Bio Paper k4
    k[4] = 10  # uM-1*s-1
    
    # Catalase reacts with H2O2 inside peroxisome
    # H2O2p ->
    # Updated 10/6
    k[5] = 34  # uM-1*s-1
    
    # Km of NADP+
    # Updated 9/22 with KS PLoS Comp Bio Paper k4
    k[6] = 57  # uM
    
    # Peroxiredoxin is oxidized by H2O2
    # Updated 10/6
    k[7] = 40  # uM-1*s-1
    
    # Oxidized peroxiredoxin is over-oxidized by H2O2
    # Updated 10/6
    k[8] = 0.072  # uM-1*s-1
    
    # Reduction of overoxidized Prx by Srx enzyme
    k[9] = 3e-3  # s-1
    
    # Self-catalyzed disulfide formation of Prx-SS from Prx-SOH
    k[10] = 15  # s-1
    
    # Peroxiredoxin is reduced by Thioredoxin
    # Updated 10/6
    k[11] = 2.1  # uM-1*s-1
    
    # Auto-oxidation of GSH
    k[12] = 7.4e-05  # s-1
    
    # Protein monothiol oxidized by H2O2
    # Updated 10/6
    k[13] = 0.01  # uM-1*s-1
    
    # Oxidized Protein monothiol glutathionylated by GSH
    # Updated 10/6
    k[14] = 0.12  # uM-1*s-1
    
    # Grx-SH de-glutathionylates Protein-SSG
    # Updated 10/6
    k[15] = 0.091  # uM-1*s-1
    
    # GSH de-glutathionylates Grx-SSG
    # Updated 10/6
    k[16] = 0.037  # uM-1*s-1
    
    # Protein dithiol oxidized by H2O2
    # Updated 10/6
    k[17] = 5e-1  # uM-1*s-1
    
    # Protein disulfide reduced by Trx1
    # Updated 10/6
    k[18] = 1e-1  # uM-1*s-1
    
    # GSSG reduced by GR
    k[19] = 3.2 * protein_abund[22] / 0.1419  # uM-1*s-1
    
    # Oxidized Thioredoxin reduced by TrxR
    k[20] = 2e1 * protein_abund[15] / 0.1816  # uM-1*s-1
    
    # Production of NADPH by G6P-DH
    k[21] = 3.75e2 * (protein_abund[20] + protein_abund[21]) / (0.2581 + 0.5159)  # uM/s
    
    # Permeability constant (peroxisomal membrane)
    k[22] = 3e-3  # cm/s
    
    # GSH synthesis
    k[23] = 4.1e-1 * protein_abund[32] / .0075  # uM/s 4.1e-7
    
    # GSSG efflux
    k[24] = 1.2e-2  # uM/s 1.2e-8
    
    # GSH + GSSG efflux
    k[25] = 1.2e-1  # uM/s 1.2e-7
    
    # NQO1 Reaction; b-lap Q->HQ
    k[28] = 161.5 * protein_abund[25] / .2905  # 1/uM-s
    
    # b-lap HQ->SQ
    k[29] = 3e2  # 1/uM-s
    
    # b-lap SQ->Q
    k[30] = 3e2  # 1/uM-s
    
    # SOD
    k[31] = 6400.0 * protein_abund[26] / 1.6641  # 1/uM-s
    
    # CPR
    k[32] = 1900 * protein_abund[24] / 0.2231  # 1/uM-s
    
    # blap perm
    k[33] = 1e-6  # cm/s
    
    # blapHQ + GSH -> blapHQ-SG
    k[34] = 0.05 * protein_abund[23] / 2.6052  # 1/uM-s
    
    # blapHQ-SG -> blapHQ-SG extracell
    k[35] = 1e-6  # cm/s
    
    if parameter_pctchange is not None:
        k = k * parameter_pctchange
    
    return k

def get_blap_x(protein_abund, blap_conc, extraC, intC, parameter_pctchange=None, species_pctchange=None):
    # Defining initial Conditions (Concentrations in uM = micromoles/Liter)
    # Defining the array, x0, that will house initial concentrations
    x0 = np.zeros(32)
    
    # H2O2media
    x0[0] = extraC
    
    # H2O2cytosol
    x0[1] = intC
    
    # GPXr
    x0[2] = protein_abund[0]
    
    # GPXo
    x0[3] = x0[2] / 5e9
    
    # GPX-SG
    x0[4] = x0[2] / 5e9
    
    # GSH
    x0[5] = 368
    
    # GSSG
    x0[6] = x0[5] / 206
    
    # Catalase
    x0[7] = protein_abund[12]
    
    # H2O2peroxisome
    x0[8] = 1.0e-10
    
    # Prx1/2-SH
    x0[9] = protein_abund[8] + protein_abund[9]
    
    # Prx1/2-SOH
    x0[10] = x0[9] / 1.92e9
    
    # Prx1/2-SOOH
    x0[11] = x0[9] / 1.92e9
    
    # Prx1/2-SS
    x0[12] = x0[9] * (.5 / 100)
    
    # Trx1-SH
    x0[13] = protein_abund[13]
    
    # Trx1-SS
    x0[14] = x0[13] / 5.7
    
    # Pr-SH
    x0[15] = 50  # 1.22e2
    
    # Pr-SOH
    x0[16] = x0[15] * (.5 / 100)
    
    # Pr-SSG
    x0[17] = x0[15] * (.5 / 100)
    
    # Grx-SH
    x0[18] = protein_abund[18]
    
    # Grx-SSG
    x0[19] = x0[18] * (.5 / 100)
    
    # Pr-(SH)2
    x0[20] = 450
    
    # Pr-SS
    x0[21] = x0[20] * (.5 / 100)
    
    # NADPH
    x0[22] = 3.0e1
    
    # NADP+
    x0[23] = x0[22] / 100
    
    # blap Q
    x0[24] = 0
    
    # blap HQ
    x0[25] = 0
    
    # blap SQ
    x0[26] = 0
    
    # superO2
    x0[27] = 0
    
    # oxygen
    x0[28] = 2.6e2
    
    # blap extracell
    x0[29] = blap_conc
    
    # blap-GSH
    x0[30] = 0
    
    # blap-GSH extracell
    x0[31] = 0
    
    if species_pctchange is not None:
        x0 = x0 * species_pctchange
        
        x0[3] = x0[2] / 5e9  # GPXo
        x0[4] = x0[2] / 5e9  # GPX-SG
        x0[6] = x0[5] / 206  # GSSG
        x0[10] = x0[9] / 1.92e9  # Prx-SOH
        x0[11] = x0[9] / 1.92e9  # Prx-SOOH
        x0[12] = x0[9] * (.5 / 100)  # Prx-SS
        x0[14] = x0[13] / 5.7  # Trx1-SS
        x0[16] = x0[15] * (.5 / 100)  # Pr-SOH
        x0[17] = x0[15] * (.5 / 100)  # Pr-SSG
        x0[19] = x0[18] * (.5 / 100)  # Grx-SSG
        x0[21] = x0[20] * (.5 / 100)  # Pr-SS
        x0[23] = x0[22] / 100  # NADP+
    
    return x0
            
def blap_ode_sys(x, t, k, x0, celldens):
    # Apply bounds to state variables to prevent them from going too high/low
    dxdt = np.zeros_like(x)
    
    x_area = celldens * 1 * 1000 * 4 * np.pi * .00001**2 * 10e-3 # in cm^-1, peroxisome area
    c_area = celldens * 1 * 4 * np.pi * .001**2 * 10e-3 - x_area  # in cm^-1, # cell membrane area
    x_vol = 1 * 1000 * 4/3 * np.pi * .00001**3  # in cm^3
    c_vol = 1 * 4/3 * np.pi * .001**3 - x_vol  # in cm^3
    m_vol = 1000 - x_vol - c_vol  # in cm^3
    
    # Cytosolic Dynamics
    dxdt[0] = k[0] * c_area * (x[1] - x[0]) / m_vol  # H2O2 extracellular
    
    dxdt[1] = (k[0] * c_area * (x[0] - x[1]) / (c_vol) - 
                k[22] * x_area * (x[1] - x[8]) / (c_vol) + 
                k[1] - k[2] * x[2] * (x[1] - x0[1]) - 
                k[7] * x[9] * (x[1] - x0[1]) - k[8] * x[10] * (x[1] - x0[1]) - 
                k[13] * x[15] * (x[1] - x0[1]) - k[17] * x[20] * (x[1] - x0[1]) + 
                k[31] * x[27]**2)  # H2O2 cytosol
    
    # dxdt[0] = k[0] * (x[1] - x[0]) * 10e-3 * 1.02e-5 * (1e9)

    # dxdt[1] = (((k[0] * x[0] * 10e-3 * 1.02e-5) / (9.16e-13))
    #            - ((k[0] * x[1] * 10e-3 * 1.02e-5) / (9.16e-13))
    #            - 2 * (k[22] * (x[1]-x[8]) * 10e-3 * 2.98e-9) / (1.53e-14)              # peroxisomal H2O2     
    #            - k[2] * x[2] * (x[1] - x0[1])                                   # GPx reduction of H2O2
    #            - k[5] * x[7] * (x[1] - x0[1])                                   # catalase reaction
    #            - k[7] * x[9] * (x[1] - x0[1])                                   # peroxiredoxin oxidation
    #            - k[8] * x[10] * (x[1] - x0[1])                                  # overoxidized peroxiredoxin
    #            - k[13] * x[15] * (x[1] - x0[1])                                 # Protein monothiol oxidation
    #            - k[17] * x[20] * (x[1] - x0[1])                                 # Protein dithiol oxidation
    #            + k[31] * x[27]**2                             
    #            )
    
    dxdt[2] = -k[2] * x[2] * (x[1] - x0[1]) + k[4] * (x[4] - x0[4]) * x[5]  # GPXr
    
    dxdt[3] = k[2] * x[2] * (x[1] - x0[1]) - k[3] * (x[3] - x0[3]) * x[5]  # GPXo
    
    dxdt[4] = k[3] * (x[3] - x0[3]) * x[5] - k[4] * (x[4] - x0[4]) * x[5]  # GPX-SG
    
    dxdt[5] = (-k[3] * (x[3] - x0[3]) * x[5] - 
                k[4] * (x[4] - x0[4]) * x[5] - 
                2 * k[12] * x[5] - k[14] * (x[16] - x0[16]) * x[5] - 
                k[34] * x[25] * x[5] - k[16] * (x[19] - x0[19]) * x[5] + 
                2 * k[19] * (x[6] - x0[6]) * x[22] + 
                k[23] - k[25] * x[5] * x[6])  # GSH
    
    dxdt[6] = (k[4] * (x[4] - x0[4]) * x[5] + k[12] * x[5] + 
                k[16] * (x[19] - x0[19]) * x[5] - 
                k[19] * (x[6] - x0[6]) * x[22] - 
                k[24] - k[25] * x[5] * x[6])  # GSSG
    
    dxdt[7] = 0  # FeCat
    
    dxdt[8] = k[22] * x_area * (x[1] - x[8]) / x_vol - k[5] * x[7] * (x[8] - x0[8])  # H2O2 peroxisome
    
    dxdt[9] = -k[7] * x[9] * (x[1] - x0[1]) + k[11] * (x[12] - x0[12]) * x[13]  # Prx-SH
    
    dxdt[10] = (k[7] * x[9] * (x[1] - x0[1]) - k[8] * x[10] * (x[1] - x0[1]) + 
                k[9] * (x[11] - x0[11]) - k[10] * x[10])  # Prx-SOH
    
    dxdt[11] = k[8] * x[10] * (x[1] - x0[1]) - k[9] * (x[11] - x0[11])  # Prx-SOOH
    
    dxdt[12] = k[10] * x[10] - k[11] * (x[12] - x0[12]) * x[13]  # Prx-SS
    
    dxdt[13] = (-k[11] * (x[12] - x0[12]) * x[13] - 
                k[18] * (x[21] - x0[21]) * x[13] + 
                k[20] * (x[14] - x0[14]) * x[22])  # Trx-SH
    
    dxdt[14] = (k[11] * (x[12] - x0[12]) * x[13] + 
                k[18] * (x[21] - x0[21]) * x[13] - 
                k[20] * (x[14] - x0[14]) * x[22])  # Trx-SS
    
    dxdt[15] = -k[13] * x[15] * (x[1] - x0[1]) + k[15] * x[18] * (x[17] - x0[17])  # Pr-SH
    
    dxdt[16] = k[13] * x[15] * (x[1] - x0[1]) - k[14] * (x[16] - x0[16]) * x[5]  # Pr-SOH
    
    dxdt[17] = k[14] * (x[16] - x0[16]) * x[5] - k[15] * x[18] * (x[17] - x0[17])  # Pr-SSG
    
    dxdt[18] = k[16] * (x[19] - x0[19]) * x[5] - k[15] * x[18] * (x[17] - x0[17])  # Grx-SH
    
    dxdt[19] = k[15] * x[18] * (x[17] - x0[17]) - k[16] * (x[19] - x0[19]) * x[5]  # Grx-SSG
    
    dxdt[20] = -k[17] * x[20] * (x[1] - x0[1]) + k[18] * (x[21] - x0[21]) * x[13]  # Pr-(SH)2
    
    dxdt[21] = k[17] * x[20] * (x[1] - x0[1]) - k[18] * (x[21] - x0[21]) * x[13]  # Pr-SS
    
    dxdt[22] = (-k[19] * (x[6] - x0[6]) * x[22] - 
                k[20] * (x[14] - x0[14]) * x[22] - 
                k[28] * x[24] * x[22] + 
                k[21] * (x[23] - x0[23]) / (k[6] + x[23]))  # NADPH
    
    dxdt[23] = (k[19] * (x[6] - x0[6]) * x[22] + 
                k[20] * (x[14] - x0[14]) * x[22] + 
                k[28] * x[24] * x[22] - 
                k[21] * (x[23] - x0[23]) / (k[6] + x[23]))  # NADP+
    
    dxdt[24] = (k[33] * c_area * (x[29] - x[24]) / c_vol + 
                k[30] * x[26] * x[28] - k[28] * x[24] * x[22] - 
                k[32] * x[24] * x[22])  # b-lap Q cyto
    
    dxdt[25] = (k[28] * x[24] * x[22] - k[29] * x[25] * x[28] - 
                k[34] * x[25] * x[5] + k[32] * x[26] * x[22])  # b-lap HQ
    
    dxdt[26] = (k[29] * x[25] * x[28] - k[30] * x[26] * x[28] + 
                k[32] * x[24] * x[22] - k[32] * x[26] * x[22])  # b-lap SQ
    
    dxdt[27] = (k[29] * x[25] * x[28] 
                + k[30] * x[26] * x[28] 
                - k[31] * x[27]**2)  # superO2
    
    dxdt[28] = -k[29] * x[25] * x[28] - k[30] * x[26] * x[28] + k[31] * x[27]**2  # O2
    
    dxdt[29] = -k[33] * (x[29] - x[24]) / m_vol  # b-lap Q extracellular
    
    dxdt[30] = k[34] * x[25] * x[5] - k[35] * c_area * (x[30] - x[31]) / c_vol  # b-lapHQ-GSH
    
    dxdt[31] = k[35] * c_area * (x[30] - x[31]) / m_vol  # b-lapHQ-GSH extracellular
    
    return dxdt


if __name__ == "__main__":
    _x0 = get_x0()
    _k = get_k()
    _dx_full = model_full(_x0.copy(), 0.0, _k, _x0.copy())
    _dx_hill = model_hill(_x0.copy(), 0.0, _k, _x0.copy())
    print(f"H2O2 index: {H2O2I_INDEX}, Ca index: {CAI_INDEX}")
    print(f"full derivatives: dH={_dx_full[H2O2I_INDEX]}, dCa={_dx_full[CAI_INDEX]}")
    print(f"hill derivatives: dH={_dx_hill[H2O2I_INDEX]}, dCa={_dx_hill[CAI_INDEX]}")
