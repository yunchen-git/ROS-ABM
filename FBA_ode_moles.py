################################################################################
# File: FBA_ode_moles.py
# Purpose:
#   Variant of the intracellular ODE model expressed in mol (amount) units
#   rather than molar concentration units. Useful for alternative parameter
#   scalings or validation runs.
#
# Key public API:
#   - get_k(), get_x0(), model(...)
#
# How it fits in the codebase:
#   Not always used by the main ABM loop; kept as an alternate model backend.
#
# Inputs/Outputs:
#   - No file I/O. Numerical RHS for SciPy integrators.
#
# Notes / gotchas:
#   - Keep consistent state ordering with FBA_ode.py when comparing results.
################################################################################
import numpy as np
from scipy.integrate import odeint, solve_ivp
import random

def get_k():
        # constants

        # Defining forward reaction rates 
        # Defining the array, k, that will house reaction rates
        k = np.zeros(33)  # Defining the array, k, that will house reaction rates

        # Rxns (Cytosolic)

        k[0] = 1.0e-5  # Permeability constant (plasma membrane) (cm/s) [unchanged]
        k[1] = 1.0076e-19  # Intracellular peroxide production (mol/s)  [was 1.1e-7 M/s]
        k[2] = 2.1e7  # GPx_red reacting with H2O2in (M-1*s-1)
        k[3] = 4e4  # GPx_ox reacting with GSH (M-1*s-1)
        k[4] = 1e7  # GPx-SSG reacting with GSH (M-1*s-1)
        k[5] = 3.4e7  # Catalase reacts with H2O2 inside peroxisome (M-1*s-1)
        k[6] = 5.2212e-17  # Km of NADP+ (mol)  [was 5.7e-5 M]
        k[7] = 4e7  # Peroxiredoxin is oxidized by H2O2 (M-1*s-1)
        k[8] = 7.2e4  # Oxidized peroxiredoxin is over-oxidized by H2O2 (M-1*s-1)
        k[9] = 3e-3  # Reduction of overoxidized Prx by Srx enzyme (s-1)
        k[10] = 15  # Self-catalyzed disulfide formation of Prx-SS from Prx-SOH (s-1)
        k[11] = 2.1e6  # Peroxiredoxin is reduced by Thioredoxin (M-1*s-1)
        k[12] = 7.4e-05  # Auto-oxidation of GSH (s-1)
        k[13] = 1e4  # Protein monothiol oxidized by H2O2 (M-1*s-1)
        k[14] = 1.2e5  # Oxidized Protein monothiol glutathionylated by GSH (M-1*s-1)
        k[15] = 9.1e4  # Grx-SH de-glutathionylates Protein-SSG (M-1*s-1)
        k[16] = 3.7e4  # GSH de-glutathionylates Grx-SSG (M-1*s-1)
        k[17] = 5e5  # Protein dithiol oxidized by H2O2 (M-1*s-1)
        k[18] = 1e5  # Protein disulfide reduced by Trx (M-1*s-1)
        k[19] = 3.2e6  # GSSG reduced by GR (M-1*s-1)
        k[20] = 2e7  # Oxidized Thioredoxin reduced by TrxR (M-1*s-1)
        k[21] = 3.435e-16  # Production of NADPH by G6P-DH (mol/s)  [was 3.75e-04 M/s]
        k[22] = 3e-3  # Permeability constant (peroxisomal membrane) (cm/s)
        k[23] = 3.7556e-19  # GSH synthesis (mol/s)  [was 4.1e-7 M/s]
        k[24] = 1.0992e-20  # GSSG efflux (mol/s)  [was 1.2e-8 M/s]
        k[25] = 1.0992e-19  # GSH + GSSG efflux (mol/s)  [was 1.2e-7 M/s]
        k[26] = 6.8242e-22  # Trx_SH efflux (mol/s)  [was 7.45e-10 M/s]
        k[27] = 6.38452e-22  # Trx_SH synthesis (mol/s)  [was 6.97e-10 M/s]
        k[28] = 0  # nox generation of superoxide (s-1, effectively zero here)
        k[29] = 0  # superoxide dismutase reduction and oxidation by superoxide (M-1*s-1)
        k[30] = 0  # intracellular superoxide production (mol/s)  [was 0 M/s]
        k[31] = 0  # nox h2o2 production (mol/s)  [was 0 M/s]
        k[32] = 0  # extracellular vesicle production (mol/s?)  [was 0 M/s]

        return k

def get_x0():
        x0 = np.zeros(30)  # Defining the array, x0, that will house initial concentrations
        # Cytoplasm
        x0[0] = 0  # H2O2media 
        x0[1] = 9.16e-22 # H2O2cytosol

        x0[2] = 4.58e-17  # GPXr 
        x0[3] = 9.16e-27  # GPXo 
        x0[4] = 9.16e-27  # GPX-SG 

        x0[5] = 3.37088e-16  # GSH 
        x0[6] = 1.63048e-18  # GSSG 

        x0[7] = 8.244e-19  # Catalase 
        x0[8] = 9.16e-23  # H2O2peroxisome 

        x0[9] = 1.75872e-17  # Prx-SH  
        x0[10] = 9.16e-27  # Prx-SOH 
        x0[11] = 9.16e-27  # Prx-SOOH 
        x0[12] = x0[9] * (0.5 / 100)  # Prx-SS  (preserve original dependency)

        x0[13] = 3.91132e-19  # Trx1-SH 
        x0[14] = 6.90664e-20  # Trx1-SS 

        x0[15] = 1.11752e-16  # Pr-SH 
        x0[16] = x0[15] * (0.5 / 100)  # Pr-SOH
        x0[17] = x0[15] * (0.5 / 100)  # Pr-SSG 

        x0[18] = 1.0992e-18  # Grx-SH 
        x0[19] = x0[18] * (0.5 / 100)  # Grx-SSG 

        x0[20] = 9.9844e-16  # Pr-(SH)2  
        x0[21] = x0[20] * (0.5 / 100)  # Pr-SS

        x0[22] = 2.748e-17  # NADPH 
        x0[23] = 2.748e-19  # NADP+

        x0[24] = 0.0 # superoxide media
        x0[25] = 0.0 # superoxide cytosol

        x0[26] = 0.0 # superoxide dismutase reduced
        x0[27] = 0.0 # superoxide dismutase oxidized

        #x0[28] = 0 # calcium intracellular
        #x0[29] = 0 # calcium extracellular

        return x0

# ODE system
def model(n, t, k, n0):
    """
    Amount-based RHS.
    Inputs n, n0 are amounts (mol). k is unchanged from get_k() (still in M, M/s, M^-1*s^-1, etc.).
    This converts n->concentrations, evaluates your original concentration RHS,
    then returns mol/s by multiplying by V.
    """
    import numpy as np

    V = 9.16e-13  # L (single compartment volume you specified)

    # concentrations (M) from amounts (mol)
    c  = n  / V
    c0 = n0 / V

    dc = np.zeros_like(c)

    # ---- ORIGINAL concentration-form RHS (unchanged, but using c/c0 instead of x/x0) ----
    dc[0] = (k[0] * (c[1] - n[0]) * 10e-3 * 1.02e-5) + k[31]

    dc[1] = (((k[0] * n[0] * 10e-3 * 1.02e-5) / (9.16e-13))
             - ((k[0] * c[1] * 10e-3 * 1.02e-5) / (9.16e-13))
             #- 2 * (k[22] * c[1] * 10e-3 * 2.98e-9) / (1.53e-14)              # peroxisomal H2O2
             - k[2] * c[2] * (c[1] - c0[1])                                   # GPx reduction of H2O2
             - k[5] * c[7] * (c[1] - c0[1])                                   # catalase reaction
             - k[7] * c[9] * (c[1] - c0[1])                                   # peroxiredoxin oxidation
             - k[8] * c[10] * (c[1] - c0[1])                                  # overoxidized peroxiredoxin
             - k[13] * c[15] * (c[1] - c0[1])                                 # Protein monothiol oxidation
             - k[17] * c[20] * (c[1] - c0[1])                                 # Protein dithiol oxidation
             + k[1]                                                           # intracellular peroxide production
             + k[29] * c[25] * c[27])                                         # SOD + superoxide

    dc[2] = (- k[2] * c[2] * (c[1] - c0[1])
             + k[4] * (c[4] - c0[4]) * c[5])
    dc[3] = k[2] * c[2] * (c[1] - c0[1]) - k[3] * (c[3] - c0[3]) * c[5]
    dc[4] = k[3] * (c[3] - c[0+3]) * c[5] - k[4] * (c[4] - c0[4]) * c[5]
    dc[5] = (- k[3] * (c[3] - c0[3]) * c[5]
             - k[4] * (c[4] - c0[4]) * c[5]
             - 2 * k[12] * c[5]
             - k[14] * (c[16] - c0[16]) * c[5]
             - k[16] * (c[19] - c0[19]) * c[5]
             + 2 * k[19] * (c[6] - c0[6]) * c[22]
             + k[23]
             - k[25])
    dc[6] = (k[4] * (c[4] - c0[4]) * c[5] + k[12] * c[5] +
             k[16] * (c[19] - c0[19]) * c[5] - k[19] * (c[6] - c0[6]) * c[22] -
             k[24] - k[25])
    dc[7] = 0
    dc[8] = (k[22] * c[1] * 10e-3 * 2.98e-9) / (1.53e-14) - k[5] * c[7] * (c[8] - c0[8])
    dc[9] = - k[7] * c[9] * (c[1] - c0[1]) + k[11] * (c[12] - c0[12]) * c[13]
    dc[10] = (k[7] * c[9] * (c[1] - c0[1]) - k[8] * c[10] * (c[1] - c0[1]) +
              k[9] * (c[11] - c0[11]) - k[10] * c[10])
    dc[11] = k[8] * c[10] * (c[1] - c0[1]) - k[9] * (c[11] - c0[11])
    dc[12] = k[10] * c[10] - k[11] * (c[12] - c0[12]) * c[13]
    dc[13] = (- k[11] * (c[12] - c0[12]) * c[13]
              - k[18] * (c[21] - c0[21]) * c[13]
              + k[20] * (c[14] - c0[14]) * c[22]
              - k[26]
              + k[27])
    dc[14] = (k[11] * (c[12] - c0[12]) * c[13] + k[18] * (c[21] - c0[21]) * c[13] -
              k[20] * (c[14] - c0[14]) * c[22])
    dc[15] = (- k[13] * c[15] * (c[1] - c0[1])
              + k[15] * c[18] * (c[17] - c0[17]))
    dc[16] = k[13] * c[15] * (c[1] - c0[1]) - k[14] * (c[16] - c0[16]) * c[5]
    dc[17] = k[14] * (c[16] - c0[16]) * c[5] - k[15] * c[18] * (c[17] - c0[17])
    dc[18] = k[16] * (c[19] - c0[19]) * c[5] - k[15] * c[18] * (c[17] - c0[17])
    dc[19] = k[15] * c[18] * (c[17] - c0[17]) - k[16] * (c[19] - c0[19]) * c[5]
    dc[20] = - k[17] * c[20] * (c[1] - c0[1]) + k[18] * (c[21] - c0[21]) * c[13]
    dc[21] = k[17] * c[20] * (c[1] - c0[1]) - k[18] * (c[21] - c0[21]) * c[13]

    # NADPH / NADP+
    dc[22] = (- k[19] * (c[6] - c0[6]) * c[22]
              - k[20] * (c[14] - c0[14]) * c[22]
              - k[28] * c[22]
              + k[21] * (c[23] - c0[23]) / (k[6] + c[23]))
    dc[23] = (k[19] * (c[6] - c0[6]) * c[22]
              + k[20] * (c[14] - c0[14]) * c[22]
              + k[28] * c[22]
              - k[21] * (c[23] - c0[23]) / (k[6] + c[23]))

    # Superoxide
    dc[24] = (k[0] * (c[25] - c[24]) * 10e-3 * 1.02e-5 * (1e9)  # media
              + k[28] * c[22])
    dc[25] = (((k[0] * c[24] * 10e-3 * 1.02e-5) / (9.16e-13))    # cytosol
              - ((k[0] * c[25] * 10e-3 * 1.02e-5) / (9.16e-13))
              - k[29] * c[25] * c[26]
              - k[29] * c[25] * c[27]
              + k[30])
    dc[26] = (k[29] * c[25] * c[27] - k[29] * c[25] * c[26])
    dc[27] = (k[29] * c[25] * c[26] - k[29] * c[25] * c[27])
    # -----------------------------------------------------------------------------

    # convert back to amounts: dn/dt = (dC/dt) * V
    dn = dc * V
    return dn