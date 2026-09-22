import numpy as np
from scipy.integrate import odeint, solve_ivp
import random

def get_k():
        k = np.zeros(63)
        k[0] = 2e-4 # Permeability constant (plasma membrane) (cm/s)
        k[1] = 1.1e-7 # Intracellular peroxide production (M/s)
        k[2] = 2.1e7 # GPx_red reacting with H2O2in (M-1*s-1)
        k[3] = 4e4 # GPx_ox reacting with GSH (M-1*s-1)
        k[4] = 1e7 # GPx-SSG reacting with GSH (M-1*s-1)
        k[5] = 3.4e7 # Catalase reacts with H2O2 inside peroxisome (M-1*s-1)
        k[6] = 5.7e-5 # Km of NADP+ (M)
        k[7] = 4e7#100e-6 # Peroxiredoxin is oxidized by H2O2 (M-1*s-1)
        k[8] = 7e4#0.0049e-6 # Oxidized peroxiredoxin is over-oxidized by H2O2 (M-1*s-1)
        k[9] = 3e-3 # Reduction of overoxidized Prx by Srx enzyme (s-1)
        k[10] = 15 # Self-catalyzed disulfide formation of Prx-SS from Prx-SOH (s-1)
        k[11] = 2.1e6 # Peroxiredoxin is reduced by Thioredoxin (M-1*s-1)
        k[12] = 7.4e-05 # Auto-oxidation of GSH (s-1)
        k[13] = 1e4 # Protein monothiol oxidized by H2O2 (M-1*s-1)
        k[14] = 1.2e5 # Oxidized Protein monothiol glutathionylated by GSH (M-1*s-1)
        k[15] = 9.1e4 # Grx-SH de-glutathionylates Protein-SSG (M-1*s-1)
        k[16] = 3.7e4 # GSH de-glutathionylates Grx-SSG (M-1*s-1)
        k[17] = 5e5 # Protein dithiol oxidized by H2O2 (M-1*s-1)
        k[18] = 1e5 # Protein disulfide reduced by Trx (M-1*s-1)
        k[19] = 3.2e6 # GSSG reduced by GR (M-1*s-1)
        k[20] = 2e7 # Oxidized Thioredoxin reduced by TrxR (M-1*s-1)
        k[21] = 3.75e-04 # Production of NADPH by G6P-DH (M/s)
        k[22] = 3e-3 # Permeability constant (peroxisomal membrane) (cm/s)
        k[23] = 4.1e-7 # GSH synthesis (M/s)
        k[24] = 1.2e-8 # GSSG efflux (M/s)
        k[25] = 1.2e-7 # GSH + GSSG efflux (M/s)
        k[26] = 7.45e-10 # Trx_SH efflux (M/s)
        k[27] = 6.97e-10 # Trx_SH synthesis (M/s)

        # IP3R: passive ER calcium flux: k[28]–k[36]
        k[28] = 0.038  # P_IP3R: passive flux coefficient (s^-1); Han et al. model benchmark, doi:10.1371/journal.pcbi.1005275
        k[29] = 1e-7  # K_IP3R_H: H2O2 half-activation constant (M)
        k[30] = 4.0  # n_IP3R_H: H2O2 activation Hill exponent (dimensionless)
        k[31] = 5e-7  # K_IP3R_Ca: cytosolic Ca half-activation constant (M)
        k[32] = 2.0  # n_IP3R_Ca: cytosolic Ca activation Hill exponent (dimensionless)
        k[33] = 5e-7  # K_IP3R_H_inhib: H2O2 half-inhibition constant (M)
        k[34] = 2.0  # m_IP3R_H_inhib: H2O2 inhibition Hill exponent (dimensionless)
        k[35] = 5e-5  # K_IP3R_Ca_inhib: cytosolic Ca half-inhibition constant (M)
        k[36] = 2.0  # m_IP3R_Ca_inhib: cytosolic Ca inhibition Hill exponent (dimensionless; R37)

        # SERCA and ER capacity: k[37]–k[43]
        k[37] = 1e-5  # V_SERCA: uptake-flux scale before ER-capacity gating (M/s)
        k[38] = 0.23e-6  # K_ser: cytosolic Ca half-activation constant (M); doi:10.1074/jbc.M114.585604
        k[39] = 2  # n_SERCA: cytosolic Ca activation Hill exponent (dimensionless)
        k[40] = 1e-5  # K_SERCA_H: H2O2 half-inhibition constant (M)
        k[41] = 2.0  # m_SERCA_H: H2O2 inhibition Hill exponent (dimensionless)
        k[42] = 1.1e-4  # ER Ca capacity used to gate SERCA uptake (M)
        k[43] = 1e-2  # Intended ER overflow-release rate constant (s^-1)

        # TRPM2: plasma-membrane calcium influx: k[44]–k[48]
        k[44] = 9e-7  # P_TRPM2: effective Ca-influx coefficient (s^-1)
        k[45] = 5e-5  # K_TRPM2_H: H2O2 half-activation constant (M)
        k[46] = 4.0  # p_TRPM2: H2O2 activation Hill exponent (dimensionless)
        k[47] = 16.9e-6  # K_TRPM2_Ca: cytosolic Ca half-activation constant (M); doi:10.1073/pnas.0811725106
        k[48] = 2  # n_TRPM2_Ca: cytosolic Ca activation Hill exponent (dimensionless)

        # SOCE: store-operated calcium influx: k[49]–k[50]
        k[49] = 1e-7  # P_SOCE: effective Ca-influx coefficient (s^-1)
        k[50] = 1e-5  # K_store: ER Ca concentration for half-maximal SOCE gating (M)

        # Cytosolic calcium extrusion: k[51]
        k[51] = 0.11  # V_out: maximum Ca-extrusion flux in J_out (M/s, as implemented)

        # NOX2: k[52]–k[53]
        k[52] = 3.7e-5  # Maximum NOX2-associated extracellular H2O2 source (M/s)
        k[53] = 1e-7  # Cytosolic H2O2 half-activation constant for NOX2 (M)

        # NOX5: calcium-dependent ROS production: k[54]–k[57]
        # Original commented alternative: 8.7e-3
        k[54] = 2.2e-4  # Vmax_NOX5: maximum cytosolic H2O2 source (M/s); doi:10.1016/j.redox.2020.101656
        k[55] = 0.71e-6  # K_Ca_NOX5: Ca half-activation constant (M); doi:10.1038/s41467-024-48467-y
        k[56] = 4.0  # m_NOX5: Ca activation Hill exponent (dimensionless)
        k[57] = 13.67e-6  # Km_NOX5_NADPH: NADPH half-saturation constant (M); doi:10.1038/s41467-024-48467-y

        # DUOX: calcium-dependent NADPH affinity: k[58]–k[62]
        k[58] = 1e-9  # Vmax_DUOX: maximum extracellular H2O2 source (M/s, as implemented)
        k[59] = 0.87e-6  # K_DUOX_Ca: midpoint of the Ca-dependent NADPH-affinity switch (M)
        k[60] = 4  # m_DUOX: Ca-dependent affinity-switch Hill exponent (dimensionless)
        k[61] = 24.31e-6  # Km_DUOX_NADPH_low_Ca: positive low-Ca limit (M)
        k[62] = 11.52e-6  # Km_DUOX_NADPH_high_Ca: high-Ca limit (M)

        return k

def get_x0():
        # initial conditions
        x0 = np.zeros(28)  # Defining the array, x0, that will house initial concentrations
        # Cytoplasm
        x0[0] = 1e-9  # H2O2media 
        x0[1] = 1e-9  # H2O2cytosol

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

        x0[24] = 1e-9 # steady state h2o2 intracellular

        x0[25] = 1e-7 # calcium intracellular
        x0[26] = 1.5e-3 # calcium extracellular
        x0[27] = 1.0e-4 # calcium ER [M] ~ 0.1 mM

        return x0

def model_noref(x, t, k, x_base):
    dxdt = np.zeros_like(x)

    H   = x[1]     # H2O2 cytosol
    C   = x[25]    # Ca cytosol
    Cext= x[26]    # Ca extracellular (constant ~1-2 mM)
    Cer = x[27]    # Ca ER pool

    # Unpack from k
    P_IP3R     = k[28]; K_IP3R_H = k[29]; nH = k[30]
    V_SERCA   = k[37]; K_ser    = k[38]; K_SERCA_H = k[40]; m_inhib = k[41]
    P_TRPM2   = k[44]; K_TRPM2_H= k[45]; p_TRPM2 = k[46]
    P_SOCE    = k[49]; K_store  = k[50]
    k_out     = k[39]
    Vmax_NOX5 = k[54]; K_Ca     = k[55]; m_nox = k[56]

    ##  Helper gates (pure functions of species & k)
    Ca_gate_IP3R = (x[25])**k[32] / (k[31]**k[32] + (x[25])**k[32]) if x[25] > 0 else 0.0 # Activated by cyt Ca
    Ca_IP3R_inhib = 1.0/(1.0 + (x[25]/k[35])**k[36]) if x[25] > 0 else 0.0 # Inhibited by high Ca
    h2o2_gate_IP3R = (H**nH)/(K_IP3R_H**nH + H**nH) if H > 0 else 0.0 # activation by H2o2
    H2O2_inhibit_IP3R = (k[33]**k[34]) / (k[33]**k[34] + H**k[34]) if H > 0 else 0.0 # inhibition by very high H2O2

    SERCA_inhb = 1.0/(1.0 + (H/K_SERCA_H)**m_inhib) if H > 0 else 1.0
    H2O2_gate_TRPM2 = (H**p_TRPM2)/(K_TRPM2_H**p_TRPM2 + H**p_TRPM2) if H > 0 else 0.0
    Ca_gate_TRPM2 = (x[25]**k[48])/((k[47]**k[48]) + (x[25]**k[48])) if x[25] > 0 else 0.0
    SOCE_drive = K_store/(K_store + Cer)       # ramps as ER calcium depletes

    # fluxes
    # Signed passive flux (M/s): positive from ER to cytosol; reverses with the gradient.
    J_IP3R = (P_IP3R * h2o2_gate_IP3R * Ca_gate_IP3R
              * Ca_IP3R_inhib * H2O2_inhibit_IP3R * (Cer - C))

    J_SERCA   = V_SERCA * (C**k[39]/(k[38]**k[39] + C**k[39])) * SERCA_inhb if C > 0 else 0.0 # cyt -> ER uptake by SERCA
    J_in_TRPM2= P_TRPM2 * H2O2_gate_TRPM2 * Ca_gate_TRPM2 * (x[26] - x[25]) if (x[26] - x[25]) > 0 else 0.0 # PM influx via TRPM2
    J_in_SOCE = P_SOCE  * SOCE_drive * (x[26] - x[25]) if (x[26] - x[25]) > 0 else 0.0 # PM influx via SOCE to restore ER

    # --- ER capacity gating for SERCA (prevents overfilling) ---
    cap = 1.0 - (x[27] / k[42])         # linear headroom fraction
    cap = 0.0 if cap < 0.0 else cap   # clamp to [0, 1]
    J_SERCA *= cap                    # SERCA stops as ER approaches capacity

    # --- Overflow if ER exceeds the cap (hard push back to cytosol) ---
    J_over = k[43] * cap * x[27] if cap < 0.0 else 0.0
    J_out = k[51] * (x[25])/ (5e-6 + (x[25]))  if x[25] > 0 else 0.0 # cytosolic Ca extrusion (PMCA/NCX lumped)
    # --- NOX5 Ca-dependent ROS production ---
    nox2 = k[52] * (x[1]**6)/(k[53]**6 + x[1]**6) if x[1] > 0 else 0.0 #nox2
    # NOX5 with NADPH limitation
    nox5 = k[54] * (x[25]**k[56])/(k[55]**k[56] + x[25]**k[56]) \
                * (x[22]/(k[57] + x[22])) if x[25] > 0 or x[22] > 0 else 0.0

    km_duox = k[61] - (k[61] - k[62]) * (x[25]**k[60])/(k[59]**k[60] + x[25]**k[60]) if x[25] > 0 else k[62]
    duox = k[58] * x[22]/(km_duox + x[22])

    dxdt[0] = (k[0]*(x[1] - x[0]) * 1e-3 * 5.3e-6 * (1e9)) + nox2 + duox   # extracellular peroxide

    dxdt[1] = (((k[0] * x[0] * 1e-3 * 5.3e-6) / (1.15e-12))
               - ((k[0] * x[1] * 1e-3 * 5.3e-6) / (1.15e-12))
               - 2 * (k[22] * x[1] * 1e-3 * 2.98e-9) / (1.53e-14)              # peroxisomal H2O2     
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
               - k[16] * x[19] * x[5]
               + 2 * k[19] * x[6] * x[22]
               + k[23] 
               - k[25])
    #GSSG
    dxdt[6] = (k[4] * x[4]* x[5] 
               + k[12] * x[5] 
               + k[16] * x[19] * x[5] 
               - (k[19] * x[6] * x[22] if x[6] > 0 and x[22] > 0 else 0.0)
               - (k[24] * x[6] / x_base[6] if x_base[6] > 0 else 0.0)
               - (k[25] * x[6] / x_base[6] if x_base[6] > 0 else 0.0))

    dxdt[7] = 0
    dxdt[8] = ((k[22] * x[1] * 1e-3 * 2.98e-9) / (1.53e-14)            # peroxisomal H2O2
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
                - (2 * nox2 if x[22] > 0 else 0.0)
                - (2 * nox5 if x[22] > 0 else 0.0)
                + (k[21] * x[23] / (k[6] + x[23]) if x[23] > 0 or x[22] > 0 else 0.0))
    #NADP+
    dxdt[23] = (k[19] * x[6] * x[22] 
                + k[20] * x[14] * x[22] 
                + 2 * nox2
                + 2 * nox5
                - (k[21] * x[23] / (k[6] + x[23]) if x[23] > 0 else 0.0))
    # Steady-state intracellular H2O2 reference (constant)
    dxdt[24] = 0

    # --- Calcium balances --
    dxdt[25] = J_IP3R + J_in_TRPM2 + J_in_SOCE + J_over - J_SERCA - J_out            # d[Ca_cyt]/dt
    dxdt[26] = 0                                                     # d[Ca_ext]/dt
    dxdt[27] = J_SERCA - J_IP3R - J_over                           # d[Ca_ER]/dt
    #print(dxdt[25], dxdt[27])

    return dxdt