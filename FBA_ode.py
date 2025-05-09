import numpy as np
from scipy.integrate import odeint, solve_ivp
import random

def get_k():
        # constants

        # Defining forward reaction rates 
        # Defining the array, k, that will house reaction rates
        k = np.zeros(32)  # Defining the array, k, that will house reaction rates

        # Rxns (Cytosolic)

        k[0] = 1.0e-5  # Permeability constant (plasma membrane) (cm/s)
        k[1] = 1.1e-7  # Intracellular peroxide production (M/s)
        k[2] = 2.1e7  # GPx_red reacting with H2O2in (M-1*s-1)
        k[3] = 4e4  # GPx_ox reacting with GSH (M-1*s-1)
        k[4] = 1e7  # GPx-SSG reacting with GSH (M-1*s-1)
        k[5] = 3.4e7  # Catalase reacts with H2O2 inside peroxisome (M-1*s-1)
        k[6] = 5.7e-5  # Km of NADP+ (M)
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
        k[21] = 3.75e-04  # Production of NADPH by G6P-DH (M/s)
        k[22] = 3e-3  # Permeability constant (peroxisomal membrane) (cm/s)
        k[23] = 4.1e-7  # GSH synthesis (M/s)
        k[24] = 1.2e-8  # GSSG efflux (M/s)
        k[25] = 1.2e-7  # GSH + GSSG efflux (M/s)
        k[26] = 7.45e-10  # Trx_SH efflux (M/s)
        k[27] = 6.97e-10  # Trx_SH synthesis (M/s)
        k[28] = 1e-5 # nox generation of superoxide ()    #Banfi B. et al, JBC 2002
        k[29] = 4e9 # superoxide dismutase reduction and oxidation by superoxide (M-1*s-1), from Fee J.A. & Bull C. The journal of biological chemistry 1986
        k[30] = 1e-4 # intracellular superoxide production (M/s)
        k[31] = 0

        return k

def get_x0():
        x0 = np.zeros(28)  # Defining the array, x0, that will house initial concentrations
        # Cytoplasm
        x0[0] = 0#1e-4  # H2O2media 
        x0[1] = 1.0e-8  # H2O2cytosol

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

        return x0

# ODE system
def model(x, t, k, x0):
    # if x0 is None:
    #     x0 = get_x0()

    dxdt = np.zeros_like(x)
    dxdt[0] = k[0] * (x[1] - x[0]) * 10e-3 * 1.02e-5 * (1e9) + k[31]

    dxdt[1] = (((k[0] * x[0] * 10e-3 * 1.02e-5) / (9.16e-13))
               - ((k[0] * x[1] * 10e-3 * 1.02e-5) / (9.16e-13))
               - 2 * (k[22] * x[1] * 10e-3 * 2.98e-9) / (1.53e-14)              # peroxisomal H2O2     
               - k[2] * x[2] * (x[1] - x0[1])                                   # GPx reduction of H2O2
               - k[5] * x[7] * (x[1] - x0[1])                                   # catalase reaction
               - k[7] * x[9] * (x[1] - x0[1])                                   # peroxiredoxin oxidation
               - k[8] * x[10] * (x[1] - x0[1])                                  # overoxidized peroxiredoxin
               - k[13] * x[15] * (x[1] - x0[1])                                 # Protein monothiol oxidation
               - k[17] * x[20] * (x[1] - x0[1])                                 # Protein dithiol oxidation
               + k[1]                                                          # intracellular peroxide production
               + k[29] * x[25] * x[27]                                      # superoxide dismutase reduction and oxidation by superoxide
               )
    
    dxdt[2] = - k[2] * x[2] * (x[1] - x0[1]) + k[4] * (x[4] - x0[4]) * x[5]
    dxdt[3] = k[2] * x[2] * (x[1] - x0[1]) - k[3] * (x[3] - x0[3]) * x[5]
    dxdt[4] = k[3] * (x[3] - x0[3]) * x[5] - k[4] * (x[4] - x0[4]) * x[5]
    dxdt[5] = (- k[3] * (x[3] - x0[3]) * x[5]
               - k[4] * (x[4] - x0[4]) * x[5]
               - 2 * k[12] * x[5]
               - k[14] * (x[16] - x0[16]) * x[5]
               - k[16] * (x[19] - x0[19]) * x[5]
               + 2 * k[19] * (x[6] - x0[6]) * x[22]
               + k[23] - k[25])
    dxdt[6] = (k[4] * (x[4] - x0[4]) * x[5] + k[12] * x[5] +
               k[16] * (x[19] - x0[19]) * x[5] - k[19] * (x[6] - x0[6]) * x[22] -
               k[24] - k[25])
    dxdt[7] = 0
    dxdt[8] = (k[22] * x[1] * 10e-3 * 2.98e-9) / (1.53e-14) - k[5] * x[7] * (x[8] - x0[8])
    dxdt[9] = - k[7] * x[9] * (x[1] - x0[1]) + k[11] * (x[12] - x0[12]) * x[13]
    dxdt[10] = (k[7] * x[9] * (x[1] - x0[1]) - k[8] * x[10] * (x[1] - x0[1]) +
                k[9] * (x[11] - x0[11]) - k[10] * x[10])
    dxdt[11] = k[8] * x[10] * (x[1] - x0[1]) - k[9] * (x[11] - x0[11])
    dxdt[12] = k[10] * x[10] - k[11] * (x[12] - x0[12]) * x[13]
    dxdt[13] = (- k[11] * (x[12] - x0[12]) * x[13] 
                - k[18] * (x[21] - x0[21]) * x[13] 
                + k[20] * (x[14] - x0[14]) * x[22] 
                - k[26] 
                + k[27])
    dxdt[14] = (k[11] * (x[12] - x0[12]) * x[13] + k[18] * (x[21] - x0[21]) * x[13] -
                k[20] * (x[14] - x0[14]) * x[22])
    
    dxdt[15] = (- k[13] * x[15] * (x[1] - x0[1]) 
                + k[15] * x[18] * (x[17] - x0[17]))
    
    dxdt[16] = k[13] * x[15] * (x[1] - x0[1]) - k[14] * (x[16] - x0[16]) * x[5]
    dxdt[17] = k[14] * (x[16] - x0[16]) * x[5] - k[15] * x[18] * (x[17] - x0[17])
    dxdt[18] = k[16] * (x[19] - x0[19]) * x[5] - k[15] * x[18] * (x[17] - x0[17])
    dxdt[19] = k[15] * x[18] * (x[17] - x0[17]) - k[16] * (x[19] - x0[19]) * x[5]
    dxdt[20] = - k[17] * x[20] * (x[1] - x0[1]) + k[18] * (x[21] - x0[21]) * x[13]
    dxdt[21] = k[17] * x[20] * (x[1] - x0[1]) - k[18] * (x[21] - x0[21]) * x[13]

    #NADPH
    dxdt[22] = (- k[19] * (x[6] - x0[6]) * x[22] 
                - k[20] * (x[14] - x0[14]) * x[22] 
                - k[28] * x[22]                                         # nox generation of superoxide
                + k[21] * (x[23] - x0[23]) / (k[6] + x[23]))
    #NADP+
    dxdt[23] = (k[19] * (x[6] - x0[6]) * x[22] 
                + k[20] * (x[14] - x0[14]) * x[22] 
                + k[28] * x[22]                                         # nox generation of superoxide and nadp+
                - k[21] * (x[23] - x0[23]) / (k[6] + x[23]))
    
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
    
    return dxdt

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
    # Establish genes of interest and index
    genes = ["GPX1", "GPX2", "GPX3", "GPX4", "GPX5", "GPX6", "GPX7", "GPX8", "PRDX1", "PRDX2",
             "PRDX2", "PRDX3", "PRDX6", "CAT", "TXN", "TXN2", "TXNRD1", "TXNRD2", "TXNRD3", "GLRX", 
             "GLRX2", "G6PD", "GLUD1", "GSR", "GSTP1", "POR", "NQO1", "SOD1", "SOD2", "SOD3", 
             "AQP3", "AQP8", "AQP9", "GCLC"]
    
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
