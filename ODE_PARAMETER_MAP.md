# ODE parameter index map

The model now has 63 parameters, `k[0]` through `k[62]`; `k[28]` onward are grouped by mechanism.
`k[0]` through `k[27]` retain their original indices, values, comments, and equation references.
R29 / `k[28]` is now the passive IP3R flux coefficient `P_IP3R = 0.038 s^-1`, using the user-selected Han et al. (2017) model benchmark.
The former R38 / `k[37]` (`K_er`) has been removed. Former `k[38]` through `k[63]` shift down by one index; their values and roles are preserved.
The IP3R calcium-inhibition exponent remains at `k[36]` (R37), with value 2.0. Species indices (`x` and `dxdt`) and the existing GSH correction are unchanged.

**Old** in the table is the original 63-parameter ordering, before any organization edits. **New** is the ordering now returned by `get_k()`.
For one-based reaction numbers, add 1 to either code index. In particular, new `k[37]` (R38) is SERCA Vmax; only the former `K_er` parameter was removed.

To migrate an array from the immediately preceding 64-parameter version, remove element 37 and set element 28 to 0.038. Earlier 63-element arrays use a different ordering and are not compatible merely because their length matches.
The active callers in `Simulation.py` and `simulationObjects.py` obtain `k` from `get_k()`; exports in `Simulation.py` use `len(k)` and now contain 63 elements.
No external hardcoded indices were found in the project source. `ODEs_new.py` is a separate unchanged model.
Existing Word tables and reference-audit reports retain their previous numbering; this code change does not update those artifacts.

## Groups

| Mechanism | New indices |
|---|---|
| Original core parameters in their original order | `k[0]–k[27]` |
| IP3R: passive ER calcium flux | `k[28]–k[36]` |
| SERCA and ER capacity | `k[37]–k[43]` |
| TRPM2: plasma-membrane calcium influx | `k[44]–k[48]` |
| SOCE: store-operated calcium influx | `k[49]–k[50]` |
| Cytosolic calcium extrusion | `k[51]` |
| NOX2 | `k[52]–k[53]` |
| NOX5: calcium-dependent ROS production | `k[54]–k[57]` |
| DUOX: calcium-dependent NADPH affinity | `k[58]–k[62]` |

## Individual parameters

| New | Old | Value | Role and units |
|---|---|---:|---|
| `k[0]` | `k[0]` | `2e-4` | Plasma-membrane H2O2 permeability (cm/s) |
| `k[1]` | `k[1]` | `1.1e-7` | Basal intracellular H2O2 production (M/s) |
| `k[2]` | `k[2]` | `2.1e7` | GPx_red oxidation by H2O2 (M^-1 s^-1) |
| `k[3]` | `k[3]` | `4e4` | GPx_ox reduction by GSH, forming GPx-SG (M^-1 s^-1) |
| `k[4]` | `k[4]` | `1e7` | GPx-SG reduction by GSH (M^-1 s^-1) |
| `k[5]` | `k[5]` | `3.4e7` | Catalase H2O2 removal; cytosolic and peroxisomal terms (M^-1 s^-1) |
| `k[6]` | `k[6]` | `5.7e-5` | NADP+ half-saturation constant for NADPH regeneration (M) |
| `k[7]` | `k[7]` | `4e7` | Prx-SH oxidation by H2O2 (M^-1 s^-1) |
| `k[8]` | `k[8]` | `7e4` | Prx-SOH hyperoxidation by H2O2 (M^-1 s^-1) |
| `k[9]` | `k[9]` | `3e-3` | Effective recovery of hyperoxidized Prx by Srx (s^-1) |
| `k[10]` | `k[10]` | `15` | Prx-SOH conversion to disulfide Prx-SS (s^-1) |
| `k[11]` | `k[11]` | `2.1e6` | Prx-SS reduction by reduced Trx (M^-1 s^-1) |
| `k[12]` | `k[12]` | `7.4e-05` | GSH auto-oxidation coefficient (s^-1) |
| `k[13]` | `k[13]` | `1e4` | Protein monothiol oxidation by H2O2 (M^-1 s^-1) |
| `k[14]` | `k[14]` | `1.2e5` | Protein-SOH glutathionylation by GSH (M^-1 s^-1) |
| `k[15]` | `k[15]` | `9.1e4` | Protein-SSG deglutathionylation by Grx-SH (M^-1 s^-1) |
| `k[16]` | `k[16]` | `3.7e4` | Grx-SSG reduction by GSH (M^-1 s^-1) |
| `k[17]` | `k[17]` | `5e5` | Protein dithiol oxidation by H2O2 (M^-1 s^-1) |
| `k[18]` | `k[18]` | `1e5` | Protein disulfide reduction by reduced Trx (M^-1 s^-1) |
| `k[19]` | `k[19]` | `3.2e6` | Effective GSSG reduction coefficient with NADPH (M^-1 s^-1) |
| `k[20]` | `k[20]` | `2e7` | Effective Trx-SS reduction coefficient with NADPH (M^-1 s^-1) |
| `k[21]` | `k[21]` | `3.75e-04` | Maximum NADPH regeneration flux (M/s) |
| `k[22]` | `k[22]` | `3e-3` | Peroxisomal-membrane H2O2 permeability (cm/s) |
| `k[23]` | `k[23]` | `4.1e-7` | GSH synthesis flux (M/s) |
| `k[24]` | `k[24]` | `1.2e-8` | Intended GSSG efflux scale, normalized to baseline GSSG (M/s) |
| `k[25]` | `k[25]` | `1.2e-7` | GSH efflux and intended additional GSSG efflux scale (M/s) |
| `k[26]` | `k[26]` | `7.45e-10` | Reduced-Trx efflux (M/s) |
| `k[27]` | `k[27]` | `6.97e-10` | Reduced-Trx synthesis (M/s) |
| `k[28]` | `k[30]` | `0.038` | P_IP3R: passive flux coefficient (s^-1); Han et al. (2017) model benchmark, doi:10.1371/journal.pcbi.1005275 |
| `k[29]` | `k[31]` | `1e-7` | K_IP3R_H: H2O2 half-activation constant (M) |
| `k[30]` | `k[32]` | `4.0` | n_IP3R_H: H2O2 activation Hill exponent (dimensionless) |
| `k[31]` | `k[53]` | `5e-7` | K_IP3R_Ca: cytosolic Ca half-activation constant (M) |
| `k[32]` | `k[50]` | `2.0` | n_IP3R_Ca: cytosolic Ca activation Hill exponent (dimensionless) |
| `k[33]` | `k[47]` | `5e-7` | K_IP3R_H_inhib: H2O2 half-inhibition constant (M) |
| `k[34]` | `k[48]` | `2.0` | m_IP3R_H_inhib: H2O2 inhibition Hill exponent (dimensionless) |
| `k[35]` | `k[49]` | `5e-5` | K_IP3R_Ca_inhib: cytosolic Ca half-inhibition constant (M) |
| `k[36]` | New (previously fixed at 2) | `2.0` | m_IP3R_Ca_inhib: cytosolic Ca inhibition Hill exponent (dimensionless) |
| `k[37]` | `k[34]` | `1e-5` | V_SERCA: uptake-flux scale before ER-capacity gating (M/s) |
| `k[38]` | `k[35]` | `0.23e-6` | K_ser: cytosolic Ca half-activation constant (M); doi:10.1074/jbc.M114.585604 |
| `k[39]` | `k[43]` | `2` | n_SERCA: cytosolic Ca activation Hill exponent (dimensionless) |
| `k[40]` | `k[36]` | `1e-5` | K_SERCA_H: H2O2 half-inhibition constant (M) |
| `k[41]` | `k[37]` | `2.0` | m_SERCA_H: H2O2 inhibition Hill exponent (dimensionless) |
| `k[42]` | `k[51]` | `1.1e-4` | ER Ca capacity used to gate SERCA uptake (M) |
| `k[43]` | `k[52]` | `1e-2` | Intended ER overflow-release rate constant (s^-1) |
| `k[44]` | `k[38]` | `9e-7` | P_TRPM2: effective Ca-influx coefficient (s^-1) |
| `k[45]` | `k[39]` | `5e-5` | K_TRPM2_H: H2O2 half-activation constant (M) |
| `k[46]` | `k[40]` | `4.0` | p_TRPM2: H2O2 activation Hill exponent (dimensionless) |
| `k[47]` | `k[61]` | `16.9e-6` | K_TRPM2_Ca: cytosolic Ca half-activation constant (M); doi:10.1073/pnas.0811725106 |
| `k[48]` | `k[62]` | `2` | n_TRPM2_Ca: cytosolic Ca activation Hill exponent (dimensionless) |
| `k[49]` | `k[41]` | `1e-7` | P_SOCE: effective Ca-influx coefficient (s^-1) |
| `k[50]` | `k[42]` | `1e-5` | K_store: ER Ca concentration for half-maximal SOCE gating (M) |
| `k[51]` | `k[54]` | `0.11` | V_out: maximum Ca-extrusion flux in J_out (M/s, as implemented) |
| `k[52]` | `k[28]` | `3.7e-5` | Maximum NOX2-associated extracellular H2O2 source (M/s) |
| `k[53]` | `k[29]` | `1e-7` | Cytosolic H2O2 half-activation constant for NOX2 (M) |
| `k[54]` | `k[44]` | `2.2e-4` | Vmax_NOX5: maximum cytosolic H2O2 source (M/s); doi:10.1016/j.redox.2020.101656 |
| `k[55]` | `k[45]` | `0.71e-6` | K_Ca_NOX5: Ca half-activation constant (M); doi:10.1038/s41467-024-48467-y |
| `k[56]` | `k[46]` | `4.0` | m_NOX5: Ca activation Hill exponent (dimensionless) |
| `k[57]` | `k[55]` | `13.67e-6` | Km_NOX5_NADPH: NADPH half-saturation constant (M); doi:10.1038/s41467-024-48467-y |
| `k[58]` | `k[56]` | `1e-9` | Vmax_DUOX: maximum extracellular H2O2 source (M/s, as implemented) |
| `k[59]` | `k[57]` | `0.87e-6` | K_DUOX_Ca: midpoint of the Ca-dependent NADPH-affinity switch (M) |
| `k[60]` | `k[58]` | `4` | m_DUOX: Ca-dependent affinity-switch Hill exponent (dimensionless) |
| `k[61]` | `k[59]` | `24.31e-6` | Km_DUOX_NADPH_low_Ca: positive low-Ca limit (M) |
| `k[62]` | `k[60]` | `11.52e-6` | Km_DUOX_NADPH_high_Ca: high-Ca limit (M) |

## Passive IP3R flux

`J_IP3R = P_IP3R * G(H, C) * (Cer - C)`, where `G` is the product of the existing four H2O2 and calcium activation/inhibition gates.
`P_IP3R` has units s^-1 and `J_IP3R` has units M/s. The former denominator `K_er + Cer` and the one-way gradient clamp are removed.
The flux is zero at equal ER and cytosolic calcium, positive for ER-to-cytosol release, and negative when the gradient reverses (subject to the same gates).

Reference: [Han et al. (2017), A mathematical model of calcium dynamics in HSY cells](https://doi.org/10.1371/journal.pcbi.1005275), Table 1 and the calcium-flux equations, gives an IP3R flux coefficient of 0.038 s^-1. Its original receiving compartment is a microdomain and its gating, buffering, and volume assumptions differ from this model. This selected value is a starting benchmark, not a calibration of this model.
The existing compartment balances and gating assumptions, including implicit IP3 availability, are retained.

## Scope

The calcium-inhibition gate remains `1 / (1 + (x[25] / k[35]) ** k[36])` for positive cytosolic calcium. The existing zero-calcium branch is unchanged.
The existing GSSG conditional-expression behavior, clamped ER overflow, and DUOX zero-Ca fallback are preserved.
Other hardcoded exponents, compartment geometry, and the Ca-extrusion half-saturation remain outside `k`.
