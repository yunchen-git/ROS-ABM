"""Opt-in JAX experimental RHS implementations.

This module is intentionally isolated from the legacy runtime path.
"""

from ros_abm.backends.jax_exp.ode.schema import (
    HILL2_CAI_INDEX,
    HILL2_H2O2I_INDEX,
    HILL2_STATE_LENGTH,
    LEGACY_STATE_LENGTH,
)

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover - environment dependent
    jax = None
    jnp = None
    _JAX_IMPORT_ERROR = exc
else:
    _JAX_IMPORT_ERROR = None


H2O2I_INDEX = 1
CAI_INDEX = 29

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


def _require_jax():
    if jnp is None:  # pragma: no cover - environment dependent
        raise ImportError("jax is required for ros_abm.backends.jax_exp modules") from _JAX_IMPORT_ERROR


def _hill_gate(y, k_half, power):
    y = jnp.maximum(y, 0.0)
    y_pow = y ** power
    return y_pow / ((k_half ** power) + y_pow + HILL_EPS)


def _resolve_param(params, name, default):
    if params is None:
        return default
    if hasattr(params, "get"):
        return params.get(name, default)
    return getattr(params, name, default)


def _where(cond, yes, no):
    return jnp.where(cond, yes, no)


def model_hill2_jax(hill_state, params=None):
    """Minimal Hill RHS on (..., 2) state shaped as [h2o2i, cai]."""
    _require_jax()

    state_arr = jnp.asarray(hill_state)
    if state_arr.shape[-1] != HILL2_STATE_LENGTH:
        raise ValueError(
            f"hill_state must have last dimension {HILL2_STATE_LENGTH}; got shape {state_arr.shape}."
        )

    a0 = _resolve_param(params, "a0", HILL_A0)
    a1 = _resolve_param(params, "a1", HILL_A1)
    k_c = _resolve_param(params, "k_c", HILL_KC)
    n = _resolve_param(params, "n", HILL_N)
    b_h = _resolve_param(params, "b_h", HILL_BH)
    c0 = _resolve_param(params, "c0", HILL_C0)
    c1 = _resolve_param(params, "c1", HILL_C1)
    k_h = _resolve_param(params, "k_h", HILL_KH)
    m = _resolve_param(params, "m", HILL_M)
    b_c = _resolve_param(params, "b_c", HILL_BC)

    h = state_arr[..., HILL2_H2O2I_INDEX]
    c = state_arr[..., HILL2_CAI_INDEX]

    dh = a0 + a1 * _hill_gate(c, k_c, n) - b_h * h
    dc = c0 + c1 * _hill_gate(h, k_h, m) - b_c * c
    return jnp.stack((dh, dc), axis=-1)


def model_hill_jax(x, t, k, x0, *args, **kwargs):
    """Hill-minimal H2O2/Ca RHS equivalent to FBA_ode.model_hill."""
    _require_jax()

    x_arr = jnp.asarray(x)
    dxdt = jnp.zeros_like(x_arr)

    h = x_arr[H2O2I_INDEX]
    c = x_arr[CAI_INDEX]

    dh = HILL_A0 + HILL_A1 * _hill_gate(c, HILL_KC, HILL_N) - HILL_BH * h
    dc = HILL_C0 + HILL_C1 * _hill_gate(h, HILL_KH, HILL_M) - HILL_BC * c

    dxdt = dxdt.at[H2O2I_INDEX].set(dh)
    dxdt = dxdt.at[CAI_INDEX].set(dc)
    return dxdt


def model_noref_full_jax(x, t, k, x_base):
    """Full 32-state RHS mirroring FBA_ode.model_noref in full mode."""
    _require_jax()

    x_arr = jnp.asarray(x)
    k_arr = jnp.asarray(k)
    x_base_arr = jnp.asarray(x_base)

    if x_arr.shape[-1] != LEGACY_STATE_LENGTH:
        raise ValueError(
            f"x must have shape ({LEGACY_STATE_LENGTH},); got {x_arr.shape}."
        )
    if x_base_arr.shape[-1] != LEGACY_STATE_LENGTH:
        raise ValueError(
            f"x_base must have shape ({LEGACY_STATE_LENGTH},); got {x_base_arr.shape}."
        )

    dxdt = jnp.zeros_like(x_arr)

    H = x_arr[1]
    C = x_arr[29]
    Cer = x_arr[31]

    V_IP3R = k_arr[34]
    K_IP3R_H = k_arr[35]
    nH = k_arr[36]
    K_er = k_arr[37]
    V_SERCA = k_arr[38]
    K_SERCA_H = k_arr[40]
    m_inhib = k_arr[41]
    P_TRPM2 = k_arr[42]
    K_TRPM2_H = k_arr[43]
    p_TRPM2 = k_arr[44]
    P_SOCE = k_arr[45]
    K_store = k_arr[46]

    c_pos = jnp.maximum(x_arr[29], 0.0)
    h_pos = jnp.maximum(H, 0.0)

    c_power_54 = c_pos ** k_arr[54]
    ca_gate_ip3r_expr = c_power_54 / (k_arr[57] ** k_arr[54] + c_power_54)
    Ca_gate_IP3R = _where(x_arr[29] > 0.0, ca_gate_ip3r_expr, 0.0)

    ca_ip3r_inhib_expr = 1.0 / (1.0 + (x_arr[29] / k_arr[53]) ** 2)
    Ca_IP3R_inhib = _where(x_arr[29] > 0.0, ca_ip3r_inhib_expr, 0.0)

    h2o2_gate_ip3r_expr = (h_pos ** nH) / (K_IP3R_H ** nH + h_pos ** nH)
    h2o2_gate_IP3R = _where(H > 0.0, h2o2_gate_ip3r_expr, 0.0)

    h2o2_inhib_ip3r_expr = (k_arr[51] ** k_arr[52]) / (k_arr[51] ** k_arr[52] + h_pos ** k_arr[52])
    H2O2_inhibit_IP3R = _where(H > 0.0, h2o2_inhib_ip3r_expr, 0.0)

    serca_inhb_expr = 1.0 / (1.0 + (H / K_SERCA_H) ** m_inhib)
    SERCA_inhb = _where(H > 0.0, serca_inhb_expr, 1.0)

    h2o2_gate_trpm2_expr = (h_pos ** p_TRPM2) / (K_TRPM2_H ** p_TRPM2 + h_pos ** p_TRPM2)
    H2O2_gate_TRPM2 = _where(H > 0.0, h2o2_gate_trpm2_expr, 0.0)

    ca_gate_trpm2_expr = (c_pos ** k_arr[69]) / (k_arr[68] ** k_arr[69] + c_pos ** k_arr[69])
    Ca_gate_TRPM2 = _where(x_arr[29] > 0.0, ca_gate_trpm2_expr, 0.0)

    SOCE_drive = K_store / (K_store + Cer)

    j_ip3r_expr = (
        V_IP3R
        * h2o2_gate_IP3R
        * Ca_gate_IP3R
        * Ca_IP3R_inhib
        * H2O2_inhibit_IP3R
        * (Cer - C)
        / (K_er + Cer)
    )
    J_IP3R = _where((Cer - C) > 0.0, j_ip3r_expr, 0.0)

    j_serca_expr = V_SERCA * (jnp.maximum(C, 0.0) ** k_arr[47] / (k_arr[39] ** k_arr[47] + jnp.maximum(C, 0.0) ** k_arr[47])) * SERCA_inhb
    J_SERCA = _where(C > 0.0, j_serca_expr, 0.0)

    j_in_trpm2_expr = P_TRPM2 * H2O2_gate_TRPM2 * Ca_gate_TRPM2 * (x_arr[30] - x_arr[29])
    J_in_TRPM2 = _where((x_arr[30] - x_arr[29]) > 0.0, j_in_trpm2_expr, 0.0)

    j_in_soce_expr = P_SOCE * SOCE_drive * (x_arr[30] - x_arr[29])
    J_in_SOCE = _where((x_arr[30] - x_arr[29]) > 0.0, j_in_soce_expr, 0.0)

    cap_raw = 1.0 - (x_arr[31] / k_arr[55])
    cap = _where(cap_raw < 0.0, 0.0, cap_raw)
    J_SERCA = J_SERCA * cap

    # Mirrors legacy semantics exactly (always zero after cap clamp).
    J_over = _where(cap < 0.0, k_arr[56] * cap * x_arr[31], 0.0)

    j_out_expr = k_arr[58] * x_arr[29] / (5e-6 + x_arr[29])
    J_out = _where(x_arr[29] > 0.0, j_out_expr, 0.0)

    nox2_expr = k_arr[32] * (h_pos ** 6) / (k_arr[33] ** 6 + h_pos ** 6)
    nox2 = _where(x_arr[1] > 0.0, nox2_expr, 0.0)

    nox5_expr = (
        k_arr[48]
        * (c_pos ** k_arr[50])
        / (k_arr[49] ** k_arr[50] + c_pos ** k_arr[50])
        * (x_arr[22] / (k_arr[62] + x_arr[22]))
    )
    nox5 = _where((x_arr[29] > 0.0) | (x_arr[22] > 0.0), nox5_expr, 0.0)

    km_duox_expr = k_arr[66] - (k_arr[66] - k_arr[67]) * (c_pos ** k_arr[65]) / (k_arr[64] ** k_arr[65] + c_pos ** k_arr[65])
    km_duox = _where(x_arr[29] > 0.0, km_duox_expr, k_arr[67])
    duox = k_arr[63] * x_arr[22] / (km_duox + x_arr[22])

    dxdt = dxdt.at[0].set((k_arr[0] * (x_arr[1] - x_arr[0]) * 10e-3 * 5.3e-6 * (1e9)) + nox2 + duox)

    dxdt = dxdt.at[1].set(
        ((k_arr[0] * x_arr[0] * 10e-3 * 5.3e-6) / (1.15e-12))
        - ((k_arr[0] * x_arr[1] * 10e-3 * 5.3e-6) / (1.15e-12))
        - 2 * (k_arr[22] * x_arr[1] * 10e-3 * 2.98e-9) / (1.53e-14)
        - k_arr[2] * x_arr[2] * x_arr[1]
        - k_arr[5] * x_arr[7] * x_arr[1]
        - k_arr[7] * x_arr[9] * x_arr[1]
        - k_arr[8] * x_arr[10] * x_arr[1]
        - k_arr[13] * x_arr[15] * x_arr[1]
        - k_arr[17] * x_arr[20] * x_arr[1]
        + k_arr[1]
        + nox5
    )

    dxdt = dxdt.at[2].set(-k_arr[2] * x_arr[2] * x_arr[1] + k_arr[4] * x_arr[4] * x_arr[5])
    dxdt = dxdt.at[3].set(k_arr[2] * x_arr[2] * x_arr[1] - k_arr[3] * x_arr[3] * x_arr[5])
    dxdt = dxdt.at[4].set(k_arr[3] * x_arr[3] * x_arr[5] - k_arr[4] * x_arr[4] * x_arr[5])

    dxdt = dxdt.at[5].set(
        -k_arr[3] * x_arr[3] * x_arr[5]
        - k_arr[4] * x_arr[4] * x_arr[5]
        - 2 * k_arr[12] * x_arr[5]
        - k_arr[14] * x_arr[16] * x_arr[5]
        - k_arr[16] * x_arr[16] * x_arr[5]
        + 2 * k_arr[19] * x_arr[6] * x_arr[22]
        + k_arr[23]
        - k_arr[25]
    )

    dx6_if = (
        k_arr[4] * x_arr[4] * x_arr[5]
        + k_arr[12] * x_arr[5]
        + k_arr[16] * x_arr[19] * x_arr[5]
        - k_arr[19] * x_arr[6] * x_arr[22]
    )
    dx6_else1 = -k_arr[24] * x_arr[6] / x_base_arr[6]
    dx6_else2 = -k_arr[25] * x_arr[6] / x_base_arr[6]
    dx6 = _where(
        (x_arr[6] > 0.0) & (x_arr[22] > 0.0),
        dx6_if,
        _where(x_base_arr[6] > 0.0, dx6_else1, _where(x_base_arr[6] > 0.0, dx6_else2, 0.0)),
    )
    dxdt = dxdt.at[6].set(dx6)

    dxdt = dxdt.at[7].set(0.0)
    dxdt = dxdt.at[8].set((k_arr[22] * x_arr[1] * 10e-3 * 2.98e-9) / (1.53e-14) - k_arr[5] * x_arr[7] * x_arr[8])
    dxdt = dxdt.at[9].set(-k_arr[7] * x_arr[9] * x_arr[1] + k_arr[11] * x_arr[12] * x_arr[13])
    dxdt = dxdt.at[10].set(k_arr[7] * x_arr[9] * x_arr[1] - k_arr[8] * x_arr[10] * x_arr[1] + k_arr[9] * x_arr[11] - k_arr[10] * x_arr[10])
    dxdt = dxdt.at[11].set(k_arr[8] * x_arr[10] * x_arr[1] - k_arr[9] * x_arr[11])
    dxdt = dxdt.at[12].set(k_arr[10] * x_arr[10] - k_arr[11] * x_arr[12] * x_arr[13])
    dxdt = dxdt.at[13].set(-k_arr[11] * x_arr[12] * x_arr[13] - k_arr[18] * x_arr[21] * x_arr[13] + k_arr[20] * x_arr[14] * x_arr[22] - k_arr[26] + k_arr[27])
    dxdt = dxdt.at[14].set(k_arr[11] * x_arr[12] * x_arr[13] + k_arr[18] * x_arr[21] * x_arr[13] - k_arr[20] * x_arr[14] * x_arr[22])
    dxdt = dxdt.at[15].set(-k_arr[13] * x_arr[15] * x_arr[1] + k_arr[15] * x_arr[18] * x_arr[17])
    dxdt = dxdt.at[16].set(k_arr[13] * x_arr[15] * x_arr[1] - k_arr[14] * x_arr[16] * x_arr[5])
    dxdt = dxdt.at[17].set(k_arr[14] * x_arr[16] * x_arr[5] - k_arr[15] * x_arr[18] * x_arr[17])
    dxdt = dxdt.at[18].set(k_arr[16] * x_arr[19] * x_arr[5] - k_arr[15] * x_arr[18] * x_arr[17])
    dxdt = dxdt.at[19].set(k_arr[15] * x_arr[18] * x_arr[17] - k_arr[16] * x_arr[19] * x_arr[5])
    dxdt = dxdt.at[20].set(-k_arr[17] * x_arr[20] * x_arr[1] + k_arr[18] * x_arr[21] * x_arr[13])
    dxdt = dxdt.at[21].set(k_arr[17] * x_arr[20] * x_arr[1] - k_arr[18] * x_arr[21] * x_arr[13])

    nadph_extra = _where(x_arr[23] > 0.0, k_arr[21] * x_arr[23] / (k_arr[6] + x_arr[23]), 0.0)
    nadph_extra = _where((x_arr[23] > 0.0) | (x_arr[22] > 0.0), nadph_extra, 0.0)
    dx22 = (
        -k_arr[19] * x_arr[6] * x_arr[22]
        - k_arr[20] * x_arr[14] * x_arr[22]
        - _where(x_arr[22] > 0.0, 2 * nox2, 0.0)
        - _where(x_arr[22] > 0.0, 2 * nox5, 0.0)
        + nadph_extra
    )
    dxdt = dxdt.at[22].set(dx22)

    dx23 = (
        k_arr[19] * x_arr[6] * x_arr[22]
        + k_arr[20] * x_arr[14] * x_arr[22]
        + 2 * nox2
        + 2 * nox5
        - _where(x_arr[23] > 0.0, k_arr[21] * x_arr[23] / (k_arr[6] + x_arr[23]), 0.0)
    )
    dxdt = dxdt.at[23].set(dx23)

    dxdt = dxdt.at[24].set(0.0)
    dxdt = dxdt.at[25].set(0.0)
    dxdt = dxdt.at[26].set(0.0)
    dxdt = dxdt.at[27].set(0.0)

    dxdt = dxdt.at[29].set(J_IP3R + J_in_TRPM2 + J_in_SOCE + J_over - J_SERCA - J_out)
    dxdt = dxdt.at[30].set(0.0)
    dxdt = dxdt.at[31].set(J_SERCA - J_IP3R - J_over)

    return dxdt
