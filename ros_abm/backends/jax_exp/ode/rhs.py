"""Opt-in JAX experimental RHS implementations.

This module is intentionally isolated from the legacy runtime path.
"""

try:
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover - environment dependent
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

