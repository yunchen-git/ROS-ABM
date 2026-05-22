"""Minimal Hill-state schema and adapters for opt-in JAX experiments."""

from __future__ import annotations

try:
    import jax.numpy as jnp
except Exception:  # pragma: no cover - environment dependent
    jnp = None
    import numpy as np

    _xp = np
else:
    _xp = jnp


HILL2_STATE_ORDER = ("h2o2i", "cai")
HILL2_H2O2I_INDEX = 0
HILL2_CAI_INDEX = 1
HILL2_STATE_LENGTH = 2

LEGACY_H2O2I_INDEX = 1
LEGACY_CAI_INDEX = 29
LEGACY_STATE_LENGTH = 32


def _validate_last_dim(arr, expected_last_dim, name):
    if arr.shape[-1] != expected_last_dim:
        raise ValueError(
            f"{name} must have last dimension {expected_last_dim}; got shape {arr.shape}."
        )


def pack_legacy32_to_hill2(x):
    """Pack legacy (..., 32) state vectors to Hill (..., 2) [h2o2i, cai]."""
    x_arr = _xp.asarray(x)
    _validate_last_dim(x_arr, LEGACY_STATE_LENGTH, "x")
    return _xp.stack(
        (
            x_arr[..., LEGACY_H2O2I_INDEX],
            x_arr[..., LEGACY_CAI_INDEX],
        ),
        axis=-1,
    )


def unpack_hill2_to_legacy32_dx(dx_hill2, template=None):
    """Unpack Hill (..., 2) derivatives into sparse legacy (..., 32) derivative vectors."""
    dx2_arr = _xp.asarray(dx_hill2)
    _validate_last_dim(dx2_arr, HILL2_STATE_LENGTH, "dx_hill2")

    if template is None:
        out_shape = dx2_arr.shape[:-1] + (LEGACY_STATE_LENGTH,)
        out = _xp.zeros(out_shape, dtype=dx2_arr.dtype)
    else:
        template_arr = _xp.asarray(template)
        _validate_last_dim(template_arr, LEGACY_STATE_LENGTH, "template")
        if template_arr.shape[:-1] != dx2_arr.shape[:-1]:
            raise ValueError(
                "template leading dimensions must match dx_hill2 leading dimensions; "
                f"got template {template_arr.shape} and dx_hill2 {dx2_arr.shape}."
            )
        out = _xp.zeros_like(template_arr)

    if hasattr(out, "at"):
        out = out.at[..., LEGACY_H2O2I_INDEX].set(dx2_arr[..., HILL2_H2O2I_INDEX])
        out = out.at[..., LEGACY_CAI_INDEX].set(dx2_arr[..., HILL2_CAI_INDEX])
        return out

    out[..., LEGACY_H2O2I_INDEX] = dx2_arr[..., HILL2_H2O2I_INDEX]
    out[..., LEGACY_CAI_INDEX] = dx2_arr[..., HILL2_CAI_INDEX]
    return out

