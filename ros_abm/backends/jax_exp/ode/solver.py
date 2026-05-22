"""Opt-in JAX ODE solver utilities for experimental backend work."""

from __future__ import annotations

import argparse

import numpy as np

from ros_abm.backends.jax_exp.ode.rhs import model_noref_full_jax

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


def _require_jax():
    if jnp is None:  # pragma: no cover - environment dependent
        raise ImportError("jax is required for ros_abm.backends.jax_exp solver modules") from _JAX_IMPORT_ERROR


def rk4_step(rhs_fn, y, t, dt):
    """Single RK4 step on state y at time t with step dt."""
    k1 = rhs_fn(y, t)
    k2 = rhs_fn(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs_fn(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs_fn(y + dt * k3, t + dt)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_rk4(rhs_fn, y0, t0=0.0, dt=1.0, num_steps=1):
    """Integrate an ODE using explicit RK4 for a fixed number of steps."""
    _require_jax()

    y_init = jnp.asarray(y0)
    dt_val = jnp.asarray(dt, dtype=y_init.dtype)
    t0_val = jnp.asarray(t0, dtype=y_init.dtype)

    if int(num_steps) < 1:
        raise ValueError(f"num_steps must be >= 1; got {num_steps}.")

    def scan_step(carry, _):
        y_curr, t_curr = carry
        y_next = rk4_step(rhs_fn, y_curr, t_curr, dt_val)
        t_next = t_curr + dt_val
        return (y_next, t_next), y_next

    (_, _), y_hist = jax.lax.scan(scan_step, (y_init, t0_val), xs=None, length=int(num_steps))
    y_all = jnp.concatenate((y_init[None, ...], y_hist), axis=0)
    t_all = t0_val + dt_val * jnp.arange(int(num_steps) + 1, dtype=y_init.dtype)
    return t_all, y_all


def solve_model_noref_full_jax(x0, k, x_base=None, t0=0.0, dt=1.0, num_steps=1):
    """Run opt-in full-mode JAX solver with legacy-sized states."""
    _require_jax()

    x0_arr = jnp.asarray(x0)
    k_arr = jnp.asarray(k)
    if x_base is None:
        x_base_arr = x0_arr
    else:
        x_base_arr = jnp.asarray(x_base)

    def rhs_fn(y, t):
        return model_noref_full_jax(y, t, k_arr, x_base_arr)

    return solve_rk4(rhs_fn, x0_arr, t0=t0, dt=dt, num_steps=num_steps)


def main(argv=None):
    """CLI for quick opt-in full JAX ODE solves."""
    parser = argparse.ArgumentParser(description="Run opt-in JAX full RHS RK4 solve for legacy 32-state ODE.")
    parser.add_argument("--dt", type=float, default=0.001, help="Fixed RK4 step size.")
    parser.add_argument("--steps", type=int, default=20, help="Number of RK4 steps.")
    parser.add_argument("--t0", type=float, default=0.0, help="Initial time.")
    args = parser.parse_args(argv)

    import FBA_ode  # local import to keep module import lightweight

    old_mode = FBA_ode.get_chem_mode()
    try:
        FBA_ode.set_chem_mode("full")
        x0 = FBA_ode.get_x0()
        k = FBA_ode.get_k()
        times, states = solve_model_noref_full_jax(x0=x0, k=k, x_base=x0, t0=args.t0, dt=args.dt, num_steps=args.steps)
    finally:
        FBA_ode.set_chem_mode(old_mode)

    times_np = np.asarray(times)
    states_np = np.asarray(states)
    final = states_np[-1]
    print(f"solver=rk4_jax steps={args.steps} dt={args.dt} t0={args.t0}")
    print(f"trajectory_shape={states_np.shape} finite={np.isfinite(states_np).all()}")
    print(f"t_final={times_np[-1]:.6g} H2O2i={final[1]:.6g} Ca={final[29]:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
