"""Numerical validation tests for the opt-in JAX Hill2 solver path only.

This file intentionally validates only the reduced 2-state Hill dynamics
implemented in `model_hill2_jax` and integrated by fixed-step RK4.
It does not validate the full 32-state RHS, runtime wiring, or Cython transport.
"""

import importlib

import pytest


np = pytest.importorskip("numpy")
scipy_integrate = pytest.importorskip("scipy.integrate")
pytest.importorskip("jax")

jax_rhs = importlib.import_module("ros_abm.backends.jax_exp.ode.rhs")
jax_solver = importlib.import_module("ros_abm.backends.jax_exp.ode.solver")


# Agreement thresholds against a high-accuracy Radau reference solve.
RADAU_REF_ATOL = 1e-12
RADAU_REF_RTOL = 1e-10
HILL2_AGREEMENT_ATOL = 1e-10
HILL2_AGREEMENT_RTOL = 1e-8


def _hill2_rhs(y, _t):
    return jax_rhs.model_hill2_jax(y)


def _integrate_hill2_rk4(y0, dt, horizon):
    steps = int(round(float(horizon) / float(dt)))
    if steps < 1:
        raise ValueError("horizon/dt must be >= 1 step for validation.")
    times, states = jax_solver.solve_rk4(
        _hill2_rhs,
        np.asarray(y0, dtype=np.float64),
        t0=0.0,
        dt=float(dt),
        num_steps=steps,
    )
    return np.asarray(times), np.asarray(states)


def _integrate_hill2_radau(y0, t_eval):
    y0_arr = np.asarray(y0, dtype=np.float64)
    t_eval_arr = np.asarray(t_eval, dtype=np.float64)

    def rhs(t, y):
        return np.asarray(jax_rhs.model_hill2_jax(y))

    sol = scipy_integrate.solve_ivp(
        rhs,
        (float(t_eval_arr[0]), float(t_eval_arr[-1])),
        y0_arr,
        method="Radau",
        t_eval=t_eval_arr,
        atol=RADAU_REF_ATOL,
        rtol=RADAU_REF_RTOL,
    )
    assert sol.success
    return sol.y.T


def test_hill2_rk4_matches_radau_reference_on_short_horizon():
    """RK4 Hill2 trajectory should agree closely with Radau on short horizons."""
    initial_conditions = [
        np.array([1e-9, 1e-7], dtype=np.float64),
        np.array([1e-6, 5e-6], dtype=np.float64),
        np.array([1e-3, 1e-3], dtype=np.float64),
    ]

    observed_max_error = 0.0
    for y0 in initial_conditions:
        times, states_rk4 = _integrate_hill2_rk4(y0=y0, dt=0.1, horizon=1.0)
        states_radau = _integrate_hill2_radau(y0=y0, t_eval=times)
        err = np.max(np.abs(states_rk4 - states_radau))
        observed_max_error = max(observed_max_error, float(err))
        np.testing.assert_allclose(
            states_rk4,
            states_radau,
            rtol=HILL2_AGREEMENT_RTOL,
            atol=HILL2_AGREEMENT_ATOL,
        )

    # Keep this as a tight sanity bound while leaving tolerance assertions above canonical.
    assert observed_max_error < 1e-8


def test_hill2_rk4_dt_halving_converges_toward_radau():
    """Halving dt should reduce final-state error for Hill2 under fixed horizon."""
    y0 = np.array([1e-3, 1e-3], dtype=np.float64)
    horizon = 5.0
    dts = (0.1, 0.05, 0.025)
    final_errors = []

    for dt in dts:
        times, states_rk4 = _integrate_hill2_rk4(y0=y0, dt=dt, horizon=horizon)
        states_radau = _integrate_hill2_radau(y0=y0, t_eval=times)
        final_error = float(np.max(np.abs(states_rk4[-1] - states_radau[-1])))
        final_errors.append(final_error)

    err_01, err_005, err_0025 = final_errors
    assert err_01 > err_005 > err_0025
    assert (err_01 / err_005) > 2.0
    assert (err_005 / err_0025) > 2.0


def test_hill2_rk4_conservative_dt_stays_finite_and_nonnegative():
    """Conservative dt should keep Hill2 states finite/nonnegative for representative ICs."""
    test_ics = [
        np.array([1e-9, 1e-7], dtype=np.float64),
        np.array([1e-6, 5e-6], dtype=np.float64),
        np.array([1e-3, 1e-3], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
    ]

    for y0 in test_ics:
        _, states = _integrate_hill2_rk4(y0=y0, dt=0.1, horizon=5.0)
        assert np.isfinite(states).all()
        assert states[:, 0].min() >= 0.0
        assert states[:, 1].min() >= 0.0


def test_hill2_rk4_random_nonnegative_initial_conditions():
    """Random nonnegative ICs should remain finite/nonnegative at conservative dt."""
    rng = np.random.default_rng(0)
    for _ in range(64):
        y0 = rng.uniform(low=[0.0, 0.0], high=[1.0, 1.0], size=(2,))
        _, states = _integrate_hill2_rk4(y0=y0, dt=0.1, horizon=5.0)
        assert np.isfinite(states).all()
        assert states[:, 0].min() >= 0.0
        assert states[:, 1].min() >= 0.0
