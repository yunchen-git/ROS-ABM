"""Hill2 parameter-sweep validation tests for the opt-in JAX experiment path.

These tests intentionally cover only reduced Hill2 dynamics and do not imply
validation of the full ROS-ABM runtime chemistry path.
"""

from __future__ import annotations

import importlib
import itertools

import pytest


np = pytest.importorskip("numpy")
scipy_integrate = pytest.importorskip("scipy.integrate")
pytest.importorskip("jax")

jax_rhs = importlib.import_module("ros_abm.backends.jax_exp.ode.rhs")
jax_solver = importlib.import_module("ros_abm.backends.jax_exp.ode.solver")


RADAU_REF_ATOL = 1e-12
RADAU_REF_RTOL = 1e-10


DEFAULT_HILL2_PARAMS = {
    "a0": float(jax_rhs.HILL_A0),
    "a1": float(jax_rhs.HILL_A1),
    "k_c": float(jax_rhs.HILL_KC),
    "n": float(jax_rhs.HILL_N),
    "b_h": float(jax_rhs.HILL_BH),
    "c0": float(jax_rhs.HILL_C0),
    "c1": float(jax_rhs.HILL_C1),
    "k_h": float(jax_rhs.HILL_KH),
    "m": float(jax_rhs.HILL_M),
    "b_c": float(jax_rhs.HILL_BC),
}


SWEEP_RANGES = {
    "a1": (0.3, 1.0),
    "k": (1e-7, 0.5),
    "b": (0.1, 0.2),
}


def _with_defaults(overrides=None):
    params = dict(DEFAULT_HILL2_PARAMS)
    if overrides:
        params.update(overrides)
    return params


def _integrate_hill2_rk4(y0, dt, horizon, params):
    steps = int(round(float(horizon) / float(dt)))
    if steps < 1:
        raise ValueError("horizon/dt must be >= 1 step for validation.")

    def rhs(y, _t):
        return jax_rhs.model_hill2_jax(y, params=params)

    times, states = jax_solver.solve_rk4(
        rhs,
        np.asarray(y0, dtype=np.float64),
        t0=0.0,
        dt=float(dt),
        num_steps=steps,
    )
    return np.asarray(times), np.asarray(states)


def _integrate_hill2_radau(y0, t_eval, params):
    y0_arr = np.asarray(y0, dtype=np.float64)
    t_eval_arr = np.asarray(t_eval, dtype=np.float64)

    def rhs(_t, y):
        return np.asarray(jax_rhs.model_hill2_jax(y, params=params))

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


def _small_param_grid():
    # Keep this intentionally small for quick guardrail checks.
    cases = []
    for a1, k_half, b_rate in itertools.product(
        SWEEP_RANGES["a1"],
        SWEEP_RANGES["k"],
        SWEEP_RANGES["b"],
    ):
        cases.append(
            _with_defaults(
                {
                    "a0": 0.0,
                    "c0": 0.0,
                    "a1": a1,
                    "c1": a1,
                    "k_c": k_half,
                    "k_h": k_half,
                    "n": 2.0,
                    "m": 2.0,
                    "b_h": b_rate,
                    "b_c": b_rate,
                }
            )
        )
    return cases


def test_default_hill2_params_are_explicitly_included_in_validation_sets():
    grid_cases = _small_param_grid()
    has_default = any(
        all(np.isclose(case[name], value) for name, value in DEFAULT_HILL2_PARAMS.items())
        for case in grid_cases
    )
    assert has_default, "Validation sweep must include the exact default Hill2 parameter point."


def test_small_parameter_grid_remains_finite():
    initial_conditions = [
        np.array([1e-9, 1e-7], dtype=np.float64),
        np.array([1e-6, 5e-6], dtype=np.float64),
        np.array([1e-3, 1e-3], dtype=np.float64),
    ]

    for case in _small_param_grid():
        for y0 in initial_conditions:
            _, states = _integrate_hill2_rk4(y0=y0, dt=0.05, horizon=2.0, params=case)
            assert np.isfinite(states).all(), f"Non-finite values for params={case}, y0={y0}."


def test_small_parameter_grid_conservative_dt_nonnegativity_for_nonnegative_sources():
    """For nonnegative-source parameter cases and nonnegative ICs, conservative dt should keep states nonnegative."""
    initial_conditions = [
        np.array([1e-9, 1e-7], dtype=np.float64),
        np.array([1e-6, 5e-6], dtype=np.float64),
        np.array([1e-3, 1e-3], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
    ]

    for case in _small_param_grid():
        assert case["a0"] >= 0.0 and case["c0"] >= 0.0
        for y0 in initial_conditions:
            _, states = _integrate_hill2_rk4(y0=y0, dt=0.01, horizon=5.0, params=case)
            assert states[:, 0].min() >= 0.0, f"Negative h2o2i for params={case}, y0={y0}."
            assert states[:, 1].min() >= 0.0, f"Negative cai for params={case}, y0={y0}."


@pytest.mark.parametrize(
    "params",
    [
        _with_defaults(),
        _with_defaults({"a1": 0.3, "c1": 0.3, "k_c": 1e-3, "k_h": 1e-3, "b_h": 0.1, "b_c": 0.1}),
    ],
)
def test_dt_halving_convergence_is_monotone_for_representative_hill2_params(params):
    y0 = np.array([1e-3, 1e-3], dtype=np.float64)
    horizon = 5.0
    dts = (0.1, 0.05, 0.025)
    final_errors = []

    for dt in dts:
        times, states_rk4 = _integrate_hill2_rk4(y0=y0, dt=dt, horizon=horizon, params=params)
        states_radau = _integrate_hill2_radau(y0=y0, t_eval=times, params=params)
        final_errors.append(float(np.max(np.abs(states_rk4[-1] - states_radau[-1]))))

    err_01, err_005, err_0025 = final_errors
    assert err_01 > err_005 > err_0025
    assert (err_01 / err_005) > 1.2
    assert (err_005 / err_0025) > 1.2
