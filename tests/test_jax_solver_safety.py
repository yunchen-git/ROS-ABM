import importlib
import inspect

import pytest


np = pytest.importorskip("numpy")
pytest.importorskip("jax")

fba_ode = importlib.import_module("FBA_ode")
jax_rhs = importlib.import_module("ros_abm.backends.jax_exp.ode.rhs")
jax_solver = importlib.import_module("ros_abm.backends.jax_exp.ode.solver")


def _hill2_rhs(y, _t):
    return jax_rhs.model_hill2_jax(y)


def test_hill2_rk4_finite_trajectory():
    hill0 = np.array([1e-9, 1e-7], dtype=np.float64)
    times, states = jax_solver.solve_rk4(_hill2_rhs, hill0, t0=0.0, dt=0.01, num_steps=500)

    times_np = np.asarray(times)
    states_np = np.asarray(states)

    assert times_np.shape == (501,)
    assert states_np.shape == (501, 2)
    assert np.isfinite(states_np).all()


def test_hill2_rk4_nonnegative_trajectory_under_conservative_dt():
    hill0 = np.array([1e-9, 1e-7], dtype=np.float64)
    _, states = jax_solver.solve_rk4(_hill2_rhs, hill0, t0=0.0, dt=0.01, num_steps=500)

    states_np = np.asarray(states)
    assert states_np[:, 0].min() >= 0.0
    assert states_np[:, 1].min() >= 0.0


def test_full_mode_rk4_negativity_is_explicitly_detected_in_safety_check():
    old_mode = fba_ode.get_chem_mode()
    try:
        fba_ode.set_chem_mode("full")
        x0 = fba_ode.get_x0()
        k = fba_ode.get_k()
        _, states = jax_solver.solve_model_noref_full_jax(
            x0=x0,
            k=k,
            x_base=x0,
            t0=0.0,
            dt=0.005,
            num_steps=1,
        )
    finally:
        fba_ode.set_chem_mode(old_mode)

    states_np = np.asarray(states)
    final_h2o2i = states_np[-1, 1]
    assert final_h2o2i < 0.0, (
        "Expected known RK4 full-mode positivity violation at dt=0.005 "
        "to remain detectable for safety documentation."
    )


def test_optional_fail_fast_diagnostic_if_supported():
    params = inspect.signature(jax_solver.solve_model_noref_full_jax).parameters
    fail_fast_name = None
    for candidate in ("fail_on_negative", "fail_on_nonfinite", "state_check", "validate_state"):
        if candidate in params:
            fail_fast_name = candidate
            break

    if fail_fast_name is None:
        pytest.skip("Optional fail-fast diagnostic is not implemented in current solver API.")

    old_mode = fba_ode.get_chem_mode()
    try:
        fba_ode.set_chem_mode("full")
        x0 = fba_ode.get_x0()
        k = fba_ode.get_k()
        kwargs = {
            "x0": x0,
            "k": k,
            "x_base": x0,
            "t0": 0.0,
            "dt": 0.005,
            "num_steps": 1,
            fail_fast_name: True,
        }
        with pytest.raises(Exception):
            jax_solver.solve_model_noref_full_jax(**kwargs)
    finally:
        fba_ode.set_chem_mode(old_mode)
