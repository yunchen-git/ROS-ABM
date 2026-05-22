import importlib
import subprocess
import sys

import pytest


np = pytest.importorskip("numpy")
pytest.importorskip("jax")

fba_ode = importlib.import_module("FBA_ode")
jax_rhs = importlib.import_module("ros_abm.backends.jax_exp.ode.rhs")
jax_solver = importlib.import_module("ros_abm.backends.jax_exp.ode.solver")


def _rk4_numpy(rhs_fn, y0, t0, dt):
    k1 = rhs_fn(y0, t0)
    k2 = rhs_fn(y0 + 0.5 * dt * k1, t0 + 0.5 * dt)
    k3 = rhs_fn(y0 + 0.5 * dt * k2, t0 + 0.5 * dt)
    k4 = rhs_fn(y0 + dt * k3, t0 + dt)
    return y0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@pytest.mark.parametrize(
    "x6,x22",
    [
        (1.78e-6, 3.0e-5),
        (-1.0e-8, 3.0e-5),
        (1.78e-6, -1.0e-8),
    ],
)
def test_model_noref_full_jax_matches_legacy_rhs(x6, x22):
    old_mode = fba_ode.get_chem_mode()
    try:
        fba_ode.set_chem_mode("full")
        x_base = fba_ode.get_x0()
        k = fba_ode.get_k()
        x = x_base.copy()
        x[1] = 2.5e-6
        x[22] = x22
        x[23] = 1.0e-6
        x[29] = 8.0e-7
        x[31] = 7.0e-5
        x[6] = x6

        legacy = fba_ode.model_noref(x, 0.0, k, x_base)
        experimental = np.asarray(jax_rhs.model_noref_full_jax(x, 0.0, k, x_base))
    finally:
        fba_ode.set_chem_mode(old_mode)

    assert experimental.shape == legacy.shape
    np.testing.assert_allclose(experimental, legacy, rtol=1e-6, atol=1e-12)


def test_solve_model_noref_full_jax_runs_and_returns_finite_trajectory():
    old_mode = fba_ode.get_chem_mode()
    try:
        fba_ode.set_chem_mode("full")
        x0 = fba_ode.get_x0()
        k = fba_ode.get_k()
        times, states = jax_solver.solve_model_noref_full_jax(
            x0=x0,
            k=k,
            x_base=x0,
            t0=0.0,
            dt=0.001,
            num_steps=20,
        )
    finally:
        fba_ode.set_chem_mode(old_mode)

    times_np = np.asarray(times)
    states_np = np.asarray(states)
    assert times_np.shape == (21,)
    assert states_np.shape == (21, 32)
    assert np.isfinite(states_np).all()


def test_solve_model_noref_full_jax_one_step_matches_numpy_rk4_reference():
    old_mode = fba_ode.get_chem_mode()
    try:
        fba_ode.set_chem_mode("full")
        x0 = fba_ode.get_x0()
        k = fba_ode.get_k()
        dt = 0.005

        def rhs_numpy(y, t):
            return fba_ode.model_noref(y, t, k, x0)

        ref_next = _rk4_numpy(rhs_numpy, x0, 0.0, dt)
        _, states = jax_solver.solve_model_noref_full_jax(
            x0=x0,
            k=k,
            x_base=x0,
            t0=0.0,
            dt=dt,
            num_steps=1,
        )
    finally:
        fba_ode.set_chem_mode(old_mode)

    np.testing.assert_allclose(np.asarray(states)[-1], ref_next, rtol=1e-6, atol=1e-10)


def test_jax_solver_cli_smoke():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ros_abm.backends.jax_exp.ode.solver",
            "--steps",
            "2",
            "--dt",
            "0.01",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "solver=rk4_jax" in proc.stdout
    assert "trajectory_shape=" in proc.stdout
