import importlib

import pytest


np = pytest.importorskip("numpy")
pytest.importorskip("jax")

fba_ode = importlib.import_module("FBA_ode")
jax_rhs = importlib.import_module("ros_abm.backends.jax_exp.ode.rhs")


@pytest.mark.parametrize(
    "h2o2i,cai",
    [
        (1e-9, 1e-7),
        (2.5e-6, 8e-6),
        (0.0, 0.0),
    ],
)
def test_model_hill_jax_matches_legacy_model_hill(h2o2i, cai):
    x_base = fba_ode.get_x0()
    k = fba_ode.get_k()

    x = x_base.copy()
    x[fba_ode.H2O2I_INDEX] = h2o2i
    x[fba_ode.CAI_INDEX] = cai

    legacy = fba_ode.model_hill(x, 0.0, k, x_base)
    experimental = jax_rhs.model_hill_jax(x, 0.0, k, x_base)
    experimental_np = np.asarray(experimental)

    assert experimental_np.shape == legacy.shape
    np.testing.assert_allclose(experimental_np, legacy, rtol=1e-6, atol=1e-12)

