import importlib

import pytest


np = pytest.importorskip("numpy")


def test_import_fba_ode_moles():
    module = importlib.import_module("FBA_ode_moles")
    assert module.__name__ == "FBA_ode_moles"


def test_import_fba_ode_moles_new_path():
    module = importlib.import_module("ros_abm.backends.legacy.ode.fba_ode_moles")
    assert module.__name__ == "ros_abm.backends.legacy.ode.fba_ode_moles"


def test_old_and_new_fba_ode_moles_paths_expose_required_api():
    old_mod = importlib.import_module("FBA_ode_moles")
    new_mod = importlib.import_module("ros_abm.backends.legacy.ode.fba_ode_moles")

    for name in ("get_k", "get_x0", "model"):
        assert hasattr(old_mod, name)
        assert hasattr(new_mod, name)
        assert callable(getattr(old_mod, name))
        assert callable(getattr(new_mod, name))


def test_get_k_exists_and_returns_expected_shape_type():
    module = importlib.import_module("FBA_ode_moles")

    assert hasattr(module, "get_k")
    k = module.get_k()

    assert isinstance(k, np.ndarray)
    assert k.shape == (33,)
    assert np.issubdtype(k.dtype, np.floating)


def test_get_x0_exists_and_returns_expected_shape_type():
    module = importlib.import_module("FBA_ode_moles")

    assert hasattr(module, "get_x0")
    x0 = module.get_x0()

    assert isinstance(x0, np.ndarray)
    assert x0.shape == (30,)
    assert np.issubdtype(x0.dtype, np.floating)


def test_model_exists():
    module = importlib.import_module("FBA_ode_moles")
    assert hasattr(module, "model")
    assert callable(module.model)


def test_model_returns_same_shaped_finite_derivative_vector():
    old_module = importlib.import_module("FBA_ode_moles")
    new_module = importlib.import_module("ros_abm.backends.legacy.ode.fba_ode_moles")

    x0 = old_module.get_x0()
    k = old_module.get_k()

    dxdt_old = old_module.model(x0.copy(), 0.0, k, x0.copy())
    dxdt_new = new_module.model(x0.copy(), 0.0, k, x0.copy())

    assert isinstance(dxdt_old, np.ndarray)
    assert dxdt_old.shape == x0.shape
    assert np.all(np.isfinite(dxdt_old))

    assert isinstance(dxdt_new, np.ndarray)
    assert dxdt_new.shape == x0.shape
    assert np.all(np.isfinite(dxdt_new))

    np.testing.assert_allclose(dxdt_old, dxdt_new, rtol=0.0, atol=0.0)
