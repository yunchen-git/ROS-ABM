import pytest

fba_ode = pytest.importorskip("FBA_ode")


def test_set_chem_mode_accepts_full_and_hill():
    assert fba_ode.set_chem_mode("full") == "full"
    assert fba_ode.get_chem_mode() == "full"

    assert fba_ode.set_chem_mode("hill") == "hill"
    assert fba_ode.get_chem_mode() == "hill"

    assert fba_ode.set_chem_mode(None) == "full"
    assert fba_ode.get_chem_mode() == "full"


def test_set_chem_mode_rejects_invalid_mode():
    with pytest.raises(ValueError):
        fba_ode.set_chem_mode("invalid-mode")


def test_model_hill_returns_same_shaped_derivative_vector():
    x0 = fba_ode.get_x0()
    k = fba_ode.get_k()

    dxdt = fba_ode.model_hill(x0.copy(), 0.0, k, x0.copy())

    assert dxdt.shape == x0.shape
