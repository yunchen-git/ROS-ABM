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


def test_explicit_model_rhs_modes_do_not_contaminate_each_other():
    x0 = fba_ode.get_x0()
    k = fba_ode.get_k()

    full_rhs = fba_ode.make_model_rhs("full", k, x0.copy())
    hill_rhs = fba_ode.make_model_rhs("hill", k, x0.copy())

    full_before = full_rhs(0.0, x0.copy())
    hill = hill_rhs(0.0, x0.copy())

    assert fba_ode.set_chem_mode("hill") == "hill"
    full_after_global_hill = full_rhs(0.0, x0.copy())
    explicit_full_after_global_hill = fba_ode.model_noref_for_mode(
        x0.copy(),
        0.0,
        k,
        x0.copy(),
        "full",
    )

    assert full_before.shape == x0.shape
    assert hill.shape == x0.shape
    assert not (full_before == hill).all()
    assert (full_before == full_after_global_hill).all()
    assert (full_before == explicit_full_after_global_hill).all()
    fba_ode.set_chem_mode("full")
