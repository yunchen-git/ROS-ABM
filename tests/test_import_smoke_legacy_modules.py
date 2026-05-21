import importlib

import pytest


def _require_core_runtime_dependencies():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    pytest.importorskip("networkx")


def test_import_simulation_smoke():
    _require_core_runtime_dependencies()
    module = importlib.import_module("Simulation")
    assert module.__name__ == "Simulation"


def test_import_model_run_smoke():
    _require_core_runtime_dependencies()
    module = importlib.import_module("Model_run")
    assert module.__name__ == "Model_run"


def test_import_model_initialparameters_smoke():
    _require_core_runtime_dependencies()
    module = importlib.import_module("Model_initialParameters")
    assert module.__name__ == "Model_initialParameters"


def test_import_fba_ode_smoke():
    _require_core_runtime_dependencies()
    module = importlib.import_module("FBA_ode")
    assert module.__name__ == "FBA_ode"


def test_import_count_directory_newdirect_smoke():
    from CountDirectory import newDirect

    assert callable(newDirect)


def test_import_intc_diff_smoke():
    _require_core_runtime_dependencies()
    module = importlib.import_module("IntC_diff")
    assert hasattr(module, "IntC_Diff")


def test_import_extc_gradient_smoke():
    _require_core_runtime_dependencies()
    module = importlib.import_module("ExtC_gradient")
    assert hasattr(module, "ExtC_Diff")
