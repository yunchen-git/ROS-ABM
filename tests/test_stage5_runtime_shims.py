import importlib
from pathlib import Path

import pytest


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _read(path):
    return path.read_text(encoding="utf-8")


def test_stage5_root_shim_sources_reference_canonical_modules():
    root = _repo_root()

    model_init_src = _read(root / "Model_initialParameters.py")
    assert "from ros_abm.backends.legacy.runtime.model_initial_parameters import *" in model_init_src

    model_run_src = _read(root / "Model_run.py")
    assert "from ros_abm.cli.model_run import *" in model_run_src
    assert "if __name__ == \"__main__\":" in model_run_src
    assert "main()" in model_run_src

    sim_wrap_src = _read(root / "ros_abm/backends/legacy/runtime/simulation.py")
    assert "from Simulation import *" in sim_wrap_src

    sim_objects_wrap_src = _read(root / "ros_abm/backends/legacy/runtime/simulation_objects.py")
    assert "from simulationObjects import *" in sim_objects_wrap_src

    fba_wrap_src = _read(root / "ros_abm/backends/legacy/ode/fba_ode.py")
    assert "from FBA_ode import *" in fba_wrap_src


def test_model_initialparameters_old_and_new_import_paths_match():
    old_mod = importlib.import_module("Model_initialParameters")
    new_mod = importlib.import_module("ros_abm.backends.legacy.runtime.model_initial_parameters")

    assert hasattr(old_mod, "main")
    assert hasattr(new_mod, "main")
    assert old_mod.main is new_mod.main

    assert hasattr(old_mod, "ModelSettings")
    assert hasattr(new_mod, "ModelSettings")
    assert old_mod.ModelSettings is new_mod.ModelSettings

    assert hasattr(old_mod, "resolve_cell_path")
    assert hasattr(new_mod, "resolve_cell_path")
    assert old_mod.resolve_cell_path is new_mod.resolve_cell_path


def test_model_run_old_and_new_import_paths_match():
    old_mod = importlib.import_module("Model_run")
    new_mod = importlib.import_module("ros_abm.cli.model_run")

    assert hasattr(old_mod, "RunSettings")
    assert hasattr(new_mod, "RunSettings")
    assert old_mod.RunSettings is new_mod.RunSettings

    assert hasattr(old_mod, "resolve_output_base_path")
    assert hasattr(new_mod, "resolve_output_base_path")
    assert old_mod.resolve_output_base_path is new_mod.resolve_output_base_path

    assert hasattr(old_mod, "main")
    assert hasattr(new_mod, "main")
    assert old_mod.main is new_mod.main


def test_stage5_wrapper_namespaces_import():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    pytest.importorskip("networkx")
    pytest.importorskip("torch")

    sim_mod = importlib.import_module("ros_abm.backends.legacy.runtime.simulation")
    sim_obj_mod = importlib.import_module("ros_abm.backends.legacy.runtime.simulation_objects")
    fba_mod = importlib.import_module("ros_abm.backends.legacy.ode.fba_ode")

    assert hasattr(sim_mod, "Simulation")
    assert hasattr(sim_obj_mod, "SimulationObject")
    assert hasattr(sim_obj_mod, "StemCell")
    assert hasattr(fba_mod, "get_k")
    assert hasattr(fba_mod, "get_x0")
