import importlib
import json
import pickle
import sys
import types

import pytest

from run_config import RunConfig, PerturbationConfig


@pytest.fixture
def simulation_with_stubs(monkeypatch):
    simulation_objects_stub = types.ModuleType("simulationObjects")
    intc_stub = types.ModuleType("IntC_diff")
    extc_stub = types.ModuleType("ExtC_gradient")
    fba_stub = types.ModuleType("FBA_ode")

    class SimulationObject:
        pass

    simulation_objects_stub.SimulationObject = SimulationObject
    intc_stub.IntC_Diff = object
    extc_stub.ExtC_Diff = object
    fba_stub.get_x0 = lambda: []
    fba_stub.get_k = lambda: []
    fba_stub.model_noref = lambda y, t, k, x0: y
    fba_stub.make_model_rhs = lambda mode, k, x0: lambda t, y: y
    fba_stub.hill_step_vectorized = lambda x_matrix, dt: x_matrix

    monkeypatch.setitem(sys.modules, "simulationObjects", simulation_objects_stub)
    monkeypatch.setitem(sys.modules, "IntC_diff", intc_stub)
    monkeypatch.setitem(sys.modules, "ExtC_gradient", extc_stub)
    monkeypatch.setitem(sys.modules, "FBA_ode", fba_stub)

    sys.modules.pop("Simulation", None)
    module = importlib.import_module("Simulation")
    yield module
    sys.modules.pop("Simulation", None)


def test_run_config_default_construction():
    cfg = RunConfig()

    assert cfg.length == 1
    assert cfg.output_base_path == "~/Documents/PhD/test"
    assert cfg.inh_val == [1.0]
    assert cfg.per_inh == [1.0]
    assert cfg.inh_time == 1000
    assert cfg.inh_time1 == 1000
    assert cfg.inhib_type is None
    assert cfg.magnitudes == [1]
    assert cfg.chem_mode == "full"
    assert cfg.use_vectorized_hill is False
    assert cfg.seed is None
    assert cfg.run_postprocess is True
    assert cfg.model.te == 300
    assert cfg.perturbation.cell_index == 877


def test_old_model_initialparameters_main_still_exists(monkeypatch):
    simulation_stub = types.ModuleType("Simulation")
    simulation_objects_stub = types.ModuleType("simulationObjects")
    intc_stub = types.ModuleType("IntC_diff")
    extc_stub = types.ModuleType("ExtC_gradient")
    fba_stub = types.ModuleType("FBA_ode")

    intc_stub.IntC_Diff = object
    extc_stub.ExtC_Diff = object

    monkeypatch.setitem(sys.modules, "Simulation", simulation_stub)
    monkeypatch.setitem(sys.modules, "simulationObjects", simulation_objects_stub)
    monkeypatch.setitem(sys.modules, "IntC_diff", intc_stub)
    monkeypatch.setitem(sys.modules, "ExtC_gradient", extc_stub)
    monkeypatch.setitem(sys.modules, "FBA_ode", fba_stub)

    sys.modules.pop("Model_initialParameters", None)
    sys.modules.pop("ros_abm.backends.legacy.runtime.model_initial_parameters", None)
    mod = importlib.import_module("Model_initialParameters")

    assert callable(mod.main)
    assert callable(mod.main_config)
    sys.modules.pop("Model_initialParameters", None)
    sys.modules.pop("ros_abm.backends.legacy.runtime.model_initial_parameters", None)


def test_perturbation_config_defaults_match_current_constants(simulation_with_stubs):
    cfg = PerturbationConfig()

    assert cfg.enabled is True
    assert cfg.cell_index == simulation_with_stubs.PERTURBATION_CELL_INDEX
    assert cfg.injection_time == simulation_with_stubs.PERTURBATION_INJECTION_TIME
    assert cfg.injection_h2o2i == simulation_with_stubs.PERTURBATION_INJECTION_H2O2I


def test_run_manifest_serializes_config(simulation_with_stubs, tmp_path):
    cfg = RunConfig(
        run_name="tiny",
        output_path=str(tmp_path),
        seed=7,
        run_postprocess=False,
    )
    cfg.model.te = 1
    cfg.model.dt = 1
    sim = simulation_with_stubs.Simulation(
        name="tiny",
        path=str(tmp_path),
        start_time=0,
        end_time=0,
        time_step=1,
        grid_size=[10, 10, 0],
        inhib_type=None,
        perturbation=cfg.perturbation,
        run_config=cfg,
    )
    (tmp_path / "tiny").mkdir()

    sim._write_run_config_manifest()

    data = json.loads((tmp_path / "tiny" / "run_config.json").read_text(encoding="utf-8"))
    assert data["seed"] == 7
    assert data["run_postprocess"] is False
    assert data["use_vectorized_hill"] is False
    assert data["model"]["te"] == 1
    assert data["model"]["dt"] == 1
    assert data["perturbation"]["enabled"] is True
    assert data["perturbation"]["cell_index"] == 877
    assert data["perturbation"]["injection_time"] == 5
    assert data["perturbation"]["injection_h2o2i"] == 0.0012


def test_timeseries_parses_gradient_and_interc_names(simulation_with_stubs, tmp_path):
    files = {
        "network0.0.gpickle": {"network": True},
        "H2O2e_0.0": {"gradient": "H2O2e"},
        "cAMP_0.0": {"interC": "cAMP"},
        "H2O2i_0.0": {"interC": "H2O2i"},
        "Ca_0.0": {"interC": "Ca"},
    }
    for name, value in files.items():
        with open(tmp_path / name, "wb") as f:
            pickle.dump(value, f)

    info = "\n".join(
        [
            "Name:tiny",
            f"Base path:{tmp_path}",
            "Platform:Linux",
            "Time range:0.0,0.0,1.0",
            "Gradient Species:H2O2e,",
            "InterC Species:cAMP,H2O2i,Ca,",
            (
                f"0.0,{tmp_path / 'network0.0.gpickle'},"
                f"{tmp_path / 'H2O2e_0.0'},,"
                f"{tmp_path / 'cAMP_0.0'},"
                f"{tmp_path / 'H2O2i_0.0'},"
                f"{tmp_path / 'Ca_0.0'}"
            ),
        ]
    )
    (tmp_path / "info.sim").write_text(info + "\n", encoding="utf-8")

    ts = simulation_with_stubs.TimeSeries(str(tmp_path))

    assert ts._grad_names == ["H2O2e"]
    assert ts._intC_names == ["cAMP", "H2O2i", "Ca"]
    assert set(ts._gradients[0.0]) == {"H2O2e"}
    assert set(ts._interC[0.0]) == {"cAMP", "H2O2i", "Ca"}
    assert ts.get_raw_agent_data() == [{"network": True}]
    assert ts.get_gradient_at_time(0.0, "H2O2e") == {"gradient": "H2O2e"}
    assert ts.get_intC_at_time(0.0, "H2O2i") == {"interC": "H2O2i"}


def test_old_main_delegates_with_compat_defaults(monkeypatch, tmp_path):
    calls = {}

    simulation_stub = types.ModuleType("Simulation")
    simulation_objects_stub = types.ModuleType("simulationObjects")
    intc_stub = types.ModuleType("IntC_diff")
    extc_stub = types.ModuleType("ExtC_gradient")
    fba_stub = types.ModuleType("FBA_ode")

    class Simulation:
        def __init__(self, *args, **kwargs):
            calls["sim_args"] = args
            calls["sim_kwargs"] = kwargs
            self.interC = []
            self.gradients = []

        def add_interC(self, interC):
            self.interC.append(interC)

        def add_gradient(self, gradient):
            self.gradients.append(gradient)

        def inc_current_ID(self):
            pass

        def add(self, sim_object):
            pass

        def collide(self, *args):
            pass

        def collide_lowDens(self):
            pass

        def run(self, *args):
            calls["run_args"] = args

    class StemCell:
        def __init__(self, *args, **kwargs):
            self.C = {}

        def setC(self, value, name):
            self.C[name] = value

        def set_gradient_source_sink_coeff(self, *args):
            pass

    class IntC_Diff:
        def __init__(self, name, *args, **kwargs):
            self.name = name

    class ExtC_Diff:
        def __init__(self, name, *args, **kwargs):
            self.name = name

    simulation_stub.Simulation = Simulation
    simulation_objects_stub.StemCell = StemCell
    intc_stub.IntC_Diff = IntC_Diff
    extc_stub.ExtC_Diff = ExtC_Diff
    fba_stub.set_chem_mode = lambda mode: calls.setdefault("chem_mode", mode)

    monkeypatch.setitem(sys.modules, "Simulation", simulation_stub)
    monkeypatch.setitem(sys.modules, "simulationObjects", simulation_objects_stub)
    monkeypatch.setitem(sys.modules, "IntC_diff", intc_stub)
    monkeypatch.setitem(sys.modules, "ExtC_gradient", extc_stub)
    monkeypatch.setitem(sys.modules, "FBA_ode", fba_stub)

    sys.modules.pop("Model_initialParameters", None)
    sys.modules.pop("ros_abm.backends.legacy.runtime.model_initial_parameters", None)
    mod = importlib.import_module("Model_initialParameters")

    cell_path = tmp_path / "cells.csv"
    with open(cell_path, "w", encoding="utf-8") as f:
        for idx in range(878):
            f.write(f"{idx},{idx % 31},{idx // 31}\n")

    mod.main(
        1.0,
        str(tmp_path),
        1.0,
        None,
        1000,
        1000,
        1,
        chem_mode="hill",
        cell_path=str(cell_path),
    )

    assert calls["run_args"] == (750, 750, 0)
    assert calls["chem_mode"] == "hill"
    perturbation = calls["sim_kwargs"]["perturbation"]
    assert perturbation.enabled is True
    assert perturbation.cell_index == 877
    assert perturbation.injection_time == 5
    assert perturbation.injection_h2o2i == 0.0012
