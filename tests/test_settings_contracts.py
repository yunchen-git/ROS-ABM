import importlib
import inspect
import sys
import types


def _load_model_run_with_stubs(monkeypatch):
    model_init_stub = types.ModuleType("Model_initialParameters")
    count_dir_stub = types.ModuleType("CountDirectory")
    count_dir_stub.newDirect = lambda _path: 1.0

    monkeypatch.setitem(sys.modules, "Model_initialParameters", model_init_stub)
    monkeypatch.setitem(sys.modules, "CountDirectory", count_dir_stub)

    sys.modules.pop("Model_run", None)
    return importlib.import_module("Model_run")


def _load_model_initialparameters_with_stubs(monkeypatch):
    simulation_stub = types.ModuleType("Simulation")
    simulation_objects_stub = types.ModuleType("simulationObjects")
    intc_stub = types.ModuleType("IntC_diff")
    extc_stub = types.ModuleType("ExtC_gradient")
    fba_stub = types.ModuleType("FBA_ode")
    numpy_stub = types.ModuleType("numpy")

    intc_stub.IntC_Diff = object
    extc_stub.ExtC_Diff = object

    monkeypatch.setitem(sys.modules, "Simulation", simulation_stub)
    monkeypatch.setitem(sys.modules, "simulationObjects", simulation_objects_stub)
    monkeypatch.setitem(sys.modules, "IntC_diff", intc_stub)
    monkeypatch.setitem(sys.modules, "ExtC_gradient", extc_stub)
    monkeypatch.setitem(sys.modules, "FBA_ode", fba_stub)
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)

    sys.modules.pop("Model_initialParameters", None)
    return importlib.import_module("Model_initialParameters")


def test_runsettings_defaults_preserve_legacy_literals(monkeypatch):
    mod = _load_model_run_with_stubs(monkeypatch)
    settings = mod.RunSettings()

    assert settings.length == 1
    assert settings.output_base_path == "~/Documents/PhD/test"
    assert settings.inh_val == [1.0]
    assert settings.per_inh == [1.0]
    assert settings.inh_time == 1000
    assert settings.inh_time1 == 1000
    assert settings.inhib_type is None
    assert settings.magnitudes == [1]
    assert settings.chem_mode == "full"
    assert settings.use_vectorized_hill is False


def test_modelsettings_defaults_preserve_legacy_literals(monkeypatch):
    mod = _load_model_initialparameters_with_stubs(monkeypatch)
    settings = mod.ModelSettings()

    assert settings.ts == 0
    assert settings.te == 300
    assert settings.dt == 1
    assert settings.x == 750
    assert settings.y == 750
    assert settings.z == 0
    assert settings.ddim == 1.0
    assert settings.sync_time == [500, 500]
    assert settings.num_cell_types == 2
    assert settings.channel_count == 1000.0
    assert settings.channel_permeability == 5e-9
    assert settings.pluripotent_params == (4.89e-6, 0.03, 1.3e-7)
    assert settings.differentiated_params == (6.0e-6, 0.04, 1.3e-7)
    assert settings.open_probabilities == (0.45, 0.85)
    assert settings.division_times_hours == (18, 51)
    assert settings.D_camp == 444
    assert settings.D_h2o2i == 1000
    assert settings.D_h2o2e == 1800
    assert settings.D_Ca == 760
    assert settings.ddim_e == 1.0
    assert settings.initial_h2o2e == 1e-9
    assert settings.perturbation_cell_index == 877
    assert settings.circle_radius == 50.0
    assert settings.cell_radius == 6.5
    assert settings.point_z == 0
    assert settings.division_set == 0
    assert settings.default_cell_state == "U"
    assert settings.special_h2o2i == 1e-9
    assert settings.special_camp == 1
    assert settings.special_ca == 1e-7
    assert settings.default_h2o2i == 1e-9
    assert settings.default_camp_base == 0.595
    assert settings.default_camp_random_scale == 0.05
    assert settings.default_ca == 1e-7


def test_model_initialparameters_main_positional_signature_preserved(monkeypatch):
    mod = _load_model_initialparameters_with_stubs(monkeypatch)
    params = list(inspect.signature(mod.main).parameters.values())

    assert [p.name for p in params[:7]] == [
        "n",
        "path",
        "inhib_value",
        "inhib_type",
        "inhib_time",
        "inhib_time1",
        "c",
    ]
    assert params[7].name == "chem_mode"
    assert params[7].default == "full"
    assert params[8].name == "cell_path"
    assert params[8].default is None
