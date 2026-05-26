import json
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import FBA_ode
from run_config import RunConfig
from species_registry import (
    EXT_GRADIENT_H2O2E,
    INTERC_SPECIES_CAMP,
    INTERC_SPECIES_CA,
    INTERC_SPECIES_H2O2I,
    ODE_STATE_CA_I,
    ODE_STATE_H2O2E,
    ODE_STATE_H2O2I,
)


HILL_RADUA_ONE_STEP_ATOL = 1e-8


def _state_matrix(pairs):
    base = FBA_ode.get_x0()
    matrix = np.vstack([base.copy() for _ in pairs])
    for row, (h2o2_i, ca_i) in enumerate(pairs):
        matrix[row, ODE_STATE_H2O2I] = h2o2_i
        matrix[row, ODE_STATE_CA_I] = ca_i
    return matrix


def test_model_hill_vectorized_derivative_matches_scalar_model_hill():
    pairs = [
        (0.0, 0.0),
        (1e-9, 1e-7),
        (1e-3, 1e-7),
        (0.1, 0.2),
        (0.5, 0.5),
    ]
    matrix = _state_matrix(pairs)
    k = FBA_ode.get_k()
    base = FBA_ode.get_x0()

    vectorized = FBA_ode.model_hill_vectorized(matrix)
    scalar = np.vstack(
        [FBA_ode.model_hill(row.copy(), 0.0, k, base.copy()) for row in matrix]
    )

    np.testing.assert_allclose(vectorized, scalar, rtol=0.0, atol=1e-14)


def test_hill_step_vectorized_leaves_non_h2o2i_non_ca_state_entries_unchanged():
    matrix = _state_matrix([(1e-9, 1e-7), (0.1, 0.2), (0.5, 0.5)])
    matrix[:, ODE_STATE_H2O2E] = np.array([2.0, 3.0, 4.0])
    before = matrix.copy()

    updated = FBA_ode.hill_step_vectorized(matrix, 1.0)

    mask = np.ones(matrix.shape[1], dtype=bool)
    mask[[ODE_STATE_H2O2I, ODE_STATE_CA_I]] = False
    np.testing.assert_array_equal(updated[:, mask], before[:, mask])


def test_vectorized_hill_one_step_matches_scalar_radau_within_1e_8():
    pairs = [
        (0.0, 0.0),
        (1e-9, 1e-7),
        (1e-3, 1e-7),
        (0.1, 0.2),
        (0.5, 0.5),
    ]
    matrix = _state_matrix(pairs)
    k = FBA_ode.get_k()
    base = FBA_ode.get_x0()

    expected = []
    for row in matrix:
        solution = solve_ivp(
            lambda t, y: FBA_ode.model_hill(y, t, k, base),
            (0, 1),
            row.copy(),
            method="Radau",
            t_eval=[0, 1],
            atol=1e-10,
            rtol=1e-8,
        )
        expected.append(solution.y[:, -1])
    expected = np.vstack(expected)

    actual = FBA_ode.hill_step_vectorized(matrix, 1.0)
    max_abs = np.max(np.abs(actual[:, [ODE_STATE_H2O2I, ODE_STATE_CA_I]] - expected[:, [ODE_STATE_H2O2I, ODE_STATE_CA_I]]))

    assert max_abs <= HILL_RADUA_ONE_STEP_ATOL, (
        f"vectorized Hill one-step drift {max_abs:g} exceeded "
        f"{HILL_RADUA_ONE_STEP_ATOL:g} vs scalar Radau"
    )


class _InterC:
    def __init__(self, name, values):
        self.name = name
        self.values = list(values)
        self.changed = []
        self.updated = []

    def update_C2(self, *args):
        self.updated.append("update_C2")

    def update_C1(self, *args):
        self.updated.append("update_C1")

    def update_C0(self, *args):
        self.updated.append("update_C0")

    def get_C(self, index):
        return self.values[index]

    def change_C(self, index, value):
        self.changed.append((index, value))


class _Gradient:
    name = EXT_GRADIENT_H2O2E

    def __init__(self, values):
        self.values = list(values)
        self.added = []

    def get_gradient_value_on_surface_shell(self, obj):
        return self.values[obj.ID]

    def add_gradient_value_on_cell_surface(self, obj, value):
        self.added.append((obj.ID, value))


def _minimal_sim(simulation_module, chem_mode, use_vectorized_hill):
    sim = object.__new__(simulation_module.Simulation)
    sim.objects = [
        SimpleNamespace(ID=0, x0=np.zeros(32)),
        SimpleNamespace(ID=1, x0=np.zeros(32)),
    ]
    sim.time = 5
    sim.chem_mode = chem_mode
    sim.use_vectorized_hill = use_vectorized_hill
    sim.perturbation = {
        "enabled": True,
        "cell_index": 1,
        "injection_time": 5,
        "injection_h2o2i": 0.0012,
    }

    camp = _InterC(INTERC_SPECIES_CAMP, [0.1, 0.2])
    h2o2i = _InterC(INTERC_SPECIES_H2O2I, [0.5, 0.6])
    ca = _InterC(INTERC_SPECIES_CA, [1.5, 1.6])
    gradient = _Gradient([7.0, 8.0])

    sim.interC = [camp, h2o2i, ca]
    sim._intC_by_name = {
        INTERC_SPECIES_CAMP: camp,
        INTERC_SPECIES_H2O2I: h2o2i,
        INTERC_SPECIES_CA: ca,
    }
    sim.gradients = [gradient]
    sim._gradients_by_name = {EXT_GRADIENT_H2O2E: gradient}
    return sim, h2o2i, ca, gradient


def test_vectorized_hill_applies_perturbation_before_hill_update(simulation_module, monkeypatch):
    captured = {}

    def fake_hill_step(x_matrix, dt):
        captured["x_matrix"] = x_matrix.copy()
        captured["dt"] = dt
        xnew = x_matrix.copy()
        xnew[:, ODE_STATE_H2O2I] += 10.0
        xnew[:, ODE_STATE_CA_I] += 20.0
        return xnew

    monkeypatch.setattr(simulation_module, "hill_step_vectorized", fake_hill_step)

    sim, h2o2i, ca, gradient = _minimal_sim(
        simulation_module,
        chem_mode="hill",
        use_vectorized_hill=True,
    )

    sim._update_intercellular_and_intracellular_chemistry(
        np.zeros((0, 2), dtype=np.int32),
        len(sim.objects),
    )

    assert captured["dt"] == 1.0
    assert captured["x_matrix"][0, ODE_STATE_H2O2I] == 0.5
    assert captured["x_matrix"][1, ODE_STATE_H2O2I] == 0.0012
    assert captured["x_matrix"][1, ODE_STATE_CA_I] == 1.6
    assert h2o2i.changed == [(0, 10.5), (1, 10.0012)]
    assert ca.changed == [(0, 21.5), (1, 21.6)]
    assert gradient.added == [(0, 0.0), (1, 0.0)]


def test_full_mode_does_not_use_vectorized_hill_backend(simulation_module, monkeypatch):
    def fail_hill_step(*args, **kwargs):
        raise AssertionError("full mode must not call the vectorized Hill backend")

    def fake_solve_ivp(rhs, t_span, x0, **kwargs):
        xnew = x0.copy()
        xnew[ODE_STATE_H2O2E] += 0.25
        xnew[ODE_STATE_H2O2I] += 1.0
        xnew[ODE_STATE_CA_I] += 2.0
        return SimpleNamespace(y=np.column_stack([x0, xnew]))

    monkeypatch.setattr(simulation_module, "hill_step_vectorized", fail_hill_step)
    monkeypatch.setattr(simulation_module, "solve_ivp", fake_solve_ivp)

    sim, h2o2i, ca, gradient = _minimal_sim(
        simulation_module,
        chem_mode="full",
        use_vectorized_hill=True,
    )

    sim._update_intercellular_and_intracellular_chemistry(
        np.zeros((0, 2), dtype=np.int32),
        len(sim.objects),
    )

    assert h2o2i.changed == [(0, 1.5), (1, 1.0012)]
    assert ca.changed == [(0, 3.5), (1, 3.6)]
    assert gradient.added == [(0, 0.25), (1, 0.25)]


def _write_cell_file(path, count=6):
    cols = 3
    spacing = 18.0
    origin = 50.0
    with open(path, "w", encoding="utf-8") as f:
        for idx in range(count):
            row = idx // cols
            col = idx % cols
            x = origin + col * spacing
            y = origin + row * spacing
            f.write(f"{idx},{x},{y}\n")


def test_vectorized_hill_runtime_smoke_writes_expected_artifacts(tmp_path):
    sys.modules.pop("Model_initialParameters", None)
    sys.modules.pop("ros_abm.backends.legacy.runtime.model_initial_parameters", None)
    sys.modules.pop("Simulation", None)
    try:
        Model_initialParameters = importlib.import_module("Model_initialParameters")
    except Exception as exc:
        pytest.skip(f"runtime imports unavailable in this environment: {exc}")

    cell_path = tmp_path / "tiny_cells.csv"
    _write_cell_file(cell_path)

    cfg = RunConfig(
        output_base_path=str(tmp_path),
        output_path=str(tmp_path),
        run_name="1.0",
        chem_mode="hill",
        use_vectorized_hill=True,
        seed=123,
        cell_path=str(cell_path),
        run_postprocess=False,
    )
    cfg.model.te = 1
    cfg.model.dt = 1
    cfg.model.perturbation_cell_index = 1
    cfg.perturbation.cell_index = 1

    Model_initialParameters.main_config(cfg)

    run_dir = Path(tmp_path) / "1.0"
    assert (run_dir / "info.sim").exists()
    assert (run_dir / "run_config.json").exists()
    data = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert data["use_vectorized_hill"] is True
    assert list(run_dir.glob("network*.gpickle"))
    assert list(run_dir.glob(f"{EXT_GRADIENT_H2O2E}_*"))
    assert list(run_dir.glob(f"{INTERC_SPECIES_CAMP}_*"))
    assert list(run_dir.glob(f"{INTERC_SPECIES_H2O2I}_*"))
    assert list(run_dir.glob(f"{INTERC_SPECIES_CA}_*"))
