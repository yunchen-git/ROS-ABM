from types import SimpleNamespace

import numpy as np
import pytest

from species_registry import (
    EXT_GRADIENT_H2O2E,
    INTERC_SPECIES_CAMP,
    INTERC_SPECIES_CA,
    INTERC_SPECIES_H2O2I,
    ODE_STATE_CA_I,
    ODE_STATE_H2O2E,
    ODE_STATE_H2O2I,
)


def test_update_calls_operator_methods_in_order(simulation_module, monkeypatch):
    calls = []
    nbs2 = object()

    def build_neighbor_pairs(self):
        calls.append("_build_neighbor_pairs")
        return nbs2

    def update_chemistry(self, actual_nbs2, n):
        calls.append(("_update_intercellular_and_intracellular_chemistry", actual_nbs2, n))

    def update_gradients(self):
        calls.append("_update_extracellular_gradients")

    def update_agent_biology(self):
        calls.append("_update_agent_biology")

    monkeypatch.setattr(simulation_module.Simulation, "_build_neighbor_pairs", build_neighbor_pairs)
    monkeypatch.setattr(
        simulation_module.Simulation,
        "_update_intercellular_and_intracellular_chemistry",
        update_chemistry,
    )
    monkeypatch.setattr(simulation_module.Simulation, "_update_extracellular_gradients", update_gradients)
    monkeypatch.setattr(simulation_module.Simulation, "_update_agent_biology", update_agent_biology)

    sim = object.__new__(simulation_module.Simulation)
    sim.objects = [object(), object()]

    sim.update()

    assert calls == [
        "_build_neighbor_pairs",
        ("_update_intercellular_and_intracellular_chemistry", nbs2, 2),
        "_update_extracellular_gradients",
        "_update_agent_biology",
    ]


def test_build_neighbor_pairs_preserves_int32_dtype(simulation_module):
    class Agent:
        def __init__(self, agent_id):
            self.ID = agent_id

    class Network:
        def __init__(self, edges):
            self._edges = edges

        def edges(self):
            return self._edges

    sim = object.__new__(simulation_module.Simulation)
    sim.network = Network([(Agent(1), Agent(3)), (Agent(5), Agent(8))])

    nbs2 = sim._build_neighbor_pairs()

    assert nbs2.dtype == np.int32
    np.testing.assert_array_equal(nbs2, np.array([[1, 3], [5, 8]], dtype=np.int32))


def test_intercellular_chemistry_uses_name_based_ca_lookup(simulation_module, monkeypatch):
    calls = []

    class InterC:
        def __init__(self, name, value):
            self.name = name
            self.value = value
            self.changed = []

        def update_C2(self, *args):
            calls.append((self.name, "update_C2"))

        def update_C1(self, *args):
            calls.append((self.name, "update_C1"))

        def update_C0(self, *args):
            calls.append((self.name, "update_C0"))

        def get_C(self, index):
            calls.append((self.name, "get_C", index))
            return self.value

        def change_C(self, index, value):
            self.changed.append((index, value))
            calls.append((self.name, "change_C", index, value))

    class H2O2eGradient:
        name = EXT_GRADIENT_H2O2E

        def __init__(self):
            self.added = []

        def get_gradient_value_on_surface_shell(self, obj):
            return 1.0

        def add_gradient_value_on_cell_surface(self, obj, value):
            self.added.append((obj, value))

    def fake_solve_ivp(rhs, t_span, x0, **kwargs):
        xnew = x0.copy()
        xnew[ODE_STATE_H2O2E] = x0[ODE_STATE_H2O2E] + 0.25
        xnew[ODE_STATE_H2O2I] = 2.5
        xnew[ODE_STATE_CA_I] = 3.5
        return SimpleNamespace(y=np.column_stack([x0, xnew]))

    monkeypatch.setattr(simulation_module, "get_x0", lambda: np.zeros(32))
    monkeypatch.setattr(simulation_module, "get_k", lambda: np.zeros(100))
    monkeypatch.setattr(simulation_module, "make_model_rhs", lambda mode, k, x_base: lambda t, y: y)
    monkeypatch.setattr(simulation_module, "solve_ivp", fake_solve_ivp)

    camp = InterC(INTERC_SPECIES_CAMP, 0.1)
    ca = InterC(INTERC_SPECIES_CA, 1.25)
    h2o2i = InterC(INTERC_SPECIES_H2O2I, 0.5)
    gradient = H2O2eGradient()

    sim = object.__new__(simulation_module.Simulation)
    sim.objects = [SimpleNamespace(x0=np.zeros(32))]
    sim.interC = [camp, ca, h2o2i]
    sim._intC_by_name = {
        INTERC_SPECIES_CAMP: camp,
        INTERC_SPECIES_CA: ca,
        INTERC_SPECIES_H2O2I: h2o2i,
    }
    sim._gradients_by_name = {EXT_GRADIENT_H2O2E: gradient}
    sim.gradients = [gradient]
    sim.time = 0
    sim.chem_mode = "full"
    sim.perturbation = {
        "enabled": False,
        "cell_index": 877,
        "injection_time": 5,
        "injection_h2o2i": 0.0012,
    }

    sim._update_intercellular_and_intracellular_chemistry(np.zeros((0, 2), dtype=np.int32), 1)

    assert (INTERC_SPECIES_CA, "get_C", 0) in calls
    assert ca.changed == [(0, 3.5)]
    assert h2o2i.changed == [(0, 2.5)]
    assert gradient.added == [(sim.objects[0], 0.25)]


@pytest.mark.parametrize(
    "collide_raises, expected_collision_calls",
    [
        (False, ["collide"]),
        (True, ["collide", "collide_lowDens"]),
    ],
)
def test_run_loop_order_preserves_collision_fallback_and_time_increment(
    simulation_module,
    monkeypatch,
    tmp_path,
    collide_raises,
    expected_collision_calls,
):
    calls = []

    class Header:
        def flush(self):
            pass

        def close(self):
            pass

    def create_header(self):
        self._header = Header()

    def record(name):
        def inner(self, *args):
            calls.append(name)
        return inner

    def collide(self, *args):
        calls.append("collide")
        if collide_raises:
            raise RuntimeError("force low density fallback")

    def save(self):
        calls.append(("save", self.time))

    monkeypatch.setattr(simulation_module.Simulation, "_write_run_config_manifest", lambda self: None)
    monkeypatch.setattr(simulation_module.Simulation, "_create_header_file", create_header)
    monkeypatch.setattr(simulation_module.Simulation, "_set_gradient_inital_conditions", lambda self: None)
    monkeypatch.setattr(simulation_module.Simulation, "update", record("update"))
    monkeypatch.setattr(simulation_module.Simulation, "update_object_queue", record("update_object_queue"))
    monkeypatch.setattr(simulation_module.Simulation, "collide", collide)
    monkeypatch.setattr(simulation_module.Simulation, "collide_lowDens", record("collide_lowDens"))
    monkeypatch.setattr(simulation_module.Simulation, "optimize", record("optimize"))
    monkeypatch.setattr(simulation_module.Simulation, "save", save)

    sim = object.__new__(simulation_module.Simulation)
    sim.name = "1.0"
    sim.path = str(tmp_path)
    sim._sep = "/"
    sim.start_time = 0
    sim.end_time = 0
    sim.time_step = 1
    sim.max_x = 10
    sim.max_y = 10
    sim.max_z = 0
    sim.objects = []
    sim.run_config = SimpleNamespace(run_postprocess=False)

    sim.run(10, 10, 0)

    assert calls == [
        "update",
        "update_object_queue",
        *expected_collision_calls,
        "optimize",
        ("save", 0),
    ]
    assert sim.time == 1
