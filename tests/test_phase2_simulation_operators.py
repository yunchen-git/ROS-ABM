from types import SimpleNamespace

import numpy as np
import pytest


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
