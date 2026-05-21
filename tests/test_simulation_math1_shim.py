import importlib
import numpy as np


def test_import_simulation_math1_old_path():
    module = importlib.import_module("simulationMath1")
    assert module.__name__ == "simulationMath1"


def test_import_simulation_math1_new_path():
    module = importlib.import_module("ros_abm.backends.legacy.runtime.simulation_math1")
    assert module.__name__ == "ros_abm.backends.legacy.runtime.simulation_math1"


def test_old_and_new_exports_are_compatible():
    old_mod = importlib.import_module("simulationMath1")
    new_mod = importlib.import_module("ros_abm.backends.legacy.runtime.simulation_math1")

    api_names = [
        "RandomPointOnSphere",
        "AddVec",
        "SubtractVec",
        "ScaleVec",
        "Mag",
        "NormVec",
        "Distance",
    ]
    for name in api_names:
        assert hasattr(old_mod, name)
        assert hasattr(new_mod, name)
        assert callable(getattr(old_mod, name))
        assert callable(getattr(new_mod, name))
        assert getattr(old_mod, name) is getattr(new_mod, name)


def test_vector_math_behavior_equivalence():
    old_mod = importlib.import_module("simulationMath1")
    new_mod = importlib.import_module("ros_abm.backends.legacy.runtime.simulation_math1")

    v1 = [1.0, 2.0, 3.0]
    v2 = [4.0, 5.0, 6.0]

    np.testing.assert_allclose(old_mod.AddVec(v1, v2), new_mod.AddVec(v1, v2))
    np.testing.assert_allclose(old_mod.SubtractVec(v2, v1), new_mod.SubtractVec(v2, v1))
    np.testing.assert_allclose(old_mod.ScaleVec(v1, 2.0), new_mod.ScaleVec(v1, 2.0))
    assert old_mod.Mag(v1) == new_mod.Mag(v1)
    np.testing.assert_allclose(old_mod.NormVec(v1), new_mod.NormVec(v1))
    assert old_mod.Distance(v1, v2) == new_mod.Distance(v1, v2)
