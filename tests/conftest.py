import importlib
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def simulation_module(monkeypatch):
    """Import Simulation with lightweight postprocess stubs."""
    read_gpickle_stub = types.ModuleType("read_gpickle")
    read_gpickle_stub.read_network = lambda path, t0, tf: None

    gradient_image_stub = types.ModuleType("gradientImage")
    gradient_image_stub.getGradientImages = lambda path: None

    monkeypatch.setitem(sys.modules, "read_gpickle", read_gpickle_stub)
    monkeypatch.setitem(sys.modules, "gradientImage", gradient_image_stub)

    sys.modules.pop("Simulation", None)
    try:
        module = importlib.import_module("Simulation")
    except Exception as exc:  # pragma: no cover - environment-dependent skip
        pytest.skip(f"Simulation import unavailable in this environment: {exc}")
    yield module
    sys.modules.pop("Simulation", None)
