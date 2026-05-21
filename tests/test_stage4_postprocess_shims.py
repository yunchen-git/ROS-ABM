import importlib.util
import sys
import types
from pathlib import Path

import pytest


np = pytest.importorskip("numpy")


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _read(path):
    return path.read_text(encoding="utf-8")


def _load_module_from_path(path, module_name, injected_modules):
    original = {}
    for name, module in injected_modules.items():
        original[name] = sys.modules.get(name)
        sys.modules[name] = module

    try:
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        module = importlib.util.module_from_spec(spec)
        assert spec is not None
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, prev in original.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def test_root_shim_sources_reference_canonical_modules_and_dispatch_points():
    root = _repo_root()

    analysis_src = _read(root / "Analysis.py")
    assert "from ros_abm.analysis.analysis import *" in analysis_src
    assert "getGradientImages(path)" in analysis_src

    gradient_src = _read(root / "gradientImage.py")
    assert "from ros_abm.analysis.gradient_image import *" in gradient_src
    assert "getGradientImages(path)" in gradient_src

    read_src = _read(root / "read_gpickle.py")
    assert "from ros_abm.analysis.read_gpickle import *" in read_src
    assert "read_network(path, 0, 200)" in read_src

    tracking_src = _read(root / "read_gpickle copy.py")
    assert "from ros_abm.analysis.read_gpickle_tracking import *" in tracking_src
    assert "run_tracking_cli(path, t0=0, tf=600)" in tracking_src

    zplot_src = _read(root / "zplot_intC_extc.py")
    assert "from ros_abm.analysis.zplot_intC_extc import *" in zplot_src
    assert "getCombinedH2O2Images3D(path, t0=0, tf=600" in zplot_src


def test_analysis_root_shim_reexports_from_canonical_module():
    root = _repo_root()
    shim_path = root / "Analysis.py"

    canonical = types.ModuleType("ros_abm.analysis.analysis")

    def fake_main(path):
        return path

    def fake_save(name, save_path, data, axis):
        return (name, save_path, axis)

    canonical.getGradientImages = fake_main
    canonical.save_2D_gradient_images = fake_save
    canonical.main = fake_main

    shim = _load_module_from_path(
        shim_path,
        "stage4_shim_analysis",
        {"ros_abm.analysis.analysis": canonical},
    )

    assert shim.getGradientImages is fake_main
    assert shim.save_2D_gradient_images is fake_save


def test_gradientimage_root_shim_reexports_from_canonical_module():
    root = _repo_root()
    shim_path = root / "gradientImage.py"

    canonical = types.ModuleType("ros_abm.analysis.gradient_image")

    def fake_get(path):
        return path

    canonical.getGradientImages = fake_get
    canonical.main = fake_get

    shim = _load_module_from_path(
        shim_path,
        "stage4_shim_gradient_image",
        {"ros_abm.analysis.gradient_image": canonical},
    )

    assert shim.getGradientImages is fake_get


def test_read_gpickle_root_shim_reexports_from_canonical_module():
    root = _repo_root()
    shim_path = root / "read_gpickle.py"

    canonical = types.ModuleType("ros_abm.analysis.read_gpickle")

    def fake_read_network(path, t0, tf):
        return (path, t0, tf)

    canonical.read_network = fake_read_network
    canonical.draw_state_image = object()

    shim = _load_module_from_path(
        shim_path,
        "stage4_shim_read_gpickle",
        {"ros_abm.analysis.read_gpickle": canonical},
    )

    assert shim.read_network is fake_read_network
    assert shim.draw_state_image is canonical.draw_state_image


def test_read_gpickle_copy_root_shim_reexports_from_canonical_module():
    root = _repo_root()
    shim_path = root / "read_gpickle copy.py"

    canonical = types.ModuleType("ros_abm.analysis.read_gpickle_tracking")

    def fake_run_tracking_cli(path, t0=0, tf=600):
        return (path, t0, tf)

    canonical.run_tracking_cli = fake_run_tracking_cli
    canonical.read_network = object()
    canonical.select_cells_by_position = object()

    shim = _load_module_from_path(
        shim_path,
        "stage4_shim_read_gpickle_copy",
        {"ros_abm.analysis.read_gpickle_tracking": canonical},
    )

    assert shim.run_tracking_cli is fake_run_tracking_cli
    assert shim.read_network is canonical.read_network
    assert shim.select_cells_by_position is canonical.select_cells_by_position


def test_zplot_root_shim_reexports_from_canonical_module():
    root = _repo_root()
    shim_path = root / "zplot_intC_extc.py"

    canonical = types.ModuleType("ros_abm.analysis.zplot_intC_extc")

    def fake_combined(path, t0=0, tf=200, plot_2d=True, plot_3d=True, make_graph=True, squished=True):
        return (path, t0, tf, plot_2d, plot_3d, make_graph, squished)

    canonical.getCombinedH2O2Images3D = fake_combined

    shim = _load_module_from_path(
        shim_path,
        "stage4_shim_zplot_intc_extc",
        {"ros_abm.analysis.zplot_intC_extc": canonical},
    )

    assert shim.getCombinedH2O2Images3D is fake_combined


def test_legacy_shared_utils_old_and_new_paths_are_behavior_equivalent():
    old_mod = __import__("legacy_shared_utils")
    new_mod = __import__("ros_abm.analysis.legacy_shared_utils", fromlist=["_"])

    assert old_mod.ZeroPad(10, 3) == new_mod.ZeroPad(10, 3)
    assert old_mod.ZeroPad(10, 0) == new_mod.ZeroPad(10, 0)

    class _Node:
        def __init__(self, x, y):
            self.location = np.array([x, y, 0.0], dtype=float)
            self.C = {"a": 1.0}

    class _Graph:
        def __init__(self):
            self._nodes = [_Node(0.0, 0.0), _Node(2.0, 0.0)]

        def nodes(self):
            return self._nodes

    graph = _Graph()
    old_center = old_mod.network_avg(graph)
    new_center = new_mod.network_avg(graph)
    np.testing.assert_allclose(old_center, new_center, rtol=0.0, atol=0.0)

    old_radius = old_mod.network_radius(graph)
    new_radius = new_mod.network_radius(graph)
    assert old_radius == new_radius

    old_species = old_mod.get_available_species(graph)
    new_species = new_mod.get_available_species(graph)
    assert old_species == new_species
