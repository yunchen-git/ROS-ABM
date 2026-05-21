import importlib
import subprocess
import sys
from pathlib import Path

import pytest


pytest.importorskip("numpy")
pytest.importorskip("matplotlib")
pytest.importorskip("networkx")


def _repo_root():
    return Path(__file__).resolve().parents[1]


def test_root_and_package_imports_expose_main_and_run_analysis():
    root_mod = importlib.import_module("analysis_wave_metrics")
    pkg_mod = importlib.import_module("ros_abm.analysis.wave_metrics")

    assert hasattr(root_mod, "main")
    assert hasattr(pkg_mod, "main")
    assert hasattr(root_mod, "run_analysis")
    assert hasattr(pkg_mod, "run_analysis")


def test_root_shim_main_is_canonical_main():
    root_mod = importlib.import_module("analysis_wave_metrics")
    pkg_mod = importlib.import_module("ros_abm.analysis.wave_metrics")

    assert root_mod.main is pkg_mod.main


def test_root_script_help_entrypoint_smoke():
    repo_root = _repo_root()
    script = repo_root / "analysis_wave_metrics.py"

    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "Wave metrics" in proc.stdout
