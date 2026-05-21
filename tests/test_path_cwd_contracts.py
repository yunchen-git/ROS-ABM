import ast
import importlib
import sys
import types
from pathlib import Path


def _repo_root():
    return Path(__file__).resolve().parents[1]


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


def test_resolve_output_base_path_preserves_default_and_override(monkeypatch):
    mod = _load_model_run_with_stubs(monkeypatch)

    assert mod.resolve_output_base_path() == mod.os.path.expanduser("~/Documents/PhD/test")
    assert mod.resolve_output_base_path("~/tmp_out") == mod.os.path.expanduser("~/tmp_out")


def test_resolve_cell_path_preserves_default_and_override(monkeypatch):
    mod = _load_model_initialparameters_with_stubs(monkeypatch)

    assert mod.resolve_cell_path(cwd="/tmp/example") == "/tmp/example/new_populated.txt"
    assert mod.resolve_cell_path(cell_path="/tmp/custom_cells.txt", cwd="/tmp/example") == "/tmp/custom_cells.txt"


def test_simulation_load_gene_expression_data_path_args_and_defaults_present():
    src = (_repo_root() / "Simulation.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    target_fn = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Simulation":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "load_gene_expression_data":
                    target_fn = child
                    break
            break

    assert target_fn is not None

    arg_names = [a.arg for a in target_fn.args.args]
    assert arg_names == ["self", "metadata_path", "expression_dir", "cwd"]

    defaults = target_fn.args.defaults
    assert len(defaults) == 3
    assert all(isinstance(d, ast.Constant) and d.value is None for d in defaults)

    assert "if metadata_path is None:" in src
    assert "/Users/huyunchen/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Melissa Kemp Lab/Reference Models/kemplab github/Raddatz AD Antioxidants 2023/AntioxidantsODE-main/2021-06-08_scrnaseqdata.csv" in src
    assert 'files = os.path.join(cwd, "HNSCC_Broad_scRNAseq")' in src
