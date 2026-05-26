import ast
from pathlib import Path


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _parse_module(rel_path):
    src = (_repo_root() / rel_path).read_text(encoding="utf-8")
    return src, ast.parse(src)


def _parse_model_initialparameters_impl():
    root = _repo_root()
    root_path = root / "Model_initialParameters.py"
    canonical_path = root / "ros_abm/backends/legacy/runtime/model_initial_parameters.py"

    root_src = root_path.read_text(encoding="utf-8")
    shim_marker = "from ros_abm.backends.legacy.runtime.model_initial_parameters import *"
    if shim_marker in root_src and canonical_path.exists():
        src = canonical_path.read_text(encoding="utf-8")
        return src, ast.parse(src)

    return root_src, ast.parse(root_src)


def _collect_module_constants(tree):
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            value = _resolve(node.value, constants)
            if value is not None:
                constants[node.targets[0].id] = value
        elif isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name) or node.value is None:
                continue
            value = _resolve(node.value, constants)
            if value is not None:
                constants[node.target.id] = value
    return constants


def _resolve(node, constants):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        resolved = _resolve(node.operand, constants)
        if isinstance(resolved, (int, float)):
            return -resolved
    if isinstance(node, (ast.Tuple, ast.List)):
        vals = [_resolve(e, constants) for e in node.elts]
        if any(v is None for v in vals):
            return None
        return tuple(vals) if isinstance(node, ast.Tuple) else vals
    return None


def _find_class(tree, class_name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"class {class_name} not found")


def _find_method(class_node, method_name):
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"method {method_name} not found")


def _attr_path(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _attr_path(node.value)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    return None


def _resolve_subscript_index(node, constants):
    if not isinstance(node, ast.Subscript):
        return None
    return _resolve(node.slice, constants)


def test_perturbation_trigger_semantics_are_preserved():
    import run_config

    src, tree = _parse_model_initialparameters_impl()
    perturbation = run_config.PerturbationConfig()

    assert perturbation.enabled is True
    assert perturbation.cell_index == 877
    assert perturbation.injection_time == 5
    assert perturbation.injection_h2o2i == 0.0012
    assert run_config.ModelConfig().perturbation_cell_index == 877
    assert "perturbation_cell_index" in src

    sim_src, sim_tree = _parse_module("Simulation.py")
    sim_cls = _find_class(sim_tree, "Simulation")
    chemistry_fn = _find_method(sim_cls, "_update_intercellular_and_intracellular_chemistry")
    update_src = ast.get_source_segment(sim_src, chemistry_fn)

    assert "self._should_apply_perturbation(j)" in update_src
    assert 'self._perturbation_value("injection_h2o2i")' in update_src
    assert "ODE_STATE_H2O2I" in update_src
    assert "PERTURBATION_CELL_INDEX" not in update_src
    assert "PERTURBATION_INJECTION_TIME" not in update_src
    assert "PERTURBATION_INJECTION_H2O2I" not in update_src


def test_interc_ordering_assumption_is_preserved():
    import species_registry

    _, tree = _parse_model_initialparameters_impl()
    constants = _collect_module_constants(tree)

    main_fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main_config":
            main_fn = node
            break
    assert main_fn is not None

    var_to_species = {}
    registered_species = []

    for node in main_fn.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "IntC_Diff" and call.args:
                species = _resolve(call.args[0], constants)
                if species is None and isinstance(call.args[0], ast.Name):
                    species = getattr(species_registry, call.args[0].id, None)
                if isinstance(species, str):
                    var_to_species[node.targets[0].id] = species
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                if call.func.value.id == "sim" and call.func.attr == "add_interC" and call.args:
                    arg = call.args[0]
                    if isinstance(arg, ast.Name) and arg.id in var_to_species:
                        registered_species.append(var_to_species[arg.id])

    assert registered_species[:3] == list(species_registry.INTERC_SPECIES_ORDER)


def test_solver_method_and_tolerances_defaults_are_preserved():
    _, tree = _parse_module("Simulation.py")
    constants = _collect_module_constants(tree)

    sim_cls = _find_class(tree, "Simulation")
    chemistry_fn = _find_method(sim_cls, "_update_intercellular_and_intracellular_chemistry")

    solve_calls = []
    for node in ast.walk(chemistry_fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "solve_ivp":
            solve_calls.append(node)

    assert solve_calls
    target_call = solve_calls[0]
    kw = {k.arg: k.value for k in target_call.keywords if k.arg}

    assert _resolve(kw["method"], constants) == "Radau"
    assert _resolve(kw["atol"], constants) == 1e-10
    assert _resolve(kw["rtol"], constants) == 1e-8
