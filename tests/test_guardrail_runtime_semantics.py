import ast
from pathlib import Path


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _parse_module(rel_path):
    src = (_repo_root() / rel_path).read_text(encoding="utf-8")
    return src, ast.parse(src)


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
    src, tree = _parse_module("Model_initialParameters.py")
    constants = _collect_module_constants(tree)

    model_settings = _find_class(tree, "ModelSettings")
    perturb_default = None
    for node in model_settings.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "perturbation_cell_index":
            perturb_default = _resolve(node.value, constants)
            break
    assert perturb_default == 877

    assert "settings.perturbation_cell_index" in src

    sim_src, sim_tree = _parse_module("Simulation.py")
    sim_constants = _collect_module_constants(sim_tree)
    sim_cls = _find_class(sim_tree, "Simulation")
    update_fn = _find_method(sim_cls, "update")

    found_gate = False
    found_injected_value = False
    for node in ast.walk(update_fn):
        if not isinstance(node, ast.If):
            continue
        compares = [n for n in ast.walk(node.test) if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], ast.Eq)]
        has_time_gate = False
        has_cell_gate = False
        for cmp_node in compares:
            left = _attr_path(cmp_node.left)
            right = _resolve(cmp_node.comparators[0], sim_constants)
            if left == "self.time" and right == 5:
                has_time_gate = True
            if left == "j" and right == 877:
                has_cell_gate = True
        if has_time_gate and has_cell_gate:
            found_gate = True
            for body_node in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                if isinstance(body_node, ast.Assign) and len(body_node.targets) == 1:
                    target = body_node.targets[0]
                    if isinstance(target, ast.Subscript) and _attr_path(target.value) == "x0":
                        idx = _resolve_subscript_index(target, sim_constants)
                        val = _resolve(body_node.value, sim_constants)
                        if idx == 1 and val == 0.0012:
                            found_injected_value = True
            break

    assert found_gate
    assert found_injected_value


def test_interc_ordering_assumption_is_preserved():
    _, tree = _parse_module("Model_initialParameters.py")
    constants = _collect_module_constants(tree)

    main_fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
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
                if isinstance(species, str):
                    var_to_species[node.targets[0].id] = species
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                if call.func.value.id == "sim" and call.func.attr == "add_interC" and call.args:
                    arg = call.args[0]
                    if isinstance(arg, ast.Name) and arg.id in var_to_species:
                        registered_species.append(var_to_species[arg.id])

    assert registered_species[:3] == ["cAMP", "H2O2i", "Ca"]


def test_solver_method_and_tolerances_defaults_are_preserved():
    _, tree = _parse_module("Simulation.py")
    constants = _collect_module_constants(tree)

    sim_cls = _find_class(tree, "Simulation")
    update_fn = _find_method(sim_cls, "update")

    solve_calls = []
    for node in ast.walk(update_fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "solve_ivp":
            solve_calls.append(node)

    assert solve_calls
    target_call = solve_calls[0]
    kw = {k.arg: k.value for k in target_call.keywords if k.arg}

    assert _resolve(kw["method"], constants) == "Radau"
    assert _resolve(kw["atol"], constants) == 1e-10
    assert _resolve(kw["rtol"], constants) == 1e-8
