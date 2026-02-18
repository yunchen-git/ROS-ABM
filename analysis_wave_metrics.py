#!/usr/bin/env python3
"""
Compute wave propagation metrics for intracellular Ca and intracellular H2O2
from saved network snapshots (network*.gpickle).

Quickstart:
  python analysis_wave_metrics.py RUN_DIR
  python analysis_wave_metrics.py RUN_DIR --out RUN_DIR/analysis
  python analysis_wave_metrics.py RUN_DIR --bolus earliest_ca --threshold quantile --q 0.9

Outputs written to OUT_DIR:
- wave_metrics_summary.json
- activation_time_map_ca.png
- activation_time_map_h2o2_intra.png
- speed_fit_ca.png
- speed_fit_h2o2_intra.png
- recruitment_curve_ca.png
- recruitment_curve_h2o2_intra.png
- bolus_neighbor_traces.png
- bolus_neighbor_traces.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Fixed intracellular indices
H_INDEX = 1
CA_INDEX = 29

# Candidate keys (case-insensitive for dict-style nodes)
POSITION_KEYS = ["pos", "position", "loc", "location"]
STATE_VECTOR_KEYS = ["x", "state", "species", "conc", "concentrations", "c"]
DIRECT_CA_KEYS = ["ca", "calcium"]
DIRECT_H_KEYS = ["h2o2", "h2o2i", "h_intra", "h"]

# Object attribute fallbacks (object-style nodes)
OBJECT_POSITION_ATTRS = ["location", "loc", "position", "pos"]
OBJECT_STATE_ATTRS = ["x", "state", "species", "conc", "concentrations", "c", "x0"]
OBJECT_ID_ATTRS = ["ID", "id"]

# Numeric extraction from network file names
NUMERIC_TOKEN_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wave metrics for intracellular Ca and H2O2")
    parser.add_argument("run_dir", type=str, help="Directory containing network*.gpickle snapshots")
    parser.add_argument("--out", dest="out_dir", type=str, default=None, help="Output directory (default: RUN_DIR/analysis)")
    parser.add_argument("--bolus", type=str, choices=["center", "earliest_ca"], default="center")
    parser.add_argument("--threshold", type=str, choices=["auto", "baseline_sd", "quantile"], default="auto")
    parser.add_argument("--baseline", dest="baseline_frac", type=float, default=0.2)
    parser.add_argument("--k", dest="k_sd", type=float, default=3.0)
    parser.add_argument("--q", dest="quantile", type=float, default=0.90)
    parser.add_argument("--farq", dest="far_quantile", type=float, default=0.90)
    parser.add_argument("--t0", dest="t0", type=float, default=None)
    parser.add_argument("--tf", dest="tf", type=float, default=None)
    return parser.parse_args()


def normalize_key_map(mapping: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in mapping.keys():
        if isinstance(key, str):
            lowered = key.lower()
            if lowered not in result:
                result[lowered] = key
    return result


def find_case_insensitive_key(mapping: dict, candidates: list[str]) -> str | None:
    lowered_map = normalize_key_map(mapping)
    for candidate in candidates:
        if candidate in lowered_map:
            return lowered_map[candidate]
    return None


def safe_float(value: object) -> float | None:
    try:
        value_f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def vector_from_object(value: object, min_len: int = 2) -> np.ndarray | None:
    if isinstance(value, (str, bytes, dict)):
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size < min_len:
        return None
    if not np.all(np.isfinite(arr[:min_len])):
        return None
    return arr


def extract_indices_from_vector(container: object) -> tuple[float, float] | None:
    if isinstance(container, dict):
        return None
    try:
        h_val = container[H_INDEX]  # type: ignore[index]
        ca_val = container[CA_INDEX]  # type: ignore[index]
    except Exception:
        return None
    h = safe_float(h_val)
    ca = safe_float(ca_val)
    if h is None or ca is None:
        return None
    return h, ca


def list_node_attributes(node: object, max_items: int = 40) -> list[str]:
    attrs: list[str] = []
    for attr in dir(node):
        if attr.startswith("__"):
            continue
        attrs.append(attr)
    attrs.sort()
    return attrs[:max_items]


def truncate_text(text: str, n: int = 80) -> str:
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def extraction_error(reason: str, node: object, attr_dict: dict) -> RuntimeError:
    dict_keys = sorted([str(k) for k in attr_dict.keys()]) if attr_dict else []
    obj_attrs = list_node_attributes(node)
    message = (
        f"{reason}. "
        f"Node repr: {truncate_text(repr(node), 120)}. "
        f"Available node-dict keys: {dict_keys if dict_keys else 'none'}. "
        f"Available node-object attrs: {obj_attrs if obj_attrs else 'none'}."
    )
    return RuntimeError(message)


def get_node_attr_dict(graph: object, node: object) -> dict:
    try:
        node_data = graph.nodes[node]  # type: ignore[index]
        if isinstance(node_data, dict):
            return node_data
    except Exception:
        pass
    return {}


def get_node_id_string(node: object, attr_dict: dict) -> str:
    for attr in OBJECT_ID_ATTRS:
        if hasattr(node, attr):
            value = getattr(node, attr)
            return str(value)
    key = find_case_insensitive_key(attr_dict, OBJECT_ID_ATTRS)
    if key is not None:
        return str(attr_dict[key])
    try:
        return str(node)
    except Exception:
        return repr(node)


def extract_position_from_dict(attr_dict: dict) -> tuple[np.ndarray, str] | None:
    if not attr_dict:
        return None

    # Vector-style position key
    key = find_case_insensitive_key(attr_dict, POSITION_KEYS)
    if key is not None:
        vec = vector_from_object(attr_dict[key], min_len=2)
        if vec is not None:
            return vec.astype(float), f"dict:{key}"

    # Split keys x/y(/z)
    kx = find_case_insensitive_key(attr_dict, ["x"])
    ky = find_case_insensitive_key(attr_dict, ["y"])
    kz = find_case_insensitive_key(attr_dict, ["z"])
    if kx is not None and ky is not None:
        x = safe_float(attr_dict[kx])
        y = safe_float(attr_dict[ky])
        if x is not None and y is not None:
            if kz is not None:
                z = safe_float(attr_dict[kz])
                if z is not None:
                    return np.array([x, y, z], dtype=float), "dict:x,y,z"
            return np.array([x, y], dtype=float), "dict:x,y"

    return None


def extract_position_from_object(node: object) -> tuple[np.ndarray, str] | None:
    for attr in OBJECT_POSITION_ATTRS:
        if hasattr(node, attr):
            vec = vector_from_object(getattr(node, attr), min_len=2)
            if vec is not None:
                return vec.astype(float), f"obj:{attr}"

    if hasattr(node, "x") and hasattr(node, "y"):
        x = safe_float(getattr(node, "x"))
        y = safe_float(getattr(node, "y"))
        if x is not None and y is not None:
            if hasattr(node, "z"):
                z = safe_float(getattr(node, "z"))
                if z is not None:
                    return np.array([x, y, z], dtype=float), "obj:x,y,z"
            return np.array([x, y], dtype=float), "obj:x,y"

    return None


def extract_direct_signals_from_mapping(mapping: dict) -> tuple[float, float] | None:
    if not mapping:
        return None
    key_ca = find_case_insensitive_key(mapping, DIRECT_CA_KEYS)
    key_h = find_case_insensitive_key(mapping, DIRECT_H_KEYS)
    if key_ca is None or key_h is None:
        return None
    ca = safe_float(mapping[key_ca])
    h = safe_float(mapping[key_h])
    if h is None or ca is None:
        return None
    return h, ca


def extract_signals_from_dict(attr_dict: dict) -> tuple[float, float, str] | None:
    if not attr_dict:
        return None

    for key_name in STATE_VECTOR_KEYS:
        key = find_case_insensitive_key(attr_dict, [key_name])
        if key is None:
            continue
        hc = extract_indices_from_vector(attr_dict[key])
        if hc is not None:
            h, ca = hc
            return h, ca, f"dict:{key}"

    direct = extract_direct_signals_from_mapping(attr_dict)
    if direct is not None:
        h, ca = direct
        return h, ca, "direct_keys"

    return None


def extract_signals_from_object(node: object) -> tuple[float, float, str] | None:
    for attr in OBJECT_STATE_ATTRS:
        if not hasattr(node, attr):
            continue
        value = getattr(node, attr)
        hc = extract_indices_from_vector(value)
        if hc is not None:
            h, ca = hc
            return h, ca, f"obj:{attr}"

        if isinstance(value, dict):
            direct = extract_direct_signals_from_mapping(value)
            if direct is not None:
                h, ca = direct
                return h, ca, f"obj:{attr}:direct_keys"

    return None


def extract_position(node: object, attr_dict: dict) -> tuple[np.ndarray, str]:
    dict_result = extract_position_from_dict(attr_dict)
    if dict_result is not None:
        return dict_result

    obj_result = extract_position_from_object(node)
    if obj_result is not None:
        return obj_result

    raise extraction_error("Could not extract node position", node, attr_dict)


def extract_signals(node: object, attr_dict: dict) -> tuple[float, float, str]:
    dict_result = extract_signals_from_dict(attr_dict)
    if dict_result is not None:
        return dict_result

    obj_result = extract_signals_from_object(node)
    if obj_result is not None:
        return obj_result

    raise extraction_error(
        "Could not extract intracellular signals (H2O2 index 1, Ca index 29)",
        node,
        attr_dict,
    )


def parse_time_token(path: str) -> float | None:
    base = os.path.basename(path)
    tokens = NUMERIC_TOKEN_RE.findall(base)
    if not tokens:
        return None
    try:
        return float(tokens[-1])
    except ValueError:
        return None


def sort_network_files(files: list[str]) -> list[str]:
    def _key(path: str) -> tuple[int, float | str, str]:
        token = parse_time_token(path)
        base = os.path.basename(path)
        if token is not None:
            return (0, token, base)
        return (1, base, base)

    return sorted(files, key=_key)


def module_redirect_candidates(module_name: str) -> list[str]:
    candidates = [module_name]
    prefixes = ["ros_abm.", "ROS_ABM.", "ros-abm.", "ROS-ABM."]
    for prefix in prefixes:
        if module_name.startswith(prefix):
            candidates.append(module_name[len(prefix) :])

    module_aliases = {
        "simulation_objects": "simulationObjects",
        "simulation_math": "simulationMath1",
        "fba_ode": "FBA_ode",
        "model_initial_parameters": "Model_initialParameters",
    }

    expanded: list[str] = list(candidates)
    for candidate in list(candidates):
        if candidate in module_aliases:
            expanded.append(module_aliases[candidate])
        if "." in candidate:
            parts = candidate.split(".")
            tail = parts[-1]
            if tail in module_aliases:
                expanded.append(module_aliases[tail])
                expanded.append(".".join(parts[:-1] + [module_aliases[tail]]))
    candidates = expanded

    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def load_with_module_redirects(path: str, encoding: str | None = None) -> object:
    class RedirectingUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> object:
            last_error: Exception | None = None
            for candidate in module_redirect_candidates(module):
                try:
                    return super().find_class(candidate, name)
                except (ModuleNotFoundError, AttributeError) as exc:
                    last_error = exc
                    continue
            if last_error is not None:
                raise last_error
            return super().find_class(module, name)

    with open(path, "rb") as handle:
        kwargs: dict[str, object] = {}
        if encoding is not None:
            kwargs["encoding"] = encoding
        unpickler = RedirectingUnpickler(handle, **kwargs)
        return unpickler.load()


def load_gpickle(path: str) -> object:
    errors: list[str] = []

    # Preferred path per requirement, when available in this networkx version.
    nx_reader = getattr(nx, "read_gpickle", None)
    if callable(nx_reader):
        try:
            graph = nx_reader(path)
            if hasattr(graph, "nodes"):
                return graph
            errors.append("networkx.read_gpickle returned non graph-like object")
        except Exception as exc_nx:
            errors.append(f"networkx.read_gpickle error: {exc_nx}")
    else:
        errors.append("networkx.read_gpickle unavailable in installed networkx version")

    # Secondary networkx path for versions where read_gpickle moved.
    try:
        nx_reader_alt = nx.readwrite.gpickle.read_gpickle  # type: ignore[attr-defined]
    except Exception as exc_lookup:
        nx_reader_alt = None
        errors.append(f"networkx.readwrite.gpickle.read_gpickle lookup error: {exc_lookup}")

    if callable(nx_reader_alt):
        try:
            graph = nx_reader_alt(path)
            if hasattr(graph, "nodes"):
                return graph
            errors.append("networkx.readwrite.gpickle.read_gpickle returned non graph-like object")
        except Exception as exc_alt:
            errors.append(f"networkx.readwrite.gpickle.read_gpickle error: {exc_alt}")

    # Raw pickle fallback with module-path redirects.
    try:
        graph = load_with_module_redirects(path, encoding=None)
    except Exception as exc_pickle:
        errors.append(f"pickle.Unpickler error: {exc_pickle}")
        try:
            graph = load_with_module_redirects(path, encoding="latin1")
        except Exception as exc_latin1:
            errors.append(f"pickle.Unpickler(..., encoding='latin1') error: {exc_latin1}")
            raise RuntimeError(f"Failed to load gpickle '{path}'. {'; '.join(errors)}") from exc_latin1

    if not hasattr(graph, "nodes"):
        raise RuntimeError(f"Loaded object from '{path}' is not graph-like (missing .nodes).")
    return graph


def source_summary(sources: set[str]) -> str:
    if not sources:
        return "unknown"
    if len(sources) == 1:
        return list(sources)[0]
    return "mixed:" + ",".join(sorted(sources))


def build_cohort(first_graph: object) -> tuple[list[str], dict[str, int], list[dict[str, object]], np.ndarray, str, str]:
    raw_positions: dict[str, np.ndarray] = {}
    position_sources: set[str] = set()
    state_sources: set[str] = set()

    for node in first_graph.nodes():  # type: ignore[attr-defined]
        attr_dict = get_node_attr_dict(first_graph, node)
        node_id = get_node_id_string(node, attr_dict)
        if node_id in raw_positions:
            raise RuntimeError(
                "Duplicate node identifier detected in first snapshot: "
                f"'{node_id}'. Ensure stable unique IDs (e.g., node.ID)."
            )

        pos, pos_source = extract_position(node, attr_dict)
        _, _, state_source = extract_signals(node, attr_dict)

        raw_positions[node_id] = pos
        position_sources.add(pos_source)
        state_sources.add(state_source)

    id_order = sorted(raw_positions.keys())
    id_to_index = {node_id: i for i, node_id in enumerate(id_order)}

    dim = 2
    for vec in raw_positions.values():
        dim = max(dim, int(vec.size))

    positions = np.zeros((len(id_order), dim), dtype=float)
    for node_id, idx in id_to_index.items():
        vec = raw_positions[node_id]
        positions[idx, : vec.size] = vec

    id_info = [{"id_str": node_id, "index_int": idx} for idx, node_id in enumerate(id_order)]

    return id_order, id_to_index, id_info, positions, source_summary(position_sources), source_summary(state_sources)


def extract_frame_signals(graph: object, id_to_index: dict[str, int]) -> tuple[np.ndarray, np.ndarray, set[str]]:
    n_cells = len(id_to_index)
    h_values = np.full(n_cells, np.nan, dtype=float)
    ca_values = np.full(n_cells, np.nan, dtype=float)
    state_sources: set[str] = set()
    seen_ids: set[str] = set()

    for node in graph.nodes():  # type: ignore[attr-defined]
        attr_dict = get_node_attr_dict(graph, node)
        node_id = get_node_id_string(node, attr_dict)

        if node_id not in id_to_index:
            continue
        if node_id in seen_ids:
            continue

        h, ca, state_source = extract_signals(node, attr_dict)
        idx = id_to_index[node_id]
        h_values[idx] = h
        ca_values[idx] = ca
        state_sources.add(state_source)
        seen_ids.add(node_id)

    if len(seen_ids) == 0:
        raise RuntimeError("No first-snapshot cohort IDs were matched in a frame. Check node ID consistency across snapshots.")

    return h_values, ca_values, state_sources


def compute_threshold(values: np.ndarray, mode: str, baseline_frac: float, k_sd: float, quantile: float) -> tuple[float, str]:
    n_cells, n_time = values.shape
    if n_time < 2:
        raise RuntimeError("Need at least 2 timepoints to compute thresholds.")

    baseline_n = int(math.ceil(max(0.0, baseline_frac) * n_time))
    baseline_n = max(2, baseline_n)
    baseline_n = min(n_time, baseline_n)

    baseline_block = values[:, :baseline_n]
    baseline_vals = baseline_block[np.isfinite(baseline_block)]

    all_vals = values[np.isfinite(values)]
    if all_vals.size == 0:
        raise RuntimeError("Signal matrix has no finite values; cannot compute threshold.")

    def baseline_sd_threshold() -> tuple[float, bool]:
        if baseline_vals.size < 2:
            return float("nan"), False
        mu = float(np.mean(baseline_vals))
        sigma = float(np.std(baseline_vals))
        theta = mu + k_sd * sigma
        valid = math.isfinite(theta) and sigma > EPS and abs(theta - mu) > EPS
        return theta, valid

    def quantile_threshold() -> float:
        q_clamped = min(1.0, max(0.0, quantile))
        return float(np.quantile(all_vals, q_clamped))

    if mode == "baseline_sd":
        theta, valid = baseline_sd_threshold()
        if not valid:
            raise RuntimeError(
                "baseline_sd threshold is degenerate (insufficient baseline variance/length). "
                "Use --threshold quantile or --threshold auto."
            )
        return theta, "baseline_sd"

    if mode == "quantile":
        return quantile_threshold(), "quantile"

    # auto
    theta, valid = baseline_sd_threshold()
    if valid:
        return theta, "baseline_sd"
    return quantile_threshold(), "quantile"


def compute_activation_times(values: np.ndarray, time_values: np.ndarray, threshold: float) -> tuple[list[float | None], float]:
    n_cells, _ = values.shape
    t_act: list[float | None] = [None] * n_cells
    n_active = 0

    for i in range(n_cells):
        trace = values[i, :]
        hits = np.where(np.isfinite(trace) & (trace > threshold))[0]
        if hits.size > 0:
            t_act[i] = float(time_values[int(hits[0])])
            n_active += 1

    fraction = n_active / float(n_cells) if n_cells > 0 else float("nan")
    return t_act, fraction


def compute_recruitment(values: np.ndarray, time_values: np.ndarray, threshold: float) -> tuple[np.ndarray, float | None, float | None]:
    n_cells, n_time = values.shape
    recruited = np.zeros(n_time, dtype=float)

    for t_idx in range(n_time):
        column = values[:, t_idx]
        recruited[t_idx] = float(np.sum(np.isfinite(column) & (column > threshold))) / float(n_cells)

    t50: float | None = None
    t90: float | None = None

    idx50 = np.where(recruited >= 0.5)[0]
    if idx50.size > 0:
        t50 = float(time_values[int(idx50[0])])

    idx90 = np.where(recruited >= 0.9)[0]
    if idx90.size > 0:
        t90 = float(time_values[int(idx90[0])])

    return recruited, t50, t90


def compute_wave_fit(t_act: list[float | None], positions: np.ndarray) -> dict[str, object]:
    valid_idx = [i for i, t in enumerate(t_act) if t is not None]

    if len(valid_idx) == 0:
        return {
            "origin_idx": None,
            "r": np.array([], dtype=float),
            "t": np.array([], dtype=float),
            "a": None,
            "b": None,
            "v": None,
            "r2": None,
            "n_points": 0,
            "fit_ok": False,
        }

    origin_idx = min(valid_idx, key=lambda i: (float(t_act[i]), i))
    origin_pos = positions[origin_idx, :]
    distances = np.linalg.norm(positions - origin_pos, axis=1)

    r = np.array([distances[i] for i in valid_idx], dtype=float)
    t = np.array([float(t_act[i]) for i in valid_idx], dtype=float)

    fit_ok = (len(valid_idx) >= 3) and (float(np.std(r)) > EPS)

    if not fit_ok:
        return {
            "origin_idx": int(origin_idx),
            "r": r,
            "t": t,
            "a": None,
            "b": None,
            "v": None,
            "r2": None,
            "n_points": int(len(valid_idx)),
            "fit_ok": False,
        }

    design = np.column_stack([np.ones_like(r), r])
    coeff, _, _, _ = np.linalg.lstsq(design, t, rcond=None)
    a = float(coeff[0])
    b = float(coeff[1])

    t_pred = a + b * r
    ss_res = float(np.sum((t - t_pred) ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    r2: float | None = None
    if ss_tot > EPS:
        r2 = 1.0 - ss_res / ss_tot

    v: float | None = None
    if b > 0:
        v = 1.0 / b

    return {
        "origin_idx": int(origin_idx),
        "r": r,
        "t": t,
        "a": a,
        "b": b,
        "v": v,
        "r2": r2,
        "n_points": int(len(valid_idx)),
        "fit_ok": True,
    }


def short_id(text: str, max_len: int = 36) -> str:
    return truncate_text(text, max_len)


def to_json_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int, np.floating, np.integer)):
        value_f = float(value)
        if math.isfinite(value_f):
            return value_f
    return None


def save_activation_map(path: str, positions_xy: np.ndarray, t_act: list[float | None], title: str) -> None:
    fig, ax = plt.subplots()

    activated_idx = [i for i, t in enumerate(t_act) if t is not None]
    if len(activated_idx) > 0:
        x = positions_xy[activated_idx, 0]
        y = positions_xy[activated_idx, 1]
        c = np.array([float(t_act[i]) for i in activated_idx], dtype=float)
        scatter = ax.scatter(x, y, c=c)
        fig.colorbar(scatter, ax=ax)
    else:
        ax.scatter(positions_xy[:, 0], positions_xy[:, 1])
        ax.text(0.5, 0.5, "No activated cells", transform=ax.transAxes, ha="center", va="center")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_speed_fit_plot(path: str, fit: dict[str, object], title: str) -> None:
    fig, ax = plt.subplots()

    r = fit["r"]
    t = fit["t"]
    ax.scatter(r, t)

    if bool(fit["fit_ok"]):
        a = float(fit["a"])
        b = float(fit["b"])
        if r.size > 0:
            r_line = np.linspace(float(np.min(r)), float(np.max(r)), 100)
            t_line = a + b * r_line
            ax.plot(r_line, t_line)
        subtitle = f"v={to_json_float(fit['v'])}, R2={to_json_float(fit['r2'])}"
        ax.set_title(f"{title}\n{subtitle}")
    else:
        ax.set_title(f"{title}\n(insufficient points or distance spread for fit)")

    ax.set_xlabel("Distance from origin")
    ax.set_ylabel("Activation time")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_recruitment_plot(path: str, time_values: np.ndarray, recruitment: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(time_values, recruitment)
    ax.set_xlabel("Time")
    ax.set_ylabel("Fraction above threshold")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def compute_lag_stats(t_act_ca: list[float | None], t_act_h: list[float | None]) -> dict[str, float | int | None]:
    deltas: list[float] = []
    for t_ca, t_h in zip(t_act_ca, t_act_h):
        if t_ca is None or t_h is None:
            continue
        deltas.append(float(t_ca - t_h))

    if len(deltas) == 0:
        return {"n": 0, "min": None, "median": None, "max": None}

    arr = np.array(deltas, dtype=float)
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def choose_bolus_index(mode: str, positions: np.ndarray, t_act_ca: list[float | None]) -> tuple[int, str]:
    centroid = np.mean(positions, axis=0)
    center_idx = int(np.argmin(np.linalg.norm(positions - centroid, axis=1)))

    if mode == "center":
        return center_idx, "center"

    active_idx = [i for i, t in enumerate(t_act_ca) if t is not None]
    if len(active_idx) == 0:
        return center_idx, "center_fallback_from_earliest_ca"

    earliest_idx = min(active_idx, key=lambda i: (float(t_act_ca[i]), i))
    return int(earliest_idx), "earliest_ca"


def select_neighbor_indices(positions: np.ndarray, bolus_idx: int, far_quantile: float) -> dict[str, tuple[int, float]]:
    distances = np.linalg.norm(positions - positions[bolus_idx, :], axis=1)

    other_indices = [i for i in range(len(distances)) if i != bolus_idx and math.isfinite(float(distances[i]))]
    if len(other_indices) == 0:
        d0 = float(distances[bolus_idx]) if math.isfinite(float(distances[bolus_idx])) else 0.0
        return {
            "bolus": (bolus_idx, 0.0),
            "near": (bolus_idx, d0),
            "mid": (bolus_idx, d0),
            "far": (bolus_idx, d0),
        }

    near_idx = min(other_indices, key=lambda i: float(distances[i]))
    d_near = float(distances[near_idx])

    q_clamped = min(1.0, max(0.0, far_quantile))
    d_all = np.array([float(distances[i]) for i in other_indices], dtype=float)
    d_far_target = float(np.quantile(d_all, q_clamped))
    far_idx = min(other_indices, key=lambda i: abs(float(distances[i]) - d_far_target))
    d_far = float(distances[far_idx])

    d_mid_target = 0.5 * (d_near + d_far)
    mid_idx = min(other_indices, key=lambda i: abs(float(distances[i]) - d_mid_target))

    return {
        "bolus": (bolus_idx, 0.0),
        "near": (near_idx, float(distances[near_idx])),
        "mid": (mid_idx, float(distances[mid_idx])),
        "far": (far_idx, float(distances[far_idx])),
    }


def save_bolus_neighbor_traces(
    path: str,
    time_values: np.ndarray,
    h_values: np.ndarray,
    ca_values: np.ndarray,
    selected: dict[str, tuple[int, float]],
    id_info: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True)

    roles = ["bolus", "near", "mid", "far"]
    for role in roles:
        idx, dist = selected[role]
        node_id = str(id_info[idx]["id_str"])
        label = f"{role}: idx={idx}, d={dist:.3g}, id={short_id(node_id)}"
        axes[0].plot(time_values, h_values[idx, :], label=label)
        axes[1].plot(time_values, ca_values[idx, :], label=label)

    axes[0].set_title("Intracellular H2O2 traces")
    axes[1].set_title("Intracellular Ca traces")
    axes[1].set_xlabel("Time")
    axes[0].set_ylabel("H2O2")
    axes[1].set_ylabel("Ca")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def classify_run(f_ca: float, r2_ca: float | None) -> str:
    if f_ca < 0.10:
        return "no_propagation"
    if r2_ca is None or r2_ca < 0.40:
        return "patchy"
    if f_ca >= 0.60 and r2_ca >= 0.40:
        return "wave"
    return "partial"


def collect_network_files(run_dir: str) -> list[str]:
    pattern = os.path.join(run_dir, "network*.gpickle")
    files = glob.glob(pattern)
    if len(files) == 0:
        raise RuntimeError(f"No files matched pattern: {pattern}")
    return sort_network_files(files)


def build_time_axis(files: list[str]) -> tuple[np.ndarray, str]:
    parsed = [parse_time_token(path) for path in files]
    if all(token is not None for token in parsed):
        return np.array([float(token) for token in parsed], dtype=float), "time_from_filename"
    return np.arange(len(files), dtype=float), "frame_index"


def filter_time_window(
    files: list[str],
    time_values: np.ndarray,
    t0: float | None,
    tf: float | None,
) -> tuple[list[str], np.ndarray]:
    mask = np.ones(time_values.size, dtype=bool)
    if t0 is not None:
        mask &= time_values >= float(t0)
    if tf is not None:
        mask &= time_values <= float(tf)

    if np.sum(mask) == 0:
        raise RuntimeError("No timepoints remain after applying --t0/--tf filter.")

    filtered_files = [files[i] for i in np.where(mask)[0]]
    filtered_times = time_values[mask]
    return filtered_files, filtered_times


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(args.out_dir if args.out_dir is not None else os.path.join(run_dir, "analysis"))
    os.makedirs(out_dir, exist_ok=True)

    files = collect_network_files(run_dir)
    time_values, time_axis_mode = build_time_axis(files)
    files, time_values = filter_time_window(files, time_values, args.t0, args.tf)

    first_graph = load_gpickle(files[0])
    id_order, id_to_index, id_info, positions, pos_source, state_source_first = build_cohort(first_graph)

    n_cells = len(id_order)
    n_time = len(files)

    h_values = np.full((n_cells, n_time), np.nan, dtype=float)
    ca_values = np.full((n_cells, n_time), np.nan, dtype=float)

    state_sources_all: set[str] = {state_source_first}

    for t_idx, path in enumerate(files):
        graph = load_gpickle(path)
        h_frame, ca_frame, frame_sources = extract_frame_signals(graph, id_to_index)
        h_values[:, t_idx] = h_frame
        ca_values[:, t_idx] = ca_frame
        state_sources_all.update(frame_sources)

    theta_ca, method_ca = compute_threshold(ca_values, args.threshold, args.baseline_frac, args.k_sd, args.quantile)
    theta_h, method_h = compute_threshold(h_values, args.threshold, args.baseline_frac, args.k_sd, args.quantile)

    t_act_ca, f_ca = compute_activation_times(ca_values, time_values, theta_ca)
    t_act_h, f_h = compute_activation_times(h_values, time_values, theta_h)

    fit_ca = compute_wave_fit(t_act_ca, positions)
    fit_h = compute_wave_fit(t_act_h, positions)

    rec_ca, t50_ca, t90_ca = compute_recruitment(ca_values, time_values, theta_ca)
    rec_h, t50_h, t90_h = compute_recruitment(h_values, time_values, theta_h)

    lag_stats = compute_lag_stats(t_act_ca, t_act_h)

    bolus_idx, bolus_mode_resolved = choose_bolus_index(args.bolus, positions, t_act_ca)
    selected = select_neighbor_indices(positions, bolus_idx, args.far_quantile)

    positions_xy = positions[:, :2]

    save_activation_map(
        os.path.join(out_dir, "activation_time_map_ca.png"),
        positions_xy,
        t_act_ca,
        "Activation time map (intracellular Ca)",
    )
    save_activation_map(
        os.path.join(out_dir, "activation_time_map_h2o2_intra.png"),
        positions_xy,
        t_act_h,
        "Activation time map (intracellular H2O2)",
    )

    save_speed_fit_plot(
        os.path.join(out_dir, "speed_fit_ca.png"),
        fit_ca,
        "Wave speed fit (intracellular Ca)",
    )
    save_speed_fit_plot(
        os.path.join(out_dir, "speed_fit_h2o2_intra.png"),
        fit_h,
        "Wave speed fit (intracellular H2O2)",
    )

    save_recruitment_plot(
        os.path.join(out_dir, "recruitment_curve_ca.png"),
        time_values,
        rec_ca,
        "Recruitment curve (intracellular Ca)",
    )
    save_recruitment_plot(
        os.path.join(out_dir, "recruitment_curve_h2o2_intra.png"),
        time_values,
        rec_h,
        "Recruitment curve (intracellular H2O2)",
    )

    save_bolus_neighbor_traces(
        os.path.join(out_dir, "bolus_neighbor_traces.png"),
        time_values,
        h_values,
        ca_values,
        selected,
        id_info,
    )

    bolus_json = {
        "bolus_mode_requested": args.bolus,
        "bolus_mode_resolved": bolus_mode_resolved,
        "selected": {},
    }

    for role, (idx, dist) in selected.items():
        bolus_json["selected"][role] = {
            "id": {
                "id_str": str(id_info[idx]["id_str"]),
                "index_int": int(id_info[idx]["index_int"]),
            },
            "distance": float(dist),
            "activation_time_ca": t_act_ca[idx],
            "activation_time_h2o2_intra": t_act_h[idx],
        }

    with open(os.path.join(out_dir, "bolus_neighbor_traces.json"), "w", encoding="utf-8") as handle:
        json.dump(bolus_json, handle, indent=2)

    r2_ca = to_json_float(fit_ca["r2"])
    classification = classify_run(float(f_ca), r2_ca)

    resolved_method = method_ca if method_ca == method_h else f"mixed:{method_ca}|{method_h}"

    summary = {
        "run_dir": run_dir,
        "out_dir": out_dir,
        "time_axis": time_axis_mode,
        "n_cells": int(n_cells),
        "n_timepoints": int(n_time),
        "thresholds": {
            "requested": args.threshold,
            "resolved_method": resolved_method,
            "resolved_method_ca": method_ca,
            "resolved_method_h2o2_intra": method_h,
            "theta_Ca": float(theta_ca),
            "theta_H": float(theta_h),
        },
        "ca": {
            "f": float(f_ca),
            "v": to_json_float(fit_ca["v"]),
            "b": to_json_float(fit_ca["b"]),
            "R2": to_json_float(fit_ca["r2"]),
            "n_points": int(fit_ca["n_points"]),
            "t50": t50_ca,
            "t90": t90_ca,
        },
        "h2o2_intra": {
            "f": float(f_h),
            "v": to_json_float(fit_h["v"]),
            "b": to_json_float(fit_h["b"]),
            "R2": to_json_float(fit_h["r2"]),
            "n_points": int(fit_h["n_points"]),
            "t50": t50_h,
            "t90": t90_h,
        },
        "lag_stats": lag_stats,
        "classification": classification,
        "bolus": bolus_json,
        "schema": {
            "position_source": pos_source,
            "state_source": source_summary(state_sources_all),
        },
    }

    with open(os.path.join(out_dir, "wave_metrics_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    summary = run_analysis(args)

    ca = summary["ca"]
    h = summary["h2o2_intra"]

    print("Wave metrics analysis complete")
    print(f"run_dir: {summary['run_dir']}")
    print(f"out_dir: {summary['out_dir']}")
    print(f"n_cells={summary['n_cells']}, n_timepoints={summary['n_timepoints']}, time_axis={summary['time_axis']}")
    print(
        "Ca: "
        f"f={ca['f']:.3f}, v={ca['v']}, R2={ca['R2']}, t50={ca['t50']}, t90={ca['t90']}"
    )
    print(
        "H2O2 intra: "
        f"f={h['f']:.3f}, v={h['v']}, R2={h['R2']}, t50={h['t50']}, t90={h['t90']}"
    )
    print(f"classification: {summary['classification']}")
    print(f"summary_json: {os.path.join(summary['out_dir'], 'wave_metrics_summary.json')}")


if __name__ == "__main__":
    main()
