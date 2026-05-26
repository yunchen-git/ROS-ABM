"""Compare scalar and vectorized Hill update costs."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import FBA_ode
from species_registry import ODE_STATE_CA_I, ODE_STATE_H2O2I

RELATIVE_ERROR_FLOOR = 1e-12


def _make_states(cell_count, seed):
    rng = np.random.default_rng(seed)
    base = FBA_ode.get_x0()
    states = np.vstack([base.copy() for _ in range(cell_count)])
    states[:, ODE_STATE_H2O2I] = rng.uniform(0.0, 0.5, size=cell_count)
    states[:, ODE_STATE_CA_I] = rng.uniform(0.0, 0.5, size=cell_count)
    return states


def _scalar_step(states):
    k = FBA_ode.get_k()
    base = FBA_ode.get_x0()
    out = np.empty_like(states)
    for idx, row in enumerate(states):
        solution = solve_ivp(
            lambda t, y: FBA_ode.model_hill(y, t, k, base),
            (0, 1),
            row,
            method="Radau",
            t_eval=[0, 1],
            atol=1e-10,
            rtol=1e-8,
        )
        out[idx] = solution.y[:, -1]
    return out


def _scalar_steps(states, steps):
    out = states.copy()
    for _ in range(steps):
        out = _scalar_step(out)
    return out


def _vectorized_steps(states, steps, substeps):
    out = states.copy()
    for _ in range(steps):
        out = FBA_ode.hill_step_vectorized(out, 1.0, substeps=substeps)
    return out


def _error_summary(actual, reference, state_index):
    abs_error = np.abs(actual[:, state_index] - reference[:, state_index])
    denom = np.maximum(np.abs(reference[:, state_index]), RELATIVE_ERROR_FLOOR)
    rel_error = abs_error / denom
    return {
        "max_abs_error": float(np.max(abs_error)),
        "max_rel_error": float(np.max(rel_error)),
        "reference_min": float(np.min(reference[:, state_index])),
        "reference_max": float(np.max(reference[:, state_index])),
    }


def _print_species_summary(name, summary):
    print(f"{name} max_abs_error: {summary['max_abs_error']:.6g}")
    print(f"{name} max_rel_error: {summary['max_rel_error']:.6g}")
    print(f"{name} reference_min: {summary['reference_min']:.6g}")
    print(f"{name} reference_max: {summary['reference_max']:.6g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--substeps", type=int, default=FBA_ode.HILL_RK4_SUBSTEPS)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    states = _make_states(args.cells, args.seed)

    t0 = time.perf_counter()
    scalar = _scalar_steps(states, args.steps)
    scalar_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    vectorized = _vectorized_steps(states, args.steps, args.substeps)
    vectorized_seconds = time.perf_counter() - t0

    h2o2i = _error_summary(vectorized, scalar, ODE_STATE_H2O2I)
    ca_i = _error_summary(vectorized, scalar, ODE_STATE_CA_I)
    speedup = scalar_seconds / vectorized_seconds if vectorized_seconds else float("inf")

    print(f"cells: {args.cells}")
    print(f"steps: {args.steps}")
    print(f"substeps: {args.substeps}")
    print(f"scalar_time: {scalar_seconds:.6f}")
    print(f"vectorized_time: {vectorized_seconds:.6f}")
    print(f"speedup: {speedup:.2f}x")
    _print_species_summary("H2O2i", h2o2i)
    _print_species_summary("Ca_i", ca_i)


if __name__ == "__main__":
    main()
