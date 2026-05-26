"""Compare scalar and vectorized Hill one-step update costs."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    states = _make_states(args.cells, args.seed)

    t0 = time.perf_counter()
    scalar = _scalar_step(states)
    scalar_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    vectorized = FBA_ode.hill_step_vectorized(states, 1.0)
    vectorized_seconds = time.perf_counter() - t0

    err = np.max(
        np.abs(
            scalar[:, [ODE_STATE_H2O2I, ODE_STATE_CA_I]]
            - vectorized[:, [ODE_STATE_H2O2I, ODE_STATE_CA_I]]
        )
    )
    speedup = scalar_seconds / vectorized_seconds if vectorized_seconds else float("inf")

    print(f"cells: {args.cells}")
    print(f"scalar_radau_seconds: {scalar_seconds:.6f}")
    print(f"vectorized_rk4_seconds: {vectorized_seconds:.6f}")
    print(f"speedup: {speedup:.2f}x")
    print(f"max_abs_h2o2i_ca_difference: {err:.6g}")


if __name__ == "__main__":
    main()
