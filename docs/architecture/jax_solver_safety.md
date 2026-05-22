# JAX Solver Safety Notes

This document describes the safety and interpretation boundaries for the opt-in JAX experimental solver.

## Runtime Contract

- The legacy runtime path remains authoritative and uses SciPy `solve_ivp` with `method="Radau"` in `Simulation.py`.
- The JAX solver in `ros_abm.backends.jax_exp.ode.solver` is a fixed-step explicit RK4 experiment.
- The JAX solver is not wired into `Simulation.py` and is not selected automatically at runtime.

## Intended Use

- JAX RK4 is currently intended for parity/debug workflows and controlled experiments.
- It is useful for checking RHS parity and deterministic stepping behavior under chosen fixed `dt`.
- It is not yet a validated replacement for the legacy adaptive stiff solver path.

## Known Limitation: Positivity

- In full-mode 32-state dynamics, explicit RK4 can produce negative concentrations for some `dt` values.
- This is a numerical integration artifact from fixed-step explicit stepping on a stiff system, not necessarily a model-definition change.
- Positivity clipping/projection is not enabled by default.
  - Rationale: clipping/projection changes the dynamics and breaks strict parity with the unclipped equations.

## Scientific-Use Preconditions

Before scientific interpretation of JAX solver results, require all of:

- `dt` convergence checks over the target time horizon.
- Comparison against the legacy Radau trajectory under matched initial conditions and parameters.
- Documented nonnegativity/finite-state behavior across the chosen parameter and initial-condition ranges.
- Explicit reporting of solver settings (`dt`, steps, horizon, dtype/precision).

## Practical Guidance

- Use conservative `dt` values and short horizons first.
- Treat large-`dt` trajectories as diagnostics unless validated by convergence and Radau comparison.
- Keep legacy runtime output as the reference baseline for scientific claims until parity is established beyond debug scope.
