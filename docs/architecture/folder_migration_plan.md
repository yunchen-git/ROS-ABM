# Folder Migration Plan (Behavior-Preserving)

## Status and scope

This document defines a staged packaging/migration plan without changing current runtime behavior.

Current runtime remains rooted in legacy top-level modules (`Model_run.py`, `Model_initialParameters.py`, `Simulation.py`, `FBA_ode.py`, `IntC_diff.pyx`, `ExtC_gradient.pyx`).

## Backend policy

- The legacy backend is the reference and default backend.
- Any future `jax_exp` backend is experimental and opt-in.
- No current script command, root import, or runtime wiring changes are implied by this scaffolding step.

## Stage outline

1. Stage 0 (current): namespace-only scaffolding under `ros_abm/`.
2. Stage 1: add compatibility import tests and migration docs (no file moves).
3. Stage 2: introduce additive compatibility shims before any runtime file moves.
4. Stage 3: move modules only behind validated shims and guardrail tests.

## Guardrails for future stages

- Preserve solver semantics and solver wiring.
- Preserve output layout and naming.
- Preserve perturbation trigger and special-cell semantics.
- Preserve postprocess hook invocation behavior at run completion.
- Preserve legacy root imports and commands until explicit deprecation approval.
