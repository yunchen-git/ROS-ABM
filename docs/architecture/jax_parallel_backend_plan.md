# JAX Parallel Backend Plan (Opt-In Only)

## Purpose

Define how to host an experimental JAX backend in parallel with the legacy engine, without changing the default runtime path.

## Backend policy

- Legacy engine remains the reference and default backend for ROS-ABM.
- `ros_abm.backends.jax_exp` is experimental and opt-in.
- No automatic backend switching is allowed.

## Namespace intent

- `ros_abm.backends.legacy`: place for compatibility-facing legacy backend namespace.
- `ros_abm.backends.jax_exp`: isolated namespace for experimental JAX work.

## Non-goals in this stage

- No JAX kernels.
- No solver replacement.
- No module moves.
- No import rewiring in runtime modules.
- No Cython build changes.

## Compatibility requirements for future implementation

- Keep legacy root imports valid.
- Keep script entry commands unchanged unless explicitly approved.
- Keep output file layout and postprocess behavior unchanged.
- Maintain pickle/runtime compatibility during any backend experiments.
