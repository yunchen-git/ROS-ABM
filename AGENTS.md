# AGENTS Guardrails

This repository is maintained as a legacy scientific simulation sandbox. Unless explicitly approved otherwise, all changes must preserve current runtime behavior.

## Hard constraints

- Preserve solver semantics.
  - Do not alter ODE RHS behavior, solver call structure, or solver tolerance settings.
- Preserve output layout.
  - Keep filenames, folder names, and saved artifact layout stable.
- Preserve special-cell perturbation semantics.
  - Keep current special-cell logic and trigger conditions intact.
- Preserve current postprocess behavior.
  - Keep automatic postprocessing hook invocation at run completion unchanged.

## Prohibited without explicit approval

- Moving or renaming existing files.
- Changing imports in runtime modules.
- Changing output paths or output naming conventions.
- Changing solver tolerances or integration method wiring.
- Changing perturbation logic or special-cell index semantics.
- Disabling, reordering, or removing postprocessing behavior currently executed by `Simulation.run(...)`.

## Scope guidance

- Prefer additive guardrails: tests, docs, and comments that describe existing behavior.
- Treat legacy assumptions as runtime contracts until they are replaced by an approved migration plan.
