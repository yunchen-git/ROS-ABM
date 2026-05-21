# ROS-ABM

ROS-ABM is an agent-based model (ABM) for intracellular/extracellular redox and calcium dynamics in a cell population.

This README is the canonical documentation and supersedes legacy documentation files previously included in this repository.

## Version Milestones

- `v1.1-chem-toggle`: introduced runtime chemistry mode toggle (`full` vs `hill`).
- `v1.2-cleanup`: applied redundancy cleanup and hygiene pass.

## What This Repository Contains

This repository contains:
- A simulation engine that evolves cell positions, neighbor topology, and chemical state over time.
- Cython-accelerated diffusion/transport modules for extracellular fields and intercellular exchange.
- A biochemical ODE module with two runtime chemistry modes:
  - `full` (default): full pathway model.
  - `hill`: reduced Hill-minimal H2O2/Ca motif.
- Post-processing scripts that render saved network and gradient outputs.

## Execution Pipeline

Main run path:

`Model_run.py` -> `Model_initialParameters.main(...)` -> `Simulation.run(...)` -> `FBA_ode` RHS + Cython diffusion modules

Key runtime modules:
- `Simulation.py`: ABM loop, save/load, network updates.
- `simulationObjects.py`: cell agent definitions (`StemCell`).
- `FBA_ode.py`: intracellular ODEs and chemistry mode dispatch.
- `IntC_diff.pyx`: intercellular transport.
- `ExtC_gradient.pyx`: extracellular diffusion field.

## Repository Layout

Core files:
- `Model_run.py`: main launcher script.
- `Model_initialParameters.py`: model wiring, initial conditions, module registration.
- `Simulation.py`: simulation loop and output writing.
- `simulationObjects.py`: agent classes.
- `FBA_ode.py`: chemistry models (`full` and `hill`).
- `IntC_diff.pyx`, `ExtC_gradient.pyx`: Cython extensions.
- `setup.py`: Cython build entry point.
- `new_populated.txt`: initial cell coordinate file used at startup.

Analysis/visualization scripts:
- `Analysis.py`
- `gradientImage.py`
- `read_gpickle.py`
- `read_gpickle copy.py`
- `zplot_intC_extc.py`

## Developer Index

1. `README.md`: canonical project documentation (setup, run, outputs, and conventions).
2. `Model_run.py`: top-level run launcher; defines run configuration and calls `Model_initialParameters.main(...)`.
3. `Model_initialParameters.py`: simulation builder/wiring; key entry point is `main(...)`.
4. `Simulation.py`: runtime engine and loader; key classes are `Simulation` and `TimeSeries`.
5. `simulationObjects.py`: agent definitions; key classes are `SimulationObject` and `StemCell`.
6. `simulationMath1.py`: math/geometry utilities; key functions include `RandomPointOnSphere`, `AddVec`, `Mag`, and `Distance`.
7. `FBA_ode.py`: intracellular chemistry module; key APIs are `get_k`, `get_x0`, `model`, `model_full`, `model_hill`, and `set_chem_mode`.
8. `FBA_ode_moles.py`: alternate mol-unit intracellular model; key APIs are `get_k`, `get_x0`, and `model`.
9. `IntC_diff.pyx`: Cython intercellular diffusion/transport module; main class `IntC_Diff` with `update_C0`, `update_C1`, `update_C2`, and `save`.
10. `ExtC_gradient.pyx`: Cython extracellular diffusion field module; main class `ExtC_Diff` with `update`, interpolation helpers, and `save`.
11. `setup.py`: Cython build script; compiles `IntC_diff.pyx` and `ExtC_gradient.pyx` via `python setup.py build_ext --inplace`.
12. `CountDirectory.py`: utility for run folder numbering; key function `newDirect(path)`.
13. `Analysis.py`: gradient post-processing helpers; key functions `save_2D_gradient_images(...)` and `getGradientImages(...)`.
14. `gradientImage.py`: thin wrapper around analysis image generation; key function `getGradientImages(path)`.
15. `read_gpickle.py`: baseline network snapshot rendering; key functions `read_network(...)`, `draw_species_image(...)`, and `draw_state_image(...)`.
16. `read_gpickle copy.py`: extended network analysis/tracking workflow; adds cell-selection and trajectory plotting helpers.
17. `zplot_intC_extc.py`: advanced combined intracellular/extracellular visualization; key entry `getCombinedH2O2Images3D(...)`.
18. `new_populated.txt`: initial cell coordinate data loaded during model initialization.

## Requirements

Recommended:
- Python 3.10+ in a virtual environment.
- A working C/C++ toolchain for Cython extension builds.

Python packages used by this codebase:
- Core runtime/build: `numpy`, `scipy`, `networkx`, `cython`, `setuptools`, `wheel`
- Visualization/analysis scripts: `matplotlib`, `opencv-python`, `vpython`, `pandas`, `torch`, `pillow`
- Canonical install surfaces: `requirements.txt` and `requirements-dev.txt`

Note:
- Postprocess hooks are still invoked automatically at run completion, but visualization dependencies are loaded lazily when those hooks execute (not during `Simulation.py` import).
- If visualization dependencies are missing, failures are expected when postprocess hooks run.
- See `docs/data_dependencies.md` for startup and external dataset requirements.

## Setup (Recommended Isolated Workflow)

Run from repository root:

```bash
cd /path/to/ROS-ABM

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy scipy networkx cython matplotlib opencv-python vpython pandas torch pillow
```

Build Cython extensions:

```bash
python setup.py build_ext --inplace
```

## Running Simulations

### Default run (full chemistry mode)

`Model_run.py` defaults to:
- `chem_mode = "full"`
- output base path: `~/Documents/PhD/test`

Run:

```bash
python Model_run.py
```

To keep all outputs inside this repo instead of your real home directory:

```bash
HOME="$PWD" python Model_run.py
```

With the `HOME="$PWD"` override, outputs are written under:
- `./Documents/PhD/test/<run_id>/`

### Hill-minimal mode

In `Model_run.py`, set:

```python
chem_mode = "hill"
```

Then run again:

```bash
python Model_run.py
```

Behavior in hill mode:
- Only these state vector entries are updated by the reduced model:
  - `x[1]`: intracellular H2O2 proxy
  - `x[29]`: intracellular Ca
- Other entries are left unchanged by the hill RHS so output formats remain compatible.

## Chemistry Toggle Details

Implemented in `FBA_ode.py`:
- `CHEM_MODE = "full"` (default)
- `set_chem_mode(mode)` with validation for `"full"` or `"hill"`
- `model_full(...)`: full pathway RHS
- `model_hill(...)`: reduced Hill H2O2/Ca motif
- `model(...)` and `model_noref(...)`: dispatch based on current `CHEM_MODE`

Wiring path:
- `Model_initialParameters.main(..., chem_mode="full")` calls `FBA_ode.set_chem_mode(chem_mode)`.
- `Model_run.py` passes `chem_mode=chem_mode` into `main(...)`.

Quick local sanity check:

```bash
python FBA_ode.py
```

This prints the configured H2O2/Ca indices and one derivative evaluation for both modes.

## Key State Indices

Selected entries in the 32-element state vector (`FBA_ode.get_x0()`):
- `x[0]`: H2O2 in media
- `x[1]`: H2O2 in cytosol
- `x[29]`: intracellular Ca
- `x[30]`: extracellular Ca
- `x[31]`: ER Ca

## Output Files and Format

Each run creates a numeric folder (for example `1.0`, `2.0`) via `CountDirectory.newDirect(...)`.

Inside a run folder:
- `info.sim`: simulation metadata and per-timepoint file index.
- `network<time>.gpickle`: network snapshots.
- `<gradient_name>_<time>`: pickled `ExtC_Diff` objects (for example `H2O2e_10.0`).
- `<interC_name>_<time>`: pickled `IntC_Diff` objects (for example `H2O2i_10.0`, `Ca_10.0`, `cAMP_10.0`).

At run completion, `Simulation.run(...)` also triggers:
- `read_gpickle.read_network(...)` image export.
- `gradientImage.getGradientImages(...)` gradient image export.

So you should also see generated image folders such as:
- `all_images_state/`
- `all_images_<species>/`
- `all_<gradient_name>/`

## Config Knobs You Will Most Likely Edit

In `Model_run.py`:
- `path`: top-level save directory.
- `chem_mode`: `"full"` or `"hill"`.
- inhibition settings (`inhib_type`, values, timing).
- `magnitudes` loop.

In `Model_initialParameters.py`:
- `ts`, `te`, `dt` (time range and step).
- `x`, `y`, `z` and grid spacing.
- diffusion constants and production/degradation parameters.
- initial seed behavior and selected perturbed cell index (`877` in current script).

## Important Runtime Notes

- Run from repository root. `Model_initialParameters.py` loads `new_populated.txt` from the current working directory (`os.getcwd()`).
- Current setup references cell index `877` for a localized perturbation. Ensure `new_populated.txt` has enough rows or adjust this index.
- Do not change on-disk output naming/layout if downstream scripts or notebooks depend on current paths.
- See `docs/data_dependencies.md` for required startup data and optional external dataset wiring.

## Minimal Smoke Test

After environment setup and Cython build:

```bash
HOME="$PWD" python Model_run.py
```

Expected checks:
- A new run directory is created under `./Documents/PhD/test/`.
- `network*.gpickle` files are written.
- Gradient/intercellular snapshot files are written.
- Post-processing image folders are generated.

## Analysis Script Entry Points

- Network/species/state rendering:

```bash
python read_gpickle.py
```

- Gradient image generation helper:

```bash
python gradientImage.py
```

- Combined intracellular/extracellular plotting (advanced):

```bash
python zplot_intC_extc.py
```

Most analysis scripts include hard-coded example paths in `__main__`; update those paths to your run directory before executing.

## Troubleshooting

`ModuleNotFoundError: No module named 'numpy'`
- Install dependencies inside the active virtual environment.

`ImportError` for `IntC_diff` or `ExtC_gradient`
- Rebuild extensions: `python setup.py build_ext --inplace`.

Extensions fail after upgrading NumPy
- Rebuild extensions in-place so binary modules match current NumPy ABI.

`FileNotFoundError` for `new_populated.txt`
- Run from repo root or fix the path construction in `Model_initialParameters.py`.

`ValueError: Invalid chem_mode`
- Use only `full` or `hill`.

Issues reading very old `.gpickle` outputs
- Some legacy snapshots may require older `networkx` for compatibility.

## Development Guidance

- Keep `full` as the default chemistry mode unless deliberately testing alternatives.
- Preserve output file names and folder structure for downstream compatibility.
- Rebuild Cython modules after changes to `.pyx`, NumPy, or compiler toolchain.
