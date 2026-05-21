# Current Execution Pipeline

This document captures the current runtime flow of the ROS-ABM legacy sandbox as implemented today.

## Entry-to-output flow

1. `Model_run.py` is the launcher entry point.
2. For each requested run, it allocates a numeric run id using `CountDirectory.newDirect(...)`.
3. It calls `Model_initialParameters.main(...)` with run id, base path, inhibition controls, and `chem_mode`.
4. `Model_initialParameters.main(...)`:
   - sets global chemistry mode in `FBA_ode`
   - builds `Simulation(...)`
   - registers intercellular and extracellular transport modules
   - loads initial cell coordinates (`new_populated.txt` from current working directory)
   - seeds the initial cell population and perturbation setup
   - calls `sim.run(...)`
5. `Simulation.run(...)`:
   - creates run directory `<base_path>/<run_id>/`
   - creates `info.sim`
   - loops over time:
     - `update()`
     - `update_object_queue()`
     - `collide(...)` (or fallback `collide_lowDens()`)
     - `optimize(...)`
     - `save()`
6. `Simulation.save()` writes:
   - `network<time>.gpickle`
   - one file per extracellular gradient (`<gradient_name>_<time>`)
   - one file per intercellular species (`<interC_name>_<time>`)
   - one index line into `info.sim`
7. After simulation completion, `Simulation.run(...)` triggers postprocessing:
   - `read_gpickle.read_network(...)`
   - `gradientImage.getGradientImages(...)`

## Pipeline sketch

`Model_run.py`  
-> `CountDirectory.newDirect(...)`  
-> `Model_initialParameters.main(...)`  
-> `Simulation.run(...)`  
-> `Simulation.update(...)` + `FBA_ode.model_noref(...)` + `IntC_Diff` / `ExtC_Diff` updates  
-> `Simulation.save(...)` snapshots  
-> postprocess hooks (`read_network`, `getGradientImages`)

## Notes

- This is a behavior snapshot, not a redesign plan.
- File names and folder layout above are treated as active contracts for downstream scripts.
