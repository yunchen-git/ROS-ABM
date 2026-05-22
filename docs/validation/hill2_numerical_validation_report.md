# Hill2 Numerical Validation Report

Date: 2026-05-22
Scope: Experimental JAX Hill2 reduced motif only (`h2o2i`, `cai`), not full ROS-ABM runtime.

## Summary

This checkpoint validates numerical behavior for the reduced Hill2 JAX pathway:
- RHS parity is exact on tested cases.
- Schema/adapters preserve index mapping and sparsity.
- Fixed-step RK4 agrees tightly with Radau at default parameters.
- Small parameter sweeps remain finite/nonnegative under conservative `dt` in tested ranges.
- A numerically sensitive low-gain/low-k regime converges with `dt` halving but is less tight than default.

This does **not** constitute biological calibration and does **not** validate the full legacy ROS-ABM runtime.

## 1. RHS Parity

Validated comparisons:
- `FBA_ode.model_hill(...)` vs `ros_abm.backends.jax_exp.ode.rhs.model_hill_jax(...)`
- `model_hill2_jax(...)` lifted back to legacy shape vs legacy-shaped JAX Hill RHS

Observed max absolute errors (representative test cases: `(1e-9,1e-7)`, `(2.5e-6,8e-6)`, `(0,0)`):
- Legacy Hill vs JAX Hill (legacy-shaped): `0.0`
- Hill2 lifted to legacy-shape vs JAX legacy-shaped Hill: `0.0`

## 2. Schema / Adapters

Validated surface:
- Pack: legacy `(..., 32)` -> Hill2 `(..., 2)`
- Unpack derivative: Hill2 `(..., 2)` -> sparse legacy `(..., 32)`
- Target indices: `h2o2i -> x[1]`, `cai -> x[29]`

Observed sparsity check:
- Max absolute value outside indices `1` and `29`: `0.0`

## 3. RK4 vs Radau Validation (Hill2 Only)

Reference:
- `scipy.solve_ivp(method="Radau", atol=1e-12, rtol=1e-10)`

Default-parameter short-horizon agreement (`dt=0.1`, horizon `1.0`):
- Max trajectory absolute error across representative ICs: `2.7011097351870017e-13`

Default-parameter dt-halving final-state errors (horizon `5.0`, IC `[1e-3,1e-3]`):
- `dt=0.1`: `4.700468530030588e-13`
- `dt=0.05`: `2.900604019126507e-14`
- `dt=0.025`: `1.6729781622537576e-15`

Interpretation:
- Default regime shows very strong RK4/Radau agreement in tested settings.

## 4. Parameter Sweep Coverage

Validated sweep grid (small guardrail grid):
- `a1=c1 in {0.3, 1.0}`
- `k_c=k_h in {1e-7, 0.5}`
- `b_h=b_c in {0.1, 0.2}`
- Fixed per sweep case: `a0=c0=0`, `n=m=2`
- Total cases: `8`

Checks performed:
- Finite trajectory checks (`dt=0.05`, horizon `2.0`) across representative ICs
- Nonnegativity checks (`dt=0.01`, horizon `5.0`) for nonnegative-source parameter cases
- Explicit inclusion check for exact default parameter point

Results:
- Sweep includes default parameter point: `True`
- All finite checks passed: `True`
- All nonnegativity checks passed: `True`
- Minimum state observed over nonnegativity sweep: `3.678933938209679e-10`

## 5. Positivity / Stability Findings

Hill2 findings (tested conservative `dt` and ranges):
- Finite trajectories observed across tested ICs and sweep cases
- Nonnegative trajectories observed in tested conservative settings

Full-mode safety finding (outside Hill2 scope but relevant to solver safety):
- Full 32-state RK4 one-step diagnostic at `dt=0.005` yields negative `H2O2i`
- Observed value: `-3.513478534366901e-07`

Interpretation:
- This supports existing safety documentation: fixed-step explicit RK4 is a debug/experimental path, not a drop-in replacement for legacy Radau in stiff/full dynamics.

## 6. Limitations

- Hill2-only validation of reduced H2O2/Ca motif.
- Not a validation of full ROS-ABM chemistry/runtime path.
- No `Simulation.py` backend wiring validated here.
- No biological calibration claims.
- Parameter units remain proxy/contract-level unless explicitly calibrated.

## Executed Test Evidence

Command:

```bash
pytest -q -rs \
  tests/test_jax_hill_parity.py \
  tests/test_jax_full_solver.py \
  tests/test_jax_solver_safety.py \
  tests/test_jax_hill2_numerical_validation.py \
  tests/test_jax_hill2_parameter_sweeps.py
```

Result:
- `29 passed, 1 skipped`
- Skipped: optional fail-fast diagnostic test when API flag is not implemented
