# Hill2 Parameter Contract (Experimental JAX Scope)

This document defines the current parameter contract for the experimental Hill2
RHS used in `ros_abm.backends.jax_exp.ode.rhs.model_hill2_jax(...)`.

Scope boundary:
- This contract applies only to the reduced 2-state H2O2/Ca motif.
- It does not validate or replace the full ROS-ABM runtime chemistry path.
- It does not alter legacy runtime defaults or solver behavior.

Legacy/runtime context:
- The legacy runtime remains the reference backend.
- Hill2 is an opt-in validation/debug model used for reduced ODE parity and
  numerical-behavior checks.

## Hill2 Equations

Given state `[h2o2i, cai]`:

- `dh/dt = a0 + a1 * hill(cai; k_c, n) - b_h * h2o2i`
- `dc/dt = c0 + c1 * hill(h2o2i; k_h, m) - b_c * cai`

with:
- `hill(y; K, p) = y^p / (K^p + y^p + eps)` after nonnegative clamping inside
  the gate function.

## Parameter Table

| Name | Default | Inferred role | Unit/status |
|---|---:|---|---|
| `a0` | `0.0` | Basal H2O2i source term | Same unit as `dh/dt`; effective units unspecified in reduced motif |
| `a1` | `1.0` | Max Cai-driven activation contribution to H2O2i production | Same unit as `dh/dt`; proxy-scale parameter unless calibrated |
| `k_c` | `0.5` | Cai half-activation scale for H2O2i gate | Same unit as Cai state if dimensional; otherwise proxy/normalized |
| `n` | `2.0` | Cai-to-H2O2i Hill exponent (cooperativity) | Dimensionless |
| `b_h` | `0.2` | First-order H2O2i loss/clearance rate | 1/time if state is dimensional; otherwise effective rate |
| `c0` | `0.0` | Basal Cai source term | Same unit as `dc/dt`; effective units unspecified in reduced motif |
| `c1` | `1.0` | Max H2O2i-driven activation contribution to Cai production | Same unit as `dc/dt`; proxy-scale parameter unless calibrated |
| `k_h` | `0.5` | H2O2i half-activation scale for Cai gate | Same unit as H2O2i state if dimensional; otherwise proxy/normalized |
| `m` | `2.0` | H2O2i-to-Cai Hill exponent (cooperativity) | Dimensionless |
| `b_c` | `0.2` | First-order Cai loss/clearance rate | 1/time if state is dimensional; otherwise effective rate |
| `eps` | `1e-12` | Numerical guard against divide-by-zero | Numerical stabilizer (not biological) |

## Dimensionality and Calibration Status

- Hill2 is a reduced validation motif and currently acts as a proxy model.
- Parameter names and mathematical roles are explicit in code.
- Absolute biological units are not fully specified for this reduced motif in
  current docs.
- Defaults should be treated as validation/proxy parameters unless calibrated
  against experimental data or a validated legacy-equivalence target.

## Recommended Sweep Ranges (Validation-Oriented)

Include the default point in every sweep and explore nearby ranges:

- `a0`, `c0`: `[0.0, 1e-6, 1e-4, 1e-2]`
- `a1`, `c1`: `[0.1, 0.3, 1.0, 3.0]`
- `k_c`, `k_h`: `[1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 0.5, 1.0]`
- `n`, `m`: `[1.0, 2.0, 3.0, 4.0, 6.0]`
- `b_h`, `b_c`: `[0.02, 0.05, 0.1, 0.2, 0.5, 1.0]`

These ranges are intended for numerical-behavior validation (finite trajectories,
conservative-step nonnegativity, and `dt` convergence), not biological claim
generation.

## Validation Scope Warning

Passing Hill2 validation tests means:
- the reduced two-state motif behaves numerically as expected under tested
  conditions.

Passing Hill2 validation tests does not mean:
- full 32-state ROS-ABM chemistry is validated,
- legacy runtime postprocess/output behavior is validated,
- or scientific calibration is complete.
