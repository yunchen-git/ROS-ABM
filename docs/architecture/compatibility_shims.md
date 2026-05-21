# Compatibility Shims

This document tracks intentional root-level compatibility shims that preserve
legacy imports and script commands while canonical implementations move under
`ros_abm/`.

These shims are temporary by design and should remain until an explicit
deprecation/removal policy is approved and downstream notebooks/scripts are
updated.

| Legacy root name | Canonical module | Reason kept | Removal status | Tests protecting it |
| --- | --- | --- | --- | --- |
| `analysis_wave_metrics.py` | `ros_abm.analysis.wave_metrics` | Preserve legacy imports and `python analysis_wave_metrics.py` entrypoint | Keep (active compatibility shim) | `tests/test_analysis_wave_metrics_shim.py` |
| `FBA_ode_moles.py` | `ros_abm.backends.legacy.ode.fba_ode_moles` | Preserve legacy module import path used by older scripts/notebooks | Keep (active compatibility shim) | `tests/test_fba_ode_moles_contract.py` |
| `CountDirectory.py` | `ros_abm.backends.legacy.runtime.count_directory` | Preserve `from CountDirectory import newDirect` in launcher/runtime-adjacent code | Keep (active compatibility shim) | `tests/test_count_directory.py`, `tests/test_count_directory_shim.py` |
| `read_gpickle copy.py` | `read_gpickle.py` (tracking helpers re-export) | Preserve legacy script filename and tracking helper compatibility | Keep (legacy compatibility surface; migration pending) | No dedicated shim tests yet |

## Notes

- Compatibility duplication here is intentional, not accidental.
- Runtime postprocess hooks still target root names:
  - `read_gpickle.read_network(...)`
  - `gradientImage.getGradientImages(...)`
- Do not remove shim files until compatibility tests and deprecation plans are in place.
