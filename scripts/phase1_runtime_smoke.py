"""Tiny real-runtime smoke test for the Phase-1 config path."""

from pathlib import Path
import tempfile
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_config import RunConfig
from species_registry import (
    EXT_GRADIENT_H2O2E,
    INTERC_SPECIES_CAMP,
    INTERC_SPECIES_CA,
    INTERC_SPECIES_H2O2I,
)
import Model_initialParameters


def _write_cell_file(path, count=878):
    cols = 31
    spacing = 18.0
    origin = 50.0
    with open(path, "w", encoding="utf-8") as f:
        for idx in range(count):
            row = idx // cols
            col = idx % cols
            x = origin + col * spacing
            y = origin + row * spacing
            f.write(f"{idx},{x},{y}\n")


def _numeric_dirs(path):
    numeric = []
    for child in path.iterdir():
        if not child.is_dir():
            continue
        try:
            float(child.name)
        except ValueError:
            continue
        numeric.append(child)
    return numeric


def main():
    output_base = Path(tempfile.mkdtemp(prefix="ros_abm_phase1_smoke_"))
    cell_path = output_base / "tiny_cells.csv"
    _write_cell_file(cell_path)

    cfg = RunConfig(
        output_base_path=str(output_base),
        length=1,
        seed=123,
        cell_path=str(cell_path),
        run_postprocess=False,
    )
    cfg.model.te = 1
    cfg.model.dt = 1

    Model_initialParameters.main_config(cfg)

    run_dirs = _numeric_dirs(output_base)
    assert run_dirs, f"no numeric run folder created in {output_base}"
    run_dir = run_dirs[0]

    assert (run_dir / "info.sim").exists()
    assert (run_dir / "run_config.json").exists()
    assert list(run_dir.glob("network*.gpickle"))
    assert list(run_dir.glob(f"{EXT_GRADIENT_H2O2E}_*"))
    assert list(run_dir.glob(f"{INTERC_SPECIES_CAMP}_*"))
    assert list(run_dir.glob(f"{INTERC_SPECIES_H2O2I}_*"))
    assert list(run_dir.glob(f"{INTERC_SPECIES_CA}_*"))

    print(f"real runtime smoke OK: {run_dir}")


if __name__ == "__main__":
    main()
