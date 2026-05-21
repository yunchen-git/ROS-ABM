from pathlib import Path


class _DummySnapshotWriter:
    def __init__(self, name):
        self.name = name

    def save(self, base_path, time_stamp):
        out_path = Path(base_path) / f"{self.name}_{time_stamp}"
        out_path.write_bytes(b"")
        return str(out_path)


def test_output_layout_contract(simulation_module, tmp_path):
    sim = simulation_module.Simulation(
        name="1.0",
        path=str(tmp_path),
        start_time=0,
        end_time=0,
        time_step=1,
        grid_size=[10, 10, 0],
        inhib_type=None,
    )

    run_dir = tmp_path / "1.0"
    run_dir.mkdir()

    sim.gradients = [_DummySnapshotWriter("H2O2e")]
    sim.interC = [
        _DummySnapshotWriter("cAMP"),
        _DummySnapshotWriter("H2O2i"),
        _DummySnapshotWriter("Ca"),
    ]

    sim._create_header_file()
    sim.time = 0.0
    sim.save()
    sim._header.close()

    assert (run_dir / "info.sim").exists()
    assert (run_dir / "network0.0.gpickle").exists()
    assert (run_dir / "H2O2e_0.0").exists()
    assert (run_dir / "cAMP_0.0").exists()
    assert (run_dir / "H2O2i_0.0").exists()
    assert (run_dir / "Ca_0.0").exists()

    header = (run_dir / "info.sim").read_text(encoding="utf-8")
    assert "Gradient Species:H2O2e," in header
    assert "InterC Species:cAMP,H2O2i,Ca," in header
    assert "network0.0.gpickle" in header
