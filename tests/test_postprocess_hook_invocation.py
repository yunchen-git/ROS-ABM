def test_postprocess_hooks_are_invoked_after_run(simulation_module, monkeypatch, tmp_path):
    calls = []

    def fake_read_network(path, t0, tf):
        calls.append(("read_network", path, t0, tf))

    def fake_get_gradient_images(path):
        calls.append(("getGradientImages", path))

    monkeypatch.setattr(simulation_module, "read_network", fake_read_network)
    monkeypatch.setattr(simulation_module, "getGradientImages", fake_get_gradient_images)

    monkeypatch.setattr(simulation_module.Simulation, "_set_gradient_inital_conditions", lambda self: None)
    monkeypatch.setattr(simulation_module.Simulation, "update", lambda self: None)
    monkeypatch.setattr(simulation_module.Simulation, "update_object_queue", lambda self: None)
    monkeypatch.setattr(simulation_module.Simulation, "collide", lambda self, x, y, z: None)
    monkeypatch.setattr(simulation_module.Simulation, "optimize", lambda self, *args, **kwargs: None)

    sim = simulation_module.Simulation(
        name="1.0",
        path=str(tmp_path),
        start_time=0,
        end_time=0,
        time_step=1,
        grid_size=[10, 10, 0],
        inhib_type=None,
    )

    sim.run(10, 10, 0)

    run_path = str(tmp_path / "1.0") + sim._sep
    assert calls == [
        ("read_network", run_path, 0, 0),
        ("getGradientImages", run_path),
    ]
