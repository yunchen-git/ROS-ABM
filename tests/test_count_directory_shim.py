import importlib


def test_import_count_directory_old_path():
    module = importlib.import_module("CountDirectory")
    assert module.__name__ == "CountDirectory"
    assert hasattr(module, "newDirect")
    assert callable(module.newDirect)


def test_import_count_directory_new_path():
    module = importlib.import_module("ros_abm.backends.legacy.runtime.count_directory")
    assert module.__name__ == "ros_abm.backends.legacy.runtime.count_directory"
    assert hasattr(module, "newDirect")
    assert callable(module.newDirect)


def test_old_and_new_newdirect_are_same_callable_or_behavior_equivalent(tmp_path):
    old_mod = importlib.import_module("CountDirectory")
    new_mod = importlib.import_module("ros_abm.backends.legacy.runtime.count_directory")

    old_fn = old_mod.newDirect
    new_fn = new_mod.newDirect

    if old_fn is new_fn:
        assert True
        return

    assert old_fn(str(tmp_path)) == new_fn(str(tmp_path))

    (tmp_path / "1.0").mkdir()
    (tmp_path / "5.5").mkdir()
    (tmp_path / "notes").mkdir()

    assert old_fn(str(tmp_path)) == new_fn(str(tmp_path))
