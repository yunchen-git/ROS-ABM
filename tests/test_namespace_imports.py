import importlib


def test_import_legacy_backend_namespace():
    module = importlib.import_module("ros_abm.backends.legacy")
    assert module.__name__ == "ros_abm.backends.legacy"


def test_import_jax_exp_backend_namespace():
    module = importlib.import_module("ros_abm.backends.jax_exp")
    assert module.__name__ == "ros_abm.backends.jax_exp"
