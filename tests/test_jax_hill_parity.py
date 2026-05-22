import importlib

import pytest


np = pytest.importorskip("numpy")
pytest.importorskip("jax")

fba_ode = importlib.import_module("FBA_ode")
jax_rhs = importlib.import_module("ros_abm.backends.jax_exp.ode.rhs")
schema = importlib.import_module("ros_abm.backends.jax_exp.ode.schema")


def test_pack_legacy32_to_hill2_single():
    x = np.zeros(schema.LEGACY_STATE_LENGTH, dtype=np.float64)
    x[schema.LEGACY_H2O2I_INDEX] = 1.25e-6
    x[schema.LEGACY_CAI_INDEX] = 3.75e-7

    hill2 = np.asarray(schema.pack_legacy32_to_hill2(x))

    assert hill2.shape == (schema.HILL2_STATE_LENGTH,)
    np.testing.assert_allclose(hill2, np.array([1.25e-6, 3.75e-7]))


def test_pack_legacy32_to_hill2_batch():
    x = np.zeros((3, schema.LEGACY_STATE_LENGTH), dtype=np.float64)
    x[:, schema.LEGACY_H2O2I_INDEX] = np.array([1e-9, 2e-9, 3e-9])
    x[:, schema.LEGACY_CAI_INDEX] = np.array([1e-7, 2e-7, 3e-7])

    hill2 = np.asarray(schema.pack_legacy32_to_hill2(x))

    assert hill2.shape == (3, schema.HILL2_STATE_LENGTH)
    np.testing.assert_allclose(hill2[:, 0], x[:, schema.LEGACY_H2O2I_INDEX])
    np.testing.assert_allclose(hill2[:, 1], x[:, schema.LEGACY_CAI_INDEX])


def test_unpack_hill2_to_legacy32_dx_single_sparsity():
    dx_hill2 = np.array([0.125, -0.25], dtype=np.float64)

    dx_legacy = np.asarray(schema.unpack_hill2_to_legacy32_dx(dx_hill2))

    assert dx_legacy.shape == (schema.LEGACY_STATE_LENGTH,)
    np.testing.assert_allclose(dx_legacy[schema.LEGACY_H2O2I_INDEX], dx_hill2[0])
    np.testing.assert_allclose(dx_legacy[schema.LEGACY_CAI_INDEX], dx_hill2[1])

    mask = np.ones(schema.LEGACY_STATE_LENGTH, dtype=bool)
    mask[[schema.LEGACY_H2O2I_INDEX, schema.LEGACY_CAI_INDEX]] = False
    np.testing.assert_allclose(dx_legacy[mask], 0.0)


def test_unpack_hill2_to_legacy32_dx_batch_shape_and_sparsity():
    dx_hill2 = np.array(
        [
            [0.1, -0.2],
            [0.3, -0.4],
            [0.5, -0.6],
        ],
        dtype=np.float64,
    )

    dx_legacy = np.asarray(schema.unpack_hill2_to_legacy32_dx(dx_hill2))

    assert dx_legacy.shape == (3, schema.LEGACY_STATE_LENGTH)
    np.testing.assert_allclose(dx_legacy[:, schema.LEGACY_H2O2I_INDEX], dx_hill2[:, 0])
    np.testing.assert_allclose(dx_legacy[:, schema.LEGACY_CAI_INDEX], dx_hill2[:, 1])

    mask = np.ones(schema.LEGACY_STATE_LENGTH, dtype=bool)
    mask[[schema.LEGACY_H2O2I_INDEX, schema.LEGACY_CAI_INDEX]] = False
    np.testing.assert_allclose(dx_legacy[:, mask], 0.0)


@pytest.mark.parametrize(
    "h2o2i,cai",
    [
        (1e-9, 1e-7),
        (2.5e-6, 8e-6),
        (0.0, 0.0),
    ],
)
def test_model_hill_jax_matches_legacy_model_hill(h2o2i, cai):
    x_base = fba_ode.get_x0()
    k = fba_ode.get_k()

    x = x_base.copy()
    x[fba_ode.H2O2I_INDEX] = h2o2i
    x[fba_ode.CAI_INDEX] = cai

    legacy = fba_ode.model_hill(x, 0.0, k, x_base)
    experimental = jax_rhs.model_hill_jax(x, 0.0, k, x_base)
    experimental_np = np.asarray(experimental)

    assert experimental_np.shape == legacy.shape
    np.testing.assert_allclose(experimental_np, legacy, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize(
    "h2o2i,cai",
    [
        (1e-9, 1e-7),
        (2.5e-6, 8e-6),
        (0.0, 0.0),
    ],
)
def test_model_hill2_jax_matches_legacy_shaped_jax_rhs(h2o2i, cai):
    x_base = fba_ode.get_x0()
    k = fba_ode.get_k()

    x = x_base.copy()
    x[fba_ode.H2O2I_INDEX] = h2o2i
    x[fba_ode.CAI_INDEX] = cai

    hill2 = schema.pack_legacy32_to_hill2(x)
    dx_hill2 = jax_rhs.model_hill2_jax(hill2)
    dx_from_hill2 = np.asarray(schema.unpack_hill2_to_legacy32_dx(dx_hill2, template=x))

    dx_legacy_shaped = np.asarray(jax_rhs.model_hill_jax(x, 0.0, k, x_base))

    np.testing.assert_allclose(dx_from_hill2, dx_legacy_shaped, rtol=1e-6, atol=1e-12)


def test_model_hill2_jax_batch_shape_stability():
    hill_batch = np.array(
        [
            [1e-9, 1e-7],
            [2.5e-6, 8e-6],
            [0.0, 0.0],
            [4e-6, 2e-6],
        ],
        dtype=np.float64,
    )

    dx_batch = np.asarray(jax_rhs.model_hill2_jax(hill_batch))

    assert dx_batch.shape == hill_batch.shape

    expected = np.stack(
        [np.asarray(jax_rhs.model_hill2_jax(hill_batch[i])) for i in range(hill_batch.shape[0])],
        axis=0,
    )
    np.testing.assert_allclose(dx_batch, expected, rtol=1e-6, atol=1e-12)
