"""Runtime-shape tests for sparse temporal learning configuration."""

import importlib.util
import pathlib

import numpy as np

EXAMPLE = pathlib.Path(__file__).resolve().with_name("15-sparse-temporal-learning.py")


def _load():
    spec = importlib.util.spec_from_file_location("_pp_prop_config_runtime", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_custom_config_controls_sparse_topology_and_scale():
    example = _load()
    common = dict(seed=0, n_epochs=1, batch_size=1, n_rec=12, degree=3)
    neuron_scaled = example._Net(example._RunConfig(**common))
    degree_scaled = example._Net(
        example._RunConfig(**common, recurrent_scale_basis="degree")
    )
    other_seed = example._Net(example._RunConfig(**(common | {"seed": 1})))
    neuron_values = np.asarray(
        example.u.get_mantissa(
            neuron_scaled.cell.rec_syn.comm.weight.value["weight"]
        )
    )
    degree_values = np.asarray(
        example.u.get_mantissa(
            degree_scaled.cell.rec_syn.comm.weight.value["weight"]
        )
    )

    assert neuron_values.size == 36
    assert degree_values.size == 36
    np.testing.assert_allclose(degree_values, neuron_values * 2.0, rtol=1e-6)
    assert not np.array_equal(
        np.asarray(neuron_scaled.cell.rec_syn.comm.spar_mat.indices),
        np.asarray(other_seed.cell.rec_syn.comm.spar_mat.indices),
    )


def test_custom_config_controls_temporal_encoding_and_mask():
    example = _load()
    config = example._RunConfig(
        seed=0,
        n_epochs=1,
        batch_size=2,
        n_rec=12,
        degree=3,
        n_step=7,
        final_window=2,
    )
    images = np.zeros((2, 64), dtype=np.float32)
    spikes = example._poisson_encode(images, 42, config)
    mask = example._loss_mask(config)

    assert spikes.shape == (7, 2, 64)
    assert np.asarray(mask).tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
