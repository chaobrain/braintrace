"""Acceptance tests for the sparse temporal learning example."""

import importlib.util
import pathlib

import numpy as np

EXAMPLE = pathlib.Path(__file__).resolve().with_name("15-sparse-temporal-learning.py")


def _load():
    spec = importlib.util.spec_from_file_location("_pp_prop_sparse_learning", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sparse_digit_learning_beats_chance_across_seeds():
    example = _load()
    result = example.main(plot=False)

    assert result["recurrent_nnz"] == example.N_REC * example.DEGREE
    assert result["minimum_accuracy"] >= 0.90
    assert result["mean_accuracy"] >= 0.95
    for seed_result in result["seed_results"]:
        assert seed_result["losses"][-1] < seed_result["losses"][0]
        assert seed_result["recurrent_values_changed"] > 0


def test_digit_split_is_fixed_and_stratified():
    example = _load()
    first = example._load_digits()
    repeated = example._load_digits()

    assert np.array_equal(first.train_images, repeated.train_images)
    assert np.array_equal(first.valid_images, repeated.valid_images)
    assert first.train_labels.size == 288
    assert first.valid_labels.size == 72
    assert np.bincount(first.train_labels).tolist() == [142, 146]
    assert np.bincount(first.valid_labels).tolist() == [36, 36]


def test_run_config_defaults_preserve_example_15_behavior():
    example = _load()
    config = example._RunConfig(seed=0, n_epochs=1, batch_size=32)

    assert config.n_rec == example.N_REC
    assert config.degree == example.DEGREE
    assert config.n_step == example.N_STEP
    assert config.final_window == 5
    assert config.learning_rate == 3e-3
    assert config.decay_or_rank == 0.95
    assert config.clip_norm == 1.0
    assert config.sparse_backend == "jax_raw"
    assert config.recurrent_scale_basis == "neurons"
