"""Validation tests for sparse temporal learning configuration."""

import importlib.util
import pathlib

import pytest

EXAMPLE = pathlib.Path(__file__).resolve().with_name("15-sparse-temporal-learning.py")


def _load():
    spec = importlib.util.spec_from_file_location("_pp_prop_config_validation", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_rec": 4, "degree": 5}, "degree"),
        ({"n_step": 4, "final_window": 5}, "final_window"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"decay_or_rank": 1.0}, "decay_or_rank"),
        ({"decay_or_rank": float("nan")}, "decay_or_rank"),
        ({"decay_or_rank": float("inf")}, "decay_or_rank"),
        ({"decay_or_rank": 0}, "decay_or_rank"),
        ({"decay_or_rank": True}, "decay_or_rank"),
        ({"clip_norm": 0.0}, "clip_norm"),
        ({"sparse_backend": ""}, "sparse_backend"),
        ({"recurrent_scale_basis": "edges"}, "recurrent_scale_basis"),
    ],
)
def test_run_config_rejects_invalid_values(overrides, message):
    example = _load()

    with pytest.raises(ValueError, match=message):
        example._RunConfig(seed=0, n_epochs=1, batch_size=1, **overrides)


@pytest.mark.parametrize("decay_or_rank", [0.0, 0.5, 0.999999, 1, 4])
def test_run_config_accepts_decay_and_rank_boundaries(decay_or_rank):
    example = _load()
    config = example._RunConfig(
        seed=0, n_epochs=1, batch_size=1, decay_or_rank=decay_or_rank
    )

    assert config.decay_or_rank == decay_or_rank
