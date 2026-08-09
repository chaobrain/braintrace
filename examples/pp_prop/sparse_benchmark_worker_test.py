"""Tests for sparse pp-prop benchmark worker accounting."""

from sparse_benchmark_config import SparseBenchmarkConfig
from sparse_benchmark_worker import (
    _RunState,
    _example_config,
    _load_learning_example,
    _max_updates,
    _record_validation,
)


def test_initial_target_is_recorded_at_zero_ticks() -> None:
    config = SparseBenchmarkConfig(target_accuracy=0.8)
    state = _RunState()

    _record_validation(state, config, 0.8)

    assert state.threshold_updates == 0
    assert state.validation_history == [
        {"update": 0, "training_ticks": 0, "accuracy": 0.8}
    ]


def test_threshold_latches_first_passing_checkpoint() -> None:
    config = SparseBenchmarkConfig(steps=7, final_window=2)
    state = _RunState(updates=1)

    _record_validation(state, config, 0.8)
    state.updates = 2
    _record_validation(state, config, 0.95)
    state.updates = 3
    _record_validation(state, config, 0.99)

    assert state.threshold_updates == 2
    assert state.validation_history[1]["training_ticks"] == 14


def test_update_budget_depends_on_mode() -> None:
    fixed = SparseBenchmarkConfig(updates=4)
    target = SparseBenchmarkConfig(
        mode="validation-target", batch_size=32, max_epochs=2
    )

    assert _max_updates(fixed, 288) == 4
    assert _max_updates(target, 288) == 18


def test_learning_config_preserves_requested_shape_and_scaling() -> None:
    benchmark = SparseBenchmarkConfig(
        neurons=12,
        degree=3,
        steps=7,
        final_window=2,
        recurrent_scale_basis="degree",
    )

    config = _example_config(_load_learning_example(), benchmark)

    assert config.n_rec == 12
    assert config.degree == 3
    assert config.n_step == 7
    assert config.final_window == 2
    assert config.recurrent_scale_basis == "degree"
