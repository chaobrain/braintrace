"""Tests for sparse pp-prop benchmark configuration."""

from pathlib import Path

import pytest

from sparse_benchmark_config import (
    SparseBenchmarkConfig,
    config_to_cli_args,
    config_to_dict,
    parse_config,
)


def test_defaults_describe_safe_fixed_work() -> None:
    config = parse_config([])

    assert config == SparseBenchmarkConfig()
    assert config.updates == 3
    assert config.recurrent_scale_basis == "degree"
    assert config.sparse_backend == "jax_raw"


def test_all_arguments_parse() -> None:
    config = parse_config(
        [
            "--mode", "validation-target", "--neurons", "1024", "--degree", "16",
            "--batch-size", "24", "--steps", "40", "--final-window", "8",
            "--seed", "7", "--max-epochs", "9", "--updates", "6",
            "--eval-interval", "2", "--target-accuracy", "0.9",
            "--learning-rate", "0.001", "--decay", "0.8", "--clip-norm", "2.5",
            "--sparse-backend", "default", "--recurrent-scale-basis", "neurons",
            "--max-rss-gib", "10", "--min-available-gib", "0",
            "--max-wall-seconds", "90",
            "--require-target", "--json-output", "result.json",
        ]
    )

    assert config.mode == "validation-target"
    assert config.neurons == 1024
    assert config.batch_size == 24
    assert config.require_target
    assert config.max_wall_seconds == 90.0
    assert config.json_output == Path("result.json")


def test_config_serializers_round_trip() -> None:
    original = SparseBenchmarkConfig(
        mode="validation-target", neurons=512, degree=4, batch_size=16,
        sparse_backend="default", recurrent_scale_basis="neurons",
        require_target=True, json_output=Path("metrics.json"),
    )

    assert parse_config(config_to_cli_args(original)) == original
    assert config_to_dict(original)["json_output"] == "metrics.json"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"neurons": 0}, "neurons"),
        ({"degree": 0}, "degree"),
        ({"neurons": 4, "degree": 5}, "degree"),
        ({"batch_size": 7}, "batch_size"),
        ({"steps": 0}, "steps"),
        ({"steps": 4, "final_window": 5}, "final_window"),
        ({"seed": -1}, "seed"),
        ({"max_epochs": 0}, "max_epochs"),
        ({"updates": 0}, "updates"),
        ({"eval_interval": 0}, "eval_interval"),
        ({"target_accuracy": 0.0}, "target_accuracy"),
        ({"target_accuracy": 1.01}, "target_accuracy"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"decay": -0.1}, "decay"),
        ({"decay": 1.0}, "decay"),
        ({"clip_norm": float("inf")}, "clip_norm"),
        ({"max_rss_gib": 0.0}, "max_rss_gib"),
        ({"min_available_gib": -1.0}, "min_available_gib"),
        ({"max_wall_seconds": 0.0}, "max_wall_seconds"),
    ],
)
def test_invalid_numeric_config_is_rejected(change: dict[str, object], message: str) -> None:
    values = config_to_dict(SparseBenchmarkConfig())
    values.update(change)

    with pytest.raises((TypeError, ValueError), match=message):
        SparseBenchmarkConfig(**values)


@pytest.mark.parametrize("option", ["--mode", "--sparse-backend", "--recurrent-scale-basis"])
def test_parser_rejects_unknown_choices(option: str) -> None:
    with pytest.raises(SystemExit):
        parse_config([option, "unknown"])
