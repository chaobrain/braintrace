"""Configuration parsing for the sparse pp-prop benchmark."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence, cast


BenchmarkMode = Literal["fixed-work", "validation-target"]
SparseBackend = Literal["jax_raw", "default"]
ScaleBasis = Literal["degree", "neurons"]

_TRAIN_EXAMPLES = 288
_MAX_CSR_INDEX = 2_147_483_647


@dataclass(frozen=True)
class SparseBenchmarkConfig:
    """Define one reproducible sparse pp-prop benchmark run.

    Attributes
    ----------
    mode : {"fixed-work", "validation-target"}
        Benchmark stopping policy.
    neurons : int
        Recurrent neuron count.
    degree : int
        Stored recurrent edges per neuron.
    batch_size : int
        Parallel training examples per update.
    steps : int
        Neural timesteps per training update.
    final_window : int
        Final timesteps receiving supervision.
    seed : int
        Model and data-order seed.
    max_epochs : int
        Validation-target epoch ceiling.
    updates : int
        Fixed-work update count.
    eval_interval : int
        Updates between validation checkpoints.
    target_accuracy : float
        Validation threshold.
    learning_rate : float
        Adam learning rate.
    decay : float
        pp-prop trace decay.
    clip_norm : float
        Gradient norm ceiling.
    sparse_backend : {"jax_raw", "default"}
        Brainevent sparse backend selection.
    recurrent_scale_basis : {"degree", "neurons"}
        Denominator used for recurrent initialization scaling.
    max_rss_gib : float
        Worker process-tree RSS ceiling.
    min_available_gib : float
        Required host-memory headroom.
    max_wall_seconds : float
        Worker wall-clock ceiling.
    require_target : bool
        Whether a target miss returns a nonzero exit.
    json_output : pathlib.Path or None
        Optional result destination.
    """

    mode: BenchmarkMode = "fixed-work"
    neurons: int = 96
    degree: int = 8
    batch_size: int = 32
    steps: int = 30
    final_window: int = 5
    seed: int = 0
    max_epochs: int = 5
    updates: int = 3
    eval_interval: int = 1
    target_accuracy: float = 0.95
    learning_rate: float = 3e-3
    decay: float = 0.95
    clip_norm: float = 1.0
    sparse_backend: SparseBackend = "jax_raw"
    recurrent_scale_basis: ScaleBasis = "degree"
    max_rss_gib: float = 12.0
    min_available_gib: float = 12.0
    max_wall_seconds: float = 1800.0
    require_target: bool = False
    json_output: Path | None = None

    def __post_init__(self) -> None:
        _require_choice("mode", self.mode, ("fixed-work", "validation-target"))
        _require_positive_int("neurons", self.neurons)
        _require_positive_int("degree", self.degree)
        _require_positive_int("batch_size", self.batch_size)
        _require_positive_int("steps", self.steps)
        _require_positive_int("final_window", self.final_window)
        _require_nonnegative_int("seed", self.seed)
        _require_positive_int("max_epochs", self.max_epochs)
        _require_positive_int("updates", self.updates)
        _require_positive_int("eval_interval", self.eval_interval)
        if self.degree > self.neurons:
            raise ValueError("degree must not exceed neurons")
        if self.neurons * self.degree > _MAX_CSR_INDEX:
            raise ValueError("neurons * degree exceeds the CSR int32 index limit")
        if _TRAIN_EXAMPLES % self.batch_size:
            raise ValueError("batch_size must divide the 288-example training split")
        if self.final_window > self.steps:
            raise ValueError("final_window must not exceed steps")
        _require_probability("target_accuracy", self.target_accuracy)
        _require_positive_float("learning_rate", self.learning_rate)
        _require_unit_interval("decay", self.decay)
        _require_positive_float("clip_norm", self.clip_norm)
        _require_choice("sparse_backend", self.sparse_backend, ("jax_raw", "default"))
        _require_choice(
            "recurrent_scale_basis", self.recurrent_scale_basis, ("degree", "neurons")
        )
        _require_positive_float("max_rss_gib", self.max_rss_gib)
        _require_nonnegative_float("min_available_gib", self.min_available_gib)
        _require_positive_float("max_wall_seconds", self.max_wall_seconds)
        if not isinstance(self.require_target, bool):
            raise TypeError("require_target must be a bool")
        if self.json_output is not None and not isinstance(self.json_output, Path):
            raise TypeError("json_output must be a pathlib.Path or None")


def parse_config(argv: Sequence[str] | None = None) -> SparseBenchmarkConfig:
    """Parse command-line arguments into a benchmark configuration.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments without the executable name. The process arguments are used
        when omitted.

    Returns
    -------
    SparseBenchmarkConfig
        Validated benchmark settings.
    """
    defaults = SparseBenchmarkConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("fixed-work", "validation-target"), default=defaults.mode
    )
    parser.add_argument("--neurons", type=int, default=defaults.neurons)
    parser.add_argument("--degree", type=int, default=defaults.degree)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--final-window", type=int, default=defaults.final_window)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--max-epochs", type=int, default=defaults.max_epochs)
    parser.add_argument("--updates", type=int, default=defaults.updates)
    parser.add_argument("--eval-interval", type=int, default=defaults.eval_interval)
    parser.add_argument("--target-accuracy", type=float, default=defaults.target_accuracy)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--decay", type=float, default=defaults.decay)
    parser.add_argument("--clip-norm", type=float, default=defaults.clip_norm)
    parser.add_argument(
        "--sparse-backend",
        choices=("jax_raw", "default"),
        default=defaults.sparse_backend,
    )
    parser.add_argument(
        "--recurrent-scale-basis",
        choices=("degree", "neurons"),
        default=defaults.recurrent_scale_basis,
    )
    parser.add_argument("--max-rss-gib", type=float, default=defaults.max_rss_gib)
    parser.add_argument("--min-available-gib", type=float, default=defaults.min_available_gib)
    parser.add_argument(
        "--max-wall-seconds", type=float, default=defaults.max_wall_seconds
    )
    parser.add_argument(
        "--require-target", action=argparse.BooleanOptionalAction, default=defaults.require_target
    )
    parser.add_argument("--json-output", type=Path, default=defaults.json_output)
    values = vars(parser.parse_args(argv))
    values["mode"] = cast(BenchmarkMode, values["mode"])
    values["sparse_backend"] = cast(SparseBackend, values["sparse_backend"])
    values["recurrent_scale_basis"] = cast(ScaleBasis, values["recurrent_scale_basis"])
    return SparseBenchmarkConfig(**values)


def config_to_dict(config: SparseBenchmarkConfig) -> dict[str, object]:
    """Return a JSON-compatible dictionary for ``config``.

    Parameters
    ----------
    config : SparseBenchmarkConfig
        Configuration to serialize.

    Returns
    -------
    dict
        Field names and JSON-compatible values.
    """
    values = cast(dict[str, object], asdict(config))
    values["json_output"] = str(config.json_output) if config.json_output else None
    return values


def config_to_cli_args(config: SparseBenchmarkConfig) -> list[str]:
    """Serialize ``config`` into arguments accepted by :func:`parse_config`.

    Parameters
    ----------
    config : SparseBenchmarkConfig
        Configuration to serialize.

    Returns
    -------
    list of str
        Complete command-line argument list.
    """
    values = config_to_dict(config)
    arguments: list[str] = []
    for name, value in values.items():
        option = f"--{name.replace('_', '-')}"
        if name == "require_target":
            arguments.append(option if value else "--no-require-target")
        elif value is not None:
            arguments.extend((option, str(value)))
    return arguments


def _require_choice(name: str, value: object, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}")


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _require_finite_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _require_positive_float(name: str, value: float) -> None:
    if _require_finite_float(name, value) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative_float(name: str, value: float) -> None:
    if _require_finite_float(name, value) < 0.0:
        raise ValueError(f"{name} must be nonnegative")


def _require_probability(name: str, value: float) -> None:
    converted = _require_finite_float(name, value)
    if not 0.0 < converted <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]")


def _require_unit_interval(name: str, value: float) -> None:
    converted = _require_finite_float(name, value)
    if not 0.0 <= converted < 1.0:
        raise ValueError(f"{name} must be in [0, 1)")
