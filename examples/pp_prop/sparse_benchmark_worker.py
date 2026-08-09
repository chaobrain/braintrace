"""Execute one configurable sparse pp-prop benchmark worker."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import pathlib
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field, replace
from typing import Any

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

try:
    from .sparse_benchmark_config import SparseBenchmarkConfig, config_to_dict
except ImportError:
    from sparse_benchmark_config import SparseBenchmarkConfig, config_to_dict


@dataclass
class _RunState:
    updates: int = 0
    examples_seen: int = 0
    losses: list[float] = field(default_factory=list)
    update_seconds: list[float] = field(default_factory=list)
    validation_history: list[dict[str, float | int]] = field(default_factory=list)
    validation_seconds: list[float] = field(default_factory=list)
    threshold_updates: int | None = None


@dataclass(frozen=True)
class _Runtime:
    example: Any
    data: Any
    run_config: Any
    experiment: Any
    train_batch: Any


@dataclass(frozen=True)
class _Outcome:
    runtime: _Runtime
    config: SparseBenchmarkConfig
    state: _RunState
    recurrent_before: np.ndarray
    setup_seconds: float
    total_seconds: float


def _load_learning_example() -> Any:
    path = pathlib.Path(__file__).with_name("15-sparse-temporal-learning.py")
    spec = importlib.util.spec_from_file_location("_sparse_learning_example", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load sparse learning example from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _example_config(example: Any, config: SparseBenchmarkConfig) -> Any:
    backend = None if config.sparse_backend == "default" else config.sparse_backend
    return example._RunConfig(
        seed=config.seed,
        n_epochs=config.max_epochs,
        batch_size=config.batch_size,
        n_rec=config.neurons,
        degree=config.degree,
        n_step=config.steps,
        final_window=config.final_window,
        learning_rate=config.learning_rate,
        decay_or_rank=config.decay,
        clip_norm=config.clip_norm,
        sparse_backend=backend,
        recurrent_scale_basis=config.recurrent_scale_basis,
    )


def _build_runtime(config: SparseBenchmarkConfig) -> tuple[_Runtime, float]:
    started = time.perf_counter()
    example = _load_learning_example()
    data = example._load_digits()
    run_config = _example_config(example, config)
    experiment = example._build_experiment(run_config)
    train_batch = example._make_train_batch(experiment, run_config)
    runtime = _Runtime(example, data, run_config, experiment, train_batch)
    return runtime, time.perf_counter() - started


def _evaluate(runtime: _Runtime) -> tuple[float, float]:
    started = time.perf_counter()
    accuracy = runtime.example._evaluate(
        runtime.experiment, runtime.data, runtime.run_config
    )
    return accuracy, time.perf_counter() - started


def _record_validation(
    state: _RunState, config: SparseBenchmarkConfig, accuracy: float
) -> None:
    state.validation_history.append(
        {
            "update": state.updates,
            "training_ticks": state.updates * config.steps,
            "accuracy": accuracy,
        }
    )
    if state.threshold_updates is None and accuracy >= config.target_accuracy:
        state.threshold_updates = state.updates


def _max_updates(config: SparseBenchmarkConfig, train_examples: int) -> int:
    if config.mode == "fixed-work":
        return config.updates
    batches_per_epoch = train_examples // config.batch_size
    return config.max_epochs * batches_per_epoch


def _epoch_data(runtime: _Runtime, config: SparseBenchmarkConfig, epoch: int):
    seed = 1000 + config.seed * 10000 + epoch
    spikes = runtime.example._poisson_encode(
        runtime.data.train_images, seed, runtime.run_config
    )
    random = np.random.default_rng(100 + config.seed + epoch)
    order = random.permutation(runtime.data.train_labels.size)
    return spikes, order


def _run_updates(runtime: _Runtime, config: SparseBenchmarkConfig) -> _RunState:
    state = _RunState()
    accuracy, elapsed = _evaluate(runtime)
    state.validation_seconds.append(elapsed)
    _record_validation(state, config, accuracy)
    print(f"initial_accuracy={accuracy:.6f}", file=sys.stderr, flush=True)
    if config.mode == "validation-target" and state.threshold_updates == 0:
        return state
    batches_per_epoch = runtime.data.train_labels.size // config.batch_size
    limit = _max_updates(config, runtime.data.train_labels.size)
    epoch_spikes = None
    order = None
    for update_index in range(limit):
        batch_index = update_index % batches_per_epoch
        epoch = update_index // batches_per_epoch
        if batch_index == 0:
            epoch_spikes, order = _epoch_data(runtime, config, epoch)
        if epoch_spikes is None or order is None:
            raise RuntimeError("epoch data were not initialized")
        start = batch_index * config.batch_size
        indices = order[start : start + config.batch_size]
        spikes = epoch_spikes[:, indices]
        labels = jnp.asarray(runtime.data.train_labels[indices])
        update_started = time.perf_counter()
        loss = runtime.train_batch(spikes, labels)
        jax.block_until_ready(loss)
        loss_value = float(loss)
        if not np.isfinite(loss_value):
            raise RuntimeError("training loss became non-finite")
        state.updates += 1
        state.examples_seen += indices.size
        state.losses.append(loss_value)
        state.update_seconds.append(time.perf_counter() - update_started)
        if state.updates % config.eval_interval:
            print(
                f"update={state.updates} loss={loss_value:.6f}",
                file=sys.stderr,
                flush=True,
            )
            continue
        accuracy, elapsed = _evaluate(runtime)
        state.validation_seconds.append(elapsed)
        _record_validation(state, config, accuracy)
        print(
            f"update={state.updates} ticks={state.updates * config.steps} "
            f"loss={loss_value:.6f} accuracy={accuracy:.6f}",
            file=sys.stderr,
            flush=True,
        )
        if config.mode == "validation-target" and state.threshold_updates is not None:
            break
    if state.validation_history[-1]["update"] != state.updates:
        accuracy, elapsed = _evaluate(runtime)
        state.validation_seconds.append(elapsed)
        _record_validation(state, config, accuracy)
    return state


def _recurrent_values(runtime: _Runtime) -> np.ndarray:
    recurrent = runtime.experiment.model.cell.rec_syn.comm.weight.value["weight"]
    return np.asarray(u.get_mantissa(recurrent))


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    directory = pathlib.Path(__file__).parent
    for name in (
        "sparse_benchmark_worker.py",
        "sparse_benchmark_config.py",
        "sparse_benchmark_supervisor.py",
        "configurable_sparse_benchmark.py",
        "16-configurable-sparse-benchmark.py",
        "15-sparse-temporal-learning.py",
        "09-operator-sparse.py",
        "_shared.py",
    ):
        digest.update((directory / name).read_bytes())
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _package_commit(name: str) -> str | None:
    try:
        direct_url = importlib.metadata.distribution(name).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if direct_url is None:
        return None
    document = json.loads(direct_url)
    return document.get("vcs_info", {}).get("commit_id")


def _environment() -> dict[str, object]:
    device = jax.devices()[0]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": _package_version("jax"),
        "braintrace": _package_version("braintrace"),
        "braintrace_commit": _package_commit("braintrace"),
        "brainstate": _package_version("brainstate"),
        "backend": jax.default_backend(),
        "device": getattr(device, "device_kind", str(device)),
        "source_sha256": _source_fingerprint(),
    }


def _timings(outcome: _Outcome) -> dict[str, object]:
    warmed = outcome.state.update_seconds[1:]
    return {
        "setup_seconds": outcome.setup_seconds,
        "cold_update_seconds": (
            outcome.state.update_seconds[0]
            if outcome.state.update_seconds
            else None
        ),
        "warm_update_median_seconds": statistics.median(warmed) if warmed else None,
        "update_seconds": outcome.state.update_seconds,
        "validation_seconds": outcome.state.validation_seconds,
        "total_worker_seconds": outcome.total_seconds,
    }


def _status(config: SparseBenchmarkConfig, state: _RunState) -> str:
    if config.require_target and state.threshold_updates is None:
        return "target_not_reached"
    if config.mode == "fixed-work":
        return "completed"
    return "target_reached" if state.threshold_updates is not None else "target_not_reached"


def _metrics(outcome: _Outcome) -> dict[str, object]:
    recurrent_after = _recurrent_values(outcome.runtime)
    recurrent_delta = recurrent_after - outcome.recurrent_before
    evaluations = len(outcome.state.validation_history)
    valid_examples = outcome.runtime.data.valid_labels.size
    padded_valid = int(
        np.ceil(valid_examples / outcome.config.batch_size)
        * outcome.config.batch_size
    )
    return {
        "updates_completed": outcome.state.updates,
        "training_ticks": outcome.state.updates * outcome.config.steps,
        "examples_seen": outcome.state.examples_seen,
        "training_sample_ticks": outcome.state.examples_seen * outcome.config.steps,
        "threshold_updates": outcome.state.threshold_updates,
        "threshold_ticks": (
            outcome.state.threshold_updates * outcome.config.steps
            if outcome.state.threshold_updates is not None
            else None
        ),
        "threshold_tick_is_checkpoint_upper_bound": True,
        "initial_validation_accuracy": outcome.state.validation_history[0]["accuracy"],
        "final_validation_accuracy": outcome.state.validation_history[-1]["accuracy"],
        "validation_history": outcome.state.validation_history,
        "validation_evaluations": evaluations,
        "validation_sample_ticks": evaluations * valid_examples * outcome.config.steps,
        "validation_padded_sample_ticks": (
            evaluations * padded_valid * outcome.config.steps
        ),
        "losses": outcome.state.losses,
        "recurrent_nnz": int(recurrent_after.size),
        "recurrent_values_changed": int(np.count_nonzero(recurrent_delta)),
        "recurrent_delta_l2": float(np.linalg.norm(recurrent_delta)),
        "dense_feedforward_values": 64 * outcome.config.neurons,
        "readout_values": outcome.config.neurons * 2,
    }


def run_benchmark(config: SparseBenchmarkConfig) -> dict[str, object]:
    """Run one benchmark configuration inside the current worker process.

    Parameters
    ----------
    config : SparseBenchmarkConfig
        Validated experiment and resource settings.

    Returns
    -------
    dict
        Schema-versioned benchmark result without supervisor telemetry.
    """
    started = time.perf_counter()
    with brainstate.environ.context(dt=1.0 * u.ms):
        runtime, setup_seconds = _build_runtime(config)
        recurrent_before = _recurrent_values(runtime).copy()
        state = _run_updates(runtime, config)
    outcome = _Outcome(
        runtime, config, state, recurrent_before, setup_seconds, 0.0
    )
    metrics = _metrics(outcome)
    environment = _environment()
    outcome = replace(outcome, total_seconds=time.perf_counter() - started)
    return {
        "schema_version": 1,
        "status": _status(config, state),
        "config": config_to_dict(config),
        "metrics": metrics,
        "timings": _timings(outcome),
        "environment": environment,
        "memory": {
            "scope": "cpu_process_tree_rss",
            "peak_rss_bytes": None,
            "guard_status": None,
            "guard_reason": None,
        },
    }
