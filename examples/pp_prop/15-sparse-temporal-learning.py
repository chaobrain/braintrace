"""Train sparse pp-prop on a fixed held-out handwritten-digit split.

The benchmark reports three deterministic runs on digits zero and one. Each
image becomes a 30-step Poisson spike train, and only the final five outputs
receive supervision.
"""

import importlib.util
import math
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Literal

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braintrace

N_STEP = 30
N_REC = 96
DEGREE = 8
N_CLASS = 2
N_EPOCH = 5
SEEDS = (0, 1, 2)


def _load_sparse_example():
    path = pathlib.Path(__file__).resolve().with_name("09-operator-sparse.py")
    spec = importlib.util.spec_from_file_location("_pp_prop_sparse_operator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load sparse pp-prop operators from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SPARSE = _load_sparse_example()


@dataclass(frozen=True)
class _DigitData:
    train_images: np.ndarray
    train_labels: np.ndarray
    valid_images: np.ndarray
    valid_labels: np.ndarray


@dataclass(frozen=True)
class _RunConfig:
    seed: int
    n_epochs: int
    batch_size: int
    n_rec: int = N_REC
    degree: int = DEGREE
    n_step: int = N_STEP
    final_window: int = 5
    learning_rate: float = 3e-3
    decay_or_rank: float | int = 0.95
    clip_norm: float = 1.0
    sparse_backend: str | None = "jax_raw"
    recurrent_scale_basis: Literal["neurons", "degree"] = "neurons"

    def __post_init__(self) -> None:
        positive = {
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "n_rec": self.n_rec,
            "degree": self.degree,
            "n_step": self.n_step,
            "final_window": self.final_window,
            "learning_rate": self.learning_rate,
            "clip_norm": self.clip_norm,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"{', '.join(invalid)} must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.degree > self.n_rec:
            raise ValueError("degree must not exceed n_rec")
        if self.final_window > self.n_step:
            raise ValueError("final_window must not exceed n_step")
        if isinstance(self.decay_or_rank, bool):
            raise ValueError("decay_or_rank must be a float decay or integer rank")
        if isinstance(self.decay_or_rank, int):
            if self.decay_or_rank < 1:
                raise ValueError("integer decay_or_rank must be at least one")
        elif isinstance(self.decay_or_rank, float):
            valid_decay = math.isfinite(self.decay_or_rank) and (
                0.0 <= self.decay_or_rank < 1.0
            )
            if not valid_decay:
                raise ValueError("float decay_or_rank must be in [0, 1)")
        else:
            raise ValueError("decay_or_rank must be a float decay or integer rank")
        if self.sparse_backend == "":
            raise ValueError("sparse_backend must be non-empty or None")
        if self.recurrent_scale_basis not in {"neurons", "degree"}:
            raise ValueError("recurrent_scale_basis must be 'neurons' or 'degree'")


@dataclass(frozen=True)
class _Experiment:
    model: brainstate.nn.Module
    learner: Any
    optimizer: Any
    batch_size: int


class _Net(brainstate.nn.Module):
    def __init__(self, config: _RunConfig):
        super().__init__()
        recurrent_scale = (
            config.n_rec / config.degree
            if config.recurrent_scale_basis == "degree"
            else 1.0
        )
        self.cell = _SPARSE.SparseLIFCell(
            n_in=64,
            n_rec=config.n_rec,
            density=config.degree / config.n_rec,
            seed=config.seed,
            ff_scale=6.0,
            rec_scale=recurrent_scale,
            sparse_backend=config.sparse_backend,
        )
        self.readout = _SPARSE._shared.LeakyReadout(
            n_rec=config.n_rec, n_out=N_CLASS
        )

    def update(self, spikes):
        return self.readout(self.cell(spikes))


def _load_digits() -> _DigitData:
    try:
        from sklearn.datasets import load_digits
    except ImportError as error:
        raise RuntimeError(
            "Example 15 requires scikit-learn; install the BrainTrace examples extra"
        ) from error
    digits = load_digits()
    selected = digits.target < N_CLASS
    images = np.asarray(digits.data[selected] / 16.0, dtype=np.float32)
    labels = np.asarray(digits.target[selected], dtype=np.int32)
    random = np.random.default_rng(42)
    train_indices = []
    valid_indices = []
    for label in range(N_CLASS):
        indices = np.flatnonzero(labels == label)
        random.shuffle(indices)
        n_valid = round(0.2 * indices.size)
        valid_indices.extend(indices[:n_valid])
        train_indices.extend(indices[n_valid:])
    return _DigitData(
        images[np.asarray(train_indices)],
        labels[np.asarray(train_indices)],
        images[np.asarray(valid_indices)],
        labels[np.asarray(valid_indices)],
    )


def _poisson_encode(
    images: np.ndarray, seed: int, config: _RunConfig
) -> jax.Array:
    random = np.random.default_rng(seed)
    probabilities = images[None, :, :] * 0.2
    spikes = random.random((config.n_step,) + images.shape) < probabilities
    return jnp.asarray(spikes, dtype=jnp.float32)


def _loss_mask(config: _RunConfig) -> jax.Array:
    return (
        jnp.arange(config.n_step) >= config.n_step - config.final_window
    ).astype(jnp.float32)


def _reset(experiment: _Experiment) -> None:
    brainstate.nn.reset_all_states(
        experiment.model, batch_size=experiment.batch_size
    )
    experiment.learner.reset_state(batch_size=experiment.batch_size)


def _build_experiment(config: _RunConfig) -> _Experiment:
    with brainstate.random.seed_context(config.seed):
        model = _Net(config)
    weights = model.states(brainstate.ParamState)
    brainstate.nn.init_all_states(model, batch_size=config.batch_size)
    learner = braintrace.compile(
        model,
        braintrace.pp_prop,
        jnp.zeros((config.batch_size, 64), dtype=jnp.float32),
        batch_size=config.batch_size,
        vmap=False,
        decay_or_rank=config.decay_or_rank,
        vjp_method="single-step",
    )
    optimizer = braintools.optim.Adam(lr=config.learning_rate)
    optimizer.register_trainable_weights(weights)
    return _Experiment(model, learner, optimizer, config.batch_size)


def _make_train_batch(experiment: _Experiment, config: _RunConfig):
    loss_mask = _loss_mask(config)

    @brainstate.transform.jit
    def train_batch(spikes, labels):
        _reset(experiment)

        def step_loss(step_spikes):
            logits = experiment.learner(step_spikes)
            return braintools.metric.softmax_cross_entropy_with_integer_labels(
                logits, labels
            ).mean()

        gradients, objective = experiment.learner.etrace_grad(
            spikes,
            step_fn=step_loss,
            mask=loss_mask,
            reduction="mean",
            loss_output="scalar",
            return_value=True,
        )
        clipped = brainstate.nn.clip_grad_norm(gradients, config.clip_norm)
        experiment.optimizer.update(clipped)
        return objective

    return train_batch


def _train(experiment: _Experiment, data: _DigitData, config: _RunConfig):
    train_batch = _make_train_batch(experiment, config)
    order_random = np.random.default_rng(100 + config.seed)
    epoch_losses = []
    for epoch in range(config.n_epochs):
        order = order_random.permutation(data.train_labels.size)
        batch_losses = []
        for batch_index, start in enumerate(
            range(0, order.size, config.batch_size)
        ):
            indices = order[start : start + config.batch_size]
            spike_seed = 1000 + config.seed * 10000 + epoch * 100 + batch_index
            spikes = _poisson_encode(
                data.train_images[indices], spike_seed, config
            )
            loss = train_batch(spikes, jnp.asarray(data.train_labels[indices]))
            jax.block_until_ready(loss)
            batch_losses.append(float(loss))
        epoch_losses.append(float(np.mean(batch_losses)))
    return epoch_losses


def _evaluate(
    experiment: _Experiment, data: _DigitData, config: _RunConfig
) -> float:
    all_spikes = _poisson_encode(data.valid_images, 9999, config)
    predictions = []
    for start in range(0, data.valid_labels.size, experiment.batch_size):
        stop = min(start + experiment.batch_size, data.valid_labels.size)
        count = stop - start
        spikes = all_spikes[:, start:stop]
        if count < experiment.batch_size:
            spikes = jnp.pad(
                spikes, ((0, 0), (0, experiment.batch_size - count), (0, 0))
            )
        _reset(experiment)
        outputs = experiment.learner.etrace_evolve(spikes, return_outputs=True)
        logits = outputs[-config.final_window :].mean(axis=0)
        jax.block_until_ready(logits)
        predictions.extend(np.asarray(jnp.argmax(logits, axis=-1))[:count])
    return float(np.mean(np.asarray(predictions) == data.valid_labels))


def _run_seed(data: _DigitData, config: _RunConfig) -> Dict:
    experiment = _build_experiment(config)
    initial_accuracy = _evaluate(experiment, data, config)
    recurrent = experiment.model.cell.rec_syn.comm.weight.value["weight"]
    recurrent_before = np.asarray(u.get_mantissa(recurrent)).copy()
    losses = _train(experiment, data, config)
    final_accuracy = _evaluate(experiment, data, config)
    recurrent_after = np.asarray(
        u.get_mantissa(experiment.model.cell.rec_syn.comm.weight.value["weight"])
    )
    recurrent_delta = recurrent_after - recurrent_before
    return {
        "seed": config.seed,
        "losses": losses,
        "initial_accuracy": initial_accuracy,
        "final_accuracy": final_accuracy,
        "recurrent_nnz": int(recurrent.size),
        "recurrent_values_changed": int(np.count_nonzero(recurrent_delta)),
    }


def _plot(results) -> None:
    import matplotlib.pyplot as plt

    for result in results:
        plt.plot(result["losses"], label=f"seed {result['seed']}")
    plt.xlabel("Epoch")
    plt.ylabel("Final-window cross-entropy")
    plt.title("15 · Sparse pp-prop held-out digit learning")
    plt.legend()
    plt.show()
    plt.close()


def main(n_epochs: int = N_EPOCH, batch_size: int = 32, plot: bool = True) -> Dict:
    data = _load_digits()
    if data.train_labels.size % batch_size:
        raise ValueError("batch_size must divide the 288-example training split")
    started = time.perf_counter()
    with brainstate.environ.context(dt=1.0 * u.ms):
        results = [
            _run_seed(data, _RunConfig(seed, n_epochs, batch_size))
            for seed in SEEDS
        ]
    accuracies = np.asarray(
        [result["final_accuracy"] for result in results], dtype=np.float64
    )
    for result in results:
        print(
            f"[15-sparse-learning] seed={result['seed']} "
            f"accuracy={result['initial_accuracy']:.3f} -> "
            f"{result['final_accuracy']:.3f} loss={result['losses'][0]:.4f} -> "
            f"{result['losses'][-1]:.4f}"
        )
    if plot:
        _plot(results)
    return {
        "losses": results[0]["losses"],
        "seed_results": results,
        "mean_accuracy": float(accuracies.mean()),
        "std_accuracy": float(accuracies.std()),
        "minimum_accuracy": float(accuracies.min()),
        "elapsed_seconds": time.perf_counter() - started,
        "recurrent_nnz": results[0]["recurrent_nnz"],
    }


if __name__ == "__main__":
    result = main()
    assert result["minimum_accuracy"] >= 0.90
    assert result["mean_accuracy"] >= 0.95
    assert all(
        seed_result["losses"][-1] < seed_result["losses"][0]
        for seed_result in result["seed_results"]
    )
