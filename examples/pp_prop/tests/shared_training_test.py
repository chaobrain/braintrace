"""Windowed training regressions for the pp-prop example helpers."""

import importlib.util
import pathlib

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp

import braintrace


SHARED = pathlib.Path(__file__).resolve().parents[1] / "_shared.py"
CONTRAST = pathlib.Path(__file__).resolve().parents[1] / "14-knob-vjp-method-contrast.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _WindowNet(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = brainstate.ParamState(jnp.eye(2, dtype=jnp.float32) * 0.2)
        self.hidden = brainstate.HiddenState(jnp.zeros(2, dtype=jnp.float32))

    def init_state(self, batch_size=None, **kwargs):
        shape = (2,) if batch_size is None else (batch_size, 2)
        self.hidden.value = jnp.zeros(shape, dtype=jnp.float32)

    def update(self, x):
        self.hidden.value = jnp.tanh(
            x + braintrace.matmul(self.hidden.value, self.weight.value)
        )
        return self.hidden.value


def test_fixed_target_helper_runs_real_windows(monkeypatch):
    shared = _load(SHARED, "_pp_prop_shared_training")
    model = _WindowNet()
    weights = model.states(brainstate.ParamState)
    optimizer = braintools.optim.Adam(lr=1e-2)
    optimizer.register_trainable_weights(weights)
    inputs = jnp.ones((4, 3, 2), dtype=jnp.float32)
    labels = jnp.array([0, 1, 0], dtype=jnp.int32)
    mask = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    captured = []
    compile_fn = braintrace.compile

    def capture(*args, **kwargs):
        learner = compile_fn(*args, **kwargs)
        captured.append(learner)
        return learner

    monkeypatch.setattr(braintrace, "compile", capture)
    before = jnp.array(model.weight.value)
    loss = shared.online_train_epoch_fixed_target(
        model,
        optimizer,
        inputs,
        labels,
        vjp_method="multi-step",
        chunk_size=2,
        vmap=False,
        loss_mask=mask,
        reduction="mean",
    )

    assert bool(jnp.isfinite(loss))
    assert int(captured[0].running_index.value) == inputs.shape[0]
    assert bool(jnp.any(model.weight.value != before))


def test_contrast_models_start_from_identical_parameters():
    contrast = _load(CONTRAST, "_pp_prop_contrast_training")
    with brainstate.environ.context(dt=1.0 * u.ms):
        first, _ = contrast._make(4, 8, 2, seed=0)
        second, _ = contrast._make(4, 8, 2, seed=0)
    first_values = first.states(brainstate.ParamState).to_dict_values()
    second_values = second.states(brainstate.ParamState).to_dict_values()

    assert first_values.keys() == second_values.keys()
    first_leaves = jax.tree.leaves(first_values)
    second_leaves = jax.tree.leaves(second_values)
    assert len(first_leaves) == len(second_leaves)
    for first_value, second_value in zip(first_leaves, second_leaves):
        assert bool(jnp.array_equal(
            u.get_mantissa(first_value), u.get_mantissa(second_value)
        ))
