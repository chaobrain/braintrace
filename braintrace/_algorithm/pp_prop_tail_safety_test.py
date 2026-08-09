import brainstate
import jax
import jax.numpy as jnp
import pytest

import braintrace
from braintrace._compiler import position_graph
from braintrace._testing import oracle_models


class _MixingTailNet(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = brainstate.ParamState(jnp.eye(3, dtype=jnp.float32))
        self.mixing = jnp.roll(jnp.eye(3, dtype=jnp.float32), 1, axis=0)
        self.hidden = brainstate.HiddenState(jnp.zeros(3, dtype=jnp.float32))

    def update(self, x):
        projected = braintrace.matmul(
            self.hidden.value + x, self.weight.value
        )
        self.hidden.value = jnp.tanh(projected @ self.mixing)
        return self.hidden.value


class _BroadcastTailNet(brainstate.nn.Module):
    def __init__(self, dimensions, weight_shape=(2,)):
        super().__init__()
        self.dimensions = dimensions
        self.weight = brainstate.ParamState(
            jnp.ones(weight_shape, dtype=jnp.float32)
        )
        self.hidden = brainstate.HiddenState(jnp.zeros((2, 2), dtype=jnp.float32))

    def update(self, x):
        projected = braintrace.element_wise(self.weight.value)
        broadcast = jax.lax.broadcast_in_dim(
            projected, self.hidden.value.shape, self.dimensions
        )
        self.hidden.value = jnp.tanh(self.hidden.value + x + broadcast)
        return self.hidden.value


def test_pp_prop_rejects_a_mixing_tail():
    model = _MixingTailNet()
    brainstate.nn.init_all_states(model)
    learner = braintrace.pp_prop(model, 0.9)

    with pytest.raises(braintrace.NotSupportedError, match='mixes hidden positions'):
        learner.compile_graph(jnp.ones(3, dtype=jnp.float32))


def test_pp_prop_rejects_a_remapped_broadcast_tail():
    model = _BroadcastTailNet((0,))
    learner = braintrace.pp_prop(model, 0.9)

    with pytest.raises(braintrace.NotSupportedError, match='broadcast_in_dim'):
        learner.compile_graph(jnp.ones((2, 2), dtype=jnp.float32))


def test_pp_prop_accepts_a_trailing_broadcast_tail():
    model = _BroadcastTailNet((1,))
    learner = braintrace.pp_prop(model, 0.9)

    learner.compile_graph(jnp.ones((2, 2), dtype=jnp.float32))

    assert learner.is_compiled


def test_pp_prop_rejects_singleton_position_expansion():
    model = _BroadcastTailNet((0, 1), weight_shape=(1, 2))
    learner = braintrace.pp_prop(model, 0.9)

    with pytest.raises(braintrace.NotSupportedError, match='broadcast_in_dim'):
        learner.compile_graph(jnp.ones((2, 2), dtype=jnp.float32))


def test_tail_proof_accepts_a_scalar_broadcast():
    def transition(hidden, scalar):
        return hidden + jax.lax.broadcast_in_dim(scalar, hidden.shape, ())

    jaxpr = jax.make_jaxpr(transition)(jnp.ones((2, 2)), jnp.asarray(1.0)).jaxpr

    assert position_graph.prove_position_preserving(jaxpr, (2, 2)) is None


def test_tail_proof_never_allocates_position_matrices(monkeypatch):
    spec = oracle_models.tanh_rnn(n_in=3, n_rec=5, seed=0)
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    learner = braintrace.pp_prop(model, 0.9)

    def fail(*args, **kwargs):
        raise AssertionError('dense position matrix allocated')

    monkeypatch.setattr(position_graph.np, 'eye', fail)
    monkeypatch.setattr(position_graph.np, 'ones', fail)
    learner.compile_graph(spec.make_inputs(2, 3, seed=1)[0])

    assert learner.is_compiled
