import brainstate
import jax
import jax.numpy as jnp
import pytest

import braintrace
from braintrace._compiler.position_graph import prove_elementwise_transform


class _WeightFnNet(brainstate.nn.Module):
    def __init__(self, weight_fn):
        super().__init__()
        self.weight_fn = weight_fn
        self.weight = brainstate.ParamState(jnp.arange(3, dtype=jnp.float32))
        self.hidden = brainstate.HiddenState(jnp.zeros(3, dtype=jnp.float32))

    def update(self, x):
        weight = braintrace.element_wise(
            self.weight.value,
            weight_fn=self.weight_fn,
        )
        self.hidden.value = jnp.tanh(self.hidden.value + x + weight)
        return self.hidden.value


class _ExternalWeightFnNet(brainstate.nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform
        self.weight = brainstate.ParamState(jnp.arange(3, dtype=jnp.float32))
        self.hidden = brainstate.HiddenState(jnp.zeros(3, dtype=jnp.float32))

    def update(self, x):
        weight = self.transform(self.weight.value)
        weight = braintrace.element_wise(weight)
        self.hidden.value = jnp.tanh(self.hidden.value + x + weight)
        return self.hidden.value


def _mix(weight):
    matrix = jnp.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=weight.dtype,
    )
    return matrix @ weight


def _reshape(weight):
    return weight.reshape((1, 3)).reshape((3,))


@pytest.mark.parametrize('weight_fn', [jnp.flip, _reshape, _mix])
def test_pp_prop_rejects_non_position_preserving_weight_fn(weight_fn):
    learner = braintrace.pp_prop(_WeightFnNet(weight_fn), 0.9)

    with pytest.raises(
        braintrace.NotSupportedError,
        match=r'element_wise\(weight_fn=.*position',
    ):
        learner.compile_graph(jnp.ones(3, dtype=jnp.float32))


@pytest.mark.parametrize(
    'weight_fn',
    [jnp.tanh, jnp.sin, lambda weight: weight * 2.0 + 1.0],
)
def test_pp_prop_accepts_position_preserving_weight_fn(weight_fn):
    learner = braintrace.pp_prop(_WeightFnNet(weight_fn), 0.9)

    learner.compile_graph(jnp.ones(3, dtype=jnp.float32))

    assert learner.is_compiled


@pytest.mark.parametrize(
    'transform',
    [lambda weight: 2.0 * weight, lambda weight: weight * jnp.array([1.0, 0.0, 1.0])],
)
def test_pp_prop_rejects_external_parameter_preprocessing(transform):
    learner = braintrace.pp_prop(_ExternalWeightFnNet(transform), 0.9)

    with pytest.raises(
        braintrace.NotSupportedError,
        match=r'directly from its ParamState.*weight_fn.*kernel_fn.*bias_fn',
    ):
        learner.compile_graph(jnp.ones(3, dtype=jnp.float32))


def test_pp_prop_accepts_direct_parameter_input():
    learner = braintrace.pp_prop(_ExternalWeightFnNet(lambda weight: weight), 0.9)

    learner.compile_graph(jnp.ones(3, dtype=jnp.float32))

    assert learner.is_compiled


def test_weight_fn_proof_traces_large_shape_abstractly():
    aval = jax.ShapeDtypeStruct((139_255,), jnp.float32)

    assert prove_elementwise_transform(jnp.tanh, aval) is None
