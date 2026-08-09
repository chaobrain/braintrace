"""Native sparse-gradient tests for the pp-prop sparse example."""

import importlib.util
import pathlib

import brainstate
import jax
import jax.numpy as jnp
import pytest

import braintrace


EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "09-operator-sparse.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("_pp_prop_sparse_gradient", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _SparseRecurrence(brainstate.nn.Module):
    def __init__(self, sparse_matrix):
        super().__init__()
        self.size = sparse_matrix.shape[0]
        self.linear = braintrace.nn.SparseLinear(sparse_matrix, b_init=None)
        self.hidden = brainstate.HiddenState(
            jnp.zeros((self.size,), dtype=jnp.float32)
        )

    def init_state(self, batch_size=None, **kwargs):
        shape = (self.size,) if batch_size is None else (batch_size, self.size)
        self.hidden.value = jnp.zeros(shape, dtype=jnp.float32)

    def update(self, drive):
        self.hidden.value = jnp.tanh(
            self.linear(self.hidden.value) + drive
        )
        return self.hidden.value


@pytest.mark.parametrize("backend", [None, "jax_raw"])
def test_sparse_backends_support_nonzero_batched_pp_prop_gradients(backend):
    module = _load_example()
    matrix = module._fixed_degree_csr(
        n_rec=32,
        density=0.125,
        scale=1.0,
        seed=0,
        backend=backend,
    )
    model = _SparseRecurrence(matrix)
    brainstate.nn.init_all_states(model, batch_size=3)
    drive = jnp.ones((3, 32), dtype=jnp.float32)
    learner = braintrace.pp_prop(model, 0.9)
    learner.compile_graph(drive)
    learner(drive)

    gradients = brainstate.transform.grad(
        lambda value: jnp.mean(learner(value) ** 2),
        model.states(brainstate.ParamState),
    )(drive)

    leaves = jax.tree.leaves(gradients)
    assert [leaf.shape for leaf in leaves] == [(128,)]
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert any(bool(jnp.any(leaf != 0)) for leaf in leaves)
