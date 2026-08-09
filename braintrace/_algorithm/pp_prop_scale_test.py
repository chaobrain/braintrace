import brainevent
import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace

NEURON_COUNT = 139_255
EIGHT_EDGE_ROWS = 25_215
EDGE_COUNT = 1_000_000

def _connectome_scale_csr() -> brainevent.CSR:
    degrees = np.full(NEURON_COUNT, 7, dtype=np.int32)
    degrees[:EIGHT_EDGE_ROWS] = 8
    indptr = np.empty(NEURON_COUNT + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(degrees, out=indptr[1:])
    eight_edge_indices = (
        np.arange(EIGHT_EDGE_ROWS, dtype=np.int32)[:, None]
        + np.arange(8, dtype=np.int32)
    ) % NEURON_COUNT
    seven_edge_indices = (
        np.arange(EIGHT_EDGE_ROWS, NEURON_COUNT, dtype=np.int32)[:, None]
        + np.arange(7, dtype=np.int32)
    ) % NEURON_COUNT
    indices = np.concatenate(
        (eight_edge_indices.reshape(-1), seven_edge_indices.reshape(-1))
    )
    values = jnp.full((EDGE_COUNT,), 0.01, dtype=jnp.float32)
    return brainevent.CSR(
        values,
        jnp.asarray(indices),
        jnp.asarray(indptr),
        shape=(NEURON_COUNT, NEURON_COUNT),
        backend='jax_raw',
    )

class _ConnectomeScaleRecurrence(brainstate.nn.Module):
    def __init__(self, sparse_matrix: brainevent.CSR):
        super().__init__()
        self.recurrence = braintrace.nn.SparseLinear(sparse_matrix, b_init=None)
        self.hidden = brainstate.HiddenState(
            jnp.zeros((NEURON_COUNT,), dtype=jnp.float32)
        )

    def update(self, drive):
        self.hidden.value = jnp.tanh(
            self.recurrence(self.hidden.value) + drive
        )
        return self.hidden.value

@pytest.mark.slow
def test_connectome_scale_pp_prop_stays_sparse_and_finite():
    sparse_matrix = _connectome_scale_csr()
    coupled_model = _ConnectomeScaleRecurrence(sparse_matrix)
    brainstate.nn.init_all_states(coupled_model)
    coupled = braintrace.pp_prop(
        coupled_model,
        0.9,
        config=braintrace.ETraceConfig(
            trace_factorization='io_factorized',
            recurrence_scope='coupled',
            decay=0.9,
        ),
    )
    drive = jnp.ones((NEURON_COUNT,), dtype=jnp.float32)

    with pytest.raises(
        braintrace.NotSupportedError,
        match=str(NEURON_COUNT ** 2),
    ):
        coupled.compile_graph(drive)

    assert not coupled.is_compiled
    model = _ConnectomeScaleRecurrence(sparse_matrix)
    brainstate.nn.init_all_states(model)
    learner = braintrace.pp_prop(model, 0.9)

    learner.compile_graph(drive)
    trace_elements = sum(
        leaf.size
        for state in list(learner.etrace_xs.values()) + list(learner.etrace_dfs.values())
        for leaf in jax.tree.leaves(state.value)
    )
    assert sparse_matrix.data.shape == (EDGE_COUNT,)
    assert trace_elements == 2 * NEURON_COUNT
    assert trace_elements * jnp.dtype(jnp.float32).itemsize < 1 << 30
    jax.block_until_ready(learner(drive))
    gradient = brainstate.transform.jit(
        brainstate.transform.grad(
            lambda value: jnp.mean(learner(value) ** 2),
            model.states(brainstate.ParamState),
        )
    )(drive)
    jax.block_until_ready(gradient)
    leaves = jax.tree.leaves(gradient)
    assert [leaf.shape for leaf in leaves] == [(EDGE_COUNT,)]
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert any(bool(jnp.any(leaf != 0)) for leaf in leaves)
