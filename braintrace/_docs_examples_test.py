# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Executable regression tests for the ``braintrace.compile`` examples in ``docs/``.

The documentation notebooks are *not* executed during the Sphinx build
(``jupyter_execute_notebooks = "off"``), so their code is otherwise never run.
This module reconstructs the *distinct* model architectures documented across the
tutorials and runs each through ``braintrace.compile`` with the same invocation
shown in the docs, asserting the documented behaviour. It guards against silent
API drift in the docs and is co-located in ``braintrace/`` so CI (``pytest
braintrace/``) executes it.

Each test names the doc page it mirrors. Architectures already covered by other
test modules (plain D-RTRL compile, vmap mechanics, ES-D-RTRL decay/rank) are not
duplicated here; only the documented *architectures* missing elsewhere are added:
``lax.select`` control-flow, recurrent ``Conv2d``, mixed ETP/plain selection,
GRU via ``braintrace.compile`` (batched + unbatched + vmap), and manual
multi-group RNNs.
"""

import jax
import jax.numpy as jnp
import brainstate

import braintrace


def test_docs_limitations_good_model_lax_select():
    """docs/advanced/limitations.ipynb — ``jax.lax.select`` control-flow workaround."""

    class GoodModel(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(jnp.ones((10, 10)))
            self.h = brainstate.HiddenState(jnp.zeros(10))

        def update(self, x):
            new_h = jax.nn.tanh(braintrace.matmul(self.h.value, self.w.value) + x)
            # jax.lax.select compiles to a single select_n primitive the compiler
            # can trace through (unlike jnp.where, which now jits a sub-jaxpr).
            self.h.value = jax.lax.select(jnp.sum(x) > 0, new_h, self.h.value)
            return self.h.value

    model = GoodModel()
    algo = braintrace.compile(model, braintrace.D_RTRL, jnp.zeros(10), batch_size=1)
    y = algo(jnp.ones(10))
    assert y.shape == (10,)
    assert bool(jnp.all(jnp.isfinite(y)))
    assert len(algo.graph.hidden_param_op_relations) == 1


def test_docs_graphviz_conv_rnn():
    """docs/tutorials/graph_visualization.ipynb — recurrent Conv2d (etp_conv)."""

    class ConvRNN(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = braintrace.nn.Conv2d(in_size=(28, 28, 8), out_channels=8,
                                             kernel_size=3, padding='SAME')
            self.h = brainstate.HiddenState(jnp.zeros((28, 28, 8)))
            self.out = braintrace.nn.Linear(8 * 28 * 28, 10)

        def update(self, x):
            self.h.value = jax.nn.tanh(self.conv(self.h.value) + x)
            return self.out(self.h.value.reshape(-1))

    model = ConvRNN()
    algo = braintrace.compile(model, braintrace.D_RTRL, jnp.zeros((28, 28, 8)), batch_size=1)
    y = algo(jnp.zeros((28, 28, 8)))
    assert y.shape == (10,)
    assert bool(jnp.all(jnp.isfinite(y)))
    # The conv kernel feeds the hidden map directly -> exactly one ETP relation,
    # carried by the convolution primitive (etp_conv).
    rels = algo.graph.hidden_param_op_relations
    assert len(rels) == 1
    assert 'conv' in str(rels[0].primitive)


def test_docs_etp_primitives_tiny_rnn_mixed_selection():
    """docs/tutorials/etp_primitives.ipynb — mixed ETP / plain primitive selection."""

    class TinyRNN(brainstate.nn.Module):
        def __init__(self, in_dim=4, hid_dim=6):
            super().__init__()
            self.in_dim = in_dim
            self.hid_dim = hid_dim
            # ETP-enabled (online learning via D-RTRL).
            self.W_rec = brainstate.ParamState(brainstate.random.randn(hid_dim, hid_dim) * 0.1)
            # Plain matmul -> learned via BPTT, excluded from online learning.
            self.W_in = brainstate.ParamState(brainstate.random.randn(in_dim, hid_dim) * 0.1)

        def init_state(self, batch_size=None, **kwargs):
            self.h = brainstate.HiddenState(jnp.zeros((batch_size or 1, self.hid_dim)))

        def update(self, x):
            input_drive = x @ self.W_in.value            # NOT marked -> excluded
            rec_drive = braintrace.matmul(self.h.value, self.W_rec.value)  # marked
            self.h.value = jax.nn.tanh(input_drive + rec_drive)
            return self.h.value

    model = TinyRNN(in_dim=4, hid_dim=6)
    algo = braintrace.compile(model, braintrace.D_RTRL, jnp.zeros((2, model.in_dim)), batch_size=2)
    y = algo(jnp.ones((2, model.in_dim)))
    assert y.shape == (2, 6)
    assert bool(jnp.all(jnp.isfinite(y)))
    # Only W_rec is routed through an ETP op -> exactly one relation; W_in excluded.
    rels = algo.graph.hidden_param_op_relations
    assert len(rels) == 1
    assert any('W_rec' in str(p) for p in rels[0].trainable_paths.values())


def test_docs_concepts_gru_net_batched():
    """docs/quickstart/concepts.ipynb — GRUNet via braintrace.compile (batched, bs=1)."""

    class GRUNet(brainstate.nn.Module):
        def __init__(self, n_in, n_rec, n_out):
            super().__init__()
            self.rnn = braintrace.nn.GRUCell(n_in, n_rec)
            self.readout = braintrace.nn.Linear(n_rec, n_out)

        def update(self, x):
            return self.readout(self.rnn(x))

    model = GRUNet(10, 64, 10)
    # The example input carries the batch axis to match batch_size=1.
    trainer = braintrace.compile(model, braintrace.D_RTRL, jnp.zeros((1, 10)), batch_size=1)

    weights = model.states(brainstate.ParamState)

    def loss_fn(x):
        return jnp.mean(trainer(x) ** 2)

    grads = brainstate.transform.grad(loss_fn, weights)(jnp.ones((1, 10)))
    leaves = jax.tree.leaves(grads)
    assert leaves and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    assert len(trainer.graph.hidden_param_op_relations) >= 1


def test_docs_batching_gru_single_sample_unbatched():
    """docs/tutorials/batching.ipynb — single-sample mode (omit batch_size)."""

    class SimpleGRU(brainstate.nn.Module):
        def __init__(self, n_in, n_rec, n_out):
            super().__init__()
            self.rnn = braintrace.nn.GRUCell(n_in, n_rec)
            self.out = braintrace.nn.Linear(n_rec, n_out)

        def update(self, x):
            return self.out(self.rnn(x))

    model = SimpleGRU(10, 64, 5)
    algo = braintrace.compile(model, braintrace.D_RTRL, jnp.zeros(10))
    out = algo(jnp.ones(10))
    assert out.shape == (5,)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_docs_batching_gru_vmap_batched():
    """docs/tutorials/batching.ipynb — vmap mode with a batched example input."""

    class SimpleGRU(brainstate.nn.Module):
        def __init__(self, n_in, n_rec, n_out):
            super().__init__()
            self.rnn = braintrace.nn.GRUCell(n_in, n_rec)
            self.out = braintrace.nn.Linear(n_rec, n_out)

        def update(self, x):
            return self.out(self.rnn(x))

    model = SimpleGRU(10, 64, 5)
    batch_size = 16
    # In vmap mode the example carries the batch axis (axis 0), stripped internally.
    algo_vmapped = braintrace.compile(
        model, braintrace.D_RTRL, jnp.zeros((batch_size, 10)),
        batch_size=batch_size, vmap=True,
    )
    out = algo_vmapped(jnp.ones((batch_size, 10)))
    assert out.shape == (batch_size, 5)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_docs_graphviz_single_layer_rnn_unbatched_structure():
    """docs/tutorials/graph_visualization.ipynb — unbatched structure (etp_mv, (32,))."""

    class SingleLayerRNN(brainstate.nn.Module):
        def __init__(self, n_in, n_rec, n_out):
            super().__init__()
            self.rnn = braintrace.nn.ValinaRNNCell(n_in, n_rec)
            self.out = braintrace.nn.Linear(n_rec, n_out)

        def update(self, x):
            return self.out(self.rnn(x))

    model = SingleLayerRNN(10, 32, 5)
    learner = braintrace.compile(model, braintrace.D_RTRL, jnp.zeros(10))
    graph = learner.graph
    # Documented structure: one hidden group of shape (32,), one matrix-vector relation.
    assert len(graph.hidden_groups) == 1
    assert tuple(graph.hidden_groups[0].varshape) == (32,)
    assert len(graph.hidden_param_op_relations) == 1
    assert 'etp_mv' in str(graph.hidden_param_op_relations[0].primitive)


def test_docs_hidden_states_two_layer_manual_multigroup():
    """docs/tutorials/hidden_states.ipynb — manual two-layer RNN, two hidden groups."""

    class TwoLayerRNN(brainstate.nn.Module):
        def __init__(self, in_size, hidden_size, out_size):
            super().__init__()
            self.w1_in = brainstate.ParamState(brainstate.random.randn(in_size, hidden_size) * 0.01)
            self.w1_rec = brainstate.ParamState(brainstate.random.randn(hidden_size, hidden_size) * 0.01)
            self.h1 = brainstate.HiddenState(jnp.zeros(hidden_size))
            self.w2_in = brainstate.ParamState(brainstate.random.randn(hidden_size, out_size) * 0.01)
            self.w2_rec = brainstate.ParamState(brainstate.random.randn(out_size, out_size) * 0.01)
            self.h2 = brainstate.HiddenState(jnp.zeros(out_size))

        def update(self, x):
            self.h1.value = jax.nn.tanh(
                x @ self.w1_in.value + braintrace.matmul(self.h1.value, self.w1_rec.value)
            )
            self.h2.value = jax.nn.tanh(
                self.h1.value @ self.w2_in.value + braintrace.matmul(self.h2.value, self.w2_rec.value)
            )
            return self.h2.value

    model = TwoLayerRNN(in_size=10, hidden_size=16, out_size=8)
    algo = braintrace.compile(model, braintrace.D_RTRL, jnp.zeros(10), batch_size=1)
    y = algo(jnp.ones(10))
    assert y.shape == (8,)
    assert bool(jnp.all(jnp.isfinite(y)))
    # Differently-shaped hidden states (16 vs 8) -> two distinct groups, two relations
    # (each recurrent weight traced; the two input weights are plain matmuls).
    assert len(algo.graph.hidden_groups) == 2
    assert len(algo.graph.hidden_param_op_relations) == 2
