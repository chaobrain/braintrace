# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Tests for the ETP compiler and graph compilation."""

import jax
import jax.numpy as jnp
import pytest
import brainstate

from braintrace.etp import (
    matmul,
    element_wise,
    etp_matmul_p,
    etp_elemwise_p,
    compile_etp_graph,
    ETPGraph,
)


# ==============================================================================
# Test models using ETP primitives
# ==============================================================================

class SimpleRNN(brainstate.nn.Module):
    """Minimal RNN using etp_matmul."""

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        w = jax.random.normal(jax.random.key(0), (in_size + hidden_size, hidden_size))
        b = jnp.zeros(hidden_size)
        self.W = brainstate.ParamState(w)
        self.b = brainstate.ParamState(b)

    def init_state(self, batch_size=None, **kwargs):
        self.h = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value], axis=-1)
        self.h.value = jax.nn.tanh(matmul(xh, self.W.value, self.b.value))
        return self.h.value


class TwoLayerRNN(brainstate.nn.Module):
    """Two-layer RNN to test multiple etp ops."""

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W1 = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size + hidden_size, hidden_size))
        )
        self.W2 = brainstate.ParamState(
            jax.random.normal(jax.random.key(1), (hidden_size + hidden_size, hidden_size))
        )

    def init_state(self, batch_size=None, **kwargs):
        self.h1 = brainstate.HiddenState(jnp.zeros(self.hidden_size))
        self.h2 = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        xh1 = jnp.concatenate([x, self.h1.value], axis=-1)
        self.h1.value = jax.nn.tanh(matmul(xh1, self.W1.value))
        xh2 = jnp.concatenate([self.h1.value, self.h2.value], axis=-1)
        self.h2.value = jax.nn.tanh(matmul(xh2, self.W2.value))
        return self.h2.value


class ElemWiseRNN(brainstate.nn.Module):
    """RNN with element-wise parameter (diagonal in hidden space)."""

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size, hidden_size))
        )
        self.tau = brainstate.ParamState(jnp.ones(hidden_size))

    def init_state(self, batch_size=None, **kwargs):
        self.h = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        inp = matmul(x, self.W.value)
        decay = element_wise(self.tau.value, fn=jax.nn.sigmoid)
        self.h.value = decay * self.h.value + (1 - decay) * jax.nn.tanh(inp)
        return self.h.value


class WeightFnRNN(brainstate.nn.Module):
    """RNN with weight_fn applied before etp_matmul."""

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.raw_W = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size + hidden_size, hidden_size))
        )

    def init_state(self, batch_size=None, **kwargs):
        self.h = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value], axis=-1)
        # Apply weight_fn outside the primitive
        w = jax.nn.softplus(self.raw_W.value)
        self.h.value = jax.nn.tanh(matmul(xh, w))
        return self.h.value


# ==============================================================================
# Tests
# ==============================================================================

class TestCompileETPGraph:
    """Tests for compile_etp_graph."""

    def test_simple_rnn(self):
        model = SimpleRNN(10, 20)
        model.init_state()
        x = jnp.zeros(10)

        graph = compile_etp_graph(model, x)

        assert isinstance(graph, ETPGraph)
        assert len(graph.hidden_groups) == 1
        assert len(graph.etp_op_relations) == 1
        assert graph.etp_op_relations[0].primitive is etp_matmul_p

    def test_two_layer_rnn(self):
        model = TwoLayerRNN(10, 20)
        model.init_state()
        x = jnp.zeros(10)

        graph = compile_etp_graph(model, x)

        assert len(graph.hidden_groups) >= 1
        assert len(graph.etp_op_relations) == 2
        for rel in graph.etp_op_relations:
            assert rel.primitive is etp_matmul_p

    def test_elemwise_rnn(self):
        model = ElemWiseRNN(10, 20)
        model.init_state()
        x = jnp.zeros(10)

        graph = compile_etp_graph(model, x)

        primitives = {rel.primitive for rel in graph.etp_op_relations}
        assert etp_matmul_p in primitives
        # The etp_elemwise_p may or may not be connected to hidden states
        # depending on the computation structure

    def test_weight_fn_rnn(self):
        """Verify that weight_fn before primitive is handled correctly."""
        model = WeightFnRNN(10, 20)
        model.init_state()
        x = jnp.zeros(10)

        graph = compile_etp_graph(model, x)

        assert len(graph.etp_op_relations) >= 1
        rel = graph.etp_op_relations[0]
        assert rel.primitive is etp_matmul_p
        # The weight path should trace back to raw_W
        assert 'raw_W' in str(rel.weight_path)

    def test_hidden_perturbation_included(self):
        model = SimpleRNN(10, 20)
        model.init_state()
        x = jnp.zeros(10)

        graph = compile_etp_graph(model, x, include_hidden_perturb=True)
        assert graph.hidden_perturb is not None

        graph2 = compile_etp_graph(model, x, include_hidden_perturb=False)
        assert graph2.hidden_perturb is None

    def test_graph_has_correct_structure(self):
        model = SimpleRNN(10, 20)
        model.init_state()
        x = jnp.zeros(10)

        graph = compile_etp_graph(model, x)

        # Check hidden group
        assert len(graph.hidden_groups) == 1
        group = graph.hidden_groups[0]
        assert group.num_state >= 1  # number of hidden state arrays in group

        # Check relation
        rel = graph.etp_op_relations[0]
        assert rel.x_var is not None  # matmul has input
        assert len(rel.hidden_groups) == 1
        assert len(rel.y_to_hidden_group_jaxprs) == 1
        assert len(rel.connected_hidden_paths) >= 1
