# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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


import unittest
from pprint import pprint

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import pytest

import braintrace
from braintrace._testing.models import (
    IF_Delta_Dense_Layer,
    LIF_ExpCo_Dense_Layer,
    ALIF_ExpCo_Dense_Layer,
    LIF_ExpCu_Dense_Layer,
    LIF_STDExpCu_Dense_Layer,
    LIF_STPExpCu_Dense_Layer,
    ALIF_ExpCu_Dense_Layer,
    ALIF_Delta_Dense_Layer,
    ALIF_STDExpCu_Dense_Layer,
    ALIF_STPExpCu_Dense_Layer,
)


class TestCompileGraphRNN(unittest.TestCase):
    def test_gru_one_layer(self):
        n_in = 3
        n_out = 4

        gru = braintrace.nn.GRUCell(n_in, n_out)
        brainstate.nn.init_all_states(gru)

        input = brainstate.random.rand(n_in)
        graph = braintrace.compile_etrace_graph(gru, input, include_hidden_perturb=False)

        self.assertTrue(isinstance(graph, braintrace.ETraceGraph))
        self.assertTrue(graph.module_info.num_var_out == 1)
        self.assertTrue(len(graph.module_info.compiled_model_states) == 4)
        self.assertTrue(len(graph.hidden_groups) == 1)

        param_states = gru.states(brainstate.ParamState)
        self.assertTrue(len(param_states) == 3)
        # Only Wz and Wh feed directly into h. Wr's output reaches h only via
        # Wh's matmul (another non-gradient-enabled ETP primitive), so ETP
        # cannot record it as a temporal relation without double-counting.
        self.assertTrue(len(graph.hidden_param_op_relations) == 2)

        pprint(graph)

    def test_lru_one_layer(self):
        n_in = 3
        n_out = 4

        lru = braintrace.nn.LRUCell(n_in, n_out)
        brainstate.nn.init_all_states(lru)

        input = brainstate.random.rand(n_in)
        graph = braintrace.compile_etrace_graph(lru, input, include_hidden_perturb=False)

        self.assertTrue(len(graph.hidden_groups) == 1)
        self.assertTrue(len(graph.hidden_groups[0].hidden_paths) == 2)

        for relation in graph.hidden_param_op_relations:
            if relation.path[0] in ['C_re', 'C_im', 'D']:
                self.assertTrue(len(relation.connected_hidden_paths) == 0)

        # pprint(graph)

    def test_lstm_one_layer(self):
        n_in = 3
        n_out = 4

        lstm = braintrace.nn.LSTMCell(n_in, n_out)
        brainstate.nn.init_all_states(lstm)

        input = brainstate.random.rand(n_in)
        graph = braintrace.compile_etrace_graph(lstm, input, include_hidden_perturb=False)

        self.assertTrue(isinstance(graph, braintrace.ETraceGraph))
        self.assertTrue(graph.module_info.num_var_out == 1)
        self.assertTrue(len(graph.hidden_groups) == 1)
        self.assertTrue(len(graph.hidden_groups[0].hidden_paths) == 2)
        self.assertTrue(len(graph.module_info.compiled_model_states) == 6)

        hid_states = lstm.states(brainstate.HiddenState)
        self.assertTrue(len(hid_states) == len(graph.hid_path_to_group))

        param_states = lstm.states(brainstate.ParamState)
        self.assertTrue(len(param_states) == len(graph.hidden_param_op_relations))

        hidden_paths = set(graph.hidden_groups[0].hidden_paths)
        for relation in graph.hidden_param_op_relations:
            if relation.path[0] == 'Wo':
                self.assertTrue(set(relation.connected_hidden_paths) == set([('h',)]))
            else:
                self.assertTrue(set(relation.connected_hidden_paths) == hidden_paths)

        # pprint(graph)

    def test_lstm_two_layers(self):
        n_in = 3
        n_out = 4

        net = brainstate.nn.Sequential(
            braintrace.nn.LSTMCell(n_in, n_out),
            brainstate.nn.ReLU(),
            braintrace.nn.LSTMCell(n_out, n_in),
        )
        brainstate.nn.init_all_states(net)

        input = brainstate.random.rand(n_in)
        graph = braintrace.compile_etrace_graph(net, input, include_hidden_perturb=False)

        self.assertTrue(isinstance(graph, braintrace.ETraceGraph))
        self.assertTrue(graph.module_info.num_var_out == 1)
        self.assertTrue(len(graph.hidden_groups) == 2)
        self.assertTrue(len(graph.hidden_groups[0].hidden_paths) == 2)
        self.assertTrue(len(graph.hidden_groups[1].hidden_paths) == 2)

        hidden_group1_path = {('layers', 0, 'c'), ('layers', 0, 'h')}
        hidden_group2_path = {('layers', 2, 'c'), ('layers', 2, 'h')}

        for relation in graph.hidden_param_op_relations:
            if relation.path[1] == 0:
                if relation.path[2] != 'Wo':
                    self.assertTrue(set(relation.connected_hidden_paths) == hidden_group1_path)
            if relation.path[1] == 2:
                if relation.path[2] != 'Wo':
                    self.assertTrue(set(relation.connected_hidden_paths) == hidden_group2_path)

        # pprint(graph)

    def test_lru_two_layers(self):
        n_in = 3
        n_out = 4

        net = brainstate.nn.Sequential(
            braintrace.nn.LRUCell(n_in, n_out),
            brainstate.nn.ReLU(),
            braintrace.nn.LRUCell(n_in, n_out),
        )
        brainstate.nn.init_all_states(net)

        input = brainstate.random.rand(n_in)
        graph = braintrace.compile_etrace_graph(net, input, include_hidden_perturb=False)

        self.assertTrue(len(graph.hidden_groups) == 2)
        self.assertTrue(len(graph.hidden_groups[0].hidden_paths) == 2)
        self.assertTrue(len(graph.hidden_groups[1].hidden_paths) == 2)
        self.assertTrue(len(graph.hidden_param_op_relations) == 10)

        layer1_hiddens = {('layers', 0, 'h_im'), ('layers', 0, 'h_re')}
        layer2_hiddens = {('layers', 2, 'h_im'), ('layers', 2, 'h_re')}

        for relation in graph.hidden_param_op_relations:
            if relation.path[1] == 0 and relation.path[2] not in ['B_im', 'B_re']:
                self.assertTrue(set(relation.connected_hidden_paths) == layer1_hiddens)
            if relation.path[1] == 2 and relation.path[2] not in ['B_im', 'B_re']:
                self.assertTrue(set(relation.connected_hidden_paths) == layer2_hiddens)

    def test_lru_two_layers_v2(self):
        # Variant where input and output sizes match. The forward BFS from
        # layer-0 output ops (C_re/C_im/D) toward layer-2 hiddens crosses
        # layer-2's B_re/B_im matmuls — another non-gradient-enabled ETP
        # primitive — and is therefore blocked. Layer-0 output ops correctly
        # retain only their layer-1 hidden connections.
        n_in = 4
        n_out = 4

        net = brainstate.nn.Sequential(
            braintrace.nn.LRUCell(n_in, n_out),
            brainstate.nn.ReLU(),
            braintrace.nn.LRUCell(n_in, n_out),
        )
        brainstate.nn.init_all_states(net)

        input = brainstate.random.rand(n_in)
        graph = braintrace.compile_etrace_graph(net, input, include_hidden_perturb=False)

        self.assertTrue(len(graph.hidden_groups) == 2)
        self.assertTrue(len(graph.hidden_groups[0].hidden_paths) == 2)
        self.assertTrue(len(graph.hidden_groups[1].hidden_paths) == 2)
        self.assertTrue(len(graph.hidden_param_op_relations) == 10)

        layer1_hiddens = {('layers', 0, 'h_im'), ('layers', 0, 'h_re')}
        layer2_hiddens = {('layers', 2, 'h_im'), ('layers', 2, 'h_re')}

        for relation in graph.hidden_param_op_relations:
            if relation.path[1] == 0 and relation.path[2] not in ['B_im', 'B_re']:
                self.assertTrue(set(relation.connected_hidden_paths) == layer1_hiddens)
            if relation.path[1] == 2 and relation.path[2] not in ['B_im', 'B_re']:
                self.assertTrue(set(relation.connected_hidden_paths) == layer2_hiddens)


class TestCompileGraphSNN(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        brainstate.environ.set(dt=0.1 * u.ms)

    def test_if_delta_dense(self):
        n_in = 3
        n_rec = 4

        net = IF_Delta_Dense_Layer(n_in, n_rec)
        brainstate.nn.init_all_states(net)

        input = brainstate.random.rand(n_in)
        graph = braintrace.compile_etrace_graph(net, input, include_hidden_perturb=False)

        pprint(graph)
        pass

    def test_lif_expco_dense_layer(self):
        n_in = 3
        n_rec = 4

        net = LIF_ExpCo_Dense_Layer(n_in, n_rec)
        brainstate.nn.init_all_states(net)

        input = brainstate.random.rand(n_in)
        graph = braintrace.compile_etrace_graph(net, input, include_hidden_perturb=False)

        pprint(graph)

    def test_alif_expco_dense_layer(self):
        n_in = 3
        n_rec = 4

        net = ALIF_ExpCo_Dense_Layer(n_in, n_rec)
        brainstate.nn.init_all_states(net)

        input = brainstate.random.rand(n_in)
        graph = braintrace.compile_etrace_graph(net, input, include_hidden_perturb=False)

        pprint(graph)

    def test_lif_expcu_dense_layer(self):
        n_in = 3
        n_rec = 4

        net = LIF_ExpCu_Dense_Layer(n_in, n_rec)
        brainstate.nn.init_all_states(net)

        input = brainstate.random.rand(n_in)
        graph = braintrace.compile_etrace_graph(net, input, include_hidden_perturb=False)

        pprint(graph)

    def test_lif_std_expcu_dense_layer(self):
        n_in = 3
        n_rec = 4

        net = LIF_STDExpCu_Dense_Layer(n_in, n_rec)
        brainstate.nn.init_all_states(net)

        input = brainstate.random.rand(n_in)
        graph = braintrace.compile_etrace_graph(net, input, include_hidden_perturb=False)

        pprint(graph)

    def test_lif_stp_expcu_dense_layer(self):
        n_in = 3
        n_rec = 4

        net = LIF_STPExpCu_Dense_Layer(n_in, n_rec)
        brainstate.nn.init_all_states(net)

        input = brainstate.random.rand(n_in)
        graph = braintrace.compile_etrace_graph(net, input, include_hidden_perturb=False)

        pprint(graph)

    def test_alif_expcu_dense_layer(self):
        n_in = 3
        n_rec = 4

        net = ALIF_ExpCu_Dense_Layer(n_in, n_rec)
        brainstate.nn.init_all_states(net)

        input = brainstate.random.rand(n_in)
        graph = braintrace.compile_etrace_graph(net, input, include_hidden_perturb=False)

        pprint(graph)

    def test_alif_delta_dense_layer(self):
        n_in = 3
        n_rec = 4

        net = ALIF_Delta_Dense_Layer(n_in, n_rec)
        brainstate.nn.init_all_states(net)

        input = brainstate.random.rand(n_in)
        graph = braintrace.compile_etrace_graph(net, input, include_hidden_perturb=False)

        pprint(graph)

    def test_alif_std_expcu_dense_layer(self):
        n_in = 3
        n_rec = 4

        net = ALIF_STDExpCu_Dense_Layer(n_in, n_rec)
        brainstate.nn.init_all_states(net)

        input = brainstate.random.rand(n_in)
        graph = braintrace.compile_etrace_graph(net, input, include_hidden_perturb=False)

        pprint(graph)

    def test_alif_stp_expcu_dense_layer(self):
        n_in = 3
        n_rec = 4

        net = ALIF_STPExpCu_Dense_Layer(n_in, n_rec)
        brainstate.nn.init_all_states(net)

        input = brainstate.random.rand(n_in)
        graph = braintrace.compile_etrace_graph(net, input, include_hidden_perturb=False)

        pprint(graph)


class TestStateConsistency(unittest.TestCase):
    pass


class TestVmappedModelCompilation:
    """A model applying `jax.vmap` over a per-sample ETP op inside update()
    must compile identically to the natively batched model (spec Phase 0)."""

    B, N_IN, N_REC = 4, 3, 8

    class _NativeCell(brainstate.nn.Module):
        def __init__(self, n_in, n_rec):
            super().__init__()
            self.w = brainstate.ParamState(
                brainstate.random.randn(n_in + n_rec, n_rec) * 0.1)

        def init_state(self, batch_size=None, **kw):
            n_rec = self.w.value.shape[1]
            self.h = brainstate.HiddenState(jnp.zeros((batch_size, n_rec)))

        def update(self, x):
            xh = jnp.concatenate([x, self.h.value], axis=-1)
            self.h.value = jnp.tanh(braintrace.matmul(xh, self.w.value))
            return self.h.value

    class _VmappedCell(brainstate.nn.Module):
        def __init__(self, n_in, n_rec):
            super().__init__()
            self.w = brainstate.ParamState(
                brainstate.random.randn(n_in + n_rec, n_rec) * 0.1)

        def init_state(self, batch_size=None, **kw):
            n_rec = self.w.value.shape[1]
            self.h = brainstate.HiddenState(jnp.zeros((batch_size, n_rec)))

        def update(self, x):
            xh = jnp.concatenate([x, self.h.value], axis=-1)
            y = jax.vmap(lambda row: braintrace.matmul(row, self.w.value))(xh)
            self.h.value = jnp.tanh(y)
            return self.h.value

    def _graph_for(self, cell_cls):
        cell = cell_cls(self.N_IN, self.N_REC)
        brainstate.nn.init_all_states(cell, batch_size=self.B)
        x = brainstate.random.randn(self.B, self.N_IN)
        return braintrace.compile_etrace_graph(cell, x)

    def test_vmapped_cell_relation_parity_with_native(self):
        from braintrace._op.dense import etp_mm_p
        g_native = self._graph_for(self._NativeCell)
        g_vmapped = self._graph_for(self._VmappedCell)
        assert len(g_native.hidden_param_op_relations) == 1
        assert len(g_vmapped.hidden_param_op_relations) == 1
        assert g_vmapped.hidden_param_op_relations[0].primitive is etp_mm_p
        assert (g_vmapped.hidden_param_op_relations[0].primitive
                is g_native.hidden_param_op_relations[0].primitive)

    def test_vmapped_cell_drtrl_gradient_parity_with_native(self):
        def build_and_grads(cell_cls):
            with brainstate.random.seed_context(42):
                model = cell_cls(self.N_IN, self.N_REC)
            learner = braintrace.compile(
                model, braintrace.D_RTRL,
                jnp.zeros((self.B, self.N_IN)), batch_size=self.B)
            weights = model.states(brainstate.ParamState)
            with brainstate.random.seed_context(7):
                xs = brainstate.random.randn(5, self.B, self.N_IN)

            def total_loss(xs):
                def step(carry, x):
                    out = learner(x)
                    return carry, jnp.mean(jnp.asarray(out) ** 2)
                _, ls = brainstate.transform.scan(step, None, xs)
                return jnp.sum(ls)

            return brainstate.transform.grad(total_loss, weights)(xs)

        g_native = build_and_grads(self._NativeCell)
        g_vmapped = build_and_grads(self._VmappedCell)
        for a, b in zip(jax.tree.leaves(g_native), jax.tree.leaves(g_vmapped)):
            assert jnp.allclose(a, b, atol=1e-5), (a - b)


class TestCompileEtraceGraphSparseNGuards(unittest.TestCase):
    """``compile_etrace_graph(..., sparse_n=...)`` must not degrade silently.

    The compiler is a *public* entry point, and it takes ``sparse_n`` and
    ``include_recurrent_mixing`` as independent keywords. Nothing about the
    combination ``sparse_n=2, include_recurrent_mixing=False`` fails: the mixing
    primitive stays out of the transition, the position analysis finds no
    cross-position coupling, every neighbourhood collapses to ``K = 1`` and the
    caller gets the diagonal rule with a SnAp label on it. Same class of failure
    as the algorithm-level guardrails, one layer lower.
    """

    @staticmethod
    def _model():
        class Net(brainstate.nn.Module):
            def __init__(self, n_in=3, n_rec=4):
                super().__init__()
                with brainstate.random.seed_context(0):
                    self.w = brainstate.ParamState(
                        brainstate.random.randn(n_rec, n_rec) * 0.5)
                    self.win = brainstate.ParamState(
                        brainstate.random.randn(n_in, n_rec) * 0.5)
                self.h = brainstate.HiddenState(jnp.zeros((n_rec,)))

            def update(self, x):
                rec = braintrace.matmul(self.h.value, self.w.value)
                self.h.value = jax.nn.tanh(rec + x @ self.win.value)
                return self.h.value

        model = Net()
        brainstate.nn.init_all_states(model)
        return model, jnp.zeros((3,))

    def test_sparse_n_without_recurrent_mixing_raises(self):
        model, x = self._model()
        with self.assertRaises(braintrace.NotSupportedError) as ctx:
            braintrace.compile_etrace_graph(model, x, sparse_n=2)
        msg = str(ctx.exception)
        assert 'include_recurrent_mixing' in msg and 'K=1' in msg

    def test_sparse_n_with_recurrent_mixing_widens(self):
        model, x = self._model()
        graph = braintrace.compile_etrace_graph(
            model, x, sparse_n=2, include_recurrent_mixing=True)
        # negative control for the test above: the same order *does* widen once
        # the mixing primitive is in the transition, so the guard is rejecting a
        # degenerate configuration rather than a valid one.
        assert [g.snap.num_neighbour for g in graph.hidden_groups] == [4]

    def test_boolean_sparse_n_is_rejected(self):
        # ``bool`` is a subclass of ``int``, so ``sparse_n=True`` would pass an
        # ``isinstance(..., int)`` check and be read as SnAp-1.
        model, x = self._model()
        with self.assertRaises(TypeError):
            braintrace.compile_etrace_graph(
                model, x, sparse_n=True, include_recurrent_mixing=True)

    def test_non_integer_and_out_of_range_sparse_n_are_rejected(self):
        model, x = self._model()
        with self.assertRaises(TypeError):
            braintrace.compile_etrace_graph(
                model, x, sparse_n=2.0, include_recurrent_mixing=True)
        with self.assertRaises(ValueError):
            braintrace.compile_etrace_graph(
                model, x, sparse_n=0, include_recurrent_mixing=True)

    def test_jacobian_ceiling_is_reachable_from_the_compiler(self):
        model, x = self._model()
        with self.assertRaises(braintrace.NotSupportedError) as ctx:
            braintrace.compile_etrace_graph(
                model, x, sparse_n=2, include_recurrent_mixing=True,
                snap_max_jacobian_elements=8)
        assert 'snap_max_jacobian_elements' in str(ctx.exception)


# ---------------------------------------------------------------------------
# #199: a weight whose credit reaches the outputs only through a non-hidden state
# ---------------------------------------------------------------------------

def _delay_net(kind='short', direct=False, n=4):
    """Two layers; layer 1 reaches layer 2 through a one-step delay line (#199).

    ``kind``: ``'short'`` holds the line in a ``ShortTermState``, ``'hidden'``
    in a ``HiddenState``, ``'none'`` has no line, ``'dead'`` drops layer 1's
    output and keeps an unrelated ``ShortTermState`` counter. ``direct`` also
    adds layer 1's output to the model output in the same step.
    """

    class Net(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = brainstate.ParamState(0.8 * jax.random.normal(jax.random.PRNGKey(0), (3, n)))
            self.w2 = brainstate.ParamState(0.8 * jax.random.normal(jax.random.PRNGKey(1), (n, n)))
            self.h1 = brainstate.HiddenState(jnp.zeros((1, n)))
            self.h2 = brainstate.HiddenState(jnp.zeros((1, n)))
            if kind in ('short', 'dead'):
                self.ring = brainstate.ShortTermState(jnp.zeros((1, n)))
            elif kind == 'hidden':
                self.ring = brainstate.HiddenState(jnp.zeros((1, n)))

        def update(self, x):
            self.h1.value = 0.9 * self.h1.value + braintrace.matmul(x, self.w1.value)
            y1 = jnp.tanh(self.h1.value)
            if kind == 'none':
                inp = y1
            elif kind == 'dead':
                inp = jnp.zeros_like(y1)
                self.ring.value = self.ring.value + 1.0
            else:
                inp = self.ring.value  # written one step ago
                self.ring.value = y1
            self.h2.value = 0.9 * self.h2.value + braintrace.matmul(inp, self.w2.value)
            out = jnp.tanh(self.h2.value)
            return out + 0.1 * y1 if direct else out

    return Net


class _ReadoutNet(brainstate.nn.Module):
    """A readout weight applied inside the step, plus a buffer nobody reads back."""

    def __init__(self, n=4):
        super().__init__()
        self.w1 = brainstate.ParamState(jax.random.normal(jax.random.PRNGKey(0), (3, n)))
        self.w_out = brainstate.ParamState(jax.random.normal(jax.random.PRNGKey(1), (n, 2)))
        self.h1 = brainstate.HiddenState(jnp.zeros((1, n)))
        self.log = brainstate.ShortTermState(jnp.zeros((1, n)))

    def update(self, x):
        self.h1.value = 0.9 * self.h1.value + braintrace.matmul(x, self.w1.value)
        self.log.value = self.h1.value
        return braintrace.matmul(jnp.tanh(self.h1.value), self.w_out.value)


def _zero_records(model_cls, x=None):
    import warnings
    from braintrace import DiagnosticKind
    model = model_cls()
    brainstate.nn.init_all_states(model)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        graph = braintrace.compile_etrace_graph(model, jnp.zeros((1, 3)) if x is None else x)
    return [r for r in graph.diagnostics if r.kind is DiagnosticKind.ONLINE_GRADIENT_ZERO]


class TestShortTermOnlyPath:
    """#199: a weight whose hidden states reach the step outputs only through a
    write into a non-hidden state (a delay line in a ``ShortTermState``) gets a
    zero online gradient; ``ONLINE_GRADIENT_ZERO`` says so."""

    def test_short_term_delay_line_is_reported(self):
        from braintrace import DiagnosticLevel
        (rec,) = _zero_records(_delay_net('short'))
        assert rec.level is DiagnosticLevel.WARNING
        assert rec.weight_path == ('w1',)
        assert ('h1',) in rec.hidden_paths
        assert rec.context['through'] == (('ring',),)
        for part in ("('ring',)", 'ShortTermState', 'HiddenState', 'BPTT'):
            assert part in rec.message

    def test_the_reported_gradient_is_zero(self):
        from braintrace._testing.oracle import bptt_param_gradients, chunked_online_param_gradients
        make = _delay_net('short')
        with brainstate.environ.context(precision=64):
            xs = jax.random.uniform(jax.random.PRNGKey(7), (8, 1, 3))
            ref = bptt_param_gradients(make, xs)
            on = chunked_online_param_gradients(
                make, xs, chunk_size=1,
                algo_factory=lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'))
        assert float(jnp.abs(ref[('w1',)]).max()) > 0
        assert float(jnp.abs(on[('w1',)]).max()) == 0.0
        assert float(jnp.abs(on[('w2',)]).max()) > 0

    @pytest.mark.parametrize('model', [
        _delay_net('hidden'), _delay_net('none'), _delay_net('short', direct=True), _ReadoutNet,
        _delay_net('dead'),
    ], ids=['hidden-line', 'no-line', 'direct-path-too', 'readout', 'dead-weight-no-line'])
    def test_other_paths_are_quiet(self, model):
        assert _zero_records(model) == []

    @pytest.mark.parametrize('algorithm', [
        lambda m: braintrace.D_RTRL(m, vjp_method='single-step'),
        lambda m: braintrace.D_RTRL(m, vjp_method='multi-step'),
        lambda m: braintrace.EProp(m),
    ], ids=['d_rtrl-single', 'd_rtrl-multi', 'eprop'])
    def test_every_online_algorithm_reports_it(self, algorithm):
        model = _delay_net('short')()
        brainstate.nn.init_all_states(model)
        with pytest.warns(UserWarning, match=r"online gradient of weight \('w1',\) is zero"):
            algorithm(model).compile_graph(jnp.zeros((1, 3)))
