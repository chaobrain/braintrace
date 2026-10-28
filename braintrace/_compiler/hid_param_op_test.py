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


from pprint import pprint

import brainstate
import jax.numpy as jnp
import pytest
import brainunit as u

import braintrace
from braintrace import find_hidden_param_op_relations_from_module
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


class TestFindRelationsFromModule:
    def test_gru_one_layer(self):
        n_in = 3
        n_out = 4

        gru = braintrace.nn.GRUCell(n_in, n_out)
        brainstate.nn.init_all_states(gru)

        input = brainstate.random.rand(n_in)
        relations = find_hidden_param_op_relations_from_module(gru, input)

        print()
        pprint(relations)
        # Only Wz and Wh feed directly into h. Wr's output reaches h only
        # through Wh's matmul (another non-gradient-enabled ETP primitive),
        # so it must not register a relation — ETP cannot decompose the
        # weight -> weight -> hidden pathway without double-counting.
        assert (len(relations) == 2)
        for relation in relations:
            assert len(relation.connected_hidden_paths) == 1
            assert relation.connected_hidden_paths[0] == ('h',)

    @pytest.mark.parametrize(
        'cls',
        [
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
        ]
    )
    def test_snn_single_layer(self, cls):
        n_in = 3
        n_out = 4
        input = brainstate.random.rand(n_in)

        print(cls)

        with brainstate.environ.context(dt=0.1 * u.ms):
            layer = cls(n_in, n_out)
            brainstate.nn.init_all_states(layer)
            relations = find_hidden_param_op_relations_from_module(layer, input)
            print(relations)

    @pytest.mark.parametrize(
        'cls',
        [
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
        ]
    )
    def test_snn_two_layers(self, cls):
        n_in = 3
        n_out = 4
        input = brainstate.random.rand(n_in)

        print()
        print(cls)

        with brainstate.environ.context(dt=0.1 * u.ms):
            layer = brainstate.nn.Sequential(cls(n_in, n_out), cls(n_out, n_out))
            brainstate.nn.init_all_states(layer)
            relations = find_hidden_param_op_relations_from_module(layer, input)
            pprint(relations)


class TestTrainableDictsDefault:
    """The new trainable_* dict fields default to empty dicts and do not
    affect the construction of existing relations. (Later tasks populate
    them during compilation.)"""

    def test_trainable_vars_default_empty(self):
        # Build a minimal relation directly; we just check the new field exists.
        from braintrace._compiler.hid_param_op import HiddenParamOpRelation
        # NamedTuple defaults: the new fields must exist and default to empty dicts.
        fields = HiddenParamOpRelation._fields
        assert 'trainable_vars' in fields
        assert 'trainable_paths' in fields
        assert 'trainable_leaf_indices' in fields
        assert 'trainable_param_states' in fields
        assert 'trainable_processing_chains' in fields

    def test_trainable_dicts_can_be_set(self):
        # Smoke: construct a relation with all required fields, including the
        # new ones populated, and read them back.
        from braintrace._compiler.hid_param_op import HiddenParamOpRelation
        r = HiddenParamOpRelation(
            primitive=None,
            x_var=None,
            y_var=None,
            hidden_groups=[],
            y_to_hidden_group_jaxprs=[],
            connected_hidden_paths=[],
            eqn_params={},
            path_classification={},
            trainable_vars={'weight': 'v'},
            trainable_paths={'weight': ('w',)},
            trainable_leaf_indices={'weight': 0},
            trainable_param_states={'weight': None},
            trainable_processing_chains={'weight': ()},
        )
        assert r.trainable_vars == {'weight': 'v'}
        assert r.trainable_paths == {'weight': ('w',)}
        assert r.trainable_leaf_indices == {'weight': 0}
        assert r.trainable_param_states == {'weight': None}
        assert r.trainable_processing_chains == {'weight': ()}


class TestTrainableInvarsPopulatedForDense:
    """The compiler fills the trainable_* dicts from the primitive's
    registered trainable-invars layout (``get_trainable_invars``). For a
    single-weight primitive such as ``etp_mm_p`` this yields the single-key
    {'weight': ...} form."""

    def test_single_weight_populated_for_mm(self):
        import brainstate
        import jax.numpy as jnp
        import braintrace
        from braintrace._compiler import (
            find_hidden_param_op_relations_from_module,
        )

        class Cell(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((4, 4)))
                self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

            def update(self, x):
                self.h.value = jnp.tanh(braintrace.matmul(self.h.value, self.w.value))
                return self.h.value

        cell = Cell()
        brainstate.nn.init_all_states(cell, batch_size=1)
        relations = find_hidden_param_op_relations_from_module(
            cell, jnp.zeros((1, 4))
        )
        assert len(relations) == 1
        r = relations[0]
        assert list(r.trainable_vars.keys()) == ['weight']
        assert r.trainable_vars['weight'] is not None
        assert r.trainable_paths['weight'] == ('w',)
        assert r.trainable_leaf_indices['weight'] == 0
        assert r.trainable_processing_chains['weight'] == ()
        assert r.trainable_param_states['weight'] is not None


class TestYToHiddenJaxprConstvarSplit:
    """The ``y -> hidden group`` jaxprs keep a recoverable const/invar split.

    Since JAX 0.11 an open jaxpr with symbolic-only constvars reports
    ``constvars == []``, so the split is recovered from the invar count instead:
    these jaxprs take exactly one invar, the relation's ``y_var``. The
    executable counterpart of this assertion lives in
    ``_testing/scenario_catalog_test.py::test_gru_y_to_hidden_groups_executes``.
    """

    def test_y_is_the_sole_invar_and_the_rest_are_constvars(self):
        import jax.numpy as jnp
        from braintrace._compatible_imports import jaxpr_all_invars, split_jaxpr_invars

        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        relations = find_hidden_param_op_relations_from_module(gru, jnp.zeros(3))

        assert len(relations) > 0
        checked = 0
        for r in relations:
            for jaxpr in r.y_to_hidden_group_jaxprs:
                constvars, invars = split_jaxpr_invars(jaxpr, 1)
                assert invars == [r.y_var]
                assert constvars == jaxpr_all_invars(jaxpr)[:-1]
                checked += 1
        assert checked > 0


class _ReceptorChannels(brainstate.nn.Module):
    """``k`` input projections; projection ``i`` drives receptor channel ``i``.

    With ``grouped=True`` the channels are one ``(n, k)`` ``HiddenGroupState``
    (an ETP output of shape ``(n,)`` drives one column of it); otherwise they
    are ``k`` separate ``(n,)`` ``HiddenState``s, the reference layout.
    """

    def __init__(self, k, grouped, n_in=3, n=5):
        super().__init__()
        self.k, self.grouped = k, grouped
        with brainstate.random.seed_context(3):
            self.W = [brainstate.ParamState(brainstate.random.randn(n_in, n) * 0.5) for _ in range(k)]
            self.decay = brainstate.random.uniform(0.5, 0.95, (n, k))
            self.gain = brainstate.random.uniform(0.05, 0.2, (n, k))
        self.V = brainstate.HiddenState(jnp.zeros(n))
        if grouped:
            self.I = brainstate.HiddenGroupState(jnp.zeros((n, k)))
        else:
            self.I = [brainstate.HiddenState(jnp.zeros(n)) for _ in range(k)]

    def update(self, x):
        # As NEST's exact multisynapse step: the membrane integrates the
        # channel currents of the previous step, then the arrivals jump them.
        currents = [braintrace.matmul(x, w.value) for w in self.W]
        if self.grouped:
            columns = [self.I.value[..., i] for i in range(self.k)]
            self.I.value = self.I.value * self.decay + jnp.stack(currents, axis=-1)
        else:
            columns = [self.I[i].value for i in range(self.k)]
            for i in range(self.k):
                self.I[i].value = self.I[i].value * self.decay[:, i] + currents[i]
        # Scale each channel before the sum, like NEST's propagator p21: an
        # equation may consume only one hidden state of the separate layout.
        total = self.gain[:, 0] * columns[0]
        for i in range(1, self.k):
            total = total + self.gain[:, i] * columns[i]
        self.V.value = 0.9 * self.V.value + total
        return jnp.tanh(self.V.value)


class _GroupALIF(brainstate.nn.Module):
    """The canonical ``HiddenGroupState``: ALIF ``(V, a)`` per neuron, input into ``V``.

    With ``grouped=False`` the components are two ``(n,)`` ``HiddenState``s.
    """

    def __init__(self, grouped, n_in=3, n=5):
        super().__init__()
        self.grouped = grouped
        with brainstate.random.seed_context(4):
            self.W = brainstate.ParamState(brainstate.random.randn(n_in, n) * 0.5)
        if grouped:
            self.st = brainstate.HiddenGroupState(jnp.zeros((n, 2)))
        else:
            self.V = brainstate.HiddenState(jnp.zeros(n))
            self.a = brainstate.HiddenState(jnp.zeros(n))

    def update(self, x):
        current = braintrace.matmul(x, self.W.value)
        if self.grouped:
            V, a = self.st.value[..., 0], self.st.value[..., 1]
        else:
            V, a = self.V.value, self.a.value
        V = 0.9 * V + current - a
        a = 0.95 * a + 0.1 * V
        if self.grouped:
            self.st.value = jnp.stack([V, a], axis=-1)
        else:
            self.V.value, self.a.value = V, a
        return jnp.tanh(V)


class _SummedMismatch(brainstate.nn.Module):
    """An ETP output of shape ``(3,)`` that reaches a ``(4,)`` hidden state only through a sum."""

    def __init__(self, grouped):
        super().__init__()
        self.grouped = grouped
        self.W = brainstate.ParamState(jnp.ones((2, 3)))
        self.h = (brainstate.HiddenGroupState(jnp.zeros((4, 2))) if grouped
                  else brainstate.HiddenState(jnp.zeros(4)))

    def update(self, x):
        drive = jnp.sum(braintrace.matmul(x, self.W.value))
        self.h.value = 0.9 * self.h.value + drive
        return self.h.value


def _shape_mismatch_records(reporter):
    return [r for r in reporter.records()
            if r.kind == braintrace.DiagnosticKind.RELATION_EXCLUDED_SHAPE_MISMATCH]


class TestHiddenGroupStateRelations:
    """An ETP output of the group's varshape drives one component of a
    ``HiddenGroupState`` (#206): the relation is kept, so the model compiles,
    and its online gradients equal those of the same model with one
    ``HiddenState`` per component."""

    @staticmethod
    def _inputs(n_in=3, steps=6):
        with brainstate.random.seed_context(1):
            return brainstate.random.rand(steps, n_in)

    @pytest.mark.parametrize('k', [1, 2, 3])
    def test_receptor_channels_relate_to_the_group_state(self, k):
        from braintrace._compiler.diagnostics import diagnostic_context
        model = _ReceptorChannels(k, grouped=True)
        brainstate.nn.init_all_states(model)
        with diagnostic_context() as reporter:
            relations = find_hidden_param_op_relations_from_module(model, self._inputs()[0])
        assert not _shape_mismatch_records(reporter)
        assert len(relations) == k
        for relation in relations:
            assert ('I',) in relation.connected_hidden_paths

    def test_alif_group_state_relates_to_its_input(self):
        from braintrace._compiler.diagnostics import diagnostic_context
        model = _GroupALIF(grouped=True)
        brainstate.nn.init_all_states(model)
        with diagnostic_context() as reporter:
            relations = find_hidden_param_op_relations_from_module(model, self._inputs()[0])
        assert not _shape_mismatch_records(reporter)
        assert len(relations) == 1
        assert relations[0].connected_hidden_paths == [('st',)]

    @pytest.mark.parametrize('factory', [
        lambda: _ReceptorChannels(3, grouped=True),
        lambda: _GroupALIF(grouped=True),
    ], ids=['receptor_channels', 'alif'])
    @pytest.mark.parametrize('algorithm', ['d_rtrl', 'eprop'])
    def test_group_state_models_compile(self, factory, algorithm):
        model = factory()
        brainstate.nn.init_all_states(model)
        learner = braintrace.compile(model, algorithm, self._inputs()[0])  # raised before #206
        kinds = [d.kind for d in learner.report.diagnostics]
        assert braintrace.DiagnosticKind.RELATION_EXCLUDED_SHAPE_MISMATCH not in kinds

    @pytest.mark.parametrize('grouped, separate', [
        (lambda: _ReceptorChannels(3, grouped=True), lambda: _ReceptorChannels(3, grouped=False)),
        (lambda: _GroupALIF(grouped=True), lambda: _GroupALIF(grouped=False)),
    ], ids=['receptor_channels', 'alif'])
    @pytest.mark.parametrize('algo_factory', [braintrace.D_RTRL, braintrace.EProp],
                             ids=['d_rtrl', 'eprop'])
    def test_group_gradients_equal_separate_states(self, grouped, separate, algo_factory):
        from braintrace._testing.oracle import (
            assert_param_gradients_close,
            online_param_gradients_singlestep_naive,
        )
        xs = self._inputs()
        g = online_param_gradients_singlestep_naive(grouped, xs, algo_factory=algo_factory)
        g_ref = online_param_gradients_singlestep_naive(separate, xs, algo_factory=algo_factory)
        assert max(float(jnp.abs(v).max()) for v in g_ref.values()) > 1e-3
        assert_param_gradients_close(g, g_ref, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize('factory', [
        lambda: _ReceptorChannels(3, grouped=True),
        lambda: _GroupALIF(grouped=True),
    ], ids=['receptor_channels', 'alif'])
    def test_group_d_rtrl_matches_bptt(self, factory):
        # D-RTRL is exact on position-diagonal cells, so BPTT is the oracle.
        from braintrace._testing.oracle import (
            assert_param_gradients_close,
            bptt_param_gradients,
            online_param_gradients_singlestep_naive,
        )
        xs = self._inputs()
        g = online_param_gradients_singlestep_naive(factory, xs, algo_factory=braintrace.D_RTRL)
        assert_param_gradients_close(g, bptt_param_gradients(factory, xs), atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize('grouped, hidden_shapes', [
        (False, ((4,),)),
        (True, ((4, 2), (4,))),
    ], ids=['hidden_state', 'group_state'])
    def test_a_true_shape_mismatch_is_still_removed(self, grouped, hidden_shapes):
        from braintrace._compiler.diagnostics import diagnostic_context
        model = _SummedMismatch(grouped)
        brainstate.nn.init_all_states(model)
        with diagnostic_context() as reporter, pytest.warns(UserWarning, match='not broadcastable'):
            relations = find_hidden_param_op_relations_from_module(model, jnp.ones(2))
        assert len(relations) == 0
        (record,) = _shape_mismatch_records(reporter)
        assert record.context['y_shape'] == (3,)
        assert record.context['hidden_shapes'] == hidden_shapes
        assert record.context['hidden_shape'] == hidden_shapes[0]
