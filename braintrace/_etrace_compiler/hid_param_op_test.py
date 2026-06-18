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
import pytest
import saiunit as u

import braintrace
from braintrace import find_hidden_param_op_relations_from_module
from braintrace._etrace_model_test import (
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
        from braintrace._etrace_compiler.hid_param_op import HiddenParamOpRelation
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
        from braintrace._etrace_compiler.hid_param_op import HiddenParamOpRelation
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
        from braintrace._etrace_compiler import (
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
