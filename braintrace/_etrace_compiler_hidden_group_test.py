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


import unittest
from pprint import pprint

import brainstate
import brainunit as u
import jax
import numpy as np
import pytest

import braintrace
from braintrace import _etrace_model_with_group_state as group_etrace_model
from braintrace._etrace_compiler_hidden_group import find_hidden_groups_from_module
from braintrace._etrace_compiler_hidden_group import group_merging
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


class TestGroupMerging(unittest.TestCase):
    def test_no_intersection(self):
        groups = [[1, 2], [3, 4], [5, 6]]
        expected = [frozenset([1, 2]),
                    frozenset([3, 4]),
                    frozenset([5, 6])]
        result = group_merging(groups, version=1)
        self.assertEqual(set(result), set(expected))
        result = group_merging(groups, version=0)
        self.assertEqual(set(result), set(expected))

    def test_single_intersection(self):
        groups = [[1, 2], [2, 3], [4, 5]]
        expected = [frozenset([1, 2, 3]),
                    frozenset([4, 5])]
        result = group_merging(groups, version=1)
        self.assertEqual(set(result), set(expected))
        result = group_merging(groups, version=0)
        self.assertEqual(set(result), set(expected))

    def test_single_intersection3(self):
        groups = [[1, 2], [1, 2], [2, 3], [4, 5]]
        expected = [frozenset([1, 2, 3]),
                    frozenset([4, 5])]
        result = group_merging(groups, version=1)
        self.assertEqual(set(result), set(expected))
        result = group_merging(groups, version=0)
        self.assertEqual(set(result), set(expected))

    def test_single_intersection2(self):
        groups = [
            [('neu', 'a'), ('neu', 'V')],
            [('neu', 'V'), ('neu', '_before_updates', 'syn', 'g')]
        ]

        expected = [frozenset({('neu', 'a'), ('neu', '_before_updates', 'syn', 'g'), ('neu', 'V')})]
        result = group_merging(groups, version=1)
        print(result)
        self.assertEqual(set(result), set(expected))
        result = group_merging(groups, version=0)
        print(result)
        self.assertEqual(set(result), set(expected))

    def test_multiple_intersections(self):
        groups = [[1, 2], [2, 3], [3, 4], [5, 6]]
        expected = [frozenset([1, 2, 3, 4]),
                    frozenset([5, 6])]
        result = group_merging(groups, version=1)
        self.assertEqual(set(result), set(expected))
        result = group_merging(groups, version=0)
        self.assertEqual(set(result), set(expected))

    def test_all_intersect(self):
        groups = [[1, 2], [2, 3], [3, 4], [4, 1]]
        expected = [frozenset([1, 2, 3, 4])]
        result = group_merging(groups, version=0)
        self.assertEqual(set(result), set(expected))
        result = group_merging(groups, version=1)
        self.assertEqual(set(result), set(expected))

    def test_empty_groups(self):
        groups = []
        expected = []
        result = group_merging(groups, version=0)
        self.assertEqual(result, expected)
        result = group_merging(groups, version=1)
        self.assertEqual(result, expected)

    def test_single_group(self):
        groups = [[1, 2, 3]]
        expected = [frozenset([1, 2, 3])]
        result = group_merging(groups, version=0)
        self.assertEqual(set(result), set(expected))
        result = group_merging(groups, version=1)
        self.assertEqual(set(result), set(expected))


class Test_find_hidden_groups_from_module:
    @pytest.mark.parametrize(
        "cls",
        [
            braintrace.nn.GRUCell,
            braintrace.nn.LSTMCell,
            braintrace.nn.LRUCell,
            braintrace.nn.MGUCell,
            braintrace.nn.MinimalRNNCell,
        ]
    )
    def test_gru_one_layer(self, cls):
        n_in = 3
        n_out = 4

        gru = cls(n_in, n_out)
        brainstate.nn.init_all_states(gru)

        input = brainstate.random.rand(n_in)
        hidden_groups, hid_path_to_group = find_hidden_groups_from_module(gru, input)

        print()
        pprint(hidden_groups)
        print()
        pprint(hid_path_to_group)
        print()

    @pytest.mark.parametrize(
        'cls,',
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
            hidden_groups, hid_path_to_group = find_hidden_groups_from_module(layer, input)

        print()
        for group in hidden_groups:
            pprint(group.hidden_paths)

        assert (len(hidden_groups) == 1)
        print()

    @pytest.mark.parametrize(
        'cls,',
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
            hidden_groups, hid_path_to_group = find_hidden_groups_from_module(layer, input)

        for group in hidden_groups:
            pprint(group.hidden_paths)

        assert (len(hidden_groups) == 2)
        # print()


class Test_module_with_group_state:
    @pytest.mark.parametrize(
        'cls_without_group,cls_with_group',
        [
            (ALIF_ExpCu_Dense_Layer, group_etrace_model.ALIF_ExpCu_Dense_Layer),
            (ALIF_Delta_Dense_Layer, group_etrace_model.ALIF_Delta_Dense_Layer),
            (ALIF_STDExpCu_Dense_Layer, group_etrace_model.ALIF_STDExpCu_Dense_Layer),
            (ALIF_STPExpCu_Dense_Layer, group_etrace_model.ALIF_STPExpCu_Dense_Layer),
        ]
    )
    def test_snn_single_layer(
        self,
        cls_without_group,
        cls_with_group,
    ):
        n_in = 3
        n_out = 4
        input = brainstate.random.rand(n_in)

        with brainstate.environ.context(dt=0.1 * u.ms):
            layer_without_group = cls_without_group(n_in, n_out)
            layer_with_group = cls_with_group(n_in, n_out)
            brainstate.nn.init_all_states(layer_without_group)
            brainstate.nn.init_all_states(layer_with_group)
            hidden_groups_without_group, _ = find_hidden_groups_from_module(layer_without_group, input)
            hidden_groups_with_group, _ = find_hidden_groups_from_module(layer_with_group, input)

        print()
        for group1, group2 in zip(hidden_groups_without_group, hidden_groups_with_group):
            assert (len(group1.hidden_paths) == len(group2.hidden_paths)) + 1
            assert (len(group1.hidden_invars) == len(group2.hidden_invars)) + 1
            assert (len(group1.hidden_outvars) == len(group2.hidden_outvars)) + 1
            assert (len(group1.hidden_states) == len(group2.hidden_states)) + 1
            assert group1.num_state == group2.num_state
            assert group1.varshape == group2.varshape

    @pytest.mark.parametrize(
        'cls_without_group,cls_with_group',
        [
            (ALIF_ExpCu_Dense_Layer, group_etrace_model.ALIF_ExpCu_Dense_Layer),
            (ALIF_Delta_Dense_Layer, group_etrace_model.ALIF_Delta_Dense_Layer),
            (ALIF_STDExpCu_Dense_Layer, group_etrace_model.ALIF_STDExpCu_Dense_Layer),
            (ALIF_STPExpCu_Dense_Layer, group_etrace_model.ALIF_STPExpCu_Dense_Layer),
        ]
    )
    def test_snn_single_layer_state_transition(
        self,
        cls_without_group,
        cls_with_group,
    ):
        rec_init = lambda shape: brainstate.random.RandomState(0).randn(*shape)
        ff_init = lambda shape: brainstate.random.RandomState(1).randn(*shape)

        n_in = 3
        n_out = 4
        input = brainstate.random.rand(n_in)

        with brainstate.environ.context(dt=0.1 * u.ms):
            layer_without_group = cls_without_group(n_in, n_out, rec_init=rec_init, ff_init=ff_init)
            layer_with_group = cls_with_group(n_in, n_out, rec_init=rec_init, ff_init=ff_init)
            brainstate.nn.init_all_states(layer_without_group)
            brainstate.nn.init_all_states(layer_with_group)

            graph_without_group = braintrace.compile_etrace_graph(layer_without_group, input)
            graph_with_group = braintrace.compile_etrace_graph(layer_with_group, input)

        out1, etrace1, other1, temp1 = graph_with_group.module_info.jaxpr_call(input)
        out2, etrace2, other2, temp2 = graph_without_group.module_info.jaxpr_call(input)

        print()
        for group_without, group_with in zip(graph_without_group.hidden_groups,
                                             graph_with_group.hidden_groups):
            hidden_vals_v1 = [temp1[invar] for invar in group_with.hidden_invars]
            input_vals_v1 = [temp1[invar] for invar in group_with.transition_jaxpr_constvars]
            out_vals_with_group = group_with.concat_hidden(group_with.transition(hidden_vals_v1, input_vals_v1))

            hidden_paths_with_group = []
            for path in group_with.hidden_paths:
                if path == ('neu', 'st'):
                    hidden_paths_with_group.append(('neu', 'V'))
                    hidden_paths_with_group.append(('neu', 'a'))
                else:
                    hidden_paths_with_group.append(path)
            a_index_map = {element: index for index, element in enumerate(group_without.hidden_paths)}
            b_indices = [a_index_map[element] for element in hidden_paths_with_group]
            b_indices = np.asarray(b_indices)

            hidden_vals_v2 = [temp2[invar] for invar in group_without.hidden_invars]
            input_vals_v2 = [temp2[invar] for invar in group_without.transition_jaxpr_constvars]
            out_vals_without_group = group_without.concat_hidden(
                group_without.transition(hidden_vals_v2, input_vals_v2))

            print(hidden_paths_with_group)
            print(out_vals_with_group)
            print(out_vals_without_group[..., b_indices])

            assert np.allclose(out_vals_with_group, out_vals_without_group[..., b_indices], atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        'cls_without_group,cls_with_group',
        [
            (ALIF_ExpCu_Dense_Layer, group_etrace_model.ALIF_ExpCu_Dense_Layer),
            (ALIF_Delta_Dense_Layer, group_etrace_model.ALIF_Delta_Dense_Layer),
            (ALIF_STDExpCu_Dense_Layer, group_etrace_model.ALIF_STDExpCu_Dense_Layer),
            (ALIF_STPExpCu_Dense_Layer, group_etrace_model.ALIF_STPExpCu_Dense_Layer),
        ]
    )
    def test_snn_two_layer_state_transition(
        self,
        cls_without_group,
        cls_with_group,
    ):
        rec_init = lambda shape: brainstate.random.RandomState(0).randn(*shape)
        ff_init = lambda shape: brainstate.random.RandomState(1).randn(*shape)

        n_in = 3
        n_out = 4
        input = brainstate.random.rand(n_in)

        with brainstate.environ.context(dt=0.1 * u.ms):
            layer_without_group = brainstate.nn.Sequential(
                cls_without_group(n_in, n_out, rec_init=rec_init, ff_init=ff_init),
                cls_without_group(n_out, n_out, rec_init=rec_init, ff_init=ff_init)
            )
            layer_with_group = brainstate.nn.Sequential(
                cls_with_group(n_in, n_out, rec_init=rec_init, ff_init=ff_init),
                cls_with_group(n_out, n_out, rec_init=rec_init, ff_init=ff_init)
            )
            brainstate.nn.init_all_states(layer_without_group)
            brainstate.nn.init_all_states(layer_with_group)

            graph_without_group = braintrace.compile_etrace_graph(layer_without_group, input)
            graph_with_group = braintrace.compile_etrace_graph(layer_with_group, input)

        out1, etrace1, other1, temp1 = graph_with_group.module_info.jaxpr_call(input)
        out2, etrace2, other2, temp2 = graph_without_group.module_info.jaxpr_call(input)

        print()
        for group_with in graph_with_group.hidden_groups:
            hidden_paths_with_group = []
            for path in group_with.hidden_paths:
                if path[-2:] == ('neu', 'st'):
                    hidden_paths_with_group.append(path[:-2] + ('neu', 'V'))
                    hidden_paths_with_group.append(path[:-2] + ('neu', 'a'))
                else:
                    hidden_paths_with_group.append(path)

            group_without = None
            for group_without in graph_without_group.hidden_groups:
                if hidden_paths_with_group[0] in group_without.hidden_paths:
                    break
            if group_without is None:
                raise ValueError('Group not found')

            # etrace variables with group state
            hidden_vals_v1 = [temp1[invar] for invar in group_with.hidden_invars]
            input_vals_v1 = [temp1[invar] for invar in group_with.transition_jaxpr_constvars]
            out_vals_with_group = group_with.concat_hidden(group_with.transition(hidden_vals_v1, input_vals_v1))

            # index mapping
            a_index_map = {element: index for index, element in enumerate(group_without.hidden_paths)}
            b_indices = [a_index_map[element] for element in hidden_paths_with_group]
            b_indices = np.asarray(b_indices)

            # etrace variables without group state
            hidden_vals_v2 = [temp2[invar] for invar in group_without.hidden_invars]
            input_vals_v2 = [temp2[invar] for invar in group_without.transition_jaxpr_constvars]
            out_vals_without_group = group_without.concat_hidden(
                group_without.transition(hidden_vals_v2, input_vals_v2))

            # comparison
            assert np.allclose(out_vals_with_group, out_vals_without_group[..., b_indices], atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        'cls_without_group,cls_with_group',
        [
            (ALIF_ExpCu_Dense_Layer, group_etrace_model.ALIF_ExpCu_Dense_Layer),
            (ALIF_Delta_Dense_Layer, group_etrace_model.ALIF_Delta_Dense_Layer),
            (ALIF_STDExpCu_Dense_Layer, group_etrace_model.ALIF_STDExpCu_Dense_Layer),
            (ALIF_STPExpCu_Dense_Layer, group_etrace_model.ALIF_STPExpCu_Dense_Layer),
        ]
    )
    def test_snn_single_layer_diagonal_jacobian(
        self,
        cls_without_group,
        cls_with_group,
    ):
        rec_init = lambda shape: brainstate.random.RandomState(0).randn(*shape)
        ff_init = lambda shape: brainstate.random.RandomState(1).randn(*shape)

        n_in = 3
        n_out = 4
        input = brainstate.random.rand(n_in)

        with brainstate.environ.context(dt=0.1 * u.ms):
            layer_without_group = cls_without_group(n_in, n_out, rec_init=rec_init, ff_init=ff_init)
            layer_with_group = cls_with_group(n_in, n_out, rec_init=rec_init, ff_init=ff_init)
            brainstate.nn.init_all_states(layer_without_group)
            brainstate.nn.init_all_states(layer_with_group)

            graph_without_group = braintrace.compile_etrace_graph(layer_without_group, input)
            graph_with_group = braintrace.compile_etrace_graph(layer_with_group, input)

        out1, etrace1, other1, temp1 = graph_with_group.module_info.jaxpr_call(input)
        out2, etrace2, other2, temp2 = graph_without_group.module_info.jaxpr_call(input)

        print()
        for group_without, group_with in zip(graph_without_group.hidden_groups,
                                             graph_with_group.hidden_groups):
            hidden_vals_v1 = [temp1[invar] for invar in group_with.hidden_invars]
            input_vals_v1 = [temp1[invar] for invar in group_with.transition_jaxpr_constvars]
            jac_with_group = group_with.diagonal_jacobian(hidden_vals_v1, input_vals_v1)

            hidden_paths_with_group = []
            for path in group_with.hidden_paths:
                if path == ('neu', 'st'):
                    hidden_paths_with_group.append(('neu', 'V'))
                    hidden_paths_with_group.append(('neu', 'a'))
                else:
                    hidden_paths_with_group.append(path)
            a_index_map = {element: index for index, element in enumerate(group_without.hidden_paths)}
            b_indices = [a_index_map[element] for element in hidden_paths_with_group]
            b_indices = np.asarray(b_indices)

            hidden_vals_v2 = [temp2[invar] for invar in group_without.hidden_invars]
            input_vals_v2 = [temp2[invar] for invar in group_without.transition_jaxpr_constvars]
            jac_without_group = group_without.diagonal_jacobian(hidden_vals_v2, input_vals_v2)
            jac_without_group = jac_without_group[..., b_indices]
            jac_without_group = jac_without_group[..., b_indices, :]

            print(hidden_paths_with_group)
            print(jac_with_group)
            print(jac_without_group)
            assert np.allclose(jac_with_group, jac_without_group, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        'cls_without_group,cls_with_group',
        [
            (ALIF_ExpCu_Dense_Layer, group_etrace_model.ALIF_ExpCu_Dense_Layer),
            (ALIF_Delta_Dense_Layer, group_etrace_model.ALIF_Delta_Dense_Layer),
            (ALIF_STDExpCu_Dense_Layer, group_etrace_model.ALIF_STDExpCu_Dense_Layer),
            (ALIF_STPExpCu_Dense_Layer, group_etrace_model.ALIF_STPExpCu_Dense_Layer),
        ]
    )
    def test_snn_two_layer_diagonal_jacobian(
        self,
        cls_without_group,
        cls_with_group,
    ):
        rec_init = lambda shape: brainstate.random.RandomState(0).randn(*shape)
        ff_init = lambda shape: brainstate.random.RandomState(1).randn(*shape)

        n_in = 3
        n_out = 4
        input = brainstate.random.rand(n_in)

        with brainstate.environ.context(dt=0.1 * u.ms):
            layer_without_group = brainstate.nn.Sequential(
                cls_without_group(n_in, n_out, rec_init=rec_init, ff_init=ff_init),
                cls_without_group(n_out, n_out, rec_init=rec_init, ff_init=ff_init)
            )
            layer_with_group = brainstate.nn.Sequential(
                cls_with_group(n_in, n_out, rec_init=rec_init, ff_init=ff_init),
                cls_with_group(n_out, n_out, rec_init=rec_init, ff_init=ff_init)
            )
            brainstate.nn.init_all_states(layer_without_group)
            brainstate.nn.init_all_states(layer_with_group)

            graph_without_group = braintrace.compile_etrace_graph(layer_without_group, input)
            graph_with_group = braintrace.compile_etrace_graph(layer_with_group, input)

        out1, etrace1, other1, temp1 = graph_with_group.module_info.jaxpr_call(input)
        out2, etrace2, other2, temp2 = graph_without_group.module_info.jaxpr_call(input)

        print()
        for group_with in graph_with_group.hidden_groups:
            hidden_paths_with_group = []
            for path in group_with.hidden_paths:
                if path[-2:] == ('neu', 'st'):
                    hidden_paths_with_group.append(path[:-2] + ('neu', 'V'))
                    hidden_paths_with_group.append(path[:-2] + ('neu', 'a'))
                else:
                    hidden_paths_with_group.append(path)

            group_without = None
            for group_without in graph_without_group.hidden_groups:
                if hidden_paths_with_group[0] in group_without.hidden_paths:
                    break
            if group_without is None:
                raise ValueError('Group not found')

            # etrace variables with group state
            hidden_vals_v1 = [temp1[invar] for invar in group_with.hidden_invars]
            input_vals_v1 = [temp1[invar] for invar in group_with.transition_jaxpr_constvars]
            jac_with_group = group_with.diagonal_jacobian(hidden_vals_v1, input_vals_v1)

            # index mapping
            a_index_map = {element: index for index, element in enumerate(group_without.hidden_paths)}
            b_indices = [a_index_map[element] for element in hidden_paths_with_group]
            b_indices = np.asarray(b_indices)

            # etrace variables without group state
            hidden_vals_v2 = [temp2[invar] for invar in group_without.hidden_invars]
            input_vals_v2 = [temp2[invar] for invar in group_without.transition_jaxpr_constvars]
            jac_without_group = group_without.diagonal_jacobian(hidden_vals_v2, input_vals_v2)
            jac_without_group = jac_without_group[..., b_indices]
            jac_without_group = jac_without_group[..., b_indices, :]

            # comparison
            assert np.allclose(jac_with_group, jac_without_group, atol=1e-3, rtol=1e-3)


class TestHiddenGroup_state_transition:
    @pytest.mark.parametrize(
        "cls",
        [
            braintrace.nn.GRUCell,
            braintrace.nn.LSTMCell,
            braintrace.nn.LRUCell,
            braintrace.nn.MGUCell,
            braintrace.nn.MinimalRNNCell,
        ]
    )
    def test_gru(self, cls):
        n_in = 3
        n_out = 4

        gru = cls(n_in, n_out)
        brainstate.nn.init_all_states(gru)

        input = brainstate.random.rand(n_in)
        hidden_groups, hid_path_to_group = find_hidden_groups_from_module(gru, input)

        print()
        for group in hidden_groups:
            print(group)
            hidden_vals = [brainstate.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [brainstate.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            out_vals = group.transition(hidden_vals, input_vals)
            print(out_vals)

    @pytest.mark.parametrize(
        'cls,',
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

        print()
        print(cls)

        with brainstate.environ.context(dt=0.1 * u.ms):
            layer = cls(n_in, n_out)
            brainstate.nn.init_all_states(layer)
            hidden_groups, hid_path_to_group = find_hidden_groups_from_module(layer, input)

        for group in hidden_groups:
            pprint(group.hidden_paths)
            hidden_vals = [brainstate.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [brainstate.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            out_vals = group.transition(hidden_vals, input_vals)
            print(out_vals)

    @pytest.mark.parametrize(
        'cls,',
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
            hidden_groups, hid_path_to_group = find_hidden_groups_from_module(layer, input)

        for group in hidden_groups:
            pprint(group.hidden_paths)
            hidden_vals = [brainstate.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [brainstate.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            out_vals = group.transition(hidden_vals, input_vals)
            print(out_vals)


class TestHiddenGroup_diagonal_jacobian:
    @pytest.mark.parametrize(
        "cls",
        [
            braintrace.nn.GRUCell,
            braintrace.nn.LSTMCell,
            braintrace.nn.LRUCell,
            braintrace.nn.MGUCell,
            braintrace.nn.MinimalRNNCell,
        ]
    )
    def test_gru(self, cls):
        n_in = 3
        n_out = 4

        gru = cls(n_in, n_out)
        brainstate.nn.init_all_states(gru)

        input = brainstate.random.rand(n_in)
        hidden_groups, hid_path_to_group = find_hidden_groups_from_module(gru, input)

        print()
        for group in hidden_groups:
            hidden_vals = [brainstate.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [brainstate.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            diag_jac = group.diagonal_jacobian(hidden_vals, input_vals)
            print(diag_jac)

    @pytest.mark.parametrize(
        "cls",
        [
            braintrace.nn.GRUCell,
            braintrace.nn.LSTMCell,
            braintrace.nn.LRUCell,
            braintrace.nn.MGUCell,
            braintrace.nn.MinimalRNNCell,
        ]
    )
    def test_gru_accuracy(self, cls):
        n_in = 1
        n_out = 1

        gru = cls(n_in, n_out)
        brainstate.nn.init_all_states(gru)

        input = brainstate.random.rand(n_in)
        hidden_groups, hid_path_to_group = find_hidden_groups_from_module(gru, input)

        print()
        for group in hidden_groups:
            hidden_vals = [brainstate.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [brainstate.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            diag_jac = group.diagonal_jacobian(hidden_vals, input_vals)
            diag_jac = u.math.squeeze(diag_jac)
            print(diag_jac)

            fn = lambda hid: group.concat_hidden(group.transition(group.split_hidden(hid), input_vals))
            jax_jac = jax.jacrev(fn)(group.concat_hidden(hidden_vals))
            jax_jac = u.math.squeeze(jax_jac)
            print(jax_jac)

            assert (u.math.allclose(diag_jac, jax_jac, atol=1e-5))

    @pytest.mark.parametrize(
        'cls,',
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

        print()
        print(cls)

        with brainstate.environ.context(dt=0.1 * u.ms):
            layer = cls(n_in, n_out)
            brainstate.nn.init_all_states(layer)
            hidden_groups, hid_path_to_group = find_hidden_groups_from_module(layer, input)

        for group in hidden_groups:
            pprint(group.hidden_paths)
            hidden_vals = [brainstate.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [brainstate.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            diag_jac = group.diagonal_jacobian(hidden_vals, input_vals)
            print(diag_jac)

    @pytest.mark.parametrize(
        'cls,',
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
    def test_snn_single_layer_accuracy(self, cls):
        n_in = 1
        n_out = 1
        input = brainstate.random.rand(n_in)

        print()
        print(cls)

        with brainstate.environ.context(dt=0.1 * u.ms):
            layer = cls(n_in, n_out)
            brainstate.nn.init_all_states(layer)
            hidden_groups, hid_path_to_group = find_hidden_groups_from_module(layer, input)

        for group in hidden_groups:
            pprint(group.hidden_paths)
            hidden_vals = [brainstate.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [brainstate.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            diag_jac = group.diagonal_jacobian(hidden_vals, input_vals)
            diag_jac = u.math.squeeze(diag_jac)
            print(diag_jac)

            fn = lambda hid: group.concat_hidden(group.transition(group.split_hidden(hid), input_vals))
            jax_jac = jax.jacrev(fn)(group.concat_hidden(hidden_vals))
            jax_jac = u.math.squeeze(jax_jac)
            print(jax_jac)

            assert (u.math.allclose(diag_jac, jax_jac, atol=1e-5))

    @pytest.mark.parametrize(
        'cls,',
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
            hidden_groups, hid_path_to_group = find_hidden_groups_from_module(layer, input)

        for group in hidden_groups:
            pprint(group.hidden_paths)
            hidden_vals = [brainstate.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [brainstate.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            diag_jac = group.diagonal_jacobian(hidden_vals, input_vals)
            print(diag_jac)

    @pytest.mark.parametrize(
        'cls,',
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
    def test_snn_two_layers_accuracy(a, cls, ):
        n_in = 1
        n_out = 1
        input = brainstate.random.rand(n_in)

        print()
        print(cls)
        with brainstate.environ.context(dt=0.1 * u.ms):
            layer = brainstate.nn.Sequential(cls(n_in, n_out), cls(n_out, n_out))
            brainstate.nn.init_all_states(layer)
            hidden_groups, hid_path_to_group = find_hidden_groups_from_module(layer, input)

        for group in hidden_groups:
            pprint(group.hidden_paths)
            hidden_vals = [brainstate.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [brainstate.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            diag_jac = group.diagonal_jacobian(hidden_vals, input_vals)
            diag_jac = u.math.squeeze(diag_jac)
            print(diag_jac)

            fn = lambda hid: group.concat_hidden(group.transition(group.split_hidden(hid), input_vals))
            jax_jac = jax.jacrev(fn)(group.concat_hidden(hidden_vals))
            jax_jac = u.math.squeeze(jax_jac)
            print(jax_jac)

            assert (u.math.allclose(diag_jac, jax_jac, atol=1e-5))
