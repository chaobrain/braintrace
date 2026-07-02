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
import warnings
from pprint import pprint

import brainstate
import jax
import numpy as np
import pytest
import brainunit as u

import braintrace
from braintrace._compiler import model4test as group_etrace_model
from braintrace._compiler.hidden_group import find_hidden_groups_from_module
from braintrace._compiler.hidden_group import group_merging
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
            hidden_groups, hid_path_to_group = find_hidden_groups_from_module(layer, input)

        print()
        for group in hidden_groups:
            pprint(group.hidden_paths)

        assert (len(hidden_groups) == 1)
        print()

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
            assert len(group1.hidden_paths) == len(group2.hidden_paths) + 1
            assert len(group1.hidden_invars) == len(group2.hidden_invars) + 1
            assert len(group1.hidden_outvars) == len(group2.hidden_outvars) + 1
            assert len(group1.hidden_states) == len(group2.hidden_states) + 1
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
            hidden_groups, hid_path_to_group = find_hidden_groups_from_module(layer, input)

        for group in hidden_groups:
            pprint(group.hidden_paths)
            hidden_vals = [brainstate.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [brainstate.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            out_vals = group.transition(hidden_vals, input_vals)
            print(out_vals)


def _true_block_diagonal(group, hidden_vals, input_vals):
    """Independent oracle for ``group.diagonal_jacobian``.

    Materializes the full recurrent Jacobian with ``jax.jacrev`` and, for every
    ``varshape`` position ``p``, picks the ``num_state x num_state`` block
    ``full[p, :, p, :]`` (dropping the cross-position terms). Uses a plain Python
    loop + ``stack`` so it shares no code with the production
    ``diagonal``/``moveaxis`` extraction it is meant to check.
    """
    fn = lambda hid: group.concat_hidden(group.transition(group.split_hidden(hid), input_vals))
    hid = group.concat_hidden(hidden_vals)
    num_state = hid.shape[-1]
    varshape = hid.shape[:-1]
    num_pos = int(np.prod(varshape)) if varshape else 1
    full = jax.jacrev(fn)(hid)  # (*varshape, num_state, *varshape, num_state)
    full = u.math.reshape(full, (num_pos, num_state, num_pos, num_state))
    blocks = u.math.stack([full[p, :, p, :] for p in range(num_pos)], axis=0)
    return u.math.reshape(blocks, (*varshape, num_state, num_state))


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

    @pytest.mark.parametrize("include_recurrent_mixing", [False, True])
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
    def test_gru_accuracy_multiunit(self, cls, include_recurrent_mixing):
        # ``diagonal_jacobian`` must equal the true per-position block diagonal
        # (never the column sum over output positions) in *both* grouping modes:
        #   - default (without recurrence): the transition is element-wise, so the
        #     block diagonal is trivially correct;
        #   - with recurrence: the transition is coupled, so the block-diagonal
        #     extraction must drop the cross-position terms (the column-sum bug).
        # The single-unit ``test_gru_accuracy`` (n_out == 1) cannot see this.
        n_in = 3
        n_out = 4

        gru = cls(n_in, n_out)
        brainstate.nn.init_all_states(gru)

        input = brainstate.random.rand(n_in)
        hidden_groups, _ = find_hidden_groups_from_module(
            gru, input, include_recurrent_mixing=include_recurrent_mixing)

        assert len(hidden_groups) >= 1
        for group in hidden_groups:
            hidden_vals = [brainstate.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [brainstate.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            diag_jac = group.diagonal_jacobian(hidden_vals, input_vals)
            truth = _true_block_diagonal(group, hidden_vals, input_vals)
            assert diag_jac.shape == (*group.varshape, group.num_state, group.num_state)
            assert u.math.allclose(diag_jac, truth, atol=1e-5)

    def test_is_diagonal_recurrence_flag(self):
        # ``is_diagonal_recurrence`` is determined purely by the grouping mode:
        # it equals ``not include_recurrent_mixing`` for every cell.
        cells = (braintrace.nn.GRUCell, braintrace.nn.LSTMCell,
                 braintrace.nn.MGUCell, braintrace.nn.MinimalRNNCell,
                 braintrace.nn.LRUCell)

        # Default mode (without recurrence): the recurrent-mixing boundary skip
        # leaves an element-wise / position-diagonal transition -> flag True.
        for cls in cells:
            model = cls(3, 4)
            brainstate.nn.init_all_states(model)
            groups, _ = find_hidden_groups_from_module(model, brainstate.random.rand(3))
            assert len(groups) >= 1
            assert all(g.is_diagonal_recurrence for g in groups), cls.__name__

        # With recurrence: the flag flips to False for *every* cell -- including
        # the LRU, whose transition is structurally diagonal (element-wise complex
        # decay, no matmul) but whose flag now follows the mode. The block-diagonal
        # extraction is correct for the diagonal case too, so this is safe (just
        # not the cheapest possible path for the LRU on this opt-in branch).
        for cls in cells:
            model = cls(3, 4)
            brainstate.nn.init_all_states(model)
            groups, _ = find_hidden_groups_from_module(
                model, brainstate.random.rand(3), include_recurrent_mixing=True)
            assert len(groups) >= 1
            assert all(not g.is_diagonal_recurrence for g in groups), cls.__name__

    @pytest.mark.parametrize(
        "cls",
        [
            braintrace.nn.GRUCell,
            braintrace.nn.LSTMCell,
            braintrace.nn.MGUCell,
            braintrace.nn.MinimalRNNCell,
            braintrace.nn.ValinaRNNCell,
        ]
    )
    def test_default_mode_transition_is_element_wise(self, cls):
        # Root-cause regression: in the default ("without recurrence") mode the
        # recurrent ETP mixing primitives must NOT appear in the transition jaxpr
        # (this is the 0.1.2 behaviour that keeps D^t contractive). A coupled
        # recurrent matmul in the transition was the cause of the trace overflow.
        model = cls(3, 4)
        brainstate.nn.init_all_states(model)
        groups, _ = find_hidden_groups_from_module(model, brainstate.random.rand(3))
        assert len(groups) >= 1
        mixing = {'etp_mv', 'etp_mm', 'etp_conv'}
        for g in groups:
            prims = {e.primitive.name for e in g.transition_jaxpr.eqns}
            assert not (prims & mixing), (cls.__name__, prims & mixing)

    @pytest.mark.parametrize(
        "cls",
        [
            braintrace.nn.GRUCell,
            braintrace.nn.LSTMCell,
            braintrace.nn.MGUCell,
            braintrace.nn.MinimalRNNCell,
            braintrace.nn.ValinaRNNCell,
        ]
    )
    def test_with_recurrence_includes_mixing(self, cls):
        # The opt-in ("with recurrence") mode traces the recurrent ETP matmul into
        # the transition jaxpr (the 0.2.x behaviour, now explicit and bounded via
        # the block-diagonal extraction).
        model = cls(3, 4)
        brainstate.nn.init_all_states(model)
        groups, _ = find_hidden_groups_from_module(
            model, brainstate.random.rand(3), include_recurrent_mixing=True)
        assert len(groups) >= 1
        mixing = {'etp_mv', 'etp_mm', 'etp_conv'}
        assert any(
            (prims := {e.primitive.name for e in g.transition_jaxpr.eqns}) & mixing
            for g in groups
        ), cls.__name__

    def test_vanilla_rnn_zero_recurrence_group(self):
        # A vanilla RNN's only h^{t-1} dependence flows through the recurrent
        # matmul; default mode excludes it, leaving no recurrent path. The hidden
        # state must still get a (singleton) group with D^t = 0 so every hidden
        # outvar carries a group index (the hid->weight compiler asserts this).
        rnn = braintrace.nn.ValinaRNNCell(3, 4, activation='tanh')
        brainstate.nn.init_all_states(rnn)
        groups, _ = find_hidden_groups_from_module(rnn, brainstate.random.rand(3))
        assert len(groups) == 1
        g = groups[0]
        assert g.is_diagonal_recurrence
        hidden_vals = [brainstate.random.rand_like(invar.aval) for invar in g.hidden_invars]
        input_vals = [brainstate.random.rand_like(invar.aval) for invar in g.transition_jaxpr_constvars]
        diag_jac = g.diagonal_jacobian(hidden_vals, input_vals)
        # D^t == 0 : the recurrence was fully excluded.
        assert u.math.allclose(diag_jac, u.math.zeros_like(diag_jac), atol=1e-6)

    def test_non_etp_reservoir_recurrence_modes(self):
        # Coverage for the *non-ETP* recurrent-weight skip (skip #2): a plain
        # ``dot_general`` that reads the hidden state (a fixed reservoir matrix)
        # couples the leading ``varshape`` positions exactly like the ETP matmul,
        # so the default ("without recurrence") mode must treat it as a boundary
        # too -- otherwise the cheap column-sum Jacobian would be applied to a
        # coupled transition and overflow. With recurrence it is traced in and the
        # block-diagonal extraction must match the independent oracle. This guards
        # the deliberate narrowness of ``_RECURRENT_WEIGHT_MIXING_PRIMITIVES``
        # (matmul/conv only -- within-position gathers stay in the transition).
        n_in, n_out = 3, 4

        class _PlainReservoir(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                # ETP input projection: feed-forward, never enters the transition.
                self.w_in = brainstate.ParamState(
                    brainstate.random.randn(n_in, n_out) * 0.1)
                # Fixed reservoir matrix -> recurrence via a *plain* dot_general
                # (a regular JAX op, not ``braintrace.matmul``).
                self.w_rec = brainstate.random.randn(n_out, n_out) * 0.3
                self.h = brainstate.HiddenState(jax.numpy.zeros(n_out))

            def update(self, x):
                rec = self.h.value @ self.w_rec  # plain dot_general reading hidden
                inp = braintrace.matmul(x, self.w_in.value)  # ETP, feed-forward
                self.h.value = jax.nn.tanh(rec + inp)
                return self.h.value

        x = brainstate.random.rand(n_in)

        # Default ("without recurrence"): the recurrent dot_general is a boundary,
        # so the transition is position-diagonal and carries no ``dot_general``.
        model = _PlainReservoir()
        brainstate.nn.init_all_states(model)
        groups, _ = find_hidden_groups_from_module(model, x)
        assert len(groups) >= 1
        assert all(g.is_diagonal_recurrence for g in groups)
        for g in groups:
            prims = {e.primitive.name for e in g.transition_jaxpr.eqns}
            assert 'dot_general' not in prims, prims

        # With recurrence: the dot_general is traced in -> coupled transition, and
        # the block-diagonal extraction matches the independent jacrev oracle.
        model = _PlainReservoir()
        brainstate.nn.init_all_states(model)
        groups, _ = find_hidden_groups_from_module(
            model, x, include_recurrent_mixing=True)
        assert len(groups) >= 1
        assert all(not g.is_diagonal_recurrence for g in groups)
        assert any(
            'dot_general' in {e.primitive.name for e in g.transition_jaxpr.eqns}
            for g in groups
        )
        for g in groups:
            hidden_vals = [brainstate.random.rand_like(iv.aval) for iv in g.hidden_invars]
            input_vals = [brainstate.random.rand_like(iv.aval)
                          for iv in g.transition_jaxpr_constvars]
            diag_jac = g.diagonal_jacobian(hidden_vals, input_vals)
            truth = _true_block_diagonal(g, hidden_vals, input_vals)
            assert diag_jac.shape == (*g.varshape, g.num_state, g.num_state)
            assert u.math.allclose(diag_jac, truth, atol=1e-5)

    @pytest.mark.parametrize("include_recurrent_mixing", [False, True])
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
    def test_snn_single_layer_accuracy_multiunit(self, cls, include_recurrent_mixing):
        # The dense SNN layers are *recurrent* (they feed spikes back through a
        # recurrent ETP weight). ``diagonal_jacobian`` must equal the true
        # per-position block diagonal in BOTH grouping modes (the n_out == 1
        # ``test_snn_single_layer_accuracy`` could not detect the column-sum bug):
        #   - default: the recurrent matmul is excluded -> diagonal transition;
        #   - with recurrence: the matmul is traced in -> coupled, exercised via
        #     the block-diagonal extraction.
        n_in = 3
        n_out = 4
        input = brainstate.random.rand(n_in)

        with brainstate.environ.context(dt=0.1 * u.ms):
            layer = cls(n_in, n_out)
            brainstate.nn.init_all_states(layer)
            hidden_groups, _ = find_hidden_groups_from_module(
                layer, input, include_recurrent_mixing=include_recurrent_mixing)

        assert len(hidden_groups) >= 1
        # The flag is mode-derived: diagonal iff recurrent mixing is excluded.
        assert all(
            g.is_diagonal_recurrence == (not include_recurrent_mixing)
            for g in hidden_groups
        )
        for group in hidden_groups:
            hidden_vals = [brainstate.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [brainstate.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            diag_jac = group.diagonal_jacobian(hidden_vals, input_vals)
            truth = _true_block_diagonal(group, hidden_vals, input_vals)
            assert diag_jac.shape == (*group.varshape, group.num_state, group.num_state)
            assert u.math.allclose(diag_jac, truth, atol=1e-5)

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
            hidden_groups, hid_path_to_group = find_hidden_groups_from_module(layer, input)

        for group in hidden_groups:
            pprint(group.hidden_paths)
            hidden_vals = [brainstate.random.rand_like(invar.aval) for invar in group.hidden_invars]
            input_vals = [brainstate.random.rand_like(invar.aval) for invar in group.transition_jaxpr_constvars]
            diag_jac = group.diagonal_jacobian(hidden_vals, input_vals)
            print(diag_jac)

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


class TestGroupOrderingAndDiagnostics:
    """Task-4 contracts: canonical member ordering, merge diagnostics, and
    varshape validation at group construction time."""

    def _lstm_minfo(self):
        lstm = braintrace.nn.LSTMCell(3, 4)
        brainstate.nn.init_all_states(lstm)
        inp = brainstate.random.rand(3)
        minfo = braintrace.extract_module_info(lstm, inp)
        return lstm, inp, minfo

    def test_lstm_group_paths_follow_compiled_state_order(self):
        # Member order inside a merged group is canonical: it follows the
        # compiled state order (hidden_path_to_outvar insertion order), not
        # the hash order of an intermediate set.
        _, inp, minfo = self._lstm_minfo()
        state_order = list(minfo.hidden_path_to_outvar.keys())

        from braintrace._compiler.hidden_group import find_hidden_groups_from_minfo
        groups, _ = find_hidden_groups_from_minfo(minfo)

        assert len(groups) == 1
        assert groups[0].hidden_paths == state_order
        # invars/outvars follow the same canonical order.
        expected_outvars = [minfo.hidden_path_to_outvar[p] for p in state_order]
        assert groups[0].hidden_outvars == expected_outvars

    def test_group_ordering_identical_across_two_compiles(self):
        def build():
            lstm = braintrace.nn.LSTMCell(3, 4)
            brainstate.nn.init_all_states(lstm)
            inp = brainstate.random.rand(3)
            groups, _ = find_hidden_groups_from_module(lstm, inp)
            return [g.hidden_paths for g in groups]

        assert build() == build()

    def test_merged_group_emits_info_record(self):
        from braintrace._compiler.diagnostics import (
            DiagnosticKind,
            DiagnosticLevel,
            diagnostic_context,
        )
        lstm, inp, _ = self._lstm_minfo()

        with diagnostic_context() as reporter:
            find_hidden_groups_from_module(lstm, inp)

        recs = [
            r for r in reporter.records()
            if r.kind is DiagnosticKind.HIDDEN_GROUP_MERGED
        ]
        assert len(recs) == 1, (
            f'LSTM h/c must yield exactly one HIDDEN_GROUP_MERGED record; '
            f'got {recs}'
        )
        assert recs[0].level is DiagnosticLevel.INFO
        assert set(recs[0].hidden_paths) == {('h',), ('c',)}

    def test_singleton_group_emits_no_merge_record(self):
        from braintrace._compiler.diagnostics import (
            DiagnosticKind,
            diagnostic_context,
        )
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        inp = brainstate.random.rand(3)

        with diagnostic_context() as reporter:
            find_hidden_groups_from_module(gru, inp)

        assert not any(
            r.kind is DiagnosticKind.HIDDEN_GROUP_MERGED
            for r in reporter.records()
        )

    def test_check_consistent_varshape_raises_on_mismatch(self):
        from braintrace._misc import NotSupportedError

        class _FakeState:
            def __init__(self, varshape):
                self.varshape = varshape
                self.num_state = 1

        group = braintrace.HiddenGroup(
            index=0,
            hidden_paths=[('a',), ('b',)],
            hidden_states=[_FakeState((3,)), _FakeState((4,))],
            hidden_invars=[],
            hidden_outvars=[],
            transition_jaxpr=None,
            transition_jaxpr_constvars=[],
        )
        with pytest.raises(NotSupportedError):
            group.check_consistent_varshape()


# ---------------------------------------------------------------------------
# Phase 3: weight-free while loops as opaque forward nodes
# ---------------------------------------------------------------------------

import jax.numpy as jnp

from braintrace import DiagnosticKind, DiagnosticLevel
from braintrace._compiler.diagnostics import diagnostic_context
from braintrace._compiler.hidden_group import (
    _transition_contains_while,
    block_diagonal_last_dim,
    find_hidden_groups_from_minfo,
    jacfwd_last_dim,
    jacrev_last_dim,
)

_WHILE_K = 3


def _settle_step(h, x):
    return h + 0.5 * jnp.tanh(x - h)


def _settle_twin(h, x):
    """Hand-composed K-step equivalent of the while fixture (no while eqn,
    so reverse-mode oracles work on it)."""
    for _ in range(_WHILE_K):
        h = _settle_step(h, x)
    return h


class WhileSettleCell(brainstate.nn.Module):
    """Weight-free ``lax.while_loop`` reading and updating the hidden state.

    Runs exactly ``_WHILE_K`` iterations of the element-wise settle step, so
    ``_settle_twin`` is its exact hand-composed equivalent.
    """

    def __init__(self, n, k=_WHILE_K):
        super().__init__()
        self.k = k
        self.h = brainstate.HiddenState(jnp.zeros(n))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        h_prev = self.h.value

        def cond_fn(s):
            return s[0] < self.k

        def body_fn(s):
            i, h = s
            return i + 1, _settle_step(h, x)

        _, h_new = jax.lax.while_loop(cond_fn, body_fn, (0, h_prev))
        self.h.value = h_new
        return h_new


class WhileMixingCell(brainstate.nn.Module):
    """While body applying a (constant) recurrent matrix to the carried
    hidden state — cross-position coupling inside the loop."""

    def __init__(self, n, k=2):
        super().__init__()
        self.k = k
        self.R = 0.1 * brainstate.random.randn(n, n)
        self.h = brainstate.HiddenState(jnp.zeros(n))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        h_prev = self.h.value

        def cond_fn(s):
            return s[0] < self.k

        def body_fn(s):
            i, h = s
            return i + 1, jnp.tanh(h @ self.R + x)

        _, h_new = jax.lax.while_loop(cond_fn, body_fn, (0, h_prev))
        self.h.value = h_new
        return h_new


class WhileInputProjCell(brainstate.nn.Module):
    """While body whose ``dot_general`` consumes only loop constants
    (``x @ W``) — no cross-position coupling of the carried hidden."""

    def __init__(self, n, k=2):
        super().__init__()
        self.k = k
        self.W = 0.1 * brainstate.random.randn(n, n)
        self.h = brainstate.HiddenState(jnp.zeros(n))

    def init_state(self, *args, **kwargs):
        self.h.value = jnp.zeros_like(self.h.value)

    def update(self, x):
        h_prev = self.h.value

        def cond_fn(s):
            return s[0] < self.k

        def body_fn(s):
            i, h = s
            return i + 1, h + 0.5 * jnp.tanh(x @ self.W - h)

        _, h_new = jax.lax.while_loop(cond_fn, body_fn, (0, h_prev))
        self.h.value = h_new
        return h_new


def _constvals_for(group, x):
    """Values for ``group.transition_jaxpr_constvars``: the float vector
    matching ``x``'s shape gets ``x``; every other constvar (e.g. the loop
    counter's initial value) gets zeros of its aval."""
    vals = []
    for v in group.transition_jaxpr_constvars:
        aval = v.aval
        if aval.shape == x.shape and jnp.issubdtype(aval.dtype, jnp.floating):
            vals.append(x)
        else:
            vals.append(jnp.zeros(aval.shape, aval.dtype))
    return vals


class TestJacfwdLastDim:
    """Forward-mode last-dim Jacobian extraction (`while` has no
    reverse-mode rule, so opaque-forward transitions need these)."""

    def test_matches_jacrev_on_position_diagonal_map(self):
        h = brainstate.random.rand(5, 2)

        def fn(hid):
            a, b = hid[..., 0], hid[..., 1]
            return jnp.stack([jnp.tanh(a) * b, a + jnp.sin(b)], axis=-1)

        fwd = jacfwd_last_dim(fn, h)
        rev = jacrev_last_dim(fn, h)
        assert fwd.shape == rev.shape == (5, 2, 2)
        assert u.math.allclose(fwd, rev, atol=1e-6)

    def test_block_diagonal_forward_mode_matches_reverse(self):
        h = brainstate.random.rand(4, 2)
        R = brainstate.random.rand(4, 4)

        def fn(hid):
            return jnp.tanh(jnp.einsum('ps,pq->qs', hid, R))

        fwd = block_diagonal_last_dim(fn, h, use_forward_mode=True)
        rev = block_diagonal_last_dim(fn, h)
        assert fwd.shape == rev.shape == (4, 2, 2)
        assert u.math.allclose(fwd, rev, atol=1e-6)

    def test_jacfwd_through_while_matches_twin(self):
        x = brainstate.random.rand(5)

        def fn_while(hid):
            def cond_fn(s):
                return s[0] < _WHILE_K

            def body_fn(s):
                i, h = s
                return i + 1, _settle_step(h, x)

            _, h_final = jax.lax.while_loop(cond_fn, body_fn, (0, hid[..., 0]))
            return h_final[..., None]

        def fn_twin(hid):
            return _settle_twin(hid[..., 0], x)[..., None]

        h = brainstate.random.rand(5, 1)
        fwd = jacfwd_last_dim(fn_while, h)
        rev = jacrev_last_dim(fn_twin, h)
        assert fwd.shape == (5, 1, 1)
        assert u.math.allclose(fwd, rev, atol=1e-6)

    def test_transition_contains_while_helper(self):
        x = brainstate.random.rand(3)

        def with_while(h):
            def body_fn(s):
                i, hh = s
                return i + 1, _settle_step(hh, x)

            return jax.lax.while_loop(lambda s: s[0] < 2, body_fn, (0, h))[1]

        jaxpr_w = jax.make_jaxpr(with_while)(x).jaxpr
        jaxpr_plain = jax.make_jaxpr(lambda h: jnp.tanh(h))(x).jaxpr
        assert _transition_contains_while(jaxpr_w)
        assert not _transition_contains_while(jaxpr_plain)


class TestWhileOpaqueForwardGroups:
    """A weight-free while reading/updating the hidden state becomes an
    opaque forward node in the hidden-to-hidden transition."""

    def _compile_settle(self, n=4):
        cell = WhileSettleCell(n)
        brainstate.nn.init_all_states(cell)
        x = brainstate.random.rand(n)
        with diagnostic_context() as reporter:
            groups, path_to_group = find_hidden_groups_from_module(cell, x)
        return cell, x, groups, path_to_group, reporter

    def test_one_group_with_while_in_transition(self):
        _, _, groups, _, reporter = self._compile_settle()
        assert len(groups) == 1
        names = [e.primitive.name for e in groups[0].transition_jaxpr.eqns]
        assert 'while' in names
        kinds = [r.kind for r in reporter.records()]
        assert DiagnosticKind.CONTROL_FLOW_OPAQUE_FWD in kinds

    def test_transition_evaluates_and_matches_twin(self):
        _, x, groups, _, _ = self._compile_settle()
        group = groups[0]
        h0 = brainstate.random.rand(*group.hidden_invars[0].aval.shape)
        out = group.transition([h0], _constvals_for(group, x))
        expect = _settle_twin(h0, x)
        assert u.math.allclose(out[0], expect, atol=1e-6)

    def test_diagonal_jacobian_matches_twin(self):
        _, x, groups, _, _ = self._compile_settle()
        group = groups[0]
        n = group.hidden_invars[0].aval.shape[0]
        h0 = brainstate.random.rand(n)

        diag = group.diagonal_jacobian([h0], _constvals_for(group, x))

        twin_full = jax.jacrev(lambda h: _settle_twin(h, x))(h0)
        expect = jnp.diagonal(twin_full)[..., None, None]
        assert diag.shape == (n, 1, 1)
        assert u.math.allclose(diag, expect, atol=1e-5)

    def test_while_hidden_error_policy_raises_in_group_pass(self):
        """``find_hidden_groups_from_minfo`` must consult the policy the
        canonicalizer ran with (threaded through ``ModuleInfo``)."""
        cell = WhileSettleCell(4)
        brainstate.nn.init_all_states(cell)
        x = brainstate.random.rand(4)
        policy = braintrace.ControlFlowPolicy(while_hidden='error')
        minfo = braintrace.extract_module_info(cell, x, control_flow=policy)
        with pytest.raises(NotImplementedError):
            find_hidden_groups_from_minfo(minfo)


class TestWhileRecurrentMixingGuard:
    """A while body applying a recurrent weight-mixing primitive to the
    carried hidden state is a boundary in the default ("without
    recurrence") grouping mode."""

    def _compile_mixing(self, include_recurrent_mixing=False, n=4):
        cell = WhileMixingCell(n)
        brainstate.nn.init_all_states(cell)
        x = brainstate.random.rand(n)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with diagnostic_context() as reporter:
                groups, path_to_group = find_hidden_groups_from_module(
                    cell, x, include_recurrent_mixing=include_recurrent_mixing,
                )
        return cell, x, groups, reporter

    def test_default_mode_falls_back_to_zero_recurrence(self):
        _, x, groups, reporter = self._compile_mixing()
        assert len(groups) == 1
        group = groups[0]
        # zero-recurrence fallback: empty transition, D == 0
        assert list(group.transition_jaxpr.eqns) == []
        h0 = brainstate.random.rand(*group.hidden_invars[0].aval.shape)
        const_vals = [
            brainstate.random.rand(*v.aval.shape)
            for v in group.transition_jaxpr_constvars
        ]
        diag = group.diagonal_jacobian([h0], const_vals)
        assert u.math.allclose(diag, jnp.zeros_like(diag))
        kinds = [r.kind for r in reporter.records()]
        assert DiagnosticKind.CONTROL_FLOW_RECURRENT_MIXING in kinds
        record = next(
            r for r in reporter.records()
            if r.kind is DiagnosticKind.CONTROL_FLOW_RECURRENT_MIXING
        )
        assert record.level is DiagnosticLevel.WARNING

    def test_include_recurrent_mixing_traces_through_while(self):
        cell, x, groups, reporter = self._compile_mixing(include_recurrent_mixing=True)
        assert len(groups) == 1
        group = groups[0]
        names = [e.primitive.name for e in group.transition_jaxpr.eqns]
        assert 'while' in names
        assert group.is_diagonal_recurrence is False
        kinds = [r.kind for r in reporter.records()]
        assert DiagnosticKind.CONTROL_FLOW_RECURRENT_MIXING not in kinds

        # forward-mode block diagonal equals the hand-composed twin's blocks.
        # The mixing cell's constvars are the counter init (int) plus two
        # float arrays distinguishable by shape: x (n,) and R (n, n).
        n = group.hidden_invars[0].aval.shape[0]
        h0 = brainstate.random.rand(n)
        const_vals = []
        for v in group.transition_jaxpr_constvars:
            aval = v.aval
            if aval.shape == x.shape and jnp.issubdtype(aval.dtype, jnp.floating):
                const_vals.append(x)
            elif aval.shape == cell.R.shape and jnp.issubdtype(aval.dtype, jnp.floating):
                const_vals.append(jnp.asarray(cell.R))
            else:
                const_vals.append(jnp.zeros(aval.shape, aval.dtype))
        diag = group.diagonal_jacobian([h0], const_vals)

        def twin(h):
            for _i in range(cell.k):
                h = jnp.tanh(h @ cell.R + x)
            return h

        full = jax.jacrev(twin)(h0)
        expect = jnp.diagonal(full)[..., None, None]
        assert diag.shape == (n, 1, 1)
        assert u.math.allclose(diag, expect, atol=1e-5)

    def test_input_projection_in_body_is_not_a_boundary(self):
        cell = WhileInputProjCell(4)
        brainstate.nn.init_all_states(cell)
        x = brainstate.random.rand(4)
        with diagnostic_context() as reporter:
            groups, _pg = find_hidden_groups_from_module(cell, x)
        assert len(groups) == 1
        names = [e.primitive.name for e in groups[0].transition_jaxpr.eqns]
        assert 'while' in names
        kinds = [r.kind for r in reporter.records()]
        assert DiagnosticKind.CONTROL_FLOW_RECURRENT_MIXING not in kinds
