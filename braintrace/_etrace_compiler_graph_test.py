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

import braintrace
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
