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

from pprint import pprint

import brainstate
import pytest
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


class TestControlFlowPolicyThreading:
    """``extract_module_info`` resolves the control-flow policy once and the
    returned ``ModuleInfo`` carries it, so downstream passes (hidden groups,
    relations, perturbation) consult the same policy the canonicalizer used."""

    def _minfo(self, **kwargs):
        gru = braintrace.nn.GRUCell(3, 4)
        brainstate.nn.init_all_states(gru)
        inputs = brainstate.random.rand(3)
        return braintrace.extract_module_info(gru, inputs, **kwargs)

    def test_default_policy_stored(self):
        from braintrace._compiler import DEFAULT_CONTROL_FLOW_POLICY
        minfo = self._minfo()
        assert minfo.control_flow is DEFAULT_CONTROL_FLOW_POLICY

    def test_custom_policy_round_trips(self):
        policy = braintrace.ControlFlowPolicy(
            cond='opaque', while_hidden='error', etp_in_control_flow='exclude',
        )
        minfo = self._minfo(control_flow=policy)
        assert minfo.control_flow is policy

    def test_add_jaxpr_outs_preserves_policy(self):
        policy = braintrace.ControlFlowPolicy(while_hidden='error')
        minfo = self._minfo(control_flow=policy)
        assert minfo.add_jaxpr_outs([]).control_flow is policy


class Test_extract_model_info:
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
    def test_rnn_one_layer(self, cls):
        n_in = 3
        n_out = 4

        rnn = cls(n_in, n_out)
        brainstate.nn.init_all_states(rnn)

        input = brainstate.random.rand(n_in)
        minfo = braintrace.extract_module_info(rnn, input)
        pprint(minfo)

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
            minfo = braintrace.extract_module_info(layer, input)
            pprint(minfo)

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
            minfo = braintrace.extract_module_info(layer, input)
            pprint(minfo)
