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
import jax.numpy as jnp
import pytest
import brainunit as u

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


class _PlasticNet(brainstate.nn.Module):
    # ``w`` is trained online; ``w_local`` is updated by a local Hebbian rule.
    def __init__(self, plastic_cls=brainstate.ParamState, write_w=False, n_in=3, n=4):
        super().__init__()
        self.n = n
        self.write_w = write_w
        self.w = brainstate.ParamState(brainstate.random.randn(n_in, n))
        self.w_local = plastic_cls(jnp.ones((n_in, n)))

    def init_state(self, *args, **kwargs):
        self.V = brainstate.HiddenState(jnp.zeros(self.n))

    def update(self, x):
        self.V.value = 0.9 * self.V.value + braintrace.matmul(x, self.w.value) + x @ self.w_local.value
        self.w_local.value = self.w_local.value + 1e-3 * jnp.outer(x, self.V.value)
        if self.write_w:
            self.w.value = self.w.value * 0.99
        return self.V.value


class _Outer(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = _PlasticNet()

    def update(self, x):
        return self.layer(x)


class TestParamStateWriteError:
    """``ParamState`` writes inside the online step raise an actionable error (#187)."""

    def _compile(self, net):
        brainstate.nn.init_all_states(net)
        return braintrace.compile(net, 'd_rtrl', jnp.ones(3))

    def test_message_names_path_and_patterns(self):
        with pytest.raises(braintrace.NotSupportedError) as info:
            self._compile(_PlasticNet())
        msg = str(info.value)
        assert "('w_local',)" in msg
        assert 'ParamState(' not in msg
        assert 'etrace_grad' in msg
        assert 'LongTermState' in msg

    def test_all_written_params_are_listed(self):
        with pytest.raises(braintrace.NotSupportedError) as info:
            self._compile(_PlasticNet(write_w=True))
        msg = str(info.value)
        assert "('w_local',)" in msg
        assert "('w',)" in msg

    def test_nested_module_path(self):
        with pytest.raises(braintrace.NotSupportedError) as info:
            self._compile(_Outer())
        assert "('layer', 'w_local')" in str(info.value)

    def test_unattached_param_falls_back_to_repr(self):
        glob = brainstate.ParamState(jnp.zeros(4))

        class Net(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jnp.ones((3, 4)))

            def init_state(self, *args, **kwargs):
                self.V = brainstate.HiddenState(jnp.zeros(4))

            def update(self, x):
                self.V.value = 0.9 * self.V.value + braintrace.matmul(x, self.w.value) + glob.value
                glob.value = glob.value + 1.
                return self.V.value

        with pytest.raises(braintrace.NotSupportedError) as info:
            self._compile(Net())
        assert 'ParamState(' in str(info.value)

    def test_long_term_state_compiles(self):
        learner = self._compile(_PlasticNet(plastic_cls=brainstate.LongTermState))
        assert learner is not None
