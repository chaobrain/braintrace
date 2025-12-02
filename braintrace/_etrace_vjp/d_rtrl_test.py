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

import brainstate
import brainunit as u
import pytest

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



class TestDiagOn2:
    @pytest.mark.parametrize(
        "cls",
        [
            # braintrace.nn.GRUCell,
            # braintrace.nn.LSTMCell,
            braintrace.nn.LRUCell,
            # braintrace.nn.MGUCell,
            # braintrace.nn.MinimalRNNCell,
        ]
    )
    def test_rnn_single_step_vjp(self, cls):
        n_in = 4
        n_rec = 5
        n_seq = 10
        model = cls(n_in, n_rec)
        model = brainstate.nn.init_all_states(model)

        inputs = brainstate.random.randn(n_seq, n_in)
        algorithm = braintrace.ParamDimVjpAlgorithm(model)
        algorithm.compile_graph(inputs[0])

        outs = brainstate.transform.for_loop(algorithm, inputs)
        print(outs.shape)

        @brainstate.transform.jit
        def grad_single_step_vjp(inp):
            return brainstate.transform.grad(
                lambda inp: algorithm(inp).sum(),
                model.states(brainstate.ParamState)
            )(inp)

        grads = grad_single_step_vjp(inputs[0])
        grads = grad_single_step_vjp(inputs[1])
        print(brainstate.util.PrettyDict(grads))

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
    def test_rnn_multi_step_vjp(self, cls):
        n_in = 4
        n_rec = 5
        n_seq = 10
        model = cls(n_in, n_rec)
        model = brainstate.nn.init_all_states(model)

        inputs = brainstate.random.randn(n_seq, n_in)
        algorithm = braintrace.ParamDimVjpAlgorithm(model, vjp_method='multi-step')
        algorithm.compile_graph(inputs[0])

        outs = algorithm(braintrace.MultiStepData(inputs))
        print(outs.shape)

        @brainstate.transform.jit
        def grad_single_step_vjp(inp):
            return brainstate.transform.grad(
                lambda inp: algorithm(braintrace.MultiStepData(inp)).sum(),
                model.states(brainstate.ParamState)
            )(inp)

        grads = grad_single_step_vjp(inputs[:1])
        print(brainstate.util.PrettyDict(grads))
        print()
        grads = grad_single_step_vjp(inputs[1:2])
        print(brainstate.util.PrettyDict(grads))

    @pytest.mark.parametrize(
        "cls",
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
    def test_snn_single_step_vjp(self, cls):
        with brainstate.environ.context(dt=0.1 * u.ms):
            print(cls)

            n_in = 4
            n_rec = 5
            n_seq = 10
            model = cls(n_in, n_rec)
            model = brainstate.nn.init_all_states(model)

            param_states = model.states(brainstate.ParamState).to_dict_values()

            inputs = brainstate.random.randn(n_seq, n_in)
            algorithm = braintrace.ParamDimVjpAlgorithm(model)
            algorithm.compile_graph(inputs[0])

            outs = brainstate.transform.for_loop(algorithm, inputs)
            print(outs.shape)

            @brainstate.transform.jit
            def grad_single_step_vjp(inp):
                return brainstate.transform.grad(
                    lambda inp: algorithm(inp).sum(),
                    model.states(brainstate.ParamState)
                )(inp)

            grads = grad_single_step_vjp(inputs[0])
            grads = grad_single_step_vjp(inputs[1])
            print(brainstate.util.PrettyDict(grads))

            for k in grads:
                assert u.get_unit(param_states[k]) == u.get_unit(grads[k])

    @pytest.mark.parametrize(
        "cls",
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
    def test_snn_multi_step_vjp(self, cls):
        with brainstate.environ.context(dt=0.1 * u.ms):
            print(cls)

            n_in = 4
            n_rec = 5
            n_seq = 10
            model = cls(n_in, n_rec)
            model = brainstate.nn.init_all_states(model)

            param_states = model.states(brainstate.ParamState).to_dict_values()

            inputs = brainstate.random.randn(n_seq, n_in)
            algorithm = braintrace.ParamDimVjpAlgorithm(model, vjp_method='multi-step')
            algorithm.compile_graph(inputs[0])

            outs = algorithm(braintrace.MultiStepData(inputs))
            print(outs.shape)

            @brainstate.transform.jit
            def grad_single_step_vjp(inp):
                return brainstate.transform.grad(
                    lambda inp: algorithm(braintrace.MultiStepData(inp)).sum(),
                    model.states(brainstate.ParamState)
                )(inp)

            grads = grad_single_step_vjp(inputs[:1])
            print(brainstate.util.PrettyDict(grads))
            print()
            grads = grad_single_step_vjp(inputs[1:2])
            print(brainstate.util.PrettyDict(grads))

            for k in grads:
                assert u.get_unit(param_states[k]) == u.get_unit(grads[k])
