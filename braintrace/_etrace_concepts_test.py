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

import brainstate
import brainunit as u

import braintrace
from braintrace._etrace_concepts import ETraceGrad


class TestETraceState(unittest.TestCase):
    def test_init(self):
        value = brainstate.random.randn(10, 10)
        state = brainstate.HiddenState(value)
        self.assertEqual(state.varshape, value.shape)
        self.assertEqual(state.num_state, 1)

    def test_check_value(self):
        with self.assertRaises(TypeError):
            brainstate.HiddenState("invalid_value")


class TestETraceGroupState(unittest.TestCase):
    def test_init(self):
        value = brainstate.random.randn(10, 10, 5)
        state = braintrace.ETraceGroupState(value)
        self.assertEqual(state.varshape, value.shape[:-1])
        self.assertEqual(state.num_state, value.shape[-1])

    def test_get_value(self):
        value = brainstate.random.randn(10, 10, 5)
        state = braintrace.ETraceGroupState(value)
        self.assertTrue(u.math.allclose(state.get_value(0), value[..., 0]))

    def test_set_value(self):
        value = brainstate.random.randn(10, 10, 5)
        state = braintrace.ETraceGroupState(value)
        new_value = brainstate.random.randn(10, 10)
        state.set_value({0: new_value})
        self.assertTrue(u.math.allclose(state.get_value(0), new_value))


class TestETraceTreeState(unittest.TestCase):
    def test_init(self):
        value = {'v': brainstate.random.randn(10, 10) * u.mV,
                 'i': brainstate.random.randn(10, 10) * u.mA}
        state = braintrace.ETraceTreeState(value)
        self.assertEqual(state.varshape, (10, 10))
        self.assertEqual(state.num_state, 2)

    def test_get_value(self):
        value = {'v': brainstate.random.randn(10, 10) * u.mV, 'i': brainstate.random.randn(10, 10) * u.mA}
        state = braintrace.ETraceTreeState(value)
        # print(state.get_value('v'), value['v'])
        self.assertTrue(u.math.allclose(state.get_value('v'), value['v'], ))

    def test_set_value(self):
        value = {'v': brainstate.random.randn(10, 10) * u.mV, 'i': brainstate.random.randn(10, 10) * u.mA}
        state = braintrace.ETraceTreeState(value)
        new_value = brainstate.random.randn(10, 10) * u.mV
        state.set_value({'v': new_value})
        self.assertTrue(u.math.allclose(state.get_value('v'), new_value))


# class TestETraceParam(unittest.TestCase):
#     def test_init(self):
#         weight = brainstate.random.randn(10, 10)
#         op = braintrace.ETraceOp(lambda x, w: x + w)
#         param = braintrace.ETraceParam(weight, op)
#         self.assertEqual(param.gradient, braintrace.ETraceGrad.adaptive)
#
#     def test_execute(self):
#         weight = brainstate.random.randn(10, 10)
#         op = braintrace.ETraceOp(lambda x, w: x + w)
#         param = braintrace.ETraceParam(weight, op)
#         x = brainstate.random.randn(10, 10)
#         result = param.execute(x)
#         self.assertTrue(u.math.allclose(result, x + weight))


class TestElemWiseParam(unittest.TestCase):
    def test_init(self):
        weight = brainstate.random.randn(10, 10)
        op = braintrace.ElemWiseOp(lambda w: w)
        param = braintrace.ElemWiseParam(weight, op)
        self.assertEqual(param.gradient, ETraceGrad.full)

    def test_execute(self):
        weight = brainstate.random.randn(10, 10)
        op = braintrace.ElemWiseOp(lambda w: w)
        param = braintrace.ElemWiseParam(weight, op)
        result = param.execute()
        self.assertTrue(u.math.allclose(result, weight))


class TestNonTempParam(unittest.TestCase):
    def test_init(self):
        weight = brainstate.random.randn(10, 10)
        op = lambda x, w: x + w
        param = braintrace.NonTempParam(weight, op)
        self.assertEqual(param.value.shape, weight.shape)

    def test_execute(self):
        weight = brainstate.random.randn(10, 10)
        op = lambda x, w: x + w
        param = braintrace.NonTempParam(weight, op)
        x = brainstate.random.randn(10, 10)
        result = param.execute(x)
        self.assertTrue(u.math.allclose(result, x + weight))


class TestFakeETraceParam(unittest.TestCase):
    def test_init(self):
        weight = brainstate.random.randn(10, 10)
        op = lambda x, w: x + w
        param = braintrace.FakeETraceParam(weight, op)
        self.assertEqual(param.value.shape, weight.shape)

    def test_execute(self):
        weight = brainstate.random.randn(10, 10)
        op = lambda x, w: x + w
        param = braintrace.FakeETraceParam(weight, op)
        x = brainstate.random.randn(10, 10)
        result = param.execute(x)
        self.assertTrue(u.math.allclose(result, x + weight))


class TestFakeElemWiseParam(unittest.TestCase):
    def test_init(self):
        weight = brainstate.random.randn(10, 10)
        op = braintrace.ElemWiseOp(lambda w: w)
        param = braintrace.FakeElemWiseParam(weight, op)
        self.assertEqual(param.value.shape, weight.shape)

    def test_execute(self):
        weight = brainstate.random.randn(10, 10)
        op = braintrace.ElemWiseOp(lambda w: w)
        param = braintrace.FakeElemWiseParam(weight, op)
        result = param.execute()
        self.assertTrue(u.math.allclose(result, weight))


if __name__ == '__main__':
    unittest.main()
