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
import jax.numpy as jnp

from braintrace._etrace_concepts import ETraceParam, ElemWiseParam
from braintrace._etrace_operators import MatMulOp, ElemWiseOp
from braintrace._state_managment import (
    assign_dict_state_values,
    assign_state_values_v2,
    sequence_split_state_values,
    split_dict_states_v2,
)


# ---------------------------------------------------------------------------
# assign_dict_state_values
# ---------------------------------------------------------------------------

class TestAssignDictStateValues(unittest.TestCase):

    def test_write_true_assigns_values(self):
        s1 = brainstate.State(jnp.array([1.0, 2.0]))
        s2 = brainstate.State(jnp.array([3.0]))
        states = {('a',): s1, ('b',): s2}
        new_vals = {('a',): jnp.array([10.0, 20.0]), ('b',): jnp.array([30.0])}

        assign_dict_state_values(states, new_vals, write=True)

        self.assertTrue(jnp.allclose(s1.value, jnp.array([10.0, 20.0])))
        self.assertTrue(jnp.allclose(s2.value, jnp.array([30.0])))

    def test_write_false_restores_values(self):
        s1 = brainstate.State(jnp.array([1.0]))
        states = {('x',): s1}
        new_vals = {('x',): jnp.array([99.0])}

        assign_dict_state_values(states, new_vals, write=False)

        self.assertTrue(jnp.allclose(s1.value, jnp.array([99.0])))

    def test_write_default_is_true(self):
        s1 = brainstate.State(jnp.array([0.0]))
        states = {('k',): s1}
        new_vals = {('k',): jnp.array([5.0])}

        assign_dict_state_values(states, new_vals)

        self.assertTrue(jnp.allclose(s1.value, jnp.array([5.0])))

    def test_mismatched_keys_raises_value_error(self):
        s1 = brainstate.State(jnp.array([1.0]))
        states = {('a',): s1}
        new_vals = {('b',): jnp.array([1.0])}

        with self.assertRaises(ValueError):
            assign_dict_state_values(states, new_vals)

    def test_extra_key_in_values_raises(self):
        s1 = brainstate.State(jnp.array([1.0]))
        states = {('a',): s1}
        new_vals = {('a',): jnp.array([1.0]), ('b',): jnp.array([2.0])}

        with self.assertRaises(ValueError):
            assign_dict_state_values(states, new_vals)

    def test_missing_key_in_values_raises(self):
        s1 = brainstate.State(jnp.array([1.0]))
        s2 = brainstate.State(jnp.array([2.0]))
        states = {('a',): s1, ('b',): s2}
        new_vals = {('a',): jnp.array([1.0])}

        with self.assertRaises(ValueError):
            assign_dict_state_values(states, new_vals)

    def test_empty_dicts(self):
        assign_dict_state_values({}, {})

    def test_multiple_states(self):
        states = {}
        new_vals = {}
        originals = {}
        for i in range(5):
            key = (f's{i}',)
            val = jnp.array([float(i)])
            states[key] = brainstate.State(val)
            new_vals[key] = jnp.array([float(i * 10)])
            originals[key] = val

        assign_dict_state_values(states, new_vals)

        for key in states:
            self.assertTrue(jnp.allclose(states[key].value, new_vals[key]))


# ---------------------------------------------------------------------------
# assign_state_values_v2
# ---------------------------------------------------------------------------

class TestAssignStateValuesV2(unittest.TestCase):

    def test_write_true_assigns(self):
        s1 = brainstate.State(jnp.array([1.0]))
        states = {0: s1}
        vals = {0: jnp.array([42.0])}

        assign_state_values_v2(states, vals, write=True)

        self.assertTrue(jnp.allclose(s1.value, jnp.array([42.0])))

    def test_write_false_restores(self):
        s1 = brainstate.State(jnp.array([1.0]))
        states = {0: s1}
        vals = {0: jnp.array([42.0])}

        assign_state_values_v2(states, vals, write=False)

        self.assertTrue(jnp.allclose(s1.value, jnp.array([42.0])))

    def test_default_write_is_true(self):
        s1 = brainstate.State(jnp.array([0.0]))
        states = {'key': s1}
        vals = {'key': jnp.array([7.0])}

        assign_state_values_v2(states, vals)

        self.assertTrue(jnp.allclose(s1.value, jnp.array([7.0])))

    def test_mismatched_keys_raises_assertion(self):
        s1 = brainstate.State(jnp.array([1.0]))
        states = {'a': s1}
        vals = {'b': jnp.array([1.0])}

        with self.assertRaises(ValueError):
            assign_state_values_v2(states, vals)

    def test_empty_dicts(self):
        assign_state_values_v2({}, {})

    def test_hashable_int_keys(self):
        s1 = brainstate.State(jnp.array([1.0]))
        s2 = brainstate.State(jnp.array([2.0]))
        states = {10: s1, 20: s2}
        vals = {10: jnp.array([100.0]), 20: jnp.array([200.0])}

        assign_state_values_v2(states, vals)

        self.assertTrue(jnp.allclose(s1.value, jnp.array([100.0])))
        self.assertTrue(jnp.allclose(s2.value, jnp.array([200.0])))

    def test_hashable_tuple_keys(self):
        s1 = brainstate.State(jnp.array([1.0]))
        states = {('layer', 'weight'): s1}
        vals = {('layer', 'weight'): jnp.array([99.0])}

        assign_state_values_v2(states, vals)

        self.assertTrue(jnp.allclose(s1.value, jnp.array([99.0])))

    def test_multidimensional_values(self):
        s1 = brainstate.State(jnp.zeros((3, 4)))
        states = {0: s1}
        new_val = jnp.ones((3, 4))
        vals = {0: new_val}

        assign_state_values_v2(states, vals)

        self.assertTrue(jnp.allclose(s1.value, new_val))


# ---------------------------------------------------------------------------
# sequence_split_state_values
# ---------------------------------------------------------------------------

class TestSequenceSplitStateValues(unittest.TestCase):

    def test_include_weight_true(self):
        param = brainstate.ParamState(jnp.array([1.0]))
        hidden = brainstate.HiddenState(jnp.array([2.0]))
        other = brainstate.ShortTermState(jnp.array([3.0]))

        states = [param, hidden, other]
        values = [jnp.array([10.0]), jnp.array([20.0]), jnp.array([30.0])]

        w, h, o = sequence_split_state_values(states, values, include_weight=True)

        self.assertEqual(len(w), 1)
        self.assertEqual(len(h), 1)
        self.assertEqual(len(o), 1)
        self.assertTrue(jnp.allclose(w[0], jnp.array([10.0])))
        self.assertTrue(jnp.allclose(h[0], jnp.array([20.0])))
        self.assertTrue(jnp.allclose(o[0], jnp.array([30.0])))

    def test_include_weight_false(self):
        param = brainstate.ParamState(jnp.array([1.0]))
        hidden = brainstate.HiddenState(jnp.array([2.0]))
        other = brainstate.ShortTermState(jnp.array([3.0]))

        states = [param, hidden, other]
        values = [jnp.array([10.0]), jnp.array([20.0]), jnp.array([30.0])]

        h, o = sequence_split_state_values(states, values, include_weight=False)

        self.assertEqual(len(h), 1)
        self.assertEqual(len(o), 1)
        self.assertTrue(jnp.allclose(h[0], jnp.array([20.0])))
        self.assertTrue(jnp.allclose(o[0], jnp.array([30.0])))

    def test_default_include_weight_is_true(self):
        param = brainstate.ParamState(jnp.array([1.0]))
        states = [param]
        values = [jnp.array([10.0])]

        result = sequence_split_state_values(states, values)

        self.assertEqual(len(result), 3)
        w, h, o = result
        self.assertEqual(len(w), 1)
        self.assertEqual(len(h), 0)
        self.assertEqual(len(o), 0)

    def test_empty_inputs(self):
        w, h, o = sequence_split_state_values([], [])
        self.assertEqual(w, [])
        self.assertEqual(h, [])
        self.assertEqual(o, [])

    def test_empty_inputs_no_weight(self):
        h, o = sequence_split_state_values([], [], include_weight=False)
        self.assertEqual(h, [])
        self.assertEqual(o, [])

    def test_only_params(self):
        p1 = brainstate.ParamState(jnp.array([1.0]))
        p2 = brainstate.ParamState(jnp.array([2.0]))
        states = [p1, p2]
        values = [jnp.array([10.0]), jnp.array([20.0])]

        w, h, o = sequence_split_state_values(states, values)

        self.assertEqual(len(w), 2)
        self.assertEqual(len(h), 0)
        self.assertEqual(len(o), 0)

    def test_only_hidden(self):
        h1 = brainstate.HiddenState(jnp.array([1.0]))
        h2 = brainstate.HiddenState(jnp.array([2.0]))
        states = [h1, h2]
        values = [jnp.array([10.0]), jnp.array([20.0])]

        w, h, o = sequence_split_state_values(states, values)

        self.assertEqual(len(w), 0)
        self.assertEqual(len(h), 2)
        self.assertEqual(len(o), 0)

    def test_only_other(self):
        o1 = brainstate.ShortTermState(jnp.array([1.0]))
        states = [o1]
        values = [jnp.array([10.0])]

        w, h, o = sequence_split_state_values(states, values)

        self.assertEqual(len(w), 0)
        self.assertEqual(len(h), 0)
        self.assertEqual(len(o), 1)

    def test_etrace_param_classified_as_weight(self):
        ep = ETraceParam(jnp.ones((3, 3)), op=MatMulOp())
        states = [ep]
        values = [jnp.array([1.0])]

        w, h, o = sequence_split_state_values(states, values)

        self.assertEqual(len(w), 1)
        self.assertEqual(len(h), 0)
        self.assertEqual(len(o), 0)

    def test_etrace_param_excluded_when_no_weight(self):
        ep = ETraceParam(jnp.ones((3, 3)), op=MatMulOp())
        states = [ep]
        values = [jnp.array([1.0])]

        h, o = sequence_split_state_values(states, values, include_weight=False)

        self.assertEqual(len(h), 0)
        self.assertEqual(len(o), 0)

    def test_mixed_ordering_preserved(self):
        p = brainstate.ParamState(jnp.array([1.0]))
        h = brainstate.HiddenState(jnp.array([2.0]))
        o = brainstate.ShortTermState(jnp.array([3.0]))
        p2 = brainstate.ParamState(jnp.array([4.0]))
        h2 = brainstate.HiddenState(jnp.array([5.0]))

        states = [p, h, o, p2, h2]
        values = [jnp.array([10.0]), jnp.array([20.0]), jnp.array([30.0]),
                  jnp.array([40.0]), jnp.array([50.0])]

        ws, hs, os = sequence_split_state_values(states, values)

        self.assertEqual(len(ws), 2)
        self.assertEqual(len(hs), 2)
        self.assertEqual(len(os), 1)
        self.assertTrue(jnp.allclose(ws[0], jnp.array([10.0])))
        self.assertTrue(jnp.allclose(ws[1], jnp.array([40.0])))
        self.assertTrue(jnp.allclose(hs[0], jnp.array([20.0])))
        self.assertTrue(jnp.allclose(hs[1], jnp.array([50.0])))
        self.assertTrue(jnp.allclose(os[0], jnp.array([30.0])))


# ---------------------------------------------------------------------------
# split_dict_states_v2
# ---------------------------------------------------------------------------

class TestSplitDictStatesV2(unittest.TestCase):

    def test_hidden_state_categorized(self):
        h = brainstate.HiddenState(jnp.array([1.0]))
        states = {('model', 'h'): h}

        etrace, hidden, param, other = split_dict_states_v2(states)

        self.assertEqual(len(etrace), 0)
        self.assertEqual(len(hidden), 1)
        self.assertEqual(len(param), 0)
        self.assertEqual(len(other), 0)
        self.assertIs(hidden[('model', 'h')], h)

    def test_param_state_categorized(self):
        p = brainstate.ParamState(jnp.array([1.0]))
        states = {('model', 'w'): p}

        etrace, hidden, param, other = split_dict_states_v2(states)

        self.assertEqual(len(etrace), 0)
        self.assertEqual(len(hidden), 0)
        self.assertEqual(len(param), 1)
        self.assertEqual(len(other), 0)
        self.assertIs(param[('model', 'w')], p)

    def test_other_state_categorized(self):
        s = brainstate.ShortTermState(jnp.array([1.0]))
        states = {('model', 's'): s}

        etrace, hidden, param, other = split_dict_states_v2(states)

        self.assertEqual(len(etrace), 0)
        self.assertEqual(len(hidden), 0)
        self.assertEqual(len(param), 0)
        self.assertEqual(len(other), 1)
        self.assertIs(other[('model', 's')], s)

    def test_etrace_param_with_is_etrace_true(self):
        ep = ETraceParam(jnp.ones((3, 3)), op=MatMulOp())
        self.assertTrue(ep.is_etrace)
        states = {('model', 'ep'): ep}

        etrace, hidden, param, other = split_dict_states_v2(states)

        self.assertEqual(len(etrace), 1)
        self.assertEqual(len(hidden), 0)
        self.assertEqual(len(param), 0)
        self.assertEqual(len(other), 0)
        self.assertIs(etrace[('model', 'ep')], ep)

    def test_etrace_param_with_is_etrace_false(self):
        ep = ETraceParam(jnp.ones((3, 3)), op=MatMulOp())
        ep.is_etrace = False
        states = {('model', 'ep'): ep}

        etrace, hidden, param, other = split_dict_states_v2(states)

        self.assertEqual(len(etrace), 0)
        self.assertEqual(len(hidden), 0)
        self.assertEqual(len(param), 1)
        self.assertEqual(len(other), 0)
        self.assertIs(param[('model', 'ep')], ep)

    def test_elem_wise_param_with_is_etrace_true(self):
        ewp = ElemWiseParam(jnp.ones(5))
        self.assertTrue(ewp.is_etrace)
        states = {('model', 'ew'): ewp}

        etrace, hidden, param, other = split_dict_states_v2(states)

        self.assertEqual(len(etrace), 1)
        self.assertEqual(len(hidden), 0)
        self.assertEqual(len(param), 0)
        self.assertEqual(len(other), 0)

    def test_elem_wise_param_with_is_etrace_false(self):
        ewp = ElemWiseParam(jnp.ones(5))
        ewp.is_etrace = False
        states = {('model', 'ew'): ewp}

        etrace, hidden, param, other = split_dict_states_v2(states)

        self.assertEqual(len(etrace), 0)
        self.assertEqual(len(hidden), 0)
        self.assertEqual(len(param), 1)
        self.assertEqual(len(other), 0)

    def test_empty_dict(self):
        etrace, hidden, param, other = split_dict_states_v2({})

        self.assertEqual(len(etrace), 0)
        self.assertEqual(len(hidden), 0)
        self.assertEqual(len(param), 0)
        self.assertEqual(len(other), 0)

    def test_mixed_states(self):
        h = brainstate.HiddenState(jnp.array([1.0]))
        p = brainstate.ParamState(jnp.array([2.0]))
        ep = ETraceParam(jnp.ones((3, 3)), op=MatMulOp())
        ep_no = ETraceParam(jnp.ones((3, 3)), op=MatMulOp())
        ep_no.is_etrace = False
        s = brainstate.ShortTermState(jnp.array([3.0]))

        states = {
            ('h',): h,
            ('p',): p,
            ('ep',): ep,
            ('ep_no',): ep_no,
            ('s',): s,
        }

        etrace, hidden, param, other = split_dict_states_v2(states)

        self.assertEqual(len(etrace), 1)
        self.assertIn(('ep',), etrace)
        self.assertEqual(len(hidden), 1)
        self.assertIn(('h',), hidden)
        self.assertEqual(len(param), 2)
        self.assertIn(('p',), param)
        self.assertIn(('ep_no',), param)
        self.assertEqual(len(other), 1)
        self.assertIn(('s',), other)

    def test_keys_preserved(self):
        h = brainstate.HiddenState(jnp.array([1.0]))
        key = ('layer1', 'cell', 'hidden')
        states = {key: h}

        _, hidden, _, _ = split_dict_states_v2(states)

        self.assertIn(key, hidden)
        self.assertIs(hidden[key], h)

    def test_multiple_hidden_states(self):
        h1 = brainstate.HiddenState(jnp.array([1.0]))
        h2 = brainstate.HiddenState(jnp.array([2.0]))
        h3 = brainstate.HiddenState(jnp.array([3.0]))
        states = {('h1',): h1, ('h2',): h2, ('h3',): h3}

        etrace, hidden, param, other = split_dict_states_v2(states)

        self.assertEqual(len(hidden), 3)
        self.assertEqual(len(etrace), 0)
        self.assertEqual(len(param), 0)
        self.assertEqual(len(other), 0)

    def test_multiple_param_states(self):
        p1 = brainstate.ParamState(jnp.array([1.0]))
        p2 = brainstate.ParamState(jnp.array([2.0]))
        states = {('w1',): p1, ('w2',): p2}

        etrace, hidden, param, other = split_dict_states_v2(states)

        self.assertEqual(len(param), 2)
        self.assertEqual(len(etrace), 0)
        self.assertEqual(len(hidden), 0)
        self.assertEqual(len(other), 0)

    def test_multiple_other_states(self):
        s1 = brainstate.ShortTermState(jnp.array([1.0]))
        s2 = brainstate.ShortTermState(jnp.array([2.0]))
        states = {('s1',): s1, ('s2',): s2}

        etrace, hidden, param, other = split_dict_states_v2(states)

        self.assertEqual(len(other), 2)
        self.assertEqual(len(etrace), 0)
        self.assertEqual(len(hidden), 0)
        self.assertEqual(len(param), 0)


if __name__ == '__main__':
    unittest.main()
