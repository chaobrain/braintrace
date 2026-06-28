# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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
import jax.numpy as jnp
import pytest

from braintrace._state_managment import (
    assign_dict_state_values,
    assign_state_values_v2,
    sequence_split_state_values,
    split_dict_states_v2,
)


class TestAssignDictStateValues:
    def test_write_assigns_values(self):
        states = {("w",): brainstate.ParamState(jnp.zeros(3))}
        assign_dict_state_values(states, {("w",): jnp.ones(3)}, write=True)
        assert jnp.allclose(states[("w",)].value, 1.0)

    def test_restore_recovers_values(self):
        st = brainstate.ParamState(jnp.zeros(3))
        assign_dict_state_values({("w",): st}, {("w",): jnp.full((3,), 2.0)}, write=False)
        assert jnp.allclose(st.value, 2.0)

    def test_key_mismatch_raises(self):
        with pytest.raises(ValueError):
            assign_dict_state_values(
                {("a",): brainstate.ParamState(jnp.zeros(1))},
                {("b",): jnp.ones(1)},
            )


class TestAssignStateValuesV2:
    def test_write_assigns_values_with_hashable_keys(self):
        states = {0: brainstate.ParamState(jnp.zeros(2))}
        assign_state_values_v2(states, {0: jnp.full((2,), 3.0)}, write=True)
        assert jnp.allclose(states[0].value, 3.0)

    def test_restore_recovers_values(self):
        st = brainstate.ParamState(jnp.zeros(2))
        assign_state_values_v2({0: st}, {0: jnp.ones(2)}, write=False)
        assert jnp.allclose(st.value, 1.0)

    def test_key_mismatch_raises(self):
        with pytest.raises(ValueError):
            assign_state_values_v2(
                {0: brainstate.ParamState(jnp.zeros(1))},
                {1: jnp.ones(1)},
            )


def _mixed_states():
    return [
        brainstate.ParamState(jnp.ones(2)),
        brainstate.HiddenState(jnp.zeros(3)),
        brainstate.LongTermState(jnp.zeros(1)),
    ]


def _mixed_values():
    return [jnp.ones(2), jnp.zeros(3), jnp.zeros(1)]


class TestSequenceSplitStateValues:
    def test_split_with_weight(self):
        weights, hidden, other = sequence_split_state_values(
            _mixed_states(), _mixed_values(), include_weight=True
        )
        assert len(weights) == 1
        assert len(hidden) == 1
        assert len(other) == 1

    def test_split_without_weight(self):
        hidden, other = sequence_split_state_values(
            _mixed_states(), _mixed_values(), include_weight=False
        )
        assert len(hidden) == 1
        assert len(other) == 1


class TestSplitDictStatesV2:
    def test_categorises_states_by_type(self):
        w = brainstate.ParamState(jnp.ones(2))
        h = brainstate.HiddenState(jnp.zeros(3))
        o = brainstate.LongTermState(jnp.zeros(1))
        states = {("w",): w, ("h",): h, ("o",): o}

        etrace_params, hidden, params, other = split_dict_states_v2(states)

        assert etrace_params == {("w",): w}
        assert hidden == {("h",): h}
        assert other == {("o",): o}
        # ParamState selection is decided by the compiler, so this stays empty.
        assert params == {}
