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

import warnings

import brainstate
import brainunit as u
import jax.numpy as jnp
from jax import make_jaxpr
import pytest

from braintrace._misc import (
    BaseEnum,
    CompilationError,
    NotSupportedError,
    check_dict_keys,
    deprecation_getattr,
    etrace_df_key,
    etrace_x_key,
    hid_group_key,
    remove_units,
    set_module_as,
    state_traceback,
    unknown_state_path,
)


def _a_jax_var():
    """Return a real jax ``Var`` extracted from a trivial jaxpr."""
    jaxpr = make_jaxpr(lambda a: a + 1.0)(jnp.zeros(3))
    return jaxpr.jaxpr.invars[0]


class TestCheckDictKeys:
    def test_matching_keys_pass_silently(self):
        check_dict_keys({"a": 1, "b": 2}, {"a": 9, "b": 8})

    def test_mismatched_keys_raise(self):
        with pytest.raises(ValueError):
            check_dict_keys({"a": 1}, {"b": 2})


class TestKeyHelpers:
    def test_hid_group_key_formats_id(self):
        assert hid_group_key(3) == "hidden_group_3"

    def test_hid_group_key_rejects_non_int(self):
        with pytest.raises(AssertionError):
            hid_group_key("x")

    def test_etrace_x_key_is_object_id(self):
        obj = object()
        assert etrace_x_key(obj) == id(obj)

    def test_etrace_df_key_pairs_var_id_with_group(self):
        var = _a_jax_var()
        assert etrace_df_key(var, 2) == (id(var), "hidden_group_2")

    def test_etrace_df_key_rejects_non_var(self):
        with pytest.raises(AssertionError):
            etrace_df_key(123, 0)

    def test_unknown_state_path(self):
        assert unknown_state_path(5) == ("_unknown_path_5",)


class TestRemoveUnits:
    def test_strips_units_from_quantity_leaves(self):
        tree = {"q": u.Quantity([1.0, 2.0, 3.0], unit=u.mV), "plain": jnp.ones(2)}
        out = remove_units(tree)
        assert not isinstance(out["q"], u.Quantity)
        assert jnp.allclose(out["q"], jnp.array([1.0, 2.0, 3.0]))
        # Non-quantity leaves are passed through unchanged.
        assert jnp.allclose(out["plain"], jnp.ones(2))


class TestDeprecationGetattr:
    def test_deprecated_attribute_warns_and_returns_replacement(self):
        sentinel = object()
        getter = deprecation_getattr(
            "mymod", {"old": ("old is deprecated, use new", sentinel)}
        )
        with pytest.warns(DeprecationWarning):
            assert getter("old") is sentinel

    def test_accelerated_deprecation_raises_attribute_error(self):
        getter = deprecation_getattr("mymod", {"gone": ("gone was removed", None)})
        with pytest.raises(AttributeError):
            getter("gone")

    def test_unknown_attribute_raises_attribute_error(self):
        getter = deprecation_getattr("mymod", {})
        with pytest.raises(AttributeError):
            getter("does_not_exist")


class TestSetModuleAs:
    def test_sets_explicit_module(self):
        @set_module_as("braintrace.sub")
        def fn():
            return 1

        assert fn.__module__ == "braintrace.sub"

    def test_defaults_to_braintrace(self):
        @set_module_as()
        def fn():
            return 1

        assert fn.__module__ == "braintrace"


class TestExceptions:
    @pytest.mark.parametrize("exc", [NotSupportedError, CompilationError])
    def test_exception_module_and_inheritance(self, exc):
        assert exc.__module__ == "braintrace"
        assert issubclass(exc, Exception)
        with pytest.raises(exc):
            raise exc("boom")


class TestStateTraceback:
    def test_traceback_string_contains_state_index(self):
        st = brainstate.ParamState(jnp.zeros(2))
        text = state_traceback([st])
        assert isinstance(text, str)
        assert "State 0" in text


class _Color(BaseEnum):
    RED = 1
    GREEN = 2


class TestBaseEnum:
    def test_get_by_name_returns_member(self):
        assert _Color.get_by_name("RED") is _Color.RED

    def test_get_by_name_unknown_raises(self):
        with pytest.raises(ValueError):
            _Color.get_by_name("BLUE")

    def test_get_passthrough_instance(self):
        assert _Color.get(_Color.GREEN) is _Color.GREEN

    def test_get_resolves_string(self):
        assert _Color.get("GREEN") is _Color.GREEN

    def test_get_rejects_other_types(self):
        with pytest.raises(ValueError):
            _Color.get(123)
