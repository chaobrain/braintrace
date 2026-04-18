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

"""Tests for the global ETP registries and flag-checking helpers.

These tests pin down the *contracts* of the registry module:
membership of every shipped primitive, lockstep population of the
flag-sets, and the True/False semantics of the three predicates that
the compiler relies on.
"""



from braintrace._compatible_imports import Primitive
from braintrace._etrace_op import (
    BATCHED_PRIMITIVES,
    ETP_PRIMITIVES,
    ETP_RULES_INIT_DRTRL,
    ETP_RULES_INIT_PP,
    ETP_RULES_XY_TO_DW,
    ETP_RULES_YW_TO_W,
    GRADIENT_ENABLED_PRIMITIVES,
    etp_conv_p,
    etp_elemwise_p,
    etp_lora_mm_p,
    etp_lora_mv_p,
    etp_mm_p,
    etp_mv_p,
    etp_sp_mm_p,
    etp_sp_mv_p,
    is_batched_primitive,
    is_etp_enable_gradient_primitive,
    is_etp_primitive,
)

_ALL_SHIPPED = (
    etp_mm_p, etp_mv_p,
    etp_elemwise_p,
    etp_conv_p,
    etp_sp_mm_p, etp_sp_mv_p,
    etp_lora_mm_p, etp_lora_mv_p,
)

_BATCHED = (etp_mm_p, etp_conv_p, etp_sp_mm_p, etp_lora_mm_p)
_UNBATCHED = (etp_mv_p, etp_elemwise_p, etp_sp_mv_p, etp_lora_mv_p)


class TestETPPrimitivesMembership:
    """Every shipped primitive lands in :data:`ETP_PRIMITIVES`."""

    def test_all_shipped_primitives_in_set(self):
        for prim in _ALL_SHIPPED:
            assert prim in ETP_PRIMITIVES, (
                f'{prim.name} missing from ETP_PRIMITIVES'
            )

    def test_set_contains_at_least_eight_entries(self):
        assert len(ETP_PRIMITIVES) >= 8, (
            f'Expected ≥8 primitives, got {len(ETP_PRIMITIVES)}'
        )


class TestRuleDictsPopulated:
    """Every shipped primitive has all four ETP rules registered."""

    def test_yw_to_w_for_every_shipped(self):
        for prim in _ALL_SHIPPED:
            assert prim in ETP_RULES_YW_TO_W, prim.name

    def test_xy_to_dw_for_every_shipped(self):
        for prim in _ALL_SHIPPED:
            assert prim in ETP_RULES_XY_TO_DW, prim.name

    def test_init_drtrl_for_every_shipped(self):
        for prim in _ALL_SHIPPED:
            assert prim in ETP_RULES_INIT_DRTRL, prim.name

    def test_init_pp_for_every_shipped(self):
        for prim in _ALL_SHIPPED:
            assert prim in ETP_RULES_INIT_PP, prim.name


class TestGradientEnabledFlag:
    """Only :data:`etp_elemwise_p` is gradient-enabled today."""

    def test_elemwise_is_gradient_enabled(self):
        assert etp_elemwise_p in GRADIENT_ENABLED_PRIMITIVES
        assert is_etp_enable_gradient_primitive(etp_elemwise_p)

    def test_no_other_primitive_is_gradient_enabled(self):
        for prim in _ALL_SHIPPED:
            if prim is etp_elemwise_p:
                continue
            assert not is_etp_enable_gradient_primitive(prim), prim.name


class TestBatchedFlag:
    """Batched primitives (``mm`` / ``conv`` / ``sp_mm`` / ``lora_mm``) carry
    the flag; their unbatched counterparts do not."""

    def test_batched_primitives_carry_flag(self):
        for prim in _BATCHED:
            assert prim in BATCHED_PRIMITIVES, prim.name
            assert is_batched_primitive(prim)

    def test_unbatched_primitives_do_not_carry_flag(self):
        for prim in _UNBATCHED:
            assert prim not in BATCHED_PRIMITIVES, prim.name
            assert not is_batched_primitive(prim)


class TestPredicatesOnNonETP:
    """The three predicates must return ``False`` for any non-ETP primitive."""

    def test_is_etp_primitive_false_for_lax_add(self):
        from jax import lax
        assert not is_etp_primitive(lax.add_p)

    def test_is_etp_enable_gradient_primitive_false_for_lax_add(self):
        from jax import lax
        assert not is_etp_enable_gradient_primitive(lax.add_p)

    def test_is_batched_primitive_false_for_lax_add(self):
        from jax import lax
        assert not is_batched_primitive(lax.add_p)

    def test_predicates_false_for_fresh_primitive(self):
        bogus = Primitive('bogus_for_test')
        assert not is_etp_primitive(bogus)
        assert not is_etp_enable_gradient_primitive(bogus)
        assert not is_batched_primitive(bogus)


class TestRegistriesAreSharedAcrossImports:
    """Importing the registries via either the package or the legacy shim
    must yield the *same* underlying objects — there must be one ``set``,
    not a copy."""

    def test_shim_and_package_share_etp_primitives(self):
        from braintrace import _etrace_op as legacy
        assert legacy.ETP_PRIMITIVES is ETP_PRIMITIVES

    def test_shim_and_package_share_rule_dicts(self):
        from braintrace import _etrace_op as legacy
        assert legacy.ETP_RULES_YW_TO_W is ETP_RULES_YW_TO_W
        assert legacy.ETP_RULES_XY_TO_DW is ETP_RULES_XY_TO_DW
        assert legacy.ETP_RULES_INIT_DRTRL is ETP_RULES_INIT_DRTRL
        assert legacy.ETP_RULES_INIT_PP is ETP_RULES_INIT_PP

    def test_shim_and_package_share_flag_sets(self):
        from braintrace import _etrace_op as legacy
        assert legacy.GRADIENT_ENABLED_PRIMITIVES is GRADIENT_ENABLED_PRIMITIVES
        assert legacy.BATCHED_PRIMITIVES is BATCHED_PRIMITIVES
