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

"""Tests for :class:`ETPPrimitiveSpec` and the spec-based registration API.

A spec is the *single source of truth* for a primitive: name, impl, four
ETP rules, invar layout, two flags. Registering a spec must populate
every relevant registry in lockstep — primitive set, batched/grad
flag-sets, four rule dicts, and the spec dict itself.

The dataclass is frozen, so attempting to mutate a field after
construction must raise. Default values for ``x_invar_index``,
``y_outvar_index``, ``batched``, ``gradient_enabled`` must match what
the dense matmul / elementwise primitives rely on at runtime.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from braintrace._etrace_op import (
    BATCHED_PRIMITIVES,
    ETP_PRIMITIVE_SPECS,
    ETP_PRIMITIVES,
    ETP_RULES_INIT_DRTRL,
    ETP_RULES_INIT_PP,
    ETP_RULES_XY_TO_DW,
    ETP_RULES_YW_TO_W,
    GRADIENT_ENABLED_PRIMITIVES,
    ETPPrimitive,
    ETPPrimitiveSpec,
    etp_conv_p,
    etp_elemwise_p,
    etp_lora_mm_p,
    etp_lora_mv_p,
    etp_mm_p,
    etp_mv_p,
    etp_sp_mm_p,
    etp_sp_mv_p,
    get_primitive_spec,
    register_primitive,
    register_primitive_spec,
)


_PRIM_COUNTER = 0


def _fresh_name(stub):
    global _PRIM_COUNTER
    _PRIM_COUNTER += 1
    return f'etp_spec_test_{stub}_{_PRIM_COUNTER}'


def _trivial_spec(stub, **overrides):
    """Build a spec with every field populated by a placeholder."""
    base = dict(
        name=_fresh_name(stub),
        impl=lambda x, w: x @ w,
        yw_to_w=lambda hd, t, **p: t,
        xy_to_dw=lambda x, hd, w, **p: w,
        init_drtrl=lambda xv, yv, wv, n: jnp.zeros((1, n)),
        init_pp=lambda xv, yv, wv, n: jnp.zeros((1, n)),
        trainable_invars_fn=lambda params: {'weight': 1},
    )
    base.update(overrides)
    return ETPPrimitiveSpec(**base)


# ---------------------------------------------------------------------------
# Dataclass shape & defaults
# ---------------------------------------------------------------------------

class TestDataclassShape:

    def test_is_frozen(self):
        spec = _trivial_spec('frozen')
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.name = 'mutated'  # type: ignore[misc]

    def test_default_x_invar_index_is_zero(self):
        spec = _trivial_spec('xidx')
        assert spec.x_invar_index == 0

    def test_default_y_outvar_index_is_zero(self):
        spec = _trivial_spec('yidx')
        assert spec.y_outvar_index == 0

    def test_default_batched_is_false(self):
        spec = _trivial_spec('batched')
        assert spec.batched is False

    def test_default_gradient_enabled_is_false(self):
        spec = _trivial_spec('grad')
        assert spec.gradient_enabled is False

    def test_x_invar_index_can_be_none(self):
        spec = _trivial_spec('xnone', x_invar_index=None)
        assert spec.x_invar_index is None


# ---------------------------------------------------------------------------
# Registration side-effects
# ---------------------------------------------------------------------------

class TestRegisterPrimitiveSpec:

    def test_returns_etp_primitive(self):
        spec = _trivial_spec('ret')
        p = register_primitive_spec(spec)
        assert isinstance(p, ETPPrimitive)

    def test_records_in_etp_primitive_specs_dict(self):
        spec = _trivial_spec('record')
        p = register_primitive_spec(spec)
        assert ETP_PRIMITIVE_SPECS[p] is spec

    def test_added_to_etp_primitives_set(self):
        spec = _trivial_spec('inset')
        p = register_primitive_spec(spec)
        assert p in ETP_PRIMITIVES

    def test_populates_all_four_rule_dicts(self):
        spec = _trivial_spec('rules')
        p = register_primitive_spec(spec)
        assert ETP_RULES_YW_TO_W[p] is spec.yw_to_w
        assert ETP_RULES_XY_TO_DW[p] is spec.xy_to_dw
        assert ETP_RULES_INIT_DRTRL[p] is spec.init_drtrl
        assert ETP_RULES_INIT_PP[p] is spec.init_pp

    def test_batched_flag_propagates(self):
        spec = _trivial_spec('bp', batched=True)
        p = register_primitive_spec(spec)
        assert p in BATCHED_PRIMITIVES

    def test_gradient_enabled_flag_propagates(self):
        spec = _trivial_spec('gp', gradient_enabled=True)
        p = register_primitive_spec(spec)
        assert p in GRADIENT_ENABLED_PRIMITIVES

    def test_primitive_name_matches_spec(self):
        name = _fresh_name('namematch')
        spec = ETPPrimitiveSpec(
            name=name,
            impl=lambda x: x,
            yw_to_w=lambda *a, **k: None,
            xy_to_dw=lambda *a, **k: None,
            init_drtrl=lambda *a, **k: None,
            init_pp=lambda *a, **k: None,
            trainable_invars_fn=lambda params: {'weight': 0},
            x_invar_index=None,
        )
        p = register_primitive_spec(spec)
        assert p.name == name

    def test_impl_runs_via_bind(self):
        def _impl(x, w):
            return x @ w
        spec = _trivial_spec('impl', impl=_impl)
        p = register_primitive_spec(spec)
        x = jnp.ones((3, 2))
        w = jnp.eye(2) * 4
        out = p.bind(x, w)
        np.testing.assert_allclose(out, jnp.ones((3, 2)) * 4)


# ---------------------------------------------------------------------------
# get_primitive_spec round-trip
# ---------------------------------------------------------------------------

class TestGetPrimitiveSpec:

    def test_round_trip_for_each_shipped_primitive(self):
        for prim in (
            etp_mm_p, etp_mv_p, etp_elemwise_p, etp_conv_p,
            etp_sp_mm_p, etp_sp_mv_p, etp_lora_mm_p, etp_lora_mv_p,
        ):
            spec = get_primitive_spec(prim)
            assert spec is not None
            assert isinstance(spec, ETPPrimitiveSpec)
            assert spec.name == prim.name

    def test_returns_none_for_legacy_register_primitive(self):
        """Primitives registered via the bare ``register_primitive`` helper
        (without a spec) must yield ``None`` from :func:`get_primitive_spec`
        — there is no spec to return."""
        p = register_primitive(_fresh_name('legacy'), lambda x: x)
        assert get_primitive_spec(p) is None

    def test_returns_none_for_completely_unknown_primitive(self):
        from braintrace._compatible_imports import Primitive
        bogus = Primitive('bogus_for_spec_test')
        assert get_primitive_spec(bogus) is None


# ---------------------------------------------------------------------------
# Spec-vs-runtime invariants for the shipped primitives
# ---------------------------------------------------------------------------

class TestShippedSpecsMatchRuntime:
    """The cached spec for each shipped primitive must reflect the actual
    flag membership — otherwise the compiler could read one truth and
    the runtime another."""

    def test_elemwise_spec_marks_gradient_enabled(self):
        spec = get_primitive_spec(etp_elemwise_p)
        assert spec.gradient_enabled is True
        assert etp_elemwise_p in GRADIENT_ENABLED_PRIMITIVES

    def test_elemwise_spec_x_invar_is_none(self):
        spec = get_primitive_spec(etp_elemwise_p)
        assert spec.x_invar_index is None

    def test_mm_spec_is_batched(self):
        spec = get_primitive_spec(etp_mm_p)
        assert spec.batched is True
        assert etp_mm_p in BATCHED_PRIMITIVES

    def test_mv_spec_is_unbatched(self):
        spec = get_primitive_spec(etp_mv_p)
        assert spec.batched is False
        assert etp_mv_p not in BATCHED_PRIMITIVES

    def test_lora_spec_x_invar_index_is_zero(self):
        for prim in (etp_lora_mm_p, etp_lora_mv_p):
            spec = get_primitive_spec(prim)
            assert spec.x_invar_index == 0
