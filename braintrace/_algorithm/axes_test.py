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

"""Tests for the learning-rule axis vocabulary (``axes.ETraceConfig``).

Coverage follows the spec's two guarantees:

* **canonicalisation** — one coordinate has exactly one spelling, and the
  rewrite is idempotent;
* **the compatibility matrix** — every rule rejects, every legal coordinate is
  admitted, and no rule fires on a spelling canonicalisation would have removed.

The second half matters more than it looks: a matrix that rejects too much is as
broken as one that rejects too little, so each rule has a *negative* control —
the nearest legal neighbour — beside it.
"""

import dataclasses

import pytest

from braintrace._algorithm.axes import ETraceConfig

# The coordinates of the five surviving presets, from the P2 spec's table.
PRESET_COORDINATES = {
    'D_RTRL': dict(),
    'OSTLRecurrent': dict(recurrence_scope='coupled'),
    'EProp': dict(trace_filter='kappa', kappa=0.9),
    'EProp_random': dict(learning_signal='random_feedback'),
    'pp_prop': dict(trace_factorization='io_factorized', decay=0.9),
    'OSTLFeedforward': dict(trace_factorization='io_factorized', decay=1e-6),
}


class TestVocabulary:

    def test_default_is_d_rtrl(self):
        cfg = ETraceConfig()
        assert cfg.trace_factorization == 'per_param'
        assert cfg.temporal_recursion == 'jacobian'
        assert cfg.recurrence_scope == 'diagonal'
        assert cfg.learning_signal == 'symmetric'
        assert cfg.trace_filter == 'none'
        assert cfg.update_schedule == 'per_step'
        assert cfg.decay is None

    @pytest.mark.parametrize('axis,value', [
        ('trace_factorization', 'per-param'),
        ('temporal_recursion', 'Jacobian'),
        ('recurrence_scope', 'diagnoal'),
        ('learning_signal', 'feedback'),
        ('trace_filter', 'lowpass'),
        ('update_schedule', 'every_step'),
    ])
    def test_unknown_value_is_rejected_by_name(self, axis, value):
        with pytest.raises(ValueError, match='is not a known value'):
            ETraceConfig(**{axis: value})

    def test_config_is_frozen(self):
        cfg = ETraceConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.recurrence_scope = 'coupled'

    def test_equality_is_by_coordinate(self):
        # Two spellings of one rule must compare equal, or "assert the preset's
        # coordinates" tests would depend on how the caller happened to write it.
        assert (ETraceConfig(temporal_recursion='scalar_leak', decay=0.0)
                == ETraceConfig(temporal_recursion='none'))

    def test_describe_names_the_non_default_axes(self):
        text = ETraceConfig(recurrence_scope='coupled').describe()
        assert "recurrence_scope='coupled'" in text
        assert 'learning_signal' not in text  # left at its default


class TestCanonicalisation:

    def test_scalar_leak_at_zero_decay_is_none(self):
        cfg = ETraceConfig(temporal_recursion='scalar_leak', decay=0.0)
        assert cfg.temporal_recursion == 'none'

    def test_none_pins_its_decay_to_zero(self):
        assert ETraceConfig(temporal_recursion='none').decay == 0.0

    def test_none_rejects_a_nonzero_decay(self):
        with pytest.raises(ValueError, match='would have no effect'):
            ETraceConfig(temporal_recursion='none', decay=0.5)

    def test_scalar_recursion_expands_and_demotes_the_x_side(self):
        # The x-side has no Jacobian, so the scalar shorthand means
        # "leak on the input factor, Jacobian on the output factor".
        cfg = ETraceConfig(trace_factorization='io_factorized', decay=0.9)
        assert cfg.temporal_recursion == ('scalar_leak', 'jacobian')
        assert cfg.recursion_x == 'scalar_leak'
        assert cfg.recursion_f == 'jacobian'

    def test_scalar_decay_expands_to_both_sides(self):
        cfg = ETraceConfig(trace_factorization='io_factorized', decay=0.9)
        assert cfg.decay == (0.9, 0.9)
        assert cfg.decay_x == 0.9 and cfg.decay_f == 0.9

    @pytest.mark.parametrize('decay,expected', [
        ((0.0, 0.9), ('none', 'jacobian')),
        ((0.9, 0.0), ('scalar_leak', 'none')),
        ((0.0, 0.0), ('none', 'none')),
    ])
    def test_a_zero_side_decay_collapses_that_side(self, decay, expected):
        """``eps <- a * (R @ eps) + (1 - a) * new`` drops ``R`` entirely at ``a = 0``.

        This is why the collapse applies to an ``'jacobian'`` f-side and not
        only to ``'scalar_leak'``: at ``a = 0`` the Jacobian is never reached.
        """
        cfg = ETraceConfig(trace_factorization='io_factorized', decay=decay)
        assert cfg.temporal_recursion == expected

    def test_rank_one_pp_prop_is_the_exact_none_coordinate(self):
        # F-29: decay_or_rank=1 maps to decay 0, which is no smearing at all.
        cfg = ETraceConfig(trace_factorization='io_factorized', decay=0.0)
        assert cfg.temporal_recursion == ('none', 'none')

    def test_zero_kappa_is_no_filter(self):
        # EProp documents kappa_filter_decay=0 as reducing exactly to D_RTRL.
        cfg = ETraceConfig(trace_filter='kappa', kappa=0.0)
        assert cfg.trace_filter == 'none'
        assert cfg.kappa is None
        assert cfg == ETraceConfig()

    @pytest.mark.parametrize('name', sorted(PRESET_COORDINATES))
    def test_canonicalisation_is_idempotent(self, name):
        """A canonical value must canonicalise to itself, or ``replace`` lies."""
        once = ETraceConfig(**PRESET_COORDINATES[name])
        twice = ETraceConfig(**{f.name: getattr(once, f.name)
                                for f in dataclasses.fields(once)})
        assert once == twice

    def test_replace_revalidates(self):
        cfg = ETraceConfig(trace_factorization='io_factorized', decay=0.9)
        assert cfg.replace(decay=0.5).decay == (0.5, 0.5)
        with pytest.raises(ValueError):
            cfg.replace(trace_filter='kappa', kappa=0.5)  # matrix rule 1


class TestCompatibilityMatrix:
    """Each rule paired with the nearest *legal* neighbour it must not reject."""

    def test_rule_1_kappa_requires_per_param(self):
        with pytest.raises(ValueError, match='not rank-1'):
            ETraceConfig(trace_factorization='io_factorized', decay=0.9,
                         trace_filter='kappa', kappa=0.5)
        ETraceConfig(trace_filter='kappa', kappa=0.5)  # legal neighbour

    def test_rule_2_scope_requires_a_consumed_jacobian(self):
        with pytest.raises(ValueError, match='never.*consumes one'):
            ETraceConfig(temporal_recursion='scalar_leak', decay=0.5,
                         recurrence_scope='coupled')
        ETraceConfig(recurrence_scope='coupled')  # legal neighbour

    def test_rule_2_reads_the_f_side_under_io_factorization(self):
        """The x-side never consumes ``D``, so a rule over both sides would
        reject every ``io_factorized`` coordinate. ``coupled`` is legal here —
        measured distinguishable from ``diagonal`` before the matrix was
        written — and illegal only once the f-side stops using the Jacobian."""
        legal = ETraceConfig(trace_factorization='io_factorized', decay=0.9,
                             recurrence_scope='coupled')
        assert legal.recursion_x == 'scalar_leak'   # x-side is not a jacobian
        assert legal.recursion_f == 'jacobian'
        with pytest.raises(ValueError, match='f-side'):
            ETraceConfig(trace_factorization='io_factorized', decay=(0.9, 0.0),
                         recurrence_scope='coupled')

    def test_rule_3_rejects_an_explicit_x_side_jacobian(self):
        # The scalar shorthand demotes; an explicit pair is a statement about
        # the x-side and is rejected rather than silently rewritten.
        with pytest.raises(ValueError, match='x-side may not be'):
            ETraceConfig(trace_factorization='io_factorized', decay=0.9,
                         temporal_recursion=('jacobian', 'jacobian'))
        assert ETraceConfig(
            trace_factorization='io_factorized', decay=0.9,
            temporal_recursion='jacobian',
        ).temporal_recursion == ('scalar_leak', 'jacobian')

    def test_rule_4_per_param_jacobian_takes_no_decay(self):
        with pytest.raises(ValueError, match='silently ignored'):
            ETraceConfig(decay=0.9)
        ETraceConfig(temporal_recursion='scalar_leak', decay=0.9)

    def test_rule_5_scalar_leak_requires_decay(self):
        with pytest.raises(ValueError, match='decay is required'):
            ETraceConfig(temporal_recursion='scalar_leak')

    def test_rule_6_io_factorization_requires_decay(self):
        with pytest.raises(ValueError, match='x-side decay is required'):
            ETraceConfig(trace_factorization='io_factorized')
        with pytest.raises(ValueError, match='f-side decay is required'):
            ETraceConfig(trace_factorization='io_factorized', decay=(0.9, None))

    @pytest.mark.parametrize('field,value', [
        ('kappa', 0.5), ('sparse_n', 2), ('window_size', 4),
    ])
    def test_rule_7_a_coefficient_needs_its_category(self, field, value):
        with pytest.raises(ValueError, match='no.*category to act on'):
            ETraceConfig(**{field: value})

    def test_rule_7_a_category_needs_its_coefficient(self):
        with pytest.raises(ValueError, match='`kappa` is required'):
            ETraceConfig(trace_filter='kappa')

    @pytest.mark.parametrize('kwargs,phase', [
        (dict(trace_factorization='random_projection'), 'P4'),
        (dict(recurrence_scope='sparse_n', sparse_n=2), 'P3'),
        (dict(learning_signal='modulatory'), 'P4'),
        (dict(learning_signal='bootstrapped'), 'P4'),
        (dict(update_schedule='window', window_size=4), 'no phase yet'),
        (dict(update_schedule='sequence_end'), 'no phase yet'),
    ])
    def test_rule_8_unimplemented_values_name_their_phase(self, kwargs, phase):
        with pytest.raises(ValueError, match='not implemented yet') as info:
            ETraceConfig(**kwargs)
        assert phase in str(info.value)


class TestCoefficientBounds:

    @pytest.mark.parametrize('decay', [-0.1, 1.0, 1.5])
    def test_decay_outside_the_unit_interval_is_rejected(self, decay):
        with pytest.raises(ValueError, match=r'\[0, 1\)'):
            ETraceConfig(temporal_recursion='scalar_leak', decay=decay)

    def test_an_integer_rank_in_decay_is_redirected_to_decay_or_rank(self):
        with pytest.raises(ValueError, match='decay_or_rank'):
            ETraceConfig(trace_factorization='io_factorized', decay=19)

    def test_zero_decay_is_admitted(self):
        # The bound is [0, 1), not (0, 1): 0 is the degenerate coordinate.
        assert ETraceConfig(temporal_recursion='scalar_leak',
                            decay=0.0).decay == 0.0

    @pytest.mark.parametrize('decay', ['0.9', True, object()])
    def test_a_non_numeric_decay_is_a_type_error(self, decay):
        # `None` is excluded on purpose: it means "unset", and rule 5 rejects it
        # with a message about the missing coefficient rather than its type.
        with pytest.raises(TypeError, match=r'must be a float in \[0, 1\)'):
            ETraceConfig(temporal_recursion='scalar_leak', decay=decay)

    def test_kappa_is_bounded_like_a_decay(self):
        with pytest.raises(ValueError, match=r'kappa.*\[0, 1\)'):
            ETraceConfig(trace_filter='kappa', kappa=1.0)

    def test_a_pair_needs_exactly_two_entries(self):
        with pytest.raises(ValueError, match='exactly two entries'):
            ETraceConfig(trace_factorization='io_factorized', decay=(0.9, 0.5, 0.1))


class TestDerivedViews:

    def test_two_sided_views_reject_a_single_sided_config(self):
        cfg = ETraceConfig()
        for attr in ('recursion_x', 'recursion_f', 'decay_x', 'decay_f'):
            with pytest.raises(AttributeError, match='io_factorized'):
                getattr(cfg, attr)

    def test_a_pair_is_rejected_under_per_param(self):
        with pytest.raises(ValueError, match='may only be a pair'):
            ETraceConfig(temporal_recursion=('scalar_leak', 'jacobian'))
        with pytest.raises(ValueError, match='may only be a pair'):
            ETraceConfig(temporal_recursion='scalar_leak', decay=(0.9, 0.9))

    @pytest.mark.parametrize('scope,expected', [
        ('diagonal', False), ('coupled', True),
    ])
    def test_include_recurrent_mixing_is_the_executors_spelling(self, scope, expected):
        assert ETraceConfig(recurrence_scope=scope).include_recurrent_mixing is expected


@pytest.mark.parametrize('name', sorted(PRESET_COORDINATES))
def test_every_preset_coordinate_is_constructible(name):
    """The matrix must admit all five surviving presets, or P2 cannot land."""
    assert isinstance(ETraceConfig(**PRESET_COORDINATES[name]), ETraceConfig)
