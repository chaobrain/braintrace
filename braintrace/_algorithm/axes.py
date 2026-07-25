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

"""The learning-rule axis vocabulary.

An online learning rule is a point in a space of independent choices, not a name.
:class:`ETraceConfig` is that point: six categorical axes carrying the
coordinate, four numeric fields carrying the coefficients a category alone
cannot express.

See ``docs/specs/2026-07-25-algorithm-axes-roadmap.md`` for the axis derivation
and ``docs/specs/2026-07-25-p2-axis-decomposition.md`` for this module's design.
"""

from __future__ import annotations

import dataclasses
import operator
from typing import Any, Dict, Optional, Tuple, Union

__all__ = ['ETraceConfig']

# --- the vocabulary ------------------------------------------------------- #

#: ``axis -> ordered legal values``. Values marked in :data:`_UNIMPLEMENTED`
#: parse but are rejected, so a typo and a not-yet-delivered feature get
#: different messages.
_VOCABULARY: Dict[str, Tuple[str, ...]] = {
    'trace_factorization': ('per_param', 'io_factorized', 'random_projection'),
    'temporal_recursion': ('jacobian', 'scalar_leak', 'none'),
    'recurrence_scope': ('diagonal', 'coupled', 'sparse_n'),
    'learning_signal': ('symmetric', 'random_feedback', 'modulatory', 'bootstrapped'),
    'trace_filter': ('none', 'kappa'),
    'update_schedule': ('per_step', 'window', 'sequence_end'),
}

#: ``(axis, value) -> where it is scheduled``. Rejected by matrix rule 8.
_UNIMPLEMENTED: Dict[Tuple[str, str], str] = {
    ('update_schedule', 'window'): 'no phase yet',
    ('update_schedule', 'sequence_end'): 'no phase yet',
}

#: ``coefficient field -> (owning axis, value that requires it)``. Drives matrix
#: rule 7 in both directions.
_COEFFICIENT_OWNERS: Dict[str, Tuple[str, str]] = {
    'kappa': ('trace_filter', 'kappa'),
    'sparse_n': ('recurrence_scope', 'sparse_n'),
    'window_size': ('update_schedule', 'window'),
}


def _check_decay(value: Any, what: str) -> float:
    """Validate one decay coefficient and return it as a float.

    The bound is ``[0, 1)``: 0 is the degenerate no-recursion coefficient (it
    canonicalises the side to ``'none'``) and 1 would make the trace a
    non-decaying sum.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'{what} must be a float in [0, 1), got {value!r}.')
    value = float(value)
    if not (0.0 <= value < 1.0):
        raise ValueError(
            f'{what} must be a float in [0, 1), got {value}. Use 0.0 for no '
            'recursion (it canonicalises to `none`); 1.0 would never discount '
            'the trace. Integer *ranks* are a preset-level convenience — pass '
            'those to the algorithm as `decay_or_rank`, which converts via '
            'decay = (rank - 1) / (rank + 1). ETraceConfig stores the decay '
            'itself, so a coordinate has one spelling.'
        )
    return value


def _check_snap_order(value: Any) -> int:
    """Validate the SnAp order ``sparse_n`` and return it as a plain ``int``.

    The order counts how many steps of influence the trace retains, so it is a
    positive integer with no upper bound: any ``n`` at or above a hidden group's
    diameter saturates, and saturation is a property of the model, not of the
    vocabulary. ``bool`` is rejected explicitly -- ``True == 1`` would otherwise
    canonicalise silently onto ``recurrence_scope='coupled'``.
    """
    if isinstance(value, bool):
        raise TypeError(
            f'sparse_n must be an integer >= 1, got {value!r}. A bool is not a '
            'SnAp order (True would silently mean n=1).'
        )
    try:
        order = operator.index(value)
    except TypeError:
        raise TypeError(
            f'sparse_n must be an integer >= 1, got {value!r}.'
        ) from None
    if order < 1:
        raise ValueError(
            f'sparse_n must be at least 1, got {order}. n=1 is SnAp-1 (the '
            "instantaneous pattern, propagated zero times) and canonicalises to "
            "recurrence_scope='coupled'; there is no smaller order. "
            "recurrence_scope='diagonal' is a different rule, not n=0."
        )
    return order


def _as_pair(value: Any, what: str) -> Tuple[Any, Any]:
    """Expand a scalar to ``(x, f)``, or validate an existing pair."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(
                f'{what} as a pair must have exactly two entries (x-side, '
                f'f-side), got {len(value)}: {value!r}.'
            )
        return tuple(value)  # type: ignore[return-value]
    return (value, value)


@dataclasses.dataclass(frozen=True)
class ETraceConfig:
    """A point in the learning-rule axis space.

    Six categorical axes carry the coordinate; four numeric fields carry the
    coefficients. Instances are canonicalised and validated at construction, so
    a coordinate has exactly one spelling and an illegal combination cannot be
    built at all.

    Parameters
    ----------
    trace_factorization : str, default 'per_param'
        How the eligibility trace is stored, and therefore which engine runs it.
        ``'per_param'`` keeps a trace per parameter element
        (:class:`~braintrace.ParamDimVjpAlgorithm`, ``O(P*H)``);
        ``'io_factorized'`` keeps an input-side and an output-side factor
        (:class:`~braintrace.IODimVjpAlgorithm`, ``O(I+O)``);
        ``'random_projection'`` keeps a rank-1 ``(hidden, parameter)`` factor
        pair carrying UORO's unbiased estimator
        (:class:`~braintrace.RandomProjectionVjpAlgorithm`,
        ``O(|theta| + P*S)`` of *carrier storage*). It is the only coordinate
        whose trace is unbiased, and it requires
        ``recurrence_scope='coupled'`` -- see rule 11.
    temporal_recursion : str or tuple of str, default 'jacobian'
        The structural operator ``R`` in the trace recurrence. ``'jacobian'``
        uses the hidden-to-hidden Jacobian ``D``, ``'scalar_leak'`` replaces it
        with ``decay * I``, ``'none'`` with ``0``. Under ``'io_factorized'``
        this is a ``(x_side, f_side)`` pair; a scalar expands to both sides,
        with an x-side ``'jacobian'`` demoted to ``'scalar_leak'`` because the
        input-side trace never involves a Jacobian.
    recurrence_scope : str, default 'diagonal'
        How much hidden-to-hidden coupling enters ``D``. ``'diagonal'`` keeps
        only each state's own recurrence; ``'coupled'`` traces recurrent mixing
        between states of a hidden group; ``'sparse_n'`` retains influence over
        an ``n``-step neighbourhood derived from the model's own transition
        (SnAp-n), with ``n`` supplied as ``sparse_n``. The last two form one
        scale: SnAp-1 *is* ``'coupled'`` (the instantaneous pattern propagated
        zero times), and ``sparse_n=1`` canonicalises onto it. ``'diagonal'``
        sits below the scale -- it drops the recurrent mixing primitive from the
        transition before differentiating -- so no ``n`` reaches it.
    learning_signal : str, default 'symmetric'
        Where the per-hidden-group signal comes from. ``'symmetric'`` uses the
        true ``dL/dh``; ``'random_feedback'`` projects it through a fixed random
        matrix (feedback alignment); ``'modulatory'`` *replaces* it with a
        user-supplied neuromodulator (three-factor learning -- one array expanded
        to every group, never a per-group sequence, and single-step only);
        ``'bootstrapped'`` leaves it alone and instead injects a learned estimate
        of the future-loss gradient at the window's *exit* cotangent (DNI), which
        reaches the plain parameters only -- the eligibility trace already
        carries the ETP parameters' cross-window credit.
    trace_filter : str, default 'none'
        Optional low-pass on the trace. ``'kappa'`` applies
        ``e_bar <- kappa * e_bar + e``, e-prop's filter.
    update_schedule : str, default 'per_step'
        When the weight gradient is emitted.
    decay : float or tuple of float, optional
        Per-step discount of the previous trace. Required by
        ``'io_factorized'`` (where it is a ``(x, f)`` pair, a scalar expanding
        to both sides) and by ``'per_param'`` with ``'scalar_leak'``. Must lie
        in ``[0, 1)``.
    kappa : float, optional
        Coefficient of ``trace_filter='kappa'``, in ``[0, 1)``.
    sparse_n : int, optional
        Coefficient of ``recurrence_scope='sparse_n'``: the SnAp order, an
        integer ``>= 1``. Any order at or above a hidden group's diameter
        saturates to full within-group RTRL, so there is no "infinity"
        spelling -- saturation is a property of the model, not the vocabulary.
    window_size : int, optional
        Coefficient of ``update_schedule='window'``.

    Raises
    ------
    ValueError
        If a field carries a value outside its vocabulary, a coefficient is out
        of range, or the combination is rejected by the compatibility matrix.
    TypeError
        If a coefficient is not a number.

    Notes
    -----
    **Canonicalisation runs before validation**, so no rule ever fires on a
    spelling that canonicalisation would have removed:

    - ``'scalar_leak'`` with ``decay == 0`` becomes ``'none'`` — they are one
      rule — and ``'none'`` pins its decay side to ``0.0``.
    - Under ``'io_factorized'`` a scalar ``temporal_recursion`` / ``decay``
      expands to a pair.
    - ``trace_filter='kappa'`` with ``kappa == 0`` becomes ``'none'``, matching
      ``EProp(kappa_filter_decay=0)``'s documented reduction to ``D_RTRL``.
    - ``recurrence_scope='sparse_n'`` with ``sparse_n == 1`` becomes
      ``'coupled'`` with no coefficient — SnAp-1 and the block-diagonal
      recursion are one rule.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> braintrace.ETraceConfig().trace_factorization
        'per_param'
        >>> # pp_prop's coordinate: a leaky input trace, a Jacobian output trace
        >>> cfg = braintrace.ETraceConfig(
        ...     trace_factorization='io_factorized', decay=0.9)
        >>> cfg.temporal_recursion
        ('scalar_leak', 'jacobian')
        >>> # a coefficient with no category is a typo, not a configuration
        >>> braintrace.ETraceConfig(kappa=0.5)          # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ValueError: `kappa=0.5` is set but `trace_filter` is 'none'...
    """

    trace_factorization: str = 'per_param'
    temporal_recursion: Union[str, Tuple[str, str]] = 'jacobian'
    recurrence_scope: str = 'diagonal'
    learning_signal: str = 'symmetric'
    trace_filter: str = 'none'
    update_schedule: str = 'per_step'
    decay: Union[float, Tuple[float, float], None] = None
    kappa: Optional[float] = None
    sparse_n: Optional[int] = None
    window_size: Optional[int] = None

    # -- construction ------------------------------------------------------ #

    def __post_init__(self) -> None:
        self._check_vocabulary()
        self._canonicalise()
        self._validate()

    def _set(self, field: str, value: Any) -> None:
        """Assign through the frozen dataclass, for canonicalisation only."""
        object.__setattr__(self, field, value)

    def _check_vocabulary(self) -> None:
        """Reject unknown values before anything tries to interpret them."""
        for axis, legal in _VOCABULARY.items():
            raw = getattr(self, axis)
            values = raw if (axis == 'temporal_recursion'
                             and isinstance(raw, (tuple, list))) else (raw,)
            for value in values:
                if value not in legal:
                    raise ValueError(
                        f'{axis}={value!r} is not a known value. Legal values: '
                        f'{", ".join(repr(v) for v in legal)}.'
                    )

    def _canonicalise(self) -> None:
        """Rewrite equivalent spellings onto one representative."""
        if self.trace_factorization == 'io_factorized':
            self._canonicalise_factorized()
        else:
            self._canonicalise_scalar()

        # SnAp-1 retains the instantaneous pattern and propagates it zero times,
        # which is precisely the per-position block-diagonal recursion `coupled`
        # already computes. One coordinate, one spelling — and `coupled` keeps
        # the established name. The order is type-checked *before* the
        # comparison so `sparse_n=True` cannot slip through as n=1.
        if self.sparse_n is not None:
            self._set('sparse_n', _check_snap_order(self.sparse_n))
            if self.sparse_n == 1 and self.recurrence_scope == 'sparse_n':
                self._set('recurrence_scope', 'coupled')
                self._set('sparse_n', None)

        # `kappa == 0` is no filter at all: EProp documents `kappa_filter_decay=0`
        # as reducing exactly to D_RTRL, so the two are one coordinate. Clearing
        # the coefficient too keeps rule 7 from firing on the rewritten form.
        if self.trace_filter == 'kappa' and self.kappa == 0:
            self._set('trace_filter', 'none')
            self._set('kappa', None)

    def _canonicalise_scalar(self) -> None:
        """Canonicalise a single-sided (``per_param``) recursion."""
        if isinstance(self.temporal_recursion, (tuple, list)):
            raise ValueError(
                f'temporal_recursion may only be a pair under '
                f'trace_factorization="io_factorized"; '
                f'{self.trace_factorization!r} has a single trace to recurse. '
                f'Got {tuple(self.temporal_recursion)!r}.'
            )
        if isinstance(self.decay, (tuple, list)):
            raise ValueError(
                f'decay may only be a pair under '
                f'trace_factorization="io_factorized"; '
                f'{self.trace_factorization!r} has a single decay. '
                f'Got {tuple(self.decay)!r}.'
            )
        if self.decay is not None:
            self._set('decay', _check_decay(self.decay, 'decay'))
        recursion, decay = self._canonicalise_side(
            self.temporal_recursion, self.decay, 'decay')
        self._set('temporal_recursion', recursion)
        self._set('decay', decay)

    def _canonicalise_factorized(self) -> None:
        """Canonicalise the ``(x, f)`` recursion pair and its decays."""
        was_scalar = not isinstance(self.temporal_recursion, (tuple, list))
        recursion_x, recursion_f = _as_pair(
            self.temporal_recursion, 'temporal_recursion')
        # The scalar shorthand means "this rule, on both sides", and the x-side
        # has no Jacobian to use — so the shorthand demotes. An *explicit* pair
        # naming an x-side jacobian is a statement about the x-side, and matrix
        # rule 3 rejects it rather than silently rewriting it.
        if was_scalar and recursion_x == 'jacobian':
            recursion_x = 'scalar_leak'

        decay_x, decay_f = _as_pair(self.decay, 'decay')
        if decay_x is not None:
            decay_x = _check_decay(decay_x, 'decay (x-side)')
        if decay_f is not None:
            decay_f = _check_decay(decay_f, 'decay (f-side)')

        recursion_x, decay_x = self._canonicalise_side(
            recursion_x, decay_x, 'decay (x-side)', zero_collapses_any=True)
        recursion_f, decay_f = self._canonicalise_side(
            recursion_f, decay_f, 'decay (f-side)', zero_collapses_any=True)

        self._set('temporal_recursion', (recursion_x, recursion_f))
        self._set('decay', (decay_x, decay_f))

    @staticmethod
    def _canonicalise_side(
        recursion: str,
        decay: Optional[float],
        what: str,
        zero_collapses_any: bool = False,
    ) -> Tuple[str, Optional[float]]:
        """Collapse one side's ``(recursion, decay)`` onto its representative.

        A zero coefficient makes ``'scalar_leak'`` structurally identical to
        ``'none'``, and ``'none'`` in turn has no coefficient other than zero.

        Parameters
        ----------
        zero_collapses_any : bool, default False
            Whether a zero coefficient collapses *any* recursion, not just
            ``'scalar_leak'``. True for the ``io_factorized`` sides, where the
            update is ``eps <- a * (R @ eps) + (1 - a) * new``: at ``a = 0`` the
            previous trace is dropped before ``R`` is ever applied, so an
            ``'jacobian'`` f-side is structurally ``'none'``. False under
            ``per_param``, where ``'jacobian'`` takes no coefficient at all and
            a ``decay`` alongside it is a user error for rule 4 to catch —
            collapsing it here would swallow that error.
        """
        if decay == 0.0 and (zero_collapses_any or recursion == 'scalar_leak'):
            recursion = 'none'
        if recursion == 'none':
            if decay not in (None, 0.0):
                raise ValueError(
                    f'temporal_recursion="none" discards the previous trace '
                    f'entirely, so {what}={decay} would have no effect. Drop it, '
                    f'or use temporal_recursion="scalar_leak" to keep a leak.'
                )
            decay = 0.0
        return recursion, decay

    # -- the compatibility matrix ------------------------------------------ #

    def _validate(self) -> None:
        """Apply the compatibility matrix to the canonical form."""
        for rule in (
            self._rule_8_unimplemented,
            self._rule_1_kappa_needs_per_param,
            self._rule_2_scope_needs_a_jacobian,
            self._rule_3_no_x_side_jacobian,
            self._rules_4_5_per_param_decay,
            self._rule_6_factorized_needs_decay,
            self._rule_7_coefficients_need_their_category,
            # Rule 11 runs before rule 9 so that a `random_projection` config
            # naming an illegal scope is explained by the factorization the user
            # chose, not by rule 9's io_factorized-specific argument about the
            # x-side having no hidden index to widen.
            self._rule_11_random_projection_needs_the_coupled_scope,
            self._rule_9_sparse_n_needs_per_param,
            self._rule_10_sparse_n_is_a_widening_order,
        ):
            rule()

    def _rule_8_unimplemented(self) -> None:
        for axis in _VOCABULARY:
            raw = getattr(self, axis)
            values = raw if isinstance(raw, tuple) else (raw,)
            for value in values:
                phase = _UNIMPLEMENTED.get((axis, value))
                if phase is not None:
                    legal = [v for v in _VOCABULARY[axis]
                             if (axis, v) not in _UNIMPLEMENTED]
                    raise ValueError(
                        f'{axis}={value!r} is a recognised axis value but is not '
                        f'implemented yet; it is scheduled for {phase}. '
                        f'Available now: {", ".join(repr(v) for v in legal)}.'
                    )

    def _rule_1_kappa_needs_per_param(self) -> None:
        if self.trace_filter == 'kappa' and self.trace_factorization != 'per_param':
            if self.trace_factorization == 'random_projection':
                why = (
                    'a rank-1 carrier, and the sum of two rank-1 traces at '
                    'different times is rank-2, so the filter cannot be applied '
                    'to the factors without changing the estimator (and '
                    'destroying its unbiasedness).'
                )
            else:
                why = (
                    'an input/output factor pair, so it cannot store the filtered '
                    'sum; filtering the two factors separately is a different '
                    'rule.'
                )
            raise ValueError(
                "trace_filter='kappa' requires "
                "trace_factorization='per_param', not "
                f"{self.trace_factorization!r}. The filtered trace "
                f'`e_bar <- kappa * e_bar + e` is not rank-1, and '
                f'{self.trace_factorization!r} keeps {why} '
                "Use trace_filter='none'."
            )

    def _rule_2_scope_needs_a_jacobian(self) -> None:
        if self.recurrence_scope == 'diagonal':
            return
        # The scope only changes what enters `D`, so it carries information only
        # where `D` is actually consumed: the single side under per_param, the
        # f-side under io_factorized (the x-side never touches a Jacobian).
        consuming = (self.recursion_f if self.is_factorized
                     else self.temporal_recursion)
        if consuming != 'jacobian':
            side = 'the f-side' if self.is_factorized else 'the'
            raise ValueError(
                f'recurrence_scope={self.recurrence_scope!r} needs {side} '
                f"temporal_recursion to be 'jacobian', but it is "
                f'{consuming!r}. The scope only changes which couplings enter '
                'the hidden-to-hidden Jacobian, and this coordinate never '
                "consumes one. Use recurrence_scope='diagonal'."
            )

    def _rule_3_no_x_side_jacobian(self) -> None:
        if self.is_factorized and self.recursion_x == 'jacobian':
            raise ValueError(
                "temporal_recursion x-side may not be 'jacobian': the "
                'input-side trace is a filtered copy of the presynaptic input '
                'and never involves a hidden-to-hidden Jacobian. Use '
                "('scalar_leak', 'jacobian') — which is what the scalar "
                "shorthand temporal_recursion='jacobian' canonicalises to."
            )

    def _rules_4_5_per_param_decay(self) -> None:
        if self.is_factorized:
            return
        if self.temporal_recursion == 'jacobian' and self.decay is not None:
            raise ValueError(
                f"temporal_recursion='jacobian' under "
                f"trace_factorization='per_param' has no discount coefficient, "
                f'so decay={self.decay} would be silently ignored. Drop it, or '
                "use temporal_recursion='scalar_leak' to make the leak the rule."
            )
        if self.temporal_recursion == 'scalar_leak' and self.decay is None:
            raise ValueError(
                "temporal_recursion='scalar_leak' is defined by its leak "
                'coefficient, so decay is required. Pass decay=<float in '
                '[0, 1)>.'
            )

    def _rule_6_factorized_needs_decay(self) -> None:
        if not self.is_factorized:
            return
        for side, value in (('x', self.decay_x), ('f', self.decay_f)):
            if value is None:
                raise ValueError(
                    "trace_factorization='io_factorized' is defined by its "
                    f'smoothing coefficients, so the {side}-side decay is '
                    'required. Pass decay=<float> for both sides or '
                    'decay=(x, f) to set them apart.'
                )

    def _rule_7_coefficients_need_their_category(self) -> None:
        for field, (axis, required) in _COEFFICIENT_OWNERS.items():
            value = getattr(self, field)
            if value is not None and getattr(self, axis) != required:
                raise ValueError(
                    f'`{field}={value!r}` is set but `{axis}` is '
                    f'{getattr(self, axis)!r}, so the coefficient has no '
                    f'category to act on. Either set {axis}={required!r} or '
                    f'drop {field}.'
                )
            if value is None and getattr(self, axis) == required:
                raise ValueError(
                    f'{axis}={required!r} is defined by its coefficient, so '
                    f'`{field}` is required.'
                )
        if self.kappa is not None:
            _check_decay(self.kappa, 'kappa')

    def _rule_9_sparse_n_needs_per_param(self) -> None:
        if self.recurrence_scope != 'sparse_n':
            return
        if self.trace_factorization != 'per_param':
            raise ValueError(
                "recurrence_scope='sparse_n' requires "
                "trace_factorization='per_param', not "
                f'{self.trace_factorization!r}. SnAp-n widens the trace\'s '
                'trailing state axis into a (neighbour, state) axis, and the '
                'factorised x-side carries no hidden index at all to widen — '
                'the whole widening would have to land on the f-side, which is '
                'a different rule and would break the O(I+O) memory the '
                "factorisation exists for. Use recurrence_scope='coupled'."
            )

    def _rule_10_sparse_n_is_a_widening_order(self) -> None:
        if self.recurrence_scope != 'sparse_n':
            return
        # Post-condition of canonicalisation: n == 1 has already been rewritten
        # onto `coupled`, so a surviving `sparse_n` coordinate genuinely widens.
        order = _check_snap_order(self.sparse_n)
        assert order >= 2, (
            f'sparse_n={order} survived canonicalisation; n=1 must have been '
            "rewritten onto recurrence_scope='coupled'."
        )

    def _rule_11_random_projection_needs_the_coupled_scope(self) -> None:
        if self.trace_factorization != 'random_projection':
            return
        if self.recurrence_scope == 'coupled':
            return
        if self.recurrence_scope == 'diagonal':
            why = (
                "'diagonal' deletes the recurrent mixing from the transition, so "
                'the rank-1 estimator would be an *unbiased estimate of an '
                'already-biased trace*: strictly more variance than the '
                'per-parameter trace, the same asymptotic error, and no memory '
                'saved -- the anchored per_param trace is already smaller than '
                "UORO's two factors."
            )
        else:  # 'sparse_n'
            why = (
                "'sparse_n' widens the trace's trailing state axis to retain n "
                'steps of influence, but the random projection retains the *whole* '
                'within-group influence in expectation already, so the widening '
                'would pay SnAp-n memory for an accuracy the estimator has '
                'without it.'
            )
        raise ValueError(
            "trace_factorization='random_projection' requires "
            f"recurrence_scope='coupled', not {self.recurrence_scope!r}. UORO "
            'rolls the *full* within-group hidden-to-hidden Jacobian, and '
            "'coupled' is the coordinate that puts the recurrent ETP mixing into "
            f'the transition for it to be full of. {why}'
        )

    # -- derived views ----------------------------------------------------- #

    @property
    def is_factorized(self) -> bool:
        """Whether the trace is stored as an input/output factor pair."""
        return self.trace_factorization == 'io_factorized'

    @property
    def recursion_x(self) -> str:
        """The x-side (input factor) recursion. ``io_factorized`` only."""
        return self._factorized_field('temporal_recursion', 0)

    @property
    def recursion_f(self) -> str:
        """The f-side (output factor) recursion. ``io_factorized`` only."""
        return self._factorized_field('temporal_recursion', 1)

    @property
    def decay_x(self) -> float:
        """The x-side smoothing coefficient. ``io_factorized`` only."""
        return self._factorized_field('decay', 0)

    @property
    def decay_f(self) -> float:
        """The f-side smoothing coefficient. ``io_factorized`` only."""
        return self._factorized_field('decay', 1)

    def _factorized_field(self, field: str, index: int) -> Any:
        if not self.is_factorized:
            raise AttributeError(
                f'{field} is only two-sided under '
                f"trace_factorization='io_factorized'; this config is "
                f'{self.trace_factorization!r}, so read `.{field}` directly.'
            )
        return getattr(self, field)[index]

    @property
    def include_recurrent_mixing(self) -> bool:
        """Whether the compiler should trace hidden-to-hidden ETP mixing.

        The graph executor's spelling of :attr:`recurrence_scope`.

        ``True`` for both non-diagonal scopes: ``'coupled'`` needs the coupled
        transition to take its per-position block diagonal, and ``'sparse_n'``
        needs the same transition to gather its widened operator out of.
        """
        return self.recurrence_scope in ('coupled', 'sparse_n')

    def replace(self, **changes: Any) -> 'ETraceConfig':
        """Return a copy with ``changes`` applied, re-canonicalised and re-checked.

        Parameters
        ----------
        **changes
            Field values to override.

        Returns
        -------
        ETraceConfig
            The new configuration.

        Notes
        -----
        The receiver is already canonical, so a field left unchanged is passed on
        in canonical form. That is only lossless because canonicalisation is
        idempotent — a canonical value always canonicalises to itself.
        """
        return dataclasses.replace(self, **changes)

    def describe(self) -> str:
        """One-line human-readable coordinate, for reports and error messages.

        Returns
        -------
        str
            The non-default axes, or ``'default'`` when the config is the
            default coordinate.
        """
        parts = []
        for axis in _VOCABULARY:
            value = getattr(self, axis)
            if value != _VOCABULARY[axis][0] or axis == 'trace_factorization':
                parts.append(f'{axis}={value!r}')
        for field in ('decay', 'kappa', 'sparse_n', 'window_size'):
            value = getattr(self, field)
            if value is not None:
                parts.append(f'{field}={value!r}')
        return ', '.join(parts) if parts else 'default'
