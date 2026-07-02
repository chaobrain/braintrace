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

r"""Global registries shared by every ETP primitive submodule.

The compiler and runtime treat membership in :data:`ETP_PRIMITIVES`
(identity-based) as the *sole* mechanism for recognising an ETP weight
operation, replacing the legacy JIT-name string matching. The four rule
dictionaries (``ETP_RULES_*``) hold every ETP-specific rule a primitive
needs.

Two boolean flag-sets — :data:`GRADIENT_ENABLED_PRIMITIVES` and
:data:`BATCHED_PRIMITIVES` — are maintained in lockstep with the
primitive set so callers can ask cheap per-primitive questions
(``is_etp_enable_gradient_primitive``, ``is_batched_primitive``) without
introspecting individual primitives.

Three metadata dictionaries — :data:`ETP_TRAINABLE_INVARS_FNS`,
:data:`ETP_X_INVAR_INDICES`, :data:`ETP_Y_OUTVAR_INDICES` — record the
per-primitive invar / outvar layout the compiler needs to locate the
weight / ``x`` / ``y`` variables on an equation. They are populated by
:func:`register_primitive` and queried through the accessor helpers
:func:`get_trainable_invars`, :func:`get_x_invar_index` and
:func:`get_y_outvar_index`.

One further metadata dictionary — :data:`ETP_FAST_PATH_RULES` — holds the
optional per-primitive closed-form param-dim D-RTRL "fast-path" kernel
bundle (:class:`FastPathRules`). Only primitives with an elementwise
``yw_to_w`` rule register one; it is queried through
:func:`get_fast_path_rules`.
"""

from typing import Callable, Dict, NamedTuple, Optional

from braintrace._compatible_imports import Primitive

__all__ = [
    'ETP_PRIMITIVES',
    'ETP_RULES_YW_TO_W',
    'ETP_RULES_XY_TO_DW',
    'ETP_RULES_INIT_DRTRL',
    'ETP_RULES_INIT_PP',
    'GRADIENT_ENABLED_PRIMITIVES',
    'BATCHED_PRIMITIVES',
    'BATCHED_COUNTERPARTS',
    'ETP_TRAINABLE_INVARS_FNS',
    'ETP_X_INVAR_INDICES',
    'ETP_Y_OUTVAR_INDICES',
    'is_etp_primitive',
    'is_etp_enable_gradient_primitive',
    'is_batched_primitive',
    'register_batched_counterpart',
    'get_batched_counterpart',
    'get_trainable_invars',
    'get_x_invar_index',
    'get_y_outvar_index',
    'FastPathRules',
    'ETP_FAST_PATH_RULES',
    'get_fast_path_rules',
]

ETP_PRIMITIVES: set = set()

ETP_RULES_YW_TO_W: Dict[Primitive, Callable] = {}
r"""D-RTRL trace propagation: ``(hidden_dim, trace, **params) -> trace``."""

ETP_RULES_XY_TO_DW: Dict[Primitive, Callable] = {}
r"""Weight gradient: ``(x, hidden_dim, w, **params) -> dw``."""

ETP_RULES_INIT_DRTRL: Dict[Primitive, Callable] = {}
r"""D-RTRL trace init: ``(x_var, y_var, weight_var, num_hidden_state) -> zeros``."""

ETP_RULES_INIT_PP: Dict[Primitive, Callable] = {}
r"""pp_prop df trace init: ``(x_var, y_var, weight_var, num_hidden_state) -> zeros``."""

GRADIENT_ENABLED_PRIMITIVES: set = set()
BATCHED_PRIMITIVES: set = set()

BATCHED_COUNTERPARTS: Dict[Primitive, Primitive] = {}
r"""Batched counterpart per unbatched ETP primitive.

Maps an unbatched primitive (e.g. ``etp_mv_p``) to the batched primitive
implementing the same operation with a leading batch axis on ``x``
(e.g. ``etp_mm_p``). Consulted by the auto-derived batching rule in
:func:`~braintrace._op._primitive.register_primitive` to keep ETP
primitive identity intact under ``jax.vmap`` (identity-preserving
promotion). Populated via :func:`register_batched_counterpart`.
"""

ETP_TRAINABLE_INVARS_FNS: Dict[Primitive, Callable] = {}
r"""Trainable-input layout: ``eqn.params -> {key: invar_index}``.

Declares the primitive's full trainable-input layout so the compiler and
executors can support N-trainable-input primitives (e.g. ``{weight, bias}``
for Linear, ``{B, A, bias}`` for LoRA).
"""

ETP_X_INVAR_INDICES: Dict[Primitive, Optional[int]] = {}
r"""Position of the input ``x`` in ``eqn.invars``, or ``None`` for primitives
that have no external input (currently only ``etp_elemwise_p``)."""

ETP_Y_OUTVAR_INDICES: Dict[Primitive, int] = {}
r"""Position of the output ``y`` in ``eqn.outvars`` (0 for all current
primitives, which have a single output)."""


def is_etp_primitive(primitive) -> bool:
    """Return True iff *primitive* was created via :func:`register_primitive`."""
    return primitive in ETP_PRIMITIVES


def is_etp_enable_gradient_primitive(primitive) -> bool:
    """Return True iff the compiler must *evaluate* this primitive instead of
    skipping it when walking through a ``pjit`` equation.

    Identity-like primitives (e.g. ``etp_elemwise_p``) must be evaluated so
    the value flows to downstream consumers; structural-marker primitives
    (e.g. ``etp_mm_p``) are skipped because their value is supplied separately.
    """
    return primitive in GRADIENT_ENABLED_PRIMITIVES


def is_batched_primitive(primitive) -> bool:
    """Return True iff *primitive* was registered with ``batched=True``."""
    return primitive in BATCHED_PRIMITIVES


def register_batched_counterpart(unbatched_p, batched_p) -> None:
    """Declare *batched_p* as the batched form of *unbatched_p* under vmap.

    Parameters
    ----------
    unbatched_p : Primitive
        An ETP primitive registered with ``batched=False``.
    batched_p : Primitive
        The ETP primitive registered with ``batched=True`` that computes the
        same operation with the batch axis leading on ``x``.

    Raises
    ------
    ValueError
        If either primitive is not an ETP primitive, if *unbatched_p* is
        registered as batched, or if *batched_p* is not registered as batched.
    """
    if unbatched_p not in ETP_PRIMITIVES or batched_p not in ETP_PRIMITIVES:
        raise ValueError(
            f'Both primitives must be ETP primitives; got '
            f'{unbatched_p} and {batched_p}.'
        )
    if unbatched_p in BATCHED_PRIMITIVES:
        raise ValueError(
            f'{unbatched_p} must be an unbatched primitive to receive a '
            f'batched counterpart.'
        )
    if batched_p not in BATCHED_PRIMITIVES:
        raise ValueError(
            f'{batched_p} must be registered with batched=True to serve as '
            f'a batched counterpart.'
        )
    BATCHED_COUNTERPARTS[unbatched_p] = batched_p


def get_batched_counterpart(primitive):
    """Return the batched counterpart of *primitive*, or ``None``.

    Parameters
    ----------
    primitive : Primitive
        The (unbatched) ETP primitive to look up.

    Returns
    -------
    Primitive or None
        The batched counterpart registered via
        :func:`register_batched_counterpart`, or ``None``.
    """
    return BATCHED_COUNTERPARTS.get(primitive)


def get_trainable_invars(primitive, eqn_params: dict) -> Dict[str, int]:
    """Return ``{key: invar_index}`` for *primitive* on an equation.

    Falls back to the single-weight ``{'weight': 1}`` layout for primitives
    registered without an explicit ``trainable_invars_fn``.
    """
    fn = ETP_TRAINABLE_INVARS_FNS.get(primitive)
    if fn is None:
        return {'weight': 1}
    return fn(eqn_params)


def get_x_invar_index(primitive) -> Optional[int]:
    """Return the index of ``x`` in ``eqn.invars`` (``None`` if no input)."""
    return ETP_X_INVAR_INDICES.get(primitive, 0)


def get_y_outvar_index(primitive) -> int:
    """Return the index of ``y`` in ``eqn.outvars``."""
    return ETP_Y_OUTVAR_INDICES.get(primitive, 0)


class FastPathRules(NamedTuple):
    """Per-primitive closed-form param-dim D-RTRL fast-path kernels + gate.

    Bundles the three closed-form einsum kernels that replace the generic
    nested-``vmap`` trace path for primitives with an *elementwise*
    ``yw_to_w`` rule (currently ``etp_mm_p`` / ``etp_mv_p`` / ``etp_elemwise_p``),
    together with a gate predicate that decides whether the fast path is
    valid for a given equation.

    Parameters
    ----------
    instant : Callable
        Instantaneous term ``diag(D_f^t) ⊗ x^t``. Signature
        ``(x, df, has_bias) -> {'weight': ..., ['bias': ...]}``.
    recurrent : Callable
        Recurrent term ``D^t · ε^{t-1}``. Signature
        ``(diag, old_bwg, num_state) -> dict``.
    solve : Callable
        Solve-time contraction ``Σ_alpha diag_like[..., alpha] · yw_to_w(ε[..., alpha])``.
        Signature ``(diag_like, etrace_data, *, fold_batch) -> dict``.
    applicable : Callable
        Gate predicate ``(eqn_params) -> bool`` — ``True`` iff the closed-form
        kernels are valid for this equation. The kernels drop the ``f'(W)``
        transform factor, so a primitive carrying an active transform hook
        (``weight_fn`` / ``bias_fn``) must report ``False`` and fall back to
        the rule path.
    """

    instant: Callable
    recurrent: Callable
    solve: Callable
    applicable: Callable


ETP_FAST_PATH_RULES: Dict[Primitive, FastPathRules] = {}
r"""Closed-form param-dim D-RTRL fast-path bundle per primitive.

Populated by :meth:`ETPPrimitive.register_etp_rules` (via its ``fast_path``
keyword). Only primitives with an elementwise ``yw_to_w`` rule register one;
conv / sparse / LoRA primitives are absent (they have no closed-form fast
path). Queried through :func:`get_fast_path_rules`.
"""


def get_fast_path_rules(primitive) -> Optional[FastPathRules]:
    """Return the :class:`FastPathRules` bundle for *primitive*, or ``None``.

    Parameters
    ----------
    primitive : Primitive
        The ETP primitive to look up.

    Returns
    -------
    FastPathRules or None
        The registered fast-path bundle, or ``None`` if *primitive* has no
        fast path (e.g. conv / sparse / LoRA).
    """
    return ETP_FAST_PATH_RULES.get(primitive)
