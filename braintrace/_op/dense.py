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

r"""Dense matmul ETP primitives.

``etp_mm_p`` is the batched primitive (``x`` shape ``(batch, in)``);
``etp_mv_p`` is the unbatched primitive (``x`` shape ``(in,)``). Both
optionally add a bias vector along the output dimension. The user-facing
:func:`matmul` selects the right primitive from ``x.ndim``.

**Forward operation**

.. math::

    y = x \, W \; (+ b)

where :math:`x \in \mathbb{R}^{B \times I}` (or :math:`\mathbb{R}^{I}`),
:math:`W \in \mathbb{R}^{I \times O}`, and :math:`b \in \mathbb{R}^{O}`.

**Role of each ETP rule**

The four ETP rules implement the hidden-to-weight Jacobian pieces needed
by D-RTRL and ES-D-RTRL (pp-prop). For a primitive producing output
:math:`y`, let :math:`\mathbf{D}_f^t = \partial h^t / \partial y^t`
(diagonal approximation) and :math:`\mathbf{D}^t = \partial h^t / \partial h^{t-1}`.

* ``xy_to_dw(x, hidden_dim, w)`` — returns :math:`\partial h / \partial W`
  via the chain rule :math:`\partial h / \partial W = (\partial h / \partial y) \cdot (\partial y / \partial W)`.
  This is the instantaneous :math:`\operatorname{diag}(\mathbf{D}_f^t) \otimes \mathbf{x}^t`
  term of the D-RTRL update:

  .. math::

      \boldsymbol{\epsilon}^t \approx \mathbf{D}^t \boldsymbol{\epsilon}^{t-1}
                                     + \operatorname{diag}(\mathbf{D}_f^t) \otimes \mathbf{x}^t

* ``dt_to_t(hidden_dim, trace)`` — multiplies the weight-shaped trace
  :math:`\boldsymbol{\epsilon}^{t-1}` elementwise by
  :math:`\partial h / \partial y` (supplied as ``hidden_dim``). This
  realises the :math:`\mathbf{D}^t \boldsymbol{\epsilon}^{t-1}` term
  *after* the executor has already contracted with :math:`\mathbf{D}^t`
  along the hidden axis, leaving only the :math:`y \to W` link to apply.

* ``init_drtrl(...)`` — allocates the D-RTRL weight-shaped trace
  :math:`\boldsymbol{\epsilon} \in \mathbb{R}^{\dots \times I \times O \times n_{\text{state}}}`.

* ``init_pp(...)`` — allocates the ES-D-RTRL output-shaped df trace
  :math:`\boldsymbol{\epsilon}_f \in \mathbb{R}^{\dots \times O \times n_{\text{state}}}`.
  In pp-prop, weight gradients are assembled at solve-time as
  :math:`\boldsymbol{\epsilon}_f \otimes \boldsymbol{\epsilon}_x`; the
  :math:`\boldsymbol{\epsilon}_x` factor is provided by the executor via
  :func:`xy_to_dw` using the stored :math:`x`-trace.

**Dict rule API (N-trainable-input refactor)**

Both primitives declare ``trainable_invars_fn``, which returns
``{'weight': 1}`` when ``has_bias=False`` and ``{'weight': 1, 'bias': 2}``
when ``has_bias=True``. The four ETP rules accept / return
``Dict[str, Array]`` instead of bare arrays so the executor can route
gradients to *both* weight and bias ``ParamState`` objects in one pass.

When ``has_bias=False`` the ``'bias'`` key is simply absent from every
dict, so the legacy (no-bias) code path is unchanged in behaviour.

**Transform hooks**

Both primitives accept two optional elementwise transform hooks in their
``eqn.params``: ``weight_fn`` (computes ``y = x @ weight_fn(w)``) and
``bias_fn`` (adds ``bias_fn(b)``). The forward and :func:`_mm_xy_to_dw` /
:func:`_mv_xy_to_dw` rules apply them; the eligibility trace and gradient
are always taken w.r.t. the **raw** weight/bias, so the transform Jacobian
:math:`f'` enters *only* through ``xy_to_dw`` via :func:`jax.vjp`. The
``dt_to_t`` rule does **not** apply :math:`f'`.

Both hooks are threaded through ``eqn.params``, which JAX treats as a
*static* (hashed-by-identity) part of the equation. Two textually identical
``lambda`` expressions are two distinct Python objects, so passing a fresh
``lambda`` each call (e.g. ``matmul(x, w, weight_fn=lambda ww: ww ** 2)``
inside a loop or inside a jitted function called repeatedly) silently
produces a cache miss / retrace every time even though the transform is
unchanged. Pass a module-level (or otherwise stably-identified) function
instead of a fresh ``lambda`` when ``weight_fn`` / ``bias_fn`` is used
inside a hot path.

**Fast path**

A closed-form param-dim D-RTRL kernel bundle (:class:`FastPathRules`,
defined as ``_DENSE_FAST_PATH`` and registered on *both* primitives) replaces
the generic nested-``vmap`` trace path with direct einsums. Because those
kernels emit the bare outer product ``x ⊗ df`` (i.e. the gradient w.r.t. the
*transformed* weight, dropping :math:`f'`), the bundle's ``applicable`` gate
disables the fast path whenever ``weight_fn`` *or* ``bias_fn`` is present.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import brainunit as u

from ._primitive import register_primitive
from ._registries import FastPathRules, register_batched_counterpart
from braintrace._typing import ArrayLike, WeightFn

__all__ = [
    'etp_mm_p',
    'etp_mv_p',
    'matmul',
]


def _etp_matmul_impl(*args: Any, has_bias: bool = False,
                     weight_fn: WeightFn | None = None,
                     bias_fn: WeightFn | None = None) -> Any:
    x, w = args[0], args[1]
    if weight_fn is not None:
        w = weight_fn(w)
    y = x @ w
    if has_bias:
        b = args[2]
        if bias_fn is not None:
            b = bias_fn(b)
        y = y + b
    return y


# ---------------------------------------------------------------------------
# etp_mm_p — batched
# ---------------------------------------------------------------------------

def _mm_trainable_invars(params: dict[str, Any]) -> dict[str, int]:
    """Return ``{key: invar_index}`` depending on ``has_bias``."""
    base = {'weight': 1}
    if params.get('has_bias', False):
        base['bias'] = 2
    return base


def _mm_dt_to_t(hidden_dim: Any, trace: dict[str, Any], *, has_bias: bool = False,
                weight_fn: WeightFn | None = None,
                bias_fn: WeightFn | None = None) -> dict[str, Any]:
    r"""Batched ``dt_to_t`` — propagate :math:`\partial h / \partial y`
    through a weight-shaped D-RTRL trace.

    **Role in D-RTRL.** Implements the multiplicative step inside
    :math:`\mathbf{D}^t \boldsymbol{\epsilon}^{t-1}`: the executor has
    already contracted the trace's hidden-axis with the hidden-to-hidden
    Jacobian, producing ``hidden_dim = ∂h/∂y`` (one slice per hidden
    state). What remains is the :math:`y \to W` chain factor, which for
    :math:`y = x W + b` is simply ``1`` on the matching ``out`` column:

    .. math::

        \frac{\partial y_{bj}}{\partial W_{ik}} = \delta_{jk} \, x_{bi},
        \qquad
        \frac{\partial y_{bj}}{\partial b_k} = \delta_{jk}.

    So the trace update along the :math:`y \to W` arrow is

    .. math::

        \epsilon^{t}_{W, bik} = (\partial h / \partial y)_{bk} \,
                                \epsilon^{t-1}_{W, bik}, \qquad
        \epsilon^{t}_{b, bk}  = (\partial h / \partial y)_{bk} \,
                                \epsilon^{t-1}_{b, bk}.

    **Two execution contexts.** Both arrive after the outer
    ``n_state``-vmap strips the trailing hidden-state axis:

    (a) trace update (batch retained):
        ``hidden_dim : (batch, out)``,
        ``trace['weight'] : (batch, in, out)``,
        ``trace['bias']   : (batch, out)``.

    (b) gradient solve (an extra batch-vmap strips the batch axis):
        ``hidden_dim : (out,)``,
        ``trace['weight'] : (in, out)``,
        ``trace['bias']   : (out,)``.

    **Broadcast rule.** ``jnp.expand_dims(hidden_dim, axis=-2)`` inserts a
    singleton at the ``in`` position in both contexts:

        (out,)       → (1, out)       broadcasts with (in, out)         ✓
        (batch, out) → (batch, 1, out) broadcasts with (batch, in, out) ✓

    Using a fixed positive axis (the old ``axis=1``) only happened to work
    for square weights; ``axis=-2`` is correct for any ``in != out``.
    """
    out = {'weight': trace['weight'] * jnp.expand_dims(hidden_dim, axis=-2)}
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _mm_xy_to_dw(x: Any, hidden_dim: Any, weights: dict[str, Any], *,
                 has_bias: bool = False, weight_fn: WeightFn | None = None,
                 bias_fn: WeightFn | None = None) -> dict[str, Any]:
    r"""Batched ``xy_to_dw`` — instantaneous hidden-to-weight Jacobian.

    **Role.** Computes :math:`\partial h / \partial W` (and
    :math:`\partial h / \partial b`) by VJP of :math:`y = x \, \text{weight\_fn}(W) + \text{bias\_fn}(b)`,
    pulling back the cotangent ``hidden_dim`` = :math:`\partial h/\partial y`.
    When ``weight_fn`` / ``bias_fn`` are ``None`` the identity is used and
    the result is identical to the undecorated matmul.

    The VJP is taken w.r.t. the **raw** weight dict (before any transform),
    so the returned gradient is :math:`\partial h / \partial W_{\text{raw}}`.

    In D-RTRL notation this is the instantaneous
    :math:`\operatorname{diag}(\mathbf{D}_f^t) \otimes \mathbf{x}^t` term
    added to :math:`\mathbf{D}^t \boldsymbol{\epsilon}^{t-1}`. In
    ES-D-RTRL it supplies the factor combined at solve-time with the
    :math:`x`-trace to form the weight gradient.

    Using ``jax.vjp`` over a *dict-valued* forward function fuses the
    weight and bias pullbacks in one pass, avoiding a second VJP call
    when ``has_bias=True``.
    """

    def _fwd(w_dict: dict[str, Any]) -> Any:
        w = w_dict['weight']
        if weight_fn is not None:
            w = weight_fn(w)
        y = x @ w
        if has_bias:
            b = w_dict['bias']
            if bias_fn is not None:
                b = bias_fn(b)
            y = y + b
        return u.get_mantissa(y)

    _, vjp_fn = jax.vjp(_fwd, weights)
    return jax.tree.map(u.get_mantissa, vjp_fn(hidden_dim)[0])


def _mm_init_drtrl(x_var: Any, y_var: Any, weight_vars: dict[str, Any],
                   num_hidden_state: int) -> dict[str, Any]:
    r"""Initialise the batched D-RTRL weight-shaped trace.

    D-RTRL stores one trace per (weight-entry, hidden-state) pair:

    .. math::

        \boldsymbol{\epsilon}_W \in
          \mathbb{R}^{B \times I \times O \times n_{\text{state}}},
        \qquad
        \boldsymbol{\epsilon}_b \in
          \mathbb{R}^{B \times O \times n_{\text{state}}}.

    Zero initialisation is consistent with :math:`\boldsymbol{\epsilon}^{0} = \mathbf{0}`
    in the recurrence
    :math:`\boldsymbol{\epsilon}^t \approx \mathbf{D}^t \boldsymbol{\epsilon}^{t-1}
                                           + \operatorname{diag}(\mathbf{D}_f^t)\otimes \mathbf{x}^t`.

    The trace dtype is derived from the participating ``x``/``y``/weight
    avals via :func:`jax.numpy.result_type` rather than left to ``jnp.zeros``'
    default (which silently follows the global x64 flag instead of the
    operands' actual dtype).
    """
    batch = x_var.aval.shape[0]
    dtype = jnp.result_type(
        x_var.aval.dtype, y_var.aval.dtype,
        *(v.aval.dtype for v in weight_vars.values()),
    )
    out = {
        'weight': jnp.zeros(
            (batch, *weight_vars['weight'].aval.shape, num_hidden_state), dtype=dtype
        )
    }
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros(
            (batch, *weight_vars['bias'].aval.shape, num_hidden_state), dtype=dtype
        )
    return out


def _mm_init_pp(x_var: Any, y_var: Any, weight_vars: dict[str, Any],
                num_hidden_state: int) -> Any:
    r"""Initialise the batched pp-prop / ES-D-RTRL output-shaped df trace.

    pp-prop factorises the eligibility trace as
    :math:`\boldsymbol{\epsilon} \approx \boldsymbol{\epsilon}_f \otimes \boldsymbol{\epsilon}_x`,
    so the trace stored per primitive is output-shaped (one entry per
    output unit, per hidden state):

    .. math::

        \boldsymbol{\epsilon}_f \in
          \mathbb{R}^{B \times O \times n_{\text{state}}}.

    The :math:`\boldsymbol{\epsilon}_x` factor lives in a separate
    executor dictionary and is combined with :math:`\boldsymbol{\epsilon}_f`
    only at gradient-solve time via :func:`xy_to_dw`.
    """
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


# ---------------------------------------------------------------------------
# Closed-form param-dim D-RTRL fast-path kernels (shared by mm and mv)
# ---------------------------------------------------------------------------

def _dense_fast_instant(x: ArrayLike, df: ArrayLike, has_bias: bool) -> dict[str, Any]:
    r"""Instantaneous term :math:`\operatorname{diag}(\mathbf{D}_f^t) \otimes \mathbf{x}^t`.

    Parameters
    ----------
    x : ArrayLike
        Presynaptic input. Shape ``(..., in)`` (``(B, in)`` for mm,
        ``(in,)`` for mv).
    df : ArrayLike
        State-to-output Jacobian :math:`\mathbf{D}_f^t`, shape
        ``(..., out, num_state)``.
    has_bias : bool
        Whether to emit a ``'bias'`` entry (equal to ``df``).

    Returns
    -------
    dict
        ``{'weight': x ⊗ df}`` (plus ``'bias': df`` when ``has_bias``).

    Notes
    -----
    The single spec ``'...i,...ka->...ika'`` covers both mm and mv: the
    ``...`` block absorbs mm's leading batch axis and matches the empty
    leading block of mv, so the outer product ``x_i ⊗ df_{ka}`` is computed
    identically in both ranks (verified numerically equal to mv's bespoke
    ``'i,ka->ika'``).
    """
    out: dict[str, Any] = {'weight': jnp.einsum('...i,...ka->...ika', x, df)}
    if has_bias:
        out['bias'] = df
    return out


def _dense_fast_recurrent(diag: jax.Array, old_bwg: dict[str, Any], num_state: int) -> dict[str, Any]:
    r"""Recurrent term :math:`\mathbf{D}^t \boldsymbol{\epsilon}^{t-1}`.

    Parameters
    ----------
    diag : jax.Array
        Hidden-to-hidden Jacobian, shape ``(..., out, num_state, num_state)``.
    old_bwg : dict
        Previous weight-shaped trace dict; ``'weight'`` has shape
        ``(..., in, out, num_state)`` and optional ``'bias'`` has shape
        ``(..., out, num_state)``.
    num_state : int
        Number of hidden states per group.

    Returns
    -------
    dict
        The contracted trace dict for the current step.

    Notes
    -----
    The contraction is ``new[...,i,k,a] = Σ_b diag[...,k,a,b] · trace[...,i,k,b]``,
    i.e. ``einsum('...kab,...ikb->...ika')`` (bias: ``'...kab,...kb->...ka'``).
    When ``num_state == 1`` both state axes are size 1, so the sum over
    ``b`` collapses to a broadcast multiply by ``diag[..., 0, 0]`` —
    bit-identical to the einsum but without a degenerate ``dot_general``.
    """
    if num_state == 1:
        d = diag[..., 0, 0]  # (..., out) — the hidden ``k`` axis
        out = {'weight': d[..., None, :, None] * old_bwg['weight']}
        if 'bias' in old_bwg:
            out['bias'] = d[..., None] * old_bwg['bias']
        return out
    out = {'weight': jnp.einsum('...kab,...ikb->...ika', diag, old_bwg['weight'])}
    if 'bias' in old_bwg:
        out['bias'] = jnp.einsum('...kab,...kb->...ka', diag, old_bwg['bias'])
    return out


def _dense_fast_solve(diag_like: ArrayLike, etrace_data: dict[str, Any], *, fold_batch: bool = False) -> dict[str, Any]:
    r"""Solve-time contraction of the learning signal with the trace.

    Parameters
    ----------
    diag_like : ArrayLike
        The :math:`\partial \mathcal{L}/\partial \mathbf{h}` group gradient,
        shape ``(..., out, num_state)`` (``(B, out, num_state)`` when batched).
    etrace_data : dict
        Weight-shaped trace dict; ``'weight'`` shape
        ``(..., in, out, num_state)`` and optional ``'bias'`` shape
        ``(..., out, num_state)``.
    fold_batch : bool, optional
        When ``True``, contract a leading batch axis ``b`` inside the einsum
        so the result is already batch-summed (avoids a ``(B, I, O)``
        intermediate). Default ``False``.

    Returns
    -------
    dict
        ``{'weight': dW}`` (plus ``'bias': db`` when present).

    Notes
    -----
    Contracts the ``num_state`` axis (``a``): ``'...ka,...ika->...ik'`` for
    the weight and ``'...ka,...ka->...k'`` for the bias. With ``fold_batch``
    the leading ``b`` axis is added to the contraction
    (``'bka,bika->ik'`` / ``'bka,bka->k'``), assuming exactly one batch axis.
    """
    w_spec = 'bka,bika->ik' if fold_batch else '...ka,...ika->...ik'
    out = {'weight': jnp.einsum(w_spec, diag_like, etrace_data['weight'])}
    if 'bias' in etrace_data:
        b_spec = 'bka,bka->k' if fold_batch else '...ka,...ka->...k'
        out['bias'] = jnp.einsum(b_spec, diag_like, etrace_data['bias'])
    return out


def _dense_fast_applicable(eqn_params: dict[str, Any]) -> bool:
    r"""Gate: is the dense fast path valid for this equation?

    Parameters
    ----------
    eqn_params : dict
        The ETP equation's ``params`` dict.

    Returns
    -------
    bool
        ``True`` iff neither ``weight_fn`` nor ``bias_fn`` is active.

    Notes
    -----
    The closed-form kernels emit the bare outer product ``x ⊗ df`` — the
    gradient w.r.t. the *transformed* weight, dropping the ``f'(W)`` factor.
    Both keys are always present in ``eqn.params`` for mm / mv, so the
    AND-both test also disables the fast path for a ``bias_fn``-only
    transform (the kernels cannot supply ``f'``).
    """
    return eqn_params.get('weight_fn') is None and eqn_params.get('bias_fn') is None


_DENSE_FAST_PATH = FastPathRules(
    _dense_fast_instant,
    _dense_fast_recurrent,
    _dense_fast_solve,
    _dense_fast_applicable,
)


etp_mm_p = register_primitive(
    'etp_mm',
    _etp_matmul_impl,
    batched=True,
    trainable_invars_fn=_mm_trainable_invars,
    x_invar_index=0,
)
etp_mm_p.register_etp_rules(
    dt_to_t=_mm_dt_to_t,
    xy_to_dw=_mm_xy_to_dw,
    init_drtrl=_mm_init_drtrl,
    init_pp=_mm_init_pp,
    fast_path=_DENSE_FAST_PATH,
)


# ---------------------------------------------------------------------------
# etp_mv_p — unbatched
# ---------------------------------------------------------------------------

def _mv_trainable_invars(params: dict[str, Any]) -> dict[str, int]:
    """Return ``{key: invar_index}`` depending on ``has_bias``."""
    base = {'weight': 1}
    if params.get('has_bias', False):
        base['bias'] = 2
    return base


def _mv_dt_to_t(hidden_dim: Any, trace: dict[str, Any], *, has_bias: bool = False,
                weight_fn: WeightFn | None = None,
                bias_fn: WeightFn | None = None) -> dict[str, Any]:
    r"""Unbatched ``dt_to_t`` — same algebra as the batched case, no batch axis.

    **Role in D-RTRL.** Realises the :math:`y \to W` chain factor within
    :math:`\mathbf{D}^t \boldsymbol{\epsilon}^{t-1}`; since
    :math:`\partial y_j / \partial W_{ik} = \delta_{jk} x_i` this reduces
    to elementwise multiplication along the ``out`` axis:

    .. math::

        \epsilon^{t}_{W, ik} = (\partial h / \partial y)_k \;
                               \epsilon^{t-1}_{W, ik}, \qquad
        \epsilon^{t}_{b, k}  = (\partial h / \partial y)_k \;
                               \epsilon^{t-1}_{b, k}.

    **Shapes (solve context after the ``n_state``-vmap):**

        ``hidden_dim``      : ``(out,)``
        ``trace['weight']`` : ``(in, out)``
        ``trace['bias']``   : ``(out,)``   (when ``has_bias=True``)

    ``jnp.expand_dims(hidden_dim, axis=0)`` turns ``(out,) → (1, out)``
    so it broadcasts against ``(in, out)`` for the weight trace.
    """
    out = {'weight': trace['weight'] * jnp.expand_dims(hidden_dim, axis=0)}
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _mv_xy_to_dw(x: Any, hidden_dim: Any, weights: dict[str, Any], *,
                 has_bias: bool = False, weight_fn: WeightFn | None = None,
                 bias_fn: WeightFn | None = None) -> dict[str, Any]:
    r"""Unbatched ``xy_to_dw`` — instantaneous :math:`\partial h / \partial W`.

    Same chain rule as the batched case with no batch axis. When ``weight_fn``
    / ``bias_fn`` are provided the VJP propagates through them automatically,
    returning the gradient w.r.t. the **raw** weight:

    .. math::

        \frac{\partial h}{\partial W_{\text{raw}, ik}} = x_i\,
            f'(W_{\text{raw}})_{ik}\, \frac{\partial h}{\partial y_k}, \qquad
        \frac{\partial h}{\partial b_{\text{raw}, k}}    =
            g'(b_{\text{raw}})_k\, \frac{\partial h}{\partial y_k}.

    Supplies :math:`\operatorname{diag}(\mathbf{D}_f^t)\otimes \mathbf{x}^t`
    for D-RTRL and the pp-prop solve-time pullback in ES-D-RTRL.
    One fused ``jax.vjp`` over a dict-valued forward returns both weight
    and bias gradients in one pass.
    """

    def _fwd(w_dict: dict[str, Any]) -> Any:
        w = w_dict['weight']
        if weight_fn is not None:
            w = weight_fn(w)
        y = x @ w
        if has_bias:
            b = w_dict['bias']
            if bias_fn is not None:
                b = bias_fn(b)
            y = y + b
        return u.get_mantissa(y)

    _, vjp_fn = jax.vjp(_fwd, weights)
    return jax.tree.map(u.get_mantissa, vjp_fn(hidden_dim)[0])


def _mv_init_drtrl(x_var: Any, y_var: Any, weight_vars: dict[str, Any],
                   num_hidden_state: int) -> dict[str, Any]:
    r"""Initialise unbatched D-RTRL weight-shaped trace.

    .. math::

        \boldsymbol{\epsilon}_W \in \mathbb{R}^{I \times O \times n_{\text{state}}}, \qquad
        \boldsymbol{\epsilon}_b \in \mathbb{R}^{O \times n_{\text{state}}}.

    Zero-initialised (matches :math:`\boldsymbol{\epsilon}^0 = \mathbf{0}`).

    The trace dtype is derived from the participating ``x``/``y``/weight
    avals via :func:`jax.numpy.result_type` rather than left to ``jnp.zeros``'
    default (which silently follows the global x64 flag instead of the
    operands' actual dtype).
    """
    dtype = jnp.result_type(
        x_var.aval.dtype, y_var.aval.dtype,
        *(v.aval.dtype for v in weight_vars.values()),
    )
    out = {
        'weight': jnp.zeros(
            (*weight_vars['weight'].aval.shape, num_hidden_state), dtype=dtype
        )
    }
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros(
            (*weight_vars['bias'].aval.shape, num_hidden_state), dtype=dtype
        )
    return out


def _mv_init_pp(x_var: Any, y_var: Any, weight_vars: dict[str, Any],
                num_hidden_state: int) -> Any:
    r"""Initialise unbatched pp-prop / ES-D-RTRL df trace.

    .. math::

        \boldsymbol{\epsilon}_f \in \mathbb{R}^{O \times n_{\text{state}}}.

    The matching :math:`\boldsymbol{\epsilon}_x` factor is held by the
    executor's x-trace dictionary and combined at solve-time.
    """
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_mv_p = register_primitive(
    'etp_mv',
    _etp_matmul_impl,
    batched=False,
    trainable_invars_fn=_mv_trainable_invars,
    x_invar_index=0,
)
etp_mv_p.register_etp_rules(
    dt_to_t=_mv_dt_to_t,
    xy_to_dw=_mv_xy_to_dw,
    init_drtrl=_mv_init_drtrl,
    init_pp=_mv_init_pp,
    fast_path=_DENSE_FAST_PATH,
)
register_batched_counterpart(etp_mv_p, etp_mm_p)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def matmul(
    x: ArrayLike,
    weight: ArrayLike,
    bias: ArrayLike | None = None,
    *,
    weight_fn: WeightFn | None = None,
    bias_fn: WeightFn | None = None,
) -> ArrayLike:
    r"""ETP-aware matrix multiplication.

    Computes :math:`y = x \mathbin{@} \text{weight\_fn}(w) \; (+ \text{bias\_fn}(b))`.
    The operation is routed through an ETP primitive so the weight (and
    optional bias) participates in eligibility-trace computation.
    Auto-dispatches to ``etp_mm_p`` (batched) or ``etp_mv_p`` (unbatched)
    based on ``x.ndim``.

    The eligibility trace and gradient are always taken w.r.t. the **raw**
    weight/bias before any transform is applied.

    Parameters
    ----------
    x : ArrayLike
        Input array, shape ``(batch, in_features)`` or ``(in_features,)``.
        Higher-rank ``x`` (``x.ndim > 2``) is rejected with a ``ValueError``:
        every ETP trace rule assumes one of these two layouts. For genuine
        tensor contractions use :func:`braintrace.einsum`.
    weight : ArrayLike
        Weight matrix, shape ``(in_features, out_features)``.
    bias : ArrayLike or None, optional
        Bias vector, shape ``(out_features,)``. Default ``None``.
    weight_fn : Callable or None, optional
        Elementwise, shape-preserving transform applied to ``weight`` *inside*
        the primitive: the op computes ``y = x @ weight_fn(weight)``. The
        eligibility trace and gradient are taken w.r.t. the raw ``weight``.
        The transform operates on the unitless mantissa; physical units are
        split off before and recombined after.
        Default ``None`` (identity).
        Pass a module-level function, not a fresh ``lambda``, if this is
        called repeatedly (e.g. once per step): ``weight_fn`` is stored as a
        static ``eqn.params`` entry hashed by object identity, so two
        textually identical ``lambda`` objects are cache misses and silently
        retrace every call.
    bias_fn : Callable or None, optional
        Elementwise transform applied to ``bias`` inside the primitive
        (``+ bias_fn(bias)``). Ignored when ``bias is None``.
        The transform operates on the unitless mantissa; physical units are
        split off before and recombined after.
        Default ``None``.
        Same re-tracing caveat as ``weight_fn``: pass a module-level
        function rather than a fresh ``lambda``.

    Returns
    -------
    ArrayLike
        Output array, shape ``(batch, out_features)`` or ``(out_features,)``.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>>
        >>> brainstate.environ.set(precision=64)
        >>> x = brainstate.random.randn(16, 3)
        >>> w = brainstate.random.randn(3, 4)
        >>> y = braintrace.matmul(x, w)
        >>> print(y.shape)
        (16, 4)
        >>>
        >>> # Unbatched input with a bias term
        >>> x1 = brainstate.random.randn(3)
        >>> b = brainstate.random.randn(4)
        >>> y1 = braintrace.matmul(x1, w, bias=b)
        >>> print(y1.shape)
        (4,)
        >>>
        >>> # Constrain weights to be non-negative via a transform
        >>> y2 = braintrace.matmul(x, w, weight_fn=lambda ww: ww ** 2)
        >>> print(y2.shape)
        (16, 4)
    """
    if x.ndim > 2:  # type: ignore[union-attr]
        raise ValueError(
            f'matmul() supports x.ndim of 1 (unbatched `(in_features,)`) or 2 '
            f'(batched `(batch, in_features)`); got x.ndim={x.ndim} '
            f'(shape={x.shape}). Every ETP trace rule for etp_mm_p / etp_mv_p '
            f'assumes one of those two layouts, so higher-rank inputs (e.g. '
            f'`(batch, time, in_features)`) are not supported -- reshape/vmap '
            f'over the extra axes before calling matmul(), or use '
            f'braintrace.einsum for genuine tensor contractions.'
        )
    p = etp_mm_p if x.ndim >= 2 else etp_mv_p  # type: ignore[union-attr]  # x is an array here; ArrayLike also admits scalars without .ndim
    x_v, x_u = u.split_mantissa_unit(x)
    weight_v, weight_u = u.split_mantissa_unit(weight)
    unit = x_u * weight_u
    if bias is not None:
        bias_v = u.Quantity(bias).to_decimal(unit)
        r = p.bind(x_v, weight_v, bias_v, has_bias=True,
                   weight_fn=weight_fn, bias_fn=bias_fn)
    else:
        r = p.bind(x_v, weight_v, has_bias=False,
                   weight_fn=weight_fn, bias_fn=bias_fn)
    return u.maybe_decimal(r * x_u * weight_u)
