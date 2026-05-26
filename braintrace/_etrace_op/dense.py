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

* ``yw_to_w(hidden_dim, trace)`` — multiplies the weight-shaped trace
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
"""

import jax
import jax.numpy as jnp
import saiunit as u

from ._primitive import register_primitive

__all__ = [
    'etp_mm_p',
    'etp_mv_p',
    'matmul',
]


def _etp_matmul_impl(*args, has_bias=False):
    x, w = args[0], args[1]
    y = x @ w
    if has_bias:
        y = y + args[2]
    return y


# ---------------------------------------------------------------------------
# etp_mm_p — batched
# ---------------------------------------------------------------------------

def _mm_trainable_invars(params):
    """Return ``{key: invar_index}`` depending on ``has_bias``."""
    base = {'weight': 1}
    if params.get('has_bias', False):
        base['bias'] = 2
    return base


def _mm_yw_to_w(hidden_dim, trace, *, has_bias=False):
    r"""Batched ``yw_to_w`` — propagate :math:`\partial h / \partial y`
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


def _mm_xy_to_dw(x, hidden_dim, weights, *, has_bias=False):
    r"""Batched ``xy_to_dw`` — instantaneous hidden-to-weight Jacobian.

    **Role.** Computes :math:`\partial h / \partial W` (and
    :math:`\partial h / \partial b`) by VJP of :math:`y = x W + b`,
    pulling back the cotangent ``hidden_dim`` = :math:`\partial h/\partial y`:

    .. math::

        \frac{\partial h}{\partial W_{ik}}
          \;=\; \sum_j \frac{\partial h}{\partial y_j}\,
                \frac{\partial y_j}{\partial W_{ik}}
          \;=\; \frac{\partial h}{\partial y_k}\, x_i ,

    .. math::

        \frac{\partial h}{\partial b_k}
          \;=\; \frac{\partial h}{\partial y_k}.

    In D-RTRL notation this is the instantaneous
    :math:`\operatorname{diag}(\mathbf{D}_f^t) \otimes \mathbf{x}^t` term
    added to :math:`\mathbf{D}^t \boldsymbol{\epsilon}^{t-1}`. In
    ES-D-RTRL it supplies the factor combined at solve-time with the
    :math:`x`-trace to form the weight gradient.

    Using ``jax.vjp`` over a *dict-valued* forward function fuses the
    weight and bias pullbacks in one pass, avoiding a second VJP call
    when ``has_bias=True``.
    """

    def _fwd(w_dict):
        y = x @ w_dict['weight']
        if has_bias:
            y = y + w_dict['bias']
        return u.get_mantissa(y)

    _, vjp_fn = jax.vjp(_fwd, weights)
    return jax.tree.map(u.get_mantissa, vjp_fn(hidden_dim)[0])


def _mm_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
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
    """
    batch = x_var.aval.shape[0]
    out = {
        'weight': jnp.zeros(
            (batch, *weight_vars['weight'].aval.shape, num_hidden_state)
        )
    }
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros(
            (batch, *weight_vars['bias'].aval.shape, num_hidden_state)
        )
    return out


def _mm_init_pp(x_var, y_var, weight_vars, num_hidden_state):
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


etp_mm_p = register_primitive(
    'etp_mm',
    _etp_matmul_impl,
    batched=True,
    trainable_invars_fn=_mm_trainable_invars,
    x_invar_index=0,
)
etp_mm_p.register_etp_rules(
    yw_to_w=_mm_yw_to_w,
    xy_to_dw=_mm_xy_to_dw,
    init_drtrl=_mm_init_drtrl,
    init_pp=_mm_init_pp,
)


# ---------------------------------------------------------------------------
# etp_mv_p — unbatched
# ---------------------------------------------------------------------------

def _mv_trainable_invars(params):
    """Return ``{key: invar_index}`` depending on ``has_bias``."""
    base = {'weight': 1}
    if params.get('has_bias', False):
        base['bias'] = 2
    return base


def _mv_yw_to_w(hidden_dim, trace, *, has_bias=False):
    r"""Unbatched ``yw_to_w`` — same algebra as the batched case, no batch axis.

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


def _mv_xy_to_dw(x, hidden_dim, weights, *, has_bias=False):
    r"""Unbatched ``xy_to_dw`` — instantaneous :math:`\partial h / \partial W`.

    Same chain rule as the batched case with no batch axis:

    .. math::

        \frac{\partial h}{\partial W_{ik}} = x_i\, \frac{\partial h}{\partial y_k}, \qquad
        \frac{\partial h}{\partial b_k}    = \frac{\partial h}{\partial y_k}.

    Supplies :math:`\operatorname{diag}(\mathbf{D}_f^t)\otimes \mathbf{x}^t`
    for D-RTRL and the pp-prop solve-time pullback in ES-D-RTRL.
    One fused ``jax.vjp`` over a dict-valued forward returns both weight
    and bias gradients in one pass.
    """

    def _fwd(w_dict):
        y = x @ w_dict['weight']
        if has_bias:
            y = y + w_dict['bias']
        return u.get_mantissa(y)

    _, vjp_fn = jax.vjp(_fwd, weights)
    return jax.tree.map(u.get_mantissa, vjp_fn(hidden_dim)[0])


def _mv_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    r"""Initialise unbatched D-RTRL weight-shaped trace.

    .. math::

        \boldsymbol{\epsilon}_W \in \mathbb{R}^{I \times O \times n_{\text{state}}}, \qquad
        \boldsymbol{\epsilon}_b \in \mathbb{R}^{O \times n_{\text{state}}}.

    Zero-initialised (matches :math:`\boldsymbol{\epsilon}^0 = \mathbf{0}`).
    """
    out = {
        'weight': jnp.zeros(
            (*weight_vars['weight'].aval.shape, num_hidden_state)
        )
    }
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros(
            (*weight_vars['bias'].aval.shape, num_hidden_state)
        )
    return out


def _mv_init_pp(x_var, y_var, weight_vars, num_hidden_state):
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
    yw_to_w=_mv_yw_to_w,
    xy_to_dw=_mv_xy_to_dw,
    init_drtrl=_mv_init_drtrl,
    init_pp=_mv_init_pp,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def matmul(x, weight, bias=None):
    r"""ETP-aware matrix multiplication.

    Computes :math:`y = x \mathbin{@} w \; (+ b)`. The operation is routed
    through an ETP primitive so the weight (and optional bias) participates
    in eligibility-trace computation. Auto-dispatches to ``etp_mm_p``
    (batched) or ``etp_mv_p`` (unbatched) based on ``x.ndim``.

    Parameters
    ----------
    x : ArrayLike
        Input array, shape ``(..., in_features)`` or ``(in_features,)``.
    weight : ArrayLike
        Weight matrix, shape ``(in_features, out_features)``.
    bias : ArrayLike or None, optional
        Bias vector, shape ``(out_features,)``. Default ``None``.

    Returns
    -------
    ArrayLike
        Output array, shape ``(..., out_features)`` or ``(out_features,)``.

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
    """
    p = etp_mm_p if x.ndim >= 2 else etp_mv_p
    x_v, x_u = u.split_mantissa_unit(x)
    weight_v, weight_u = u.split_mantissa_unit(weight)
    unit = x_u * weight_u
    if bias is not None:
        bias_v = u.Quantity(bias).to_decimal(unit)
        r = p.bind(x_v, weight_v, bias_v, has_bias=True)
    else:
        r = p.bind(x_v, weight_v, has_bias=False)
    return u.maybe_decimal(r * x_u * weight_u)
