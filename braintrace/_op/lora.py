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

r"""LoRA (Low-Rank Adaptation) ETP primitives.

``etp_lora_mm_p`` (batched) and ``etp_lora_mv_p`` (unbatched) compute
:math:`y = \alpha \cdot x \mathbin{@} B \mathbin{@} A` plus an optional
bias. The trace and gradient state are pytrees with ``lora_b``, ``lora_a``
(and optionally ``bias``) leaves; the originating ``ParamState`` holds
all factors as a pytree, e.g. ``{'lora_b': B, 'lora_a': A, 'bias': b}``.

**Forward operation**

.. math::

    y = \alpha \, x \, B \, A \;(+ b), \qquad
    B \in \mathbb{R}^{I \times r}, \;
    A \in \mathbb{R}^{r \times O}, \;
    r \ll \min(I, O).

The intermediate :math:`z = x B \in \mathbb{R}^{\dots \times r}` is what
flows through :math:`A` to produce :math:`y`. Both :math:`A` and
:math:`B` are trainable; :math:`\alpha` is a scalar scaling (static).

**Role of each ETP rule**

Let :math:`g = \partial h / \partial y`. The chain rule yields

.. math::

    \frac{\partial h}{\partial A_{r,k}}
      \;=\; g_k \cdot \alpha \cdot (x B)_{r}, \qquad
    \frac{\partial h}{\partial B_{i,r}}
      \;=\; \alpha \sum_k g_k\, A_{r,k}\, x_i, \qquad
    \frac{\partial h}{\partial b_k}
      \;=\; g_k.

* ``xy_to_dw`` — VJP of :math:`y = \alpha\, x B A + b` over the whole
  dict ``{'lora_b', 'lora_a', 'bias'}``. JAX's autodiff delivers all
  three pullbacks from a single ``jax.vjp`` call, giving the
  instantaneous :math:`\operatorname{diag}(\mathbf{D}_f^t)\otimes \mathbf{x}^t`
  term of D-RTRL (and the solve-time factor of ES-D-RTRL).

* ``yw_to_w`` — only propagates :math:`g` through the :math:`A` factor
  (plus elementwise through the bias). Intuition: :math:`A` is the
  "output-facing" factor, so :math:`\partial y / \partial A` attaches a
  :math:`g`-shaped scaling to the :math:`A` trace, exactly like dense
  matmul's :math:`y \to W` link. :math:`B` is "input-facing" and has
  no such :math:`y`-dependent scaling in the linearised view — its
  trace is carried unchanged through the :math:`y \to W` step. (The
  full :math:`B` gradient *does* depend on :math:`A`, but that
  dependence enters via ``xy_to_dw``, not through the trace
  propagation.)

* ``init_drtrl`` — allocates separate leaves for :math:`\boldsymbol{\epsilon}_B`,
  :math:`\boldsymbol{\epsilon}_A`, and optionally :math:`\boldsymbol{\epsilon}_b`,
  each of shape ``(*factor_shape, n_state)`` (plus batch prefix in the
  batched primitive).

* ``init_pp`` — output-shaped df trace; same as dense.

**Dict rule API (N-trainable-input refactor)**

Both primitives declare ``trainable_invars_fn``, which returns
``{'lora_b': 1, 'lora_a': 2}`` when ``has_bias=False`` and
``{'lora_b': 1, 'lora_a': 2, 'bias': 3}`` when ``has_bias=True``.
Keys ``'lora_b'`` / ``'lora_a'`` match the pytree leaf names in
``braintrace.nn.LoRALinear``'s merged ``ParamState``.
"""

import brainunit as u
import jax
import jax.numpy as jnp

from ._primitive import register_primitive

__all__ = [
    'etp_lora_mm_p',
    'etp_lora_mv_p',
    'lora_matmul',
]


def _etp_lora_impl(*args, alpha=1.0, has_bias=False, b_fn=None, a_fn=None, bias_fn=None):
    x, B, A = args[0], args[1], args[2]
    if b_fn is not None:
        B = b_fn(B)
    if a_fn is not None:
        A = a_fn(A)
    y = alpha * (x @ B @ A)
    if has_bias:
        b = args[3]
        if bias_fn is not None:
            b = bias_fn(b)
        y = y + b
    return y


def _lora_trainable_invars(params):
    """Return ``{key: invar_index}`` for LoRA's trainable inputs."""
    base = {'lora_b': 1, 'lora_a': 2}
    if params.get('has_bias', False):
        base['bias'] = 3
    return base


def _lora_mm_yw_to_w(
    hidden_dim, trace,
    *, alpha=1.0, has_bias=False, b_fn=None, a_fn=None, bias_fn=None
):
    r"""Batched LoRA ``yw_to_w`` — propagate :math:`\partial h / \partial y`
    through the :math:`y \to A` link.

    **Role in D-RTRL.** Realises the :math:`y \to (A, B, b)` chain factor
    of :math:`\mathbf{D}^t \boldsymbol{\epsilon}^{t-1}` for the LoRA op.
    Differentiating :math:`y_k = \alpha \sum_r (xB)_r A_{r,k}` gives

    .. math::

        \frac{\partial y_k}{\partial A_{r, k'}} = \delta_{k k'}\, \alpha\, (xB)_r,
        \qquad
        \frac{\partial y_k}{\partial B_{i, r}} =
          \alpha\, A_{r, k}\, x_i.

    After the executor has already absorbed the :math:`\mathbf{D}^t`
    contraction along the hidden axis, only the :math:`y \to` link
    remains for ``yw_to_w``. For :math:`A` this link is a simple
    broadcast of :math:`g = \partial h / \partial y` across the ``rank``
    axis of the trace:

    .. math::

        \epsilon^t_{A, r, k} = g_k\, \epsilon^{t-1}_{A, r, k}.

    For :math:`B`, the :math:`y \to B` link additionally carries an
    :math:`A` factor which *does* depend on :math:`y` via the hidden
    state. In the D-RTRL diagonal approximation used here, this
    cross-coupling is absorbed into the instantaneous contribution
    supplied by ``xy_to_dw`` each step rather than carried through the
    trace. Consequently the :math:`B`-trace is left unchanged by
    :func:`yw_to_w` — propagation only touches :math:`A` (and the
    bias, which is diagonal as usual).

    **Broadcast rule.** ``jnp.expand_dims(hidden_dim, axis=-2)`` inserts
    a singleton at the ``rank`` position in both execution contexts:

        (out,)        → (1, out)         broadcasts with (rank, out)        ✓
        (batch, out)  → (batch, 1, out)  broadcasts with (batch, rank, out) ✓

    **Shapes.**
        trace['lora_b'] : ``(..., in, rank)``   — unchanged
        trace['lora_a'] : ``(..., rank, out)``  — scaled by ``g``
        trace['bias']   : ``(..., out)``        — elementwise :math:`g`
    """
    trace_A = trace['lora_a'] * jnp.expand_dims(hidden_dim, axis=-2)
    out = {'lora_b': trace['lora_b'], 'lora_a': trace_A}
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _lora_mv_yw_to_w(
    hidden_dim, trace,
    *, alpha=1.0, has_bias=False, b_fn=None, a_fn=None, bias_fn=None
):
    r"""Unbatched LoRA ``yw_to_w`` — identical algebra with no batch axis.

    Trace shapes:
        ``trace['lora_b'] : (in, rank, n_state)``   — unchanged
        ``trace['lora_a'] : (rank, out, n_state)``  — scaled by :math:`g`
        ``trace['bias']   : (out, n_state)``        — elementwise :math:`g`

    ``jnp.expand_dims(hidden_dim, axis=0)`` turns ``(out,) → (1, out)``
    so it broadcasts against the ``(rank, out)`` leading axes of the
    :math:`A` trace. As in the batched case, only :math:`A` (and the
    bias) are touched; the :math:`B`-trace propagates unchanged (its
    :math:`y \to B` chain factor is deferred to ``xy_to_dw``).
    """
    trace_A = trace['lora_a'] * jnp.expand_dims(hidden_dim, axis=0)
    out = {'lora_b': trace['lora_b'], 'lora_a': trace_A}
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _lora_xy_to_dw(
    x, hidden_dim, weights,
    *, alpha=1.0, has_bias=False, b_fn=None, a_fn=None, bias_fn=None
):
    r"""Instantaneous LoRA Jacobian via fused VJP.

    **Role in D-RTRL / ES-D-RTRL.** Produces the full instantaneous
    :math:`\partial h / \partial \{A, B, b\}` term in one ``jax.vjp``
    pass. Using :math:`g = \partial h / \partial y`:

    .. math::

        \frac{\partial h}{\partial A_{r, k}}
          = \alpha\, (x\,b\_fn(B))_r\, g_k \cdot a\_fn'(A_{r,k}),

    .. math::

        \frac{\partial h}{\partial B_{i, r}}
          = \alpha\, \sum_k a\_fn(A)_{r, k}\, g_k\, x_i \cdot b\_fn'(B_{i,r}),

    .. math::

        \frac{\partial h}{\partial b_k}
          = g_k \cdot bias\_fn'(b_k).

    All three are computed simultaneously by differentiating

    .. code-block:: python

        def _fwd(w):
            B = b_fn(w['lora_b']) if b_fn else w['lora_b']
            A = a_fn(w['lora_a']) if a_fn else w['lora_a']
            return alpha * (x @ B @ A) + (bias_fn(w['bias']) if bias_fn else w['bias'])

    and pulling back the cotangent ``hidden_dim``. When all three transform
    functions are ``None``, the output is bit-identical to the un-transformed
    case. In D-RTRL this is the
    :math:`\operatorname{diag}(\mathbf{D}_f^t)\otimes \mathbf{x}^t`
    contribution; in ES-D-RTRL it is the pullback applied at solve-time
    to combine :math:`\boldsymbol{\epsilon}_f^t` with
    :math:`\boldsymbol{\epsilon}_x^t` into the weight gradient.
    """

    def _fwd(w):
        B = w['lora_b']
        A = w['lora_a']
        if b_fn is not None:
            B = b_fn(B)
        if a_fn is not None:
            A = a_fn(A)
        y = alpha * (x @ B @ A)
        if has_bias:
            b = w['bias']
            if bias_fn is not None:
                b = bias_fn(b)
            y = y + b
        return u.get_mantissa(y)

    _, vjp_fn = jax.vjp(_fwd, weights)
    return jax.tree.map(u.get_mantissa, vjp_fn(hidden_dim)[0])


def _lora_mm_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    r"""Initialise batched LoRA D-RTRL trace.

    Each LoRA factor gets its own trace leaf:

    .. math::

        \boldsymbol{\epsilon}_B \in \mathbb{R}^{B \times I \times r \times n_{\text{state}}}, \quad
        \boldsymbol{\epsilon}_A \in \mathbb{R}^{B \times r \times O \times n_{\text{state}}}, \quad
        \boldsymbol{\epsilon}_b \in \mathbb{R}^{B \times O \times n_{\text{state}}}.

    Memory cost :math:`\mathcal{O}(B\, r\, (I + O))` versus
    :math:`\mathcal{O}(B\, I\, O)` for a dense layer — the whole point
    of LoRA. Zero-initialised.
    """
    batch = x_var.aval.shape[0]
    B_shape = weight_vars['lora_b'].aval.shape
    A_shape = weight_vars['lora_a'].aval.shape
    out = {
        'lora_b': jnp.zeros((batch, *B_shape, num_hidden_state)),
        'lora_a': jnp.zeros((batch, *A_shape, num_hidden_state)),
    }
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros(
            (batch, *weight_vars['bias'].aval.shape, num_hidden_state)
        )
    return out


def _lora_mm_init_pp(x_var, y_var, weight_vars, num_hidden_state):
    r"""Initialise batched LoRA pp-prop / ES-D-RTRL df trace.

    .. math::

        \boldsymbol{\epsilon}_f \in \mathbb{R}^{B \times O \times n_{\text{state}}}.

    Same shape as dense — pp-prop factorisation does not care how
    :math:`W = \alpha B A` is stored. The :math:`\boldsymbol{\epsilon}_x`
    factor is the raw :math:`x`; the :math:`B, A, b` split is handled by
    :func:`_lora_xy_to_dw` at solve-time.
    """
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


def _lora_mv_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    r"""Initialise unbatched LoRA D-RTRL trace.

    .. math::

        \boldsymbol{\epsilon}_B \in \mathbb{R}^{I \times r \times n_{\text{state}}}, \quad
        \boldsymbol{\epsilon}_A \in \mathbb{R}^{r \times O \times n_{\text{state}}}, \quad
        \boldsymbol{\epsilon}_b \in \mathbb{R}^{O \times n_{\text{state}}}.

    Zero-initialised.
    """
    B_shape = weight_vars['lora_b'].aval.shape
    A_shape = weight_vars['lora_a'].aval.shape
    out = {
        'lora_b': jnp.zeros((*B_shape, num_hidden_state)),
        'lora_a': jnp.zeros((*A_shape, num_hidden_state)),
    }
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros(
            (*weight_vars['bias'].aval.shape, num_hidden_state)
        )
    return out


def _lora_mv_init_pp(x_var, y_var, weight_vars, num_hidden_state):
    r"""Initialise unbatched LoRA pp-prop / ES-D-RTRL df trace.

    .. math::

        \boldsymbol{\epsilon}_f \in \mathbb{R}^{O \times n_{\text{state}}}.
    """
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_lora_mm_p = register_primitive(
    'etp_lora_mm',
    _etp_lora_impl,
    batched=True,
    trainable_invars_fn=_lora_trainable_invars,
    x_invar_index=0,
)
etp_lora_mm_p.register_etp_rules(
    yw_to_w=_lora_mm_yw_to_w,
    xy_to_dw=_lora_xy_to_dw,
    init_drtrl=_lora_mm_init_drtrl,
    init_pp=_lora_mm_init_pp,
)

etp_lora_mv_p = register_primitive(
    'etp_lora_mv',
    _etp_lora_impl,
    batched=False,
    trainable_invars_fn=_lora_trainable_invars,
    x_invar_index=0,
)
etp_lora_mv_p.register_etp_rules(
    yw_to_w=_lora_mv_yw_to_w,
    xy_to_dw=_lora_xy_to_dw,
    init_drtrl=_lora_mv_init_drtrl,
    init_pp=_lora_mv_init_pp,
)


def lora_matmul(x, B, A, *, alpha=1.0, bias=None, b_fn=None, a_fn=None, bias_fn=None):
    r"""ETP-aware LoRA (Low-Rank Adaptation) matrix multiplication.

    Computes :math:`y = \alpha \cdot x \mathbin{@} b\_fn(B) \mathbin{@} a\_fn(A) \; (+ bias\_fn(b))`,
    routing both low-rank factors (and the optional bias) through an ETP
    primitive so they participate in eligibility-trace computation.
    Auto-dispatches batched/unbatched based on ``x.ndim``.

    Parameters
    ----------
    x : ArrayLike
        Input array, shape ``(..., in_features)`` or ``(in_features,)``.
    B : ArrayLike
        Low-rank matrix :math:`B`, shape ``(in_features, rank)``.
    A : ArrayLike
        Low-rank matrix :math:`A`, shape ``(rank, out_features)``.
    alpha : float, optional
        Scalar scaling factor :math:`\alpha`. Default ``1.0``.
    bias : ArrayLike or None, optional
        Bias vector, shape ``(out_features,)``. Default ``None``.
    b_fn : callable or None, optional
        Elementwise transform applied to the ``B`` factor before the
        matrix multiplication.  ``b_fn(B)`` must return an array of the
        same shape as ``B``.  ``None`` means identity (no transform).
        The VJP of ``b_fn`` is auto-composed inside ``xy_to_dw`` so that
        gradients w.r.t. the raw ``lora_b`` weights are correct.
        The transform operates on the unitless mantissa; physical units are
        split off before and recombined after.
    a_fn : callable or None, optional
        Elementwise transform applied to the ``A`` factor before the
        matrix multiplication.  ``a_fn(A)`` must return an array of the
        same shape as ``A``.  ``None`` means identity.
        The transform operates on the unitless mantissa; physical units are
        split off before and recombined after.
    bias_fn : callable or None, optional
        Elementwise transform applied to ``bias`` before adding.
        ``None`` means identity.
        The transform operates on the unitless mantissa; physical units are
        split off before and recombined after.

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
        >>> x = brainstate.random.randn(16, 8)
        >>> B = brainstate.random.randn(8, 2)
        >>> A = brainstate.random.randn(2, 4)
        >>> y = braintrace.lora_matmul(x, B, A, alpha=0.5)
        >>> print(y.shape)
        (16, 4)
    """
    p = etp_lora_mm_p if x.ndim >= 2 else etp_lora_mv_p
    x_v, x_u = u.split_mantissa_unit(x)
    B_v, B_u = u.split_mantissa_unit(B)
    A_v, A_u = u.split_mantissa_unit(A)
    unit = x_u * B_u * A_u
    if bias is not None:
        bias_v = u.Quantity(bias).to_decimal(unit)
        r = p.bind(x_v, B_v, A_v, bias_v, alpha=alpha, has_bias=True,
                   b_fn=b_fn, a_fn=a_fn, bias_fn=bias_fn)
    else:
        r = p.bind(x_v, B_v, A_v, alpha=alpha, has_bias=False,
                   b_fn=b_fn, a_fn=a_fn, bias_fn=bias_fn)
    return u.maybe_decimal(r * x_u * B_u * A_u)
