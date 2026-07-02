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
  three pullbacks from a single ``jax.vjp`` call. This **param-shaped**
  rule is what the IO-dim (ES-D-RTRL) algorithm applies at solve time.

* ``instant_drtrl`` / ``yw_to_w`` / ``solve_drtrl`` — the param-dim
  D-RTRL trace machinery. No :math:`B`-shaped ``(in, rank)`` trace can
  be exact: the hidden-to-hidden discount :math:`\mathbf{D}^t` acts on
  the *output* axis, which :math:`B`'s shape lacks (:math:`\partial h /
  \partial B` couples to the output only through :math:`A`). The trace
  stored under the ``'lora_b'`` key is therefore a **dense-style trace
  of the effective weight** :math:`W_{\text{eff}} = \alpha\, b\_fn(B)\,
  a\_fn(A)` of shape ``(batch?, in, out, n_state)``:

  - ``instant_drtrl`` adds :math:`x_i\, (\mathbf{D}_f^t)_o` to
    :math:`\boldsymbol{\epsilon}_{W}` (the exact dense instantaneous
    term for :math:`y = x\,W_{\text{eff}}`), while the :math:`A` and
    bias entries reuse the exact ``xy_to_dw`` pullbacks.
  - ``yw_to_w`` scales *every* trace along the output axis by
    :math:`g = \partial h / \partial y` (the dense :math:`y \to W`
    link) — including the :math:`W_{\text{eff}}` trace.
  - ``solve_drtrl`` contracts the learning signal with each trace and
    chains :math:`W_{\text{eff}}` back to the raw factor:
    :math:`\nabla_B = \operatorname{VJP}_{b\_fn}\!\big(\alpha\, G\,
    a\_fn(A)^\top\big)` with :math:`G_{io} = \sum_t g_t^{(o)}
    \boldsymbol{\epsilon}_{W,t}^{(io)}`. The chain is linear in
    :math:`G`, so applying it per step / per state / per batch and
    summing is exact for parameters held fixed over the gradient
    window.

* ``init_drtrl`` — allocates :math:`\boldsymbol{\epsilon}_W` of shape
  ``(batch?, in, out, n_state)`` under the ``'lora_b'`` key (the key
  keeps its name so gradient routing is untouched), plus the
  :math:`A`-shaped :math:`\boldsymbol{\epsilon}_A` and optionally the
  bias-shaped :math:`\boldsymbol{\epsilon}_b`.

* ``init_pp`` — output-shaped df trace; same as dense.

**Dict rule API (N-trainable-input refactor)**

Both primitives declare ``trainable_invars_fn``, which returns
``{'lora_b': 1, 'lora_a': 2}`` when ``has_bias=False`` and
``{'lora_b': 1, 'lora_a': 2, 'bias': 3}`` when ``has_bias=True``.

**Naming convention.** In this module ``lora_b`` is the *input-facing*
``(in, rank)`` factor (the first matrix applied to ``x``) and ``lora_a``
the *output-facing* ``(rank, out)`` factor — the classic LoRA-paper
:math:`B A` order for :math:`y = x B A`. Note that
:class:`braintrace.nn.LoRA` (following upstream ``brainstate.nn.LoRA``)
names its ``ParamState`` leaves the other way round: its ``'lora_a'``
leaf is the ``(in, rank)`` factor that flows into this primitive's
``lora_b`` operand, and its ``'lora_b'`` leaf the ``(rank, out)`` factor
that flows into ``lora_a``. Gradient routing is by dataflow (invar
position), not by leaf name, so the transposed naming is cosmetic.

**Transform hooks**

Both primitives accept three optional elementwise transform hooks in their
``eqn.params``: ``b_fn`` and ``a_fn`` (per-factor, computing
``y = alpha * x @ b_fn(B) @ a_fn(A)``) and ``bias_fn`` (adds ``bias_fn(b)``).
Note the per-factor names ``b_fn`` / ``a_fn`` rather than a single
``weight_fn``. The forward impl and :func:`_lora_xy_to_dw` apply them; the
gradients are always taken w.r.t. the **raw** factors / bias. In the
param-dim D-RTRL path, ``a_fn'`` and ``bias_fn'`` enter through
:func:`_lora_instant_drtrl` (which reuses the fused ``xy_to_dw`` VJP for
the :math:`A` / bias trace entries), while ``b_fn'`` enters at solve time
through :func:`_lora_solve_drtrl` via :func:`jax.vjp` (the effective-weight
trace itself is transform-free). In the IO-dim path all three Jacobians
enter through ``xy_to_dw``'s single fused VJP. The ``yw_to_w`` rule and the
trace initialisers are transform-free.

These primitives have **no fast path** — they always use the generic rule
path, which threads each :math:`f'` correctly when a transform hook is
present.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import brainunit as u

from ._primitive import register_primitive
from ._registries import ETP_RULES_INSTANT_DRTRL, ETP_RULES_SOLVE_DRTRL
from braintrace._typing import ArrayLike, WeightFn

__all__ = [
    'etp_lora_mm_p',
    'etp_lora_mv_p',
    'lora_matmul',
]


def _etp_lora_impl(*args: Any, alpha: float = 1.0, has_bias: bool = False,
                   b_fn: WeightFn | None = None, a_fn: WeightFn | None = None,
                   bias_fn: WeightFn | None = None) -> Any:
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


def _lora_trainable_invars(params: dict[str, Any]) -> dict[str, int]:
    """Return ``{key: invar_index}`` for LoRA's trainable inputs."""
    base = {'lora_b': 1, 'lora_a': 2}
    if params.get('has_bias', False):
        base['bias'] = 3
    return base


def _lora_mm_yw_to_w(hidden_dim: Any, trace: dict[str, Any], *, alpha: float = 1.0,
                     has_bias: bool = False, b_fn: WeightFn | None = None,
                     a_fn: WeightFn | None = None, bias_fn: WeightFn | None = None) -> dict[str, Any]:
    r"""Batched LoRA ``yw_to_w`` — propagate :math:`\partial h / \partial y`
    through every trace along the output axis.

    **Role in D-RTRL.** Realises the :math:`y \to` chain factor of
    :math:`\mathbf{D}^t \boldsymbol{\epsilon}^{t-1}` for the LoRA op,
    after the executor has already absorbed the :math:`\mathbf{D}^t`
    contraction along the hidden axis into ``hidden_dim`` :math:`= g`.

    The ``'lora_b'`` entry stores the **effective-weight trace**
    :math:`\boldsymbol{\epsilon}_W` for :math:`W_{\text{eff}} =
    \alpha\, b\_fn(B)\, a\_fn(A)` (see the module docstring): since
    :math:`y = x\, W_{\text{eff}}`, its :math:`y \to W_{\text{eff}}`
    link is the dense-matmul one,

    .. math::

        \epsilon^t_{W, io} = g_o\, \epsilon^{t-1}_{W, io},

    identical to the (verified-exact) dense ``mm`` rule. The
    :math:`A`-trace keeps its exact per-factor recurrence — for
    :math:`y_k = \alpha \sum_r (x\,b\_fn(B))_r\, a\_fn(A)_{r,k}` the
    :math:`y \to A` link broadcasts :math:`g` across the ``rank`` axis:

    .. math::

        \epsilon^t_{A, rk} = g_k\, \epsilon^{t-1}_{A, rk},

    and the bias trace is elementwise as usual. No trace propagates
    unchanged: a raw :math:`B`-shaped trace cannot be discounted
    correctly (the discount acts on the output axis :math:`B` lacks),
    which is exactly why :math:`\boldsymbol{\epsilon}_W` replaced it.

    **Broadcast rule.** ``jnp.expand_dims(hidden_dim, axis=-2)`` inserts
    a singleton before the output axis in both execution contexts, for
    both matrix-shaped traces:

        (out,)        → (1, out)         broadcasts with (in|rank, out)        ✓
        (batch, out)  → (batch, 1, out)  broadcasts with (batch, in|rank, out) ✓

    **Shapes.**
        trace['lora_b'] : ``(..., in, out)``    — :math:`\boldsymbol{\epsilon}_W`, scaled by ``g``
        trace['lora_a'] : ``(..., rank, out)``  — scaled by ``g``
        trace['bias']   : ``(..., out)``        — elementwise :math:`g`
    """
    g = jnp.expand_dims(hidden_dim, axis=-2)
    out = {'lora_b': trace['lora_b'] * g, 'lora_a': trace['lora_a'] * g}
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _lora_mv_yw_to_w(hidden_dim: Any, trace: dict[str, Any], *, alpha: float = 1.0,
                     has_bias: bool = False, b_fn: WeightFn | None = None,
                     a_fn: WeightFn | None = None, bias_fn: WeightFn | None = None) -> dict[str, Any]:
    r"""Unbatched LoRA ``yw_to_w`` — identical algebra with no batch axis.

    Trace shapes (recurrence context, after the ``n_state``-vmap):
        ``trace['lora_b'] : (in, out)``    — :math:`\boldsymbol{\epsilon}_W`, scaled by :math:`g`
        ``trace['lora_a'] : (rank, out)``  — scaled by :math:`g`
        ``trace['bias']   : (out,)``       — elementwise :math:`g`

    ``jnp.expand_dims(hidden_dim, axis=-2)`` turns ``(out,) → (1, out)``
    so it broadcasts against both the ``(in, out)`` effective-weight
    trace and the ``(rank, out)`` :math:`A` trace. See
    :func:`_lora_mm_yw_to_w` for the algebra.
    """
    g = jnp.expand_dims(hidden_dim, axis=-2)
    out = {'lora_b': trace['lora_b'] * g, 'lora_a': trace['lora_a'] * g}
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _lora_xy_to_dw(x: Any, hidden_dim: Any, weights: dict[str, Any], *, alpha: float = 1.0,
                   has_bias: bool = False, b_fn: WeightFn | None = None,
                   a_fn: WeightFn | None = None, bias_fn: WeightFn | None = None) -> dict[str, Any]:
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

    def _fwd(w: dict[str, Any]) -> Any:
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


def _lora_instant_drtrl(x: Any, hidden_dim: Any, weights: dict[str, Any], *,
                        alpha: float = 1.0, has_bias: bool = False,
                        b_fn: WeightFn | None = None, a_fn: WeightFn | None = None,
                        bias_fn: WeightFn | None = None) -> dict[str, Any]:
    r"""Trace-structured instantaneous term for param-dim D-RTRL (mm and mv).

    **Role.** Supplies the :math:`\operatorname{diag}(\mathbf{D}_f^t)
    \otimes \mathbf{x}^t` term added to :math:`\mathbf{D}^t
    \boldsymbol{\epsilon}^{t-1}` each step, in the *trace* structure
    rather than the parameter structure:

    * ``'lora_b'`` holds the effective-weight increment. Since
      :math:`y = x\, W_{\text{eff}}` with :math:`W_{\text{eff}} =
      \alpha\, b\_fn(B)\, a\_fn(A)`, the exact dense instantaneous term is
      the outer product

      .. math::

          \frac{\partial h}{\partial W_{\text{eff}, io}}
            = x_i\, g_o, \qquad g = \partial h / \partial y,

      with **no** :math:`\alpha` / :math:`b\_fn` / :math:`a\_fn` factor —
      those live inside :math:`W_{\text{eff}}` and are chained back to the
      raw :math:`B` only at solve time (:func:`_lora_solve_drtrl`).

    * ``'lora_a'`` and ``'bias'`` reuse the exact param-shaped
      :func:`_lora_xy_to_dw` pullbacks (which thread ``a_fn'`` /
      ``bias_fn'`` through the fused VJP); their per-factor traces are
      already exact under the dense-style ``yw_to_w`` recurrence. The
      VJP's unused ``'lora_b'`` pullback is dead code eliminated under
      ``jit``.

    **Shapes.** The algorithm vmaps over the batch axis (mm) and the
    trailing ``num_state`` axis, so this rule always sees batch-free
    slices: ``x : (in,)``, ``hidden_dim : (out,)``, returning
    ``{'lora_b': (in, out), 'lora_a': (rank, out)[, 'bias': (out,)]}``.
    """
    out = dict(_lora_xy_to_dw(
        x, hidden_dim, weights,
        alpha=alpha, has_bias=has_bias, b_fn=b_fn, a_fn=a_fn, bias_fn=bias_fn,
    ))
    x_v = u.get_mantissa(x)
    g_v = u.get_mantissa(hidden_dim)
    out['lora_b'] = jnp.expand_dims(x_v, axis=-1) * jnp.expand_dims(g_v, axis=-2)
    return out


def _lora_solve_drtrl(dg_hidden: Any, trace: dict[str, Any], weights: dict[str, Any], *,
                      alpha: float = 1.0, has_bias: bool = False,
                      b_fn: WeightFn | None = None, a_fn: WeightFn | None = None,
                      bias_fn: WeightFn | None = None) -> dict[str, Any]:
    r"""Solve-time weight gradients from the LoRA D-RTRL traces (mm and mv).

    **Role.** Contracts the learning signal :math:`g = \partial \mathcal{L} /
    \partial h` with each eligibility trace and returns **param-shaped**
    gradients keyed by trainable name:

    * ``'lora_b'`` — chain the effective-weight trace back to the raw
      factor. With :math:`G_{io} = g_o\, \boldsymbol{\epsilon}_{W, io}`
      (per slice) and :math:`W_{\text{eff}} = \alpha\, b\_fn(B)\,
      a\_fn(A)`:

      .. math::

          \nabla_{b\_fn(B)} = \alpha\, G\, a\_fn(A)^\top, \qquad
          \nabla_B = \operatorname{VJP}_{b\_fn}\big(\nabla_{b\_fn(B)}\big).

      The chain is linear in :math:`G`, so the algorithm's per-state /
      per-batch application followed by summation is exact for
      parameters held fixed over the gradient window.

    * ``'lora_a'`` / ``'bias'`` — the same contraction the generic
      ``yw_to_w`` solve path produced (it was exact): broadcast-multiply
      the signal along the output axis. ``a_fn'`` / ``bias_fn'`` are
      **not** re-applied here — they already entered the traces through
      :func:`_lora_instant_drtrl`.

    **Shapes.** The algorithm vmaps over the batch axis (mm) and the
    trailing ``num_state`` axis, so this rule sees batch-free,
    state-free slices: ``dg_hidden : (out,)`` (possibly with leading
    broadcast axes from a batched hidden state feeding the unbatched
    mv primitive — summed away below, which is exact by linearity),
    ``trace['lora_b'] : (in, out)``, ``trace['lora_a'] : (rank, out)``,
    ``trace['bias'] : (out,)``. Returns ``{'lora_b': (in, rank),
    'lora_a': (rank, out)[, 'bias': (out,)]}`` — ``'lora_a'`` /
    ``'bias'`` may keep leading broadcast axes, which the algorithm's
    trailing reduction collapses exactly as it did for the generic path.
    """
    g = jnp.expand_dims(u.get_mantissa(dg_hidden), axis=-2)

    # Effective-weight contraction G[i, o] = g[o] * eps_W[i, o]; collapse any
    # leading broadcast axes so the b_fn VJP sees a (in, out)-shaped cotangent
    # source (exact by linearity of the chain in G).
    G = trace['lora_b'] * g
    if G.ndim > 2:
        G = jnp.sum(G, axis=tuple(range(G.ndim - 2)))

    B = jnp.asarray(u.get_mantissa(weights['lora_b']))
    A = jnp.asarray(u.get_mantissa(weights['lora_a']))
    A_eff = jnp.asarray(a_fn(A)) if a_fn is not None else A
    dB_eff = alpha * (G @ A_eff.T)
    if b_fn is not None:
        _, vjp_b = jax.vjp(b_fn, B)
        dB = vjp_b(dB_eff)[0]
    else:
        dB = dB_eff

    out = {'lora_b': dB, 'lora_a': trace['lora_a'] * g}
    if has_bias:
        out['bias'] = trace['bias'] * u.get_mantissa(dg_hidden)
    return out


def _lora_mm_init_drtrl(x_var: Any, y_var: Any, weight_vars: dict[str, Any],
                        num_hidden_state: int) -> dict[str, Any]:
    r"""Initialise batched LoRA D-RTRL trace.

    The ``'lora_b'`` leaf holds the dense-style **effective-weight**
    trace :math:`\boldsymbol{\epsilon}_W` for :math:`W_{\text{eff}} =
    \alpha\, b\_fn(B)\, a\_fn(A)` (see the module docstring — no
    :math:`B`-shaped trace can be exact); ``'lora_a'`` / ``'bias'`` keep
    their factor shapes:

    .. math::

        \boldsymbol{\epsilon}_W \in \mathbb{R}^{B \times I \times O \times n_{\text{state}}}, \quad
        \boldsymbol{\epsilon}_A \in \mathbb{R}^{B \times r \times O \times n_{\text{state}}}, \quad
        \boldsymbol{\epsilon}_b \in \mathbb{R}^{B \times O \times n_{\text{state}}}.

    The effective-weight trace costs :math:`\mathcal{O}(B\, I\, O)` —
    dense-trace memory is the price of exact ``lora_b`` gradients under
    param-dim D-RTRL (the low-rank parameter count is unchanged).
    Zero-initialised.

    The trace dtype is derived from the participating ``x``/``y``/weight
    avals via :func:`jax.numpy.result_type` rather than left to ``jnp.zeros``'
    default (which silently follows the global x64 flag instead of the
    operands' actual dtype).
    """
    batch = x_var.aval.shape[0]
    B_shape = weight_vars['lora_b'].aval.shape
    A_shape = weight_vars['lora_a'].aval.shape
    dtype = jnp.result_type(
        x_var.aval.dtype, y_var.aval.dtype,
        *(v.aval.dtype for v in weight_vars.values()),
    )
    out = {
        'lora_b': jnp.zeros(
            (batch, B_shape[0], A_shape[-1], num_hidden_state), dtype=dtype
        ),
        'lora_a': jnp.zeros((batch, *A_shape, num_hidden_state), dtype=dtype),
    }
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros(
            (batch, *weight_vars['bias'].aval.shape, num_hidden_state), dtype=dtype
        )
    return out


def _lora_mm_init_pp(x_var: Any, y_var: Any, weight_vars: dict[str, Any],
                     num_hidden_state: int) -> Any:
    r"""Initialise batched LoRA pp-prop / ES-D-RTRL df trace.

    .. math::

        \boldsymbol{\epsilon}_f \in \mathbb{R}^{B \times O \times n_{\text{state}}}.

    Same shape as dense — pp-prop factorisation does not care how
    :math:`W = \alpha B A` is stored. The :math:`\boldsymbol{\epsilon}_x`
    factor is the raw :math:`x`; the :math:`B, A, b` split is handled by
    :func:`_lora_xy_to_dw` at solve-time.
    """
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


def _lora_mv_init_drtrl(x_var: Any, y_var: Any, weight_vars: dict[str, Any],
                        num_hidden_state: int) -> dict[str, Any]:
    r"""Initialise unbatched LoRA D-RTRL trace.

    The ``'lora_b'`` leaf holds the effective-weight trace
    :math:`\boldsymbol{\epsilon}_W` (see :func:`_lora_mm_init_drtrl`);
    no batch axis anywhere:

    .. math::

        \boldsymbol{\epsilon}_W \in \mathbb{R}^{I \times O \times n_{\text{state}}}, \quad
        \boldsymbol{\epsilon}_A \in \mathbb{R}^{r \times O \times n_{\text{state}}}, \quad
        \boldsymbol{\epsilon}_b \in \mathbb{R}^{O \times n_{\text{state}}}.

    Zero-initialised.

    The trace dtype is derived from the participating ``x``/``y``/weight
    avals via :func:`jax.numpy.result_type` rather than left to ``jnp.zeros``'
    default (which silently follows the global x64 flag instead of the
    operands' actual dtype).
    """
    B_shape = weight_vars['lora_b'].aval.shape
    A_shape = weight_vars['lora_a'].aval.shape
    dtype = jnp.result_type(
        x_var.aval.dtype, y_var.aval.dtype,
        *(v.aval.dtype for v in weight_vars.values()),
    )
    out = {
        'lora_b': jnp.zeros(
            (B_shape[0], A_shape[-1], num_hidden_state), dtype=dtype
        ),
        'lora_a': jnp.zeros((*A_shape, num_hidden_state), dtype=dtype),
    }
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros(
            (*weight_vars['bias'].aval.shape, num_hidden_state), dtype=dtype
        )
    return out


def _lora_mv_init_pp(x_var: Any, y_var: Any, weight_vars: dict[str, Any],
                     num_hidden_state: int) -> Any:
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
# Param-dim D-RTRL overrides: the trace structure ('lora_b' holds the
# effective-weight trace) differs from the parameter structure, so the
# instantaneous term and the solve-time contraction cannot be expressed by
# xy_to_dw / yw_to_w alone. IO-dim (ES-D-RTRL) keeps using xy_to_dw.
ETP_RULES_INSTANT_DRTRL[etp_lora_mm_p] = _lora_instant_drtrl
ETP_RULES_SOLVE_DRTRL[etp_lora_mm_p] = _lora_solve_drtrl

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
ETP_RULES_INSTANT_DRTRL[etp_lora_mv_p] = _lora_instant_drtrl
ETP_RULES_SOLVE_DRTRL[etp_lora_mv_p] = _lora_solve_drtrl


def lora_matmul(
    x: ArrayLike,
    B: ArrayLike,
    A: ArrayLike,
    *,
    alpha: float = 1.0,
    bias: ArrayLike | None = None,
    b_fn: WeightFn | None = None,
    a_fn: WeightFn | None = None,
    bias_fn: WeightFn | None = None,
) -> ArrayLike:
    r"""ETP-aware LoRA (Low-Rank Adaptation) matrix multiplication.

    Computes :math:`y = \alpha \cdot x \mathbin{@} b\_fn(B) \mathbin{@} a\_fn(A) \; (+ bias\_fn(b))`,
    routing both low-rank factors (and the optional bias) through an ETP
    primitive so they participate in eligibility-trace computation.
    Auto-dispatches batched/unbatched based on ``x.ndim``.

    Parameters
    ----------
    x : ArrayLike
        Input array, shape ``(batch, in_features)`` or ``(in_features,)``.
        Higher-rank ``x`` (``x.ndim > 2``) is rejected with a ``ValueError``:
        every ETP trace rule assumes one of these two layouts.
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
        Pass a module-level function, not a fresh ``lambda``, if this is
        called repeatedly: the hook is stored as a static ``eqn.params``
        entry hashed by object identity, so two textually identical
        ``lambda`` objects are cache misses and silently retrace every call.
    a_fn : callable or None, optional
        Elementwise transform applied to the ``A`` factor before the
        matrix multiplication.  ``a_fn(A)`` must return an array of the
        same shape as ``A``.  ``None`` means identity.
        The transform operates on the unitless mantissa; physical units are
        split off before and recombined after.
        Same re-tracing caveat as ``b_fn``: pass a module-level function
        rather than a fresh ``lambda``.
    bias_fn : callable or None, optional
        Elementwise transform applied to ``bias`` before adding.
        ``None`` means identity.
        The transform operates on the unitless mantissa; physical units are
        split off before and recombined after.
        Same re-tracing caveat as ``b_fn``: pass a module-level function
        rather than a fresh ``lambda``.

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
        >>> x = brainstate.random.randn(16, 8)
        >>> B = brainstate.random.randn(8, 2)
        >>> A = brainstate.random.randn(2, 4)
        >>> y = braintrace.lora_matmul(x, B, A, alpha=0.5)
        >>> print(y.shape)
        (16, 4)
    """
    if x.ndim > 2:  # type: ignore[union-attr]
        raise ValueError(
            f'lora_matmul() supports x.ndim of 1 (unbatched `(in_features,)`) or 2 '
            f'(batched `(batch, in_features)`); got x.ndim={x.ndim} '
            f'(shape={x.shape}). Every ETP trace rule for etp_lora_mm_p / '
            f'etp_lora_mv_p assumes one of those two layouts, so higher-rank '
            f'inputs (e.g. `(batch, time, in_features)`) are not supported -- '
            f'reshape/vmap over the extra axes before calling lora_matmul().'
        )
    p = etp_lora_mm_p if x.ndim >= 2 else etp_lora_mv_p  # type: ignore[union-attr]  # x is an array here; ArrayLike also admits scalars without .ndim
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
