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

r"""Element-wise ETP primitive — identity marker for diagonal weight ops.

``etp_elemwise_p`` is the only ``gradient_enabled=True`` primitive: the
compiler *evaluates* it when walking ``y -> h`` because its value flows
identity-like into the downstream consumer. The supplied ``fn`` is
applied to the weight by the user-facing wrapper *before* the primitive
binds, so the primitive itself is the identity.

**Forward operation**

.. math::

    y = \mathrm{fn}(w),

evaluated in Python by the :func:`element_wise` wrapper *before* the
primitive binds. The primitive body is the identity map on :math:`y`.
All non-linearity and broadcasting in ``fn`` are therefore opaque to
the primitive — what the ETP rules see is simply :math:`y` flowing
through, and :math:`\partial y / \partial w` has already been absorbed
into the upstream jaxpr by JAX's standard VJP machinery (because
``gradient_enabled=True`` tells the compiler to *descend into* this
primitive when composing Jacobians).

**Role of each ETP rule** (with :math:`y = w` in the primitive's own view)

* ``xy_to_dw(hidden_dim)`` — returns :math:`\partial h / \partial w` which
  for the identity is just the cotangent itself:

  .. math::

      \frac{\partial h}{\partial y} = \frac{\partial h}{\partial w},

  so ``xy_to_dw`` returns ``hidden_dim`` unchanged. This is the
  instantaneous :math:`\operatorname{diag}(\mathbf{D}_f^t)\otimes 1` for
  D-RTRL (no :math:`x` factor since the op has no separate input).

* ``yw_to_w(hidden_dim, trace)`` — elementwise product
  :math:`\epsilon^t = (\partial h/\partial y) \odot \epsilon^{t-1}`.
  The primitive is *diagonal*, so broadcasting is trivial: both trace
  and ``hidden_dim`` share the same shape.

* ``init_drtrl`` — weight-shaped trace :math:`\boldsymbol{\epsilon} \in
  \mathbb{R}^{\dots \times n_{\text{state}}}` (same leading shape as
  :math:`y`).

* ``init_pp`` — pp-prop df trace, same shape as ``init_drtrl`` since for
  an identity op the weight-shape coincides with the output-shape.

**Transform hooks**

The primitive accepts a single optional elementwise transform hook,
``weight_fn``, in its ``eqn.params`` (computing ``y = weight_fn(w)``; there
is no ``x`` input and no bias for this op). The eligibility trace and
gradient are taken w.r.t. the **raw** weight, so the transform Jacobian
:math:`f'` enters *only* through :func:`_elemwise_xy_to_dw` via
:func:`jax.vjp`; the ``yw_to_w`` rule does **not** apply :math:`f'`.

**Fast path**

A closed-form param-dim D-RTRL kernel bundle (:class:`FastPathRules`,
registered on ``etp_elemwise_p``) replaces the generic nested-``vmap`` trace
path with diagonal einsums. Because those kernels return the bare ``df``
(dropping :math:`f'`), the bundle's ``applicable`` gate disables the fast
path whenever ``weight_fn`` is present.
"""

import jax
import jax.numpy as jnp
import brainunit as u

from ._primitive import register_primitive
from ._registries import FastPathRules

__all__ = [
    'etp_elemwise_p',
    'element_wise',
]


def _etp_elemwise_impl(w, weight_fn=None):
    return w if weight_fn is None else weight_fn(w)


def _elem_trainable_invars(params):
    return {'weight': 0}


def _elemwise_yw_to_w(hidden_dim, trace, *, weight_fn=None):
    r"""Diagonal trace propagation for an identity-like op.

    **Role in D-RTRL.** Realises the :math:`y \to w` chain factor inside
    :math:`\mathbf{D}^t \boldsymbol{\epsilon}^{t-1}` for an identity op.
    Because :math:`y = w` has :math:`\partial y / \partial w = I`, the
    propagation reduces to an elementwise multiply:

    .. math::

        \epsilon^t_{\text{pre}} = (\partial h / \partial y) \odot \epsilon^{t-1}.

    This is invoked per hidden-state slice; both operands share shape
    ``(*y_shape,)``.

    Args:
        hidden_dim: Cotangent :math:`\partial h / \partial y`, shape matches weight.
        trace: ``{'weight': ...}``, array of the same shape.

    Returns:
        ``{'weight': hidden_dim ⊙ trace['weight']}``.
    """
    return {'weight': trace['weight'] * hidden_dim}


def _elemwise_xy_to_dw(x, hidden_dim, weights, *, weight_fn=None):
    r"""Instantaneous Jacobian for the identity marker.

    **Role in D-RTRL / ES-D-RTRL.** For :math:`y = \text{weight\_fn}(w)`,

    .. math::

        \frac{\partial h}{\partial w} = \frac{\partial h}{\partial y} \cdot \text{weight\_fn}'(w),

    computed via VJP. When ``weight_fn`` is ``None`` (identity), the
    contribution is simply the hidden cotangent itself.

    Because the op is diagonal, ``weight_fn`` is element-wise and its Jacobian is
    ``diag(weight_fn'(w))``. We therefore extract the per-element derivative once
    (cotangent of ones, shape ``(*w_shape,)``) and broadcast-multiply it against
    ``hidden_dim``. This is identical to ``vjp_fn(hidden_dim)`` in the unbatched
    case (``vjp_fn(c) = c ⊙ weight_fn'(w)`` for a diagonal Jacobian) but, unlike a
    direct ``vjp_fn(hidden_dim)``, it accepts a ``hidden_dim`` carrying extra
    leading axes (e.g. a batch axis under :class:`brainstate.mixin.Batching`),
    which the unbatched VJP would reject as a shape mismatch.

    Args:
        x: Unused (``x_invar_index=None``).
        hidden_dim: Cotangent :math:`\partial h / \partial y`. Shape matches the
            weight, optionally with extra leading axes (e.g. a batch axis).
        weights: Dict with key 'weight' (the raw weight mantissa).
        weight_fn: Element-wise function applied inside the primitive. When
            ``None``, behaves as identity.

    Returns:
        ``{'weight': hidden_dim ⊙ weight_fn'(w)}``, or
        ``{'weight': hidden_dim}`` when ``weight_fn is None``.
    """
    # ∂h/∂w = (∂h/∂y) · weight_fn'(w). For the identity (weight_fn None) this is
    # just the cotangent itself.
    if weight_fn is None:
        return {'weight': hidden_dim}
    w = weights['weight']
    out, vjp_fn = jax.vjp(weight_fn, w)
    # Diagonal Jacobian: vjp(ones) yields the per-element derivative weight_fn'(w),
    # shaped like w. Broadcast it against hidden_dim's (possibly batched) leading axes.
    deriv = u.get_mantissa(vjp_fn(jnp.ones_like(out))[0])
    return {'weight': hidden_dim * deriv}


def _elemwise_init_drtrl(x_var, y_var, weight_vars, num_hidden_state, group=None):
    r"""Initialise the D-RTRL weight-shaped trace for an identity op.

    Weight-shape and output-shape coincide for the identity, so

    .. math::

        \boldsymbol{\epsilon}_w \in \mathbb{R}^{\text{(y-shape)} \times n_{\text{state}}}.

    Zero-initialised (matches :math:`\boldsymbol{\epsilon}^0 = \mathbf{0}`).

    Unlike matmul, the elemwise output *is* the weight, so ``y_var`` carries no
    batch axis. The trace, however, lives in the hidden-state (group) position
    space and acquires whatever leading axes the hidden state has — in
    particular a batch axis under :class:`brainstate.mixin.Batching`. The trace
    leading shape is therefore taken from ``group.varshape`` (which equals the
    y-shape in the unbatched / per-lane ``vmap`` case, and prepends the batch
    under ``Batching()``). This keeps the trace shape consistent with the
    instantaneous term ``df`` (shape ``(*group.varshape, n_state)``) so the scan
    carry stays well-typed.

    Args:
        x_var: Unused (``x_invar_index=None``).
        y_var: Output variable descriptor with shape.
        weight_vars: Dict with key 'weight' (unused in body).
        num_hidden_state: Number of hidden states :math:`n_{\text{state}}`.
        group: The owning :class:`HiddenGroup`. When supplied, its (possibly
            batched) ``varshape`` is used for the trace leading axes; falls back
            to ``y_var`` shape when ``None`` (e.g. direct unit tests).

    Returns:
        ``{'weight': zeros(*leading_shape, n_state)}``.
    """
    leading = tuple(group.varshape) if group is not None else tuple(y_var.aval.shape)
    return {'weight': jnp.zeros((*leading, num_hidden_state))}


def _elemwise_init_pp(x_var, y_var, weight_vars, num_hidden_state, group=None):
    r"""Initialise the pp-prop df trace for an identity op.

    .. math::

        \boldsymbol{\epsilon}_f \in \mathbb{R}^{\text{(group.varshape)} \times n_{\text{state}}}.

    Same shape as ``init_drtrl`` because the op is diagonal. In
    ES-D-RTRL there is no separate :math:`\boldsymbol{\epsilon}_x`
    factor (``x_invar_index=None``); the weight gradient is assembled
    directly from :math:`\boldsymbol{\epsilon}_f`. As for ``init_drtrl`` the
    leading axes come from ``group.varshape`` so the trace carries the batch
    axis under :class:`brainstate.mixin.Batching`.

    Returns:
        Single array of shape ``(*leading_shape, n_state)``.
    """
    leading = tuple(group.varshape) if group is not None else tuple(y_var.aval.shape)
    return jnp.zeros((*leading, num_hidden_state), dtype=y_var.aval.dtype)


# ---------------------------------------------------------------------------
# Closed-form param-dim D-RTRL fast-path kernels (diagonal, no x, no bias)
# ---------------------------------------------------------------------------

def _elemwise_fast_instant(x, df, has_bias):
    r"""Instantaneous term for the diagonal identity op.

    Parameters
    ----------
    x : Any
        Unused (the elemwise op has no ``x`` input). Accepted for a uniform
        kernel signature.
    df : ArrayLike
        State-to-output Jacobian :math:`\mathbf{D}_f^t`, shape
        ``(..., num_state)``.
    has_bias : bool
        Unused (the elemwise op has no bias).

    Returns
    -------
    dict
        ``{'weight': df}``.

    Notes
    -----
    With no input factor and an identity ``y = w``, the instantaneous
    Jacobian :math:`\operatorname{diag}(\mathbf{D}_f^t) \otimes 1` reduces
    to ``df`` itself.
    """
    return {'weight': df}


def _elemwise_fast_recurrent(diag, old_bwg, num_state):
    r"""Recurrent term :math:`\mathbf{D}^t \boldsymbol{\epsilon}^{t-1}` (diagonal).

    Parameters
    ----------
    diag : ArrayLike
        Hidden-to-hidden Jacobian, shape ``(..., num_state, num_state)``.
    old_bwg : dict
        Previous trace dict; ``'weight'`` shape ``(..., num_state)``.
    num_state : int
        Number of hidden states per group.

    Returns
    -------
    dict
        ``{'weight': D^t · ε^{t-1}}``.

    Notes
    -----
    The contraction is ``einsum('...ab,...b->...a')`` over the ``num_state``
    axis. When ``num_state == 1`` the sum collapses to a broadcast multiply
    by ``diag[..., 0, :]`` (the size-1 ``beta`` axis is kept to align with
    the trace) — bit-identical to the einsum.
    """
    if num_state == 1:
        return {'weight': diag[..., 0, :] * old_bwg['weight']}
    return {'weight': jnp.einsum('...ab,...b->...a', diag, old_bwg['weight'])}


def _elemwise_fast_solve(diag_like, etrace_data, *, fold_batch=False):
    r"""Solve-time contraction of the learning signal with the trace (diagonal).

    Parameters
    ----------
    diag_like : ArrayLike
        The :math:`\partial \mathcal{L}/\partial \mathbf{h}` group gradient,
        shape ``(..., num_state)`` (``(B, ..., num_state)`` when batched).
    etrace_data : dict
        Trace dict; ``'weight'`` shape matches ``diag_like``.
    fold_batch : bool, optional
        When ``True``, contract the leading batch axis ``b`` inside the
        einsum so the result is already batch-summed. Default ``False``.

    Returns
    -------
    dict
        ``{'weight': dW}``.

    Notes
    -----
    Contracts every shared axis to a scalar-per-weight via
    ``'...a,...a->...'`` (and ``'b...a,b...a->...'`` under ``fold_batch``).
    """
    spec = 'b...a,b...a->...' if fold_batch else '...a,...a->...'
    return {'weight': jnp.einsum(spec, diag_like, etrace_data['weight'])}


def _elemwise_fast_applicable(eqn_params):
    r"""Gate: is the elemwise fast path valid for this equation?

    Parameters
    ----------
    eqn_params : dict
        The ETP equation's ``params`` dict.

    Returns
    -------
    bool
        ``True`` iff ``weight_fn`` is absent / ``None``.

    Notes
    -----
    The closed-form kernels return the bare ``df`` (dropping the ``f'(w)``
    transform factor), so any active ``weight_fn`` must fall back to the rule
    path (which applies ``f'`` via :func:`jax.vjp`). The op has no
    ``bias_fn``.
    """
    return eqn_params.get('weight_fn') is None


etp_elemwise_p = register_primitive(
    'etp_elemwise',
    _etp_elemwise_impl,
    batched=False,
    gradient_enabled=True,
    trainable_invars_fn=_elem_trainable_invars,
    x_invar_index=None,
)
etp_elemwise_p.register_etp_rules(
    yw_to_w=_elemwise_yw_to_w,
    xy_to_dw=_elemwise_xy_to_dw,
    init_drtrl=_elemwise_init_drtrl,
    init_pp=_elemwise_init_pp,
    fast_path=FastPathRules(
        _elemwise_fast_instant,
        _elemwise_fast_recurrent,
        _elemwise_fast_solve,
        _elemwise_fast_applicable,
    ),
)


def element_wise(weight, *, weight_fn=None):
    r"""ETP-aware element-wise operation.

    Applies ``weight_fn`` to ``weight`` *inside* the ETP primitive, so
    the transform is visible to the eligibility-trace compiler. The
    operation is treated as *diagonal* in the hidden-state space; the
    weight participates in trace computation as a per-element trainable
    parameter.

    Parameters
    ----------
    weight : ArrayLike
        Weight parameter (may be a :class:`brainunit.Quantity`).
    weight_fn : Callable or None, optional
        Element-wise function applied to the raw weight mantissa inside
        the primitive. When ``None`` (default), the identity is used and
        the output equals ``weight`` exactly.
        The transform operates on the unitless mantissa; physical units are
        split off before and recombined after.

    Returns
    -------
    ArrayLike
        ``weight_fn(weight)`` (or ``weight`` when ``weight_fn`` is
        ``None``), with the same shape as ``weight``.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>>
        >>> brainstate.environ.set(precision=64)
        >>> w = brainstate.random.randn(5)
        >>> y = braintrace.element_wise(w)
        >>> print(y.shape)
        (5,)
        >>>
        >>> # Apply a non-linearity to the weight
        >>> import jax.numpy as jnp
        >>> y1 = braintrace.element_wise(w, weight_fn=jnp.tanh)
        >>> print(y1.shape)
        (5,)
    """
    w_v, w_u = u.split_mantissa_unit(weight)
    r = etp_elemwise_p.bind(w_v, weight_fn=weight_fn)
    return u.maybe_decimal(r * w_u)
