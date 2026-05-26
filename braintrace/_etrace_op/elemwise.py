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
"""

import jax.numpy as jnp
import saiunit as u

from ._primitive import register_primitive

__all__ = [
    'etp_elemwise_p',
    'element_wise',
]


def _etp_elemwise_impl(y):
    return y


def _elem_trainable_invars(params):
    return {'weight': 0}


def _elemwise_yw_to_w(hidden_dim, trace):
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


def _elemwise_xy_to_dw(x, hidden_dim, weights):
    r"""Instantaneous Jacobian for the identity marker.

    **Role in D-RTRL / ES-D-RTRL.** For :math:`y = w` (identity),

    .. math::

        \frac{\partial h}{\partial w} = \frac{\partial h}{\partial y},

    so the instantaneous contribution is simply the hidden cotangent
    itself. In the D-RTRL update this feeds the
    :math:`\operatorname{diag}(\mathbf{D}_f^t)\otimes \mathbf{x}^t` term
    with :math:`\mathbf{x}^t \equiv 1` (no separate input). The chain
    rule through the *external* ``fn`` supplied to :func:`element_wise`
    is taken care of by JAX on the ops *before* this primitive binds
    (``gradient_enabled=True`` propagates standard VJPs through the
    primitive rather than masking them).

    Args:
        x: Unused (``x_invar_index=None``).
        hidden_dim: Cotangent :math:`\partial h / \partial y`, shape matches weight.
        weights: Dict with key 'weight' (unused in body; matches dict API).

    Returns:
        ``{'weight': hidden_dim}``.
    """
    return {'weight': hidden_dim}


def _elemwise_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    r"""Initialise the D-RTRL weight-shaped trace for an identity op.

    Weight-shape and output-shape coincide for the identity, so

    .. math::

        \boldsymbol{\epsilon}_w \in \mathbb{R}^{\text{(y-shape)} \times n_{\text{state}}}.

    Zero-initialised (matches :math:`\boldsymbol{\epsilon}^0 = \mathbf{0}`).

    Args:
        x_var: Unused (``x_invar_index=None``).
        y_var: Output variable descriptor with shape.
        weight_vars: Dict with key 'weight' (unused in body).
        num_hidden_state: Number of hidden states :math:`n_{\text{state}}`.

    Returns:
        ``{'weight': zeros(*y_shape, n_state)}``.
    """
    y_shape = y_var.aval.shape
    return {'weight': jnp.zeros((*y_shape, num_hidden_state))}


def _elemwise_init_pp(x_var, y_var, weight_vars, num_hidden_state):
    r"""Initialise the pp-prop df trace for an identity op.

    .. math::

        \boldsymbol{\epsilon}_f \in \mathbb{R}^{\text{(y-shape)} \times n_{\text{state}}}.

    Same shape as ``init_drtrl`` because the op is diagonal. In
    ES-D-RTRL there is no separate :math:`\boldsymbol{\epsilon}_x`
    factor (``x_invar_index=None``); the weight gradient is assembled
    directly from :math:`\boldsymbol{\epsilon}_f`.

    Returns:
        Single array of shape ``(*y_shape, n_state)``.
    """
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


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
)


def element_wise(weight, fn=lambda w: w):
    r"""ETP-aware element-wise operation.

    Applies ``fn`` to ``weight`` and passes the result through a marker
    primitive. The operation is treated as *diagonal* in the hidden-state
    space, so the weight participates in eligibility-trace computation as a
    per-element trainable parameter.

    Parameters
    ----------
    weight : ArrayLike
        Weight parameter.
    fn : Callable, optional
        Element-wise function applied to ``weight`` before the primitive
        binds. Default is the identity ``lambda w: w``.

    Returns
    -------
    ArrayLike
        ``fn(weight)``, with the same shape as ``weight``.

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
        >>> y1 = braintrace.element_wise(w, fn=jnp.tanh)
        >>> print(y1.shape)
        (5,)
    """
    y = fn(weight)
    y_v, y_u = u.split_mantissa_unit(y)
    r = etp_elemwise_p.bind(y_v)
    return u.maybe_decimal(r * y_u)
