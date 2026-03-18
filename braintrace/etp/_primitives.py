# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

r"""
JAX primitives for Eligibility Trace Propagation (ETP).

Design decisions
----------------
We keep thin **custom primitives** (``etp_matmul_p``, …) rather than
registering rules on existing JAX primitives (``lax.dot_general_p``,
…) because we need to *identify* which operations in a Jaxpr involve
traced parameters.  A ``lax.dot_general_p`` may appear dozens of times;
only the ones the user marked with ``etp_matmul()`` should participate
in trace propagation.

**JVP / transpose / batching** rules are **not** hand-written.  Each
primitive's ``impl`` delegates to standard JAX ops (``@``, ``lax.conv_general_dilated``, …),
so we let JAX differentiate *through* the impl:

* **JVP**: ``jax.jvp(impl, primals, tangents)`` — 3 lines per primitive.
* **Transpose (VJP)**: auto-derived by JAX from the JVP rule via linearization.  No explicit registration needed.
* **Batching**: ``jax.vmap(impl, in_axes=dims)`` — 3 lines per primitive.

This avoids duplicating the derivative formulas that JAX already knows.

ETP-specific rule registries (sparse-transform pattern)
-------------------------------------------------------
``etp_rules_yw_to_w``  :  ``Primitive → (hidden_dim, trace, **p) → trace_like``
``etp_rules_xy_to_dw`` :  ``Primitive → (x, hidden_dim, w, **p) → dw``

These are the *only* rules we write by hand because they encode
operator-specific structure exploited by D-RTRL and ES-D-RTRL
(e.g. ``trace * hidden_dim[None, :]`` for matmul).
"""

from functools import partial
from typing import Callable, Dict

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.interpreters import mlir, batching, ad

from braintrace._compatible_imports import Primitive

__all__ = [
    # primitives
    'etp_matmul_p',
    'etp_elemwise_p',
    'etp_conv_p',
    # registry helpers
    'register_primitive',
    'ETP_PRIMITIVES',
    'is_etp_primitive',
    # rule registries
    'etp_rules_yw_to_w',
    'etp_rules_xy_to_dw',
]

# ======================================================================
# Primitive & rule registries
# ======================================================================

ETP_PRIMITIVES: set = set()

etp_rules_yw_to_w: Dict[Primitive, Callable] = {}
r"""D-RTRL trace propagation rule.  ``(hidden_dim, trace, **p) → trace_like``."""

etp_rules_xy_to_dw: Dict[Primitive, Callable] = {}
r"""Weight gradient rule (D-RTRL + ES-D-RTRL).  ``(x, hidden_dim, w, **p) → dw``."""


def is_etp_primitive(primitive):
    """Check whether a JAX primitive is an ETP primitive."""
    return primitive in ETP_PRIMITIVES


# ======================================================================
# Helpers: auto-derive JVP, batching from impl
# ======================================================================

def register_primitive(name, impl_fn):
    """Create an ETP primitive with all JAX rules auto-derived from *impl_fn*.

    Registered automatically:

    * **impl** — eager execution.
    * **abstract_eval** — via ``jax.eval_shape(impl)``.
    * **lowering** — via ``mlir.lower_fun(impl)``.
    * **JVP** — via ``jax.jvp(impl)``.
    * **transpose** — derived by JAX from the JVP (no registration needed).
    * **batching** — via ``jax.vmap(impl)``.

    The only rules you write by hand are the ETP-specific
    ``yw_to_w`` / ``xy_to_dw`` (registered in the global dicts).
    """
    p = Primitive(name)
    ETP_PRIMITIVES.add(p)

    # impl
    p.def_impl(impl_fn)

    # abstract_eval — auto-derived from impl via jax.eval_shape
    @p.def_abstract_eval
    def _abstract(*args, **params):
        shapes = tuple(ShapedArray(a.shape, a.dtype) for a in args)
        out = jax.eval_shape(partial(impl_fn, **params), *shapes)
        return ShapedArray(out.shape, out.dtype)

    # lowering
    mlir.register_lowering(
        p, mlir.lower_fun(impl_fn, multiple_results=False),
    )

    # JVP — auto-derived from impl
    def _jvp(primals, tangents, **params):
        tans = tuple(
            jnp.zeros(pr.shape, pr.dtype) if isinstance(t, ad.Zero) else t
            for pr, t in zip(primals, tangents)
        )
        return jax.jvp(partial(impl_fn, **params), primals, tans)

    ad.primitive_jvps[p] = _jvp

    # batching — auto-derived from impl
    def _batching(args, dims, **params):
        return jax.vmap(partial(impl_fn, **params), in_axes=dims)(*args), 0

    batching.primitive_batchers[p] = _batching

    return p


# ======================================================================
# etp_matmul_p  —  y = x @ w (+ b)
# ======================================================================

def _etp_matmul_impl(*args, has_bias=False):
    x, w = args[0], args[1]
    y = x @ w
    if has_bias:
        y = y + args[2]
    return y


def _matmul_yw_to_w(hidden_dim, trace, *, has_bias=False):
    r"""``trace[i, j] *= hidden_dim[j]``  (broadcast along axis 0)."""
    return trace * jnp.expand_dims(hidden_dim, axis=0)


def _matmul_xy_to_dw(x, hidden_dim, w, *, has_bias=False):
    r"""VJP of ``y = x @ w`` w.r.t. ``w``, cotangent = ``hidden_dim``."""
    _, vjp_fn = jax.vjp(lambda w_: x @ w_, w)
    return vjp_fn(hidden_dim)[0]


etp_matmul_p = register_primitive('etp_matmul', _etp_matmul_impl)
etp_rules_yw_to_w[etp_matmul_p] = _matmul_yw_to_w
etp_rules_xy_to_dw[etp_matmul_p] = _matmul_xy_to_dw


# ======================================================================
# etp_elemwise_p  —  identity marker for diagonal weight ops
# ======================================================================

def _etp_elemwise_impl(y):
    return y


def _elemwise_yw_to_w(hidden_dim, trace):
    r"""Element-wise multiply."""
    return trace * hidden_dim


def _elemwise_xy_to_dw(x, hidden_dim, w):
    r"""Identity marker — gradient is just ``hidden_dim``.
    Chain rule through ``fn`` is handled by JAX on the ops
    before the primitive."""
    return hidden_dim


etp_elemwise_p = register_primitive('etp_elemwise', _etp_elemwise_impl)
etp_rules_yw_to_w[etp_elemwise_p] = _elemwise_yw_to_w
etp_rules_xy_to_dw[etp_elemwise_p] = _elemwise_xy_to_dw


# ======================================================================
# etp_conv_p  —  y = conv(x, w) (+ b)
# ======================================================================

def _etp_conv_impl(
    *args,
    has_bias=False,
    strides=(1,),
    padding='SAME',
    lhs_dilation=None,
    rhs_dilation=None,
    feature_group_count=1,
    batch_group_count=1,
    dimension_numbers=None,
):
    x, w = args[0], args[1]
    y = jax.lax.conv_general_dilated(
        lhs=x, rhs=w,
        window_strides=strides, padding=padding,
        lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        dimension_numbers=dimension_numbers,
    )
    if has_bias:
        y = y + args[2]
    return y


def _conv_yw_to_w(hidden_dim, trace, **params):
    r"""Broadcast ``hidden_dim`` across all dims except output channel."""
    n_expand = trace.ndim - hidden_dim.ndim
    for _ in range(n_expand):
        hidden_dim = jnp.expand_dims(hidden_dim, axis=0)
    return trace * hidden_dim


def _conv_xy_to_dw(x, hidden_dim, w, **params):
    r"""VJP of ``y = conv(x, w)`` w.r.t. ``w``."""
    conv_kw = {k: v for k, v in params.items() if k != 'has_bias'}

    def _fwd(w_):
        return jax.lax.conv_general_dilated(x, w_, **conv_kw)

    _, vjp_fn = jax.vjp(_fwd, w)
    return vjp_fn(hidden_dim)[0]


etp_conv_p = register_primitive('etp_conv', _etp_conv_impl)
etp_rules_yw_to_w[etp_conv_p] = _conv_yw_to_w
etp_rules_xy_to_dw[etp_conv_p] = _conv_xy_to_dw
