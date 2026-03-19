# Copyright 2024-2025 BrainX Ecosystem Limited. All Rights Reserved.
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
ETP (Eligibility Trace Propagation) primitives and rule registries.

This module defines JAX custom primitives that mark weight operations in
the computational graph. The compiler identifies these primitives by type
(``eqn.primitive in ETP_PRIMITIVES``), replacing the old approach of
matching JIT function names.

Primitives
----------
Dense matmul:
    ``etp_mm_p``  — batched:  y = x @ w (+ b),  x is (batch, in)
    ``etp_mv_p``  — unbatched: y = x @ w (+ b),  x is (in,)

Element-wise:
    ``etp_elemwise_p`` — identity marker for diagonal weight ops

Convolution:
    ``etp_conv_p`` — y = conv(x, kernel) (+ b), always expects batch dim

Sparse matmul:
    ``etp_sp_mm_p`` / ``etp_sp_mv_p``

LoRA:
    ``etp_lora_mm_p`` / ``etp_lora_mv_p``

Rule Registries
---------------
``etp_rules_yw_to_w``   : trace propagation (D-RTRL)
``etp_rules_xy_to_dw``  : weight gradient (D-RTRL + ES-D-RTRL)
``etp_rules_init_drtrl``  : D-RTRL trace initialization (parameter-dim)
``etp_rules_init_pp``     : pp_prop df trace initialization (IO-dim)

User API
--------
``matmul(x, w, bias)``       — auto-dispatches mm/mv based on x.ndim
``element_wise(w, fn)``      — element-wise marker
``conv(x, kernel, bias, ..)`` — convolution
``sparse_matmul(x, w, ..)``  — sparse matmul
``lora_matmul(x, B, A, ..)`` — LoRA matmul
"""

from functools import partial
from typing import Callable, Dict, Optional, Sequence, Any

import jax
import jax.numpy as jnp
import saiunit as u
from jax.core import ShapedArray
from jax.interpreters import mlir, batching, ad

from braintrace._compatible_imports import Primitive

__all__ = [
    # primitives
    'etp_mm_p',
    'etp_mv_p',
    'etp_elemwise_p',
    'etp_conv_p',
    'etp_sp_mm_p',
    'etp_sp_mv_p',
    'etp_lora_mm_p',
    'etp_lora_mv_p',

    # rule registries
    'etp_rules_yw_to_w',
    'etp_rules_xy_to_dw',
    'etp_rules_init_drtrl',
    'etp_rules_init_pp',

    # helpers
    'register_primitive',
    'is_etp_primitive',
    'is_batched_primitive',
    'ETP_PRIMITIVES',
    'BATCHED_PRIMITIVES',

    # user API
    'matmul',
    'element_wise',
    'conv',
    'sparse_matmul',
    'lora_matmul',
]

# ======================================================================
# Primitive & rule registries
# ======================================================================

ETP_PRIMITIVES: set = set()

etp_rules_yw_to_w: Dict[Primitive, Callable] = {}
r"""D-RTRL trace propagation: ``(hidden_dim, trace, **params) → trace``."""

etp_rules_xy_to_dw: Dict[Primitive, Callable] = {}
r"""Weight gradient: ``(x, hidden_dim, w, **params) → dw``."""

etp_rules_init_drtrl: Dict[Primitive, Callable] = {}
r"""D-RTRL trace initialization: ``(x_var, y_var, weight, num_hidden_state) → zeros``.

Returns parameter-dimension trace state. Shape depends on the weight
(e.g. ``(batch, *w_shape, ns)`` for batched matmul).
"""

etp_rules_init_pp: Dict[Primitive, Callable] = {}
r"""pp_prop (IO-dim) df trace initialization: ``(x_var, y_var, weight, num_hidden_state) → zeros``.

Returns IO-dimension df trace state. Shape is typically
``(*y_shape, num_hidden_state)``.
"""


def is_etp_primitive(primitive):
    """Check whether a JAX primitive is an ETP primitive."""
    return primitive in ETP_PRIMITIVES


BATCHED_PRIMITIVES: set = set()  # populated as primitives are registered


def is_batched_primitive(primitive):
    """Check whether an ETP primitive operates on batched data."""
    return primitive in BATCHED_PRIMITIVES


# ======================================================================
# Primitive registration helper
# ======================================================================

def register_primitive(name, impl_fn, *, batched=False):
    """Create an ETP primitive with all JAX rules auto-derived from *impl_fn*.

    Registered automatically:

    - **impl** — eager execution
    - **abstract_eval** — via ``jax.eval_shape(impl)``
    - **lowering** — via ``mlir.lower_fun(impl)``
    - **JVP** — via ``jax.jvp(impl)``
    - **transpose** — derived by JAX from the JVP
    - **batching** — via ``jax.vmap(impl)``

    Only ETP-specific rules (``yw_to_w``, ``xy_to_dw``, ``init_drtrl``,
    ``init_pp``) need hand-writing.

    Args:
        name: Primitive name (e.g., ``'etp_mm'``).
        impl_fn: Implementation function.
        batched: Whether this primitive operates on batched data.

    Returns:
        The registered ``Primitive``.
    """
    p = Primitive(name)
    ETP_PRIMITIVES.add(p)
    if batched:
        BATCHED_PRIMITIVES.add(p)

    # impl
    p.def_impl(impl_fn)

    # abstract_eval
    @p.def_abstract_eval
    def _abstract(*args, **params):
        shapes = tuple(ShapedArray(a.shape, a.dtype) for a in args)
        out = jax.eval_shape(partial(impl_fn, **params), *shapes)
        return ShapedArray(out.shape, out.dtype)

    # lowering
    mlir.register_lowering(
        p, mlir.lower_fun(impl_fn, multiple_results=False),
    )

    # JVP
    def _jvp(primals, tangents, **params):
        tans = tuple(
            jnp.zeros(pr.shape, pr.dtype) if isinstance(t, ad.Zero) else t
            for pr, t in zip(primals, tangents)
        )
        return jax.jvp(partial(impl_fn, **params), primals, tans)

    ad.primitive_jvps[p] = _jvp

    # batching
    def _batching(args, dims, **params):
        return jax.vmap(partial(impl_fn, **params), in_axes=dims)(*args), 0

    batching.primitive_batchers[p] = _batching

    return p


# ======================================================================
# etp_mm_p / etp_mv_p  —  y = x @ w (+ b)
# ======================================================================

def _etp_matmul_impl(*args, has_bias=False):
    x, w = args[0], args[1]
    y = x @ w
    if has_bias:
        y = y + args[2]
    return y


# --- mm (batched) ---


def _mm_yw_to_w(hidden_dim, trace, *, has_bias=False):
    r"""Batched: hidden_dim (batch, out), trace (batch, in, out, n_state)."""
    return trace * jnp.expand_dims(hidden_dim, axis=1)


def _mm_xy_to_dw(x, hidden_dim, w, *, has_bias=False):
    r"""VJP of ``y = x @ w``, per-sample via vmap."""
    _, vjp_fn = jax.vjp(lambda w_: x @ w_, w)
    return vjp_fn(hidden_dim)[0]


def _mm_init_drtrl(x_var, y_var, weight, num_hidden_state):
    batch = x_var.aval.shape[0]
    w_shape = jnp.shape(weight.value)
    return jnp.zeros((batch, *w_shape, num_hidden_state))


def _mm_init_pp(x_var, y_var, weight, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_mm_p = register_primitive('etp_mm', _etp_matmul_impl, batched=True)
etp_rules_yw_to_w[etp_mm_p] = _mm_yw_to_w
etp_rules_xy_to_dw[etp_mm_p] = _mm_xy_to_dw
etp_rules_init_drtrl[etp_mm_p] = _mm_init_drtrl
etp_rules_init_pp[etp_mm_p] = _mm_init_pp


# --- mv (unbatched) ---


def _mv_yw_to_w(hidden_dim, trace, *, has_bias=False):
    r"""Unbatched: hidden_dim (out,), trace (in, out, n_state)."""
    return trace * jnp.expand_dims(hidden_dim, axis=0)


def _mv_xy_to_dw(x, hidden_dim, w, *, has_bias=False):
    r"""VJP of ``y = x @ w``."""
    _, vjp_fn = jax.vjp(lambda w_: x @ w_, w)
    return vjp_fn(hidden_dim)[0]


def _mv_init_drtrl(x_var, y_var, weight, num_hidden_state):
    w_shape = jnp.shape(weight.value)
    return jnp.zeros((*w_shape, num_hidden_state))


def _mv_init_pp(x_var, y_var, weight, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_mv_p = register_primitive('etp_mv', _etp_matmul_impl, batched=False)
etp_rules_yw_to_w[etp_mv_p] = _mv_yw_to_w
etp_rules_xy_to_dw[etp_mv_p] = _mv_xy_to_dw
etp_rules_init_drtrl[etp_mv_p] = _mv_init_drtrl
etp_rules_init_pp[etp_mv_p] = _mv_init_pp


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


def _elemwise_init_drtrl(x_var, y_var, weight, num_hidden_state):
    y_shape = y_var.aval.shape
    return jnp.zeros((*y_shape, num_hidden_state))


def _elemwise_init_pp(x_var, y_var, weight, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_elemwise_p = register_primitive('etp_elemwise', _etp_elemwise_impl)
etp_rules_yw_to_w[etp_elemwise_p] = _elemwise_yw_to_w
etp_rules_xy_to_dw[etp_elemwise_p] = _elemwise_xy_to_dw
etp_rules_init_drtrl[etp_elemwise_p] = _elemwise_init_drtrl
etp_rules_init_pp[etp_elemwise_p] = _elemwise_init_pp


# ======================================================================
# etp_conv_p  —  y = conv(x, kernel) (+ b)
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
    x, kernel = args[0], args[1]
    y = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=kernel,
        window_strides=strides,
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
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


def _conv_init_drtrl(x_var, y_var, weight, num_hidden_state):
    batch = x_var.aval.shape[0]
    kernel_shape = jnp.shape(weight.value)
    return jnp.zeros((batch, *kernel_shape, num_hidden_state))


def _conv_init_pp(x_var, y_var, weight, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_conv_p = register_primitive('etp_conv', _etp_conv_impl, batched=True)
etp_rules_yw_to_w[etp_conv_p] = _conv_yw_to_w
etp_rules_xy_to_dw[etp_conv_p] = _conv_xy_to_dw
etp_rules_init_drtrl[etp_conv_p] = _conv_init_drtrl
etp_rules_init_pp[etp_conv_p] = _conv_init_pp


# ======================================================================
# etp_sp_mm_p / etp_sp_mv_p  —  sparse matmul
# ======================================================================

def _etp_sp_matmul_impl(*args, sparse_mat=None, has_bias=False):
    x, weight_data = args[0], args[1]
    w = sparse_mat.with_data(weight_data)
    y = x @ w
    if has_bias:
        y = y + args[2]
    return y


def _sp_mm_yw_to_w(hidden_dim, trace, *, sparse_mat=None, has_bias=False):
    return sparse_mat.yw_to_w_transposed(hidden_dim, trace)


def _sp_mv_yw_to_w(hidden_dim, trace, *, sparse_mat=None, has_bias=False):
    return sparse_mat.yw_to_w_transposed(hidden_dim, trace)


def _sp_xy_to_dw(x, hidden_dim, w, *, sparse_mat=None, has_bias=False):
    _, vjp_fn = jax.vjp(lambda w_: x @ sparse_mat.with_data(w_), w)
    return vjp_fn(hidden_dim)[0]


def _sp_mm_init_drtrl(x_var, y_var, weight, num_hidden_state):
    batch = x_var.aval.shape[0]
    nnz = jnp.shape(weight.value)[0]
    return jnp.zeros((batch, nnz, num_hidden_state))


def _sp_mm_init_pp(x_var, y_var, weight, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


def _sp_mv_init_drtrl(x_var, y_var, weight, num_hidden_state):
    nnz = jnp.shape(weight.value)[0]
    return jnp.zeros((nnz, num_hidden_state))


def _sp_mv_init_pp(x_var, y_var, weight, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_sp_mm_p = register_primitive('etp_sp_mm', _etp_sp_matmul_impl, batched=True)
etp_rules_yw_to_w[etp_sp_mm_p] = _sp_mm_yw_to_w
etp_rules_xy_to_dw[etp_sp_mm_p] = _sp_xy_to_dw
etp_rules_init_drtrl[etp_sp_mm_p] = _sp_mm_init_drtrl
etp_rules_init_pp[etp_sp_mm_p] = _sp_mm_init_pp

etp_sp_mv_p = register_primitive('etp_sp_mv', _etp_sp_matmul_impl, batched=False)
etp_rules_yw_to_w[etp_sp_mv_p] = _sp_mv_yw_to_w
etp_rules_xy_to_dw[etp_sp_mv_p] = _sp_xy_to_dw
etp_rules_init_drtrl[etp_sp_mv_p] = _sp_mv_init_drtrl
etp_rules_init_pp[etp_sp_mv_p] = _sp_mv_init_pp


# ======================================================================
# etp_lora_mm_p / etp_lora_mv_p  —  y = alpha * x @ B @ A (+ b)
# ======================================================================

def _etp_lora_impl(*args, alpha=1.0, has_bias=False):
    x, B, A = args[0], args[1], args[2]
    y = alpha * (x @ B @ A)
    if has_bias:
        y = y + args[3]
    return y


def _lora_mm_yw_to_w(hidden_dim, trace, *, alpha=1.0, has_bias=False):
    r"""LoRA batched: trace is ``{B: (batch, in, rank, ns), A: (batch, rank, out, ns)}``.
    Only propagate through A (B frozen during trace propagation)."""
    trace_A = trace['A']
    trace_A = trace_A * jnp.expand_dims(hidden_dim, axis=1)
    return {'B': trace['B'], 'A': trace_A}


def _lora_mv_yw_to_w(hidden_dim, trace, *, alpha=1.0, has_bias=False):
    r"""LoRA unbatched: trace is ``{B: (in, rank, ns), A: (rank, out, ns)}``."""
    trace_A = trace['A']
    trace_A = trace_A * jnp.expand_dims(hidden_dim, axis=0)
    return {'B': trace['B'], 'A': trace_A}


def _lora_xy_to_dw(x, hidden_dim, w_B, w_A, *, alpha=1.0, has_bias=False):
    r"""VJP of ``y = alpha * x @ B @ A``."""

    def _fwd(B, A):
        return alpha * (x @ B @ A)

    _, vjp_fn = jax.vjp(_fwd, w_B, w_A)
    dB, dA = vjp_fn(hidden_dim)
    return {'B': dB, 'A': dA}


def _lora_mm_init_drtrl(x_var, y_var, weight, num_hidden_state):
    # weight.value is a dict {'B': ..., 'A': ...}
    w = weight.value
    batch = x_var.aval.shape[0]
    return {
        'B': jnp.zeros((batch, *jnp.shape(w['B']), num_hidden_state)),
        'A': jnp.zeros((batch, *jnp.shape(w['A']), num_hidden_state)),
    }


def _lora_mm_init_pp(x_var, y_var, weight, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


def _lora_mv_init_drtrl(x_var, y_var, weight, num_hidden_state):
    w = weight.value
    return {
        'B': jnp.zeros((*jnp.shape(w['B']), num_hidden_state)),
        'A': jnp.zeros((*jnp.shape(w['A']), num_hidden_state)),
    }


def _lora_mv_init_pp(x_var, y_var, weight, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_lora_mm_p = register_primitive('etp_lora_mm', _etp_lora_impl, batched=True)
etp_rules_yw_to_w[etp_lora_mm_p] = _lora_mm_yw_to_w
etp_rules_xy_to_dw[etp_lora_mm_p] = _lora_xy_to_dw
etp_rules_init_drtrl[etp_lora_mm_p] = _lora_mm_init_drtrl
etp_rules_init_pp[etp_lora_mm_p] = _lora_mm_init_pp

etp_lora_mv_p = register_primitive('etp_lora_mv', _etp_lora_impl, batched=False)
etp_rules_yw_to_w[etp_lora_mv_p] = _lora_mv_yw_to_w
etp_rules_xy_to_dw[etp_lora_mv_p] = _lora_xy_to_dw
etp_rules_init_drtrl[etp_lora_mv_p] = _lora_mv_init_drtrl
etp_rules_init_pp[etp_lora_mv_p] = _lora_mv_init_pp


# ======================================================================
# User-facing functions
# ======================================================================

def matmul(x, weight, bias=None):
    r"""ETP-aware matrix multiplication.

    Computes :math:`y = x \mathbin{@} w \; (+ b)`.

    Auto-dispatches to ``etp_mm_p`` (batched) or ``etp_mv_p`` (unbatched)
    based on ``x.ndim``.

    Args:
        x: Input array, shape ``(..., in_features)`` or ``(in_features,)``.
        weight: Weight matrix, shape ``(in_features, out_features)``.
        bias: Optional bias vector, shape ``(out_features,)``.

    Returns:
        Output array.
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


def element_wise(weight, fn=lambda w: w):
    r"""ETP-aware element-wise operation.

    Applies ``fn`` to ``weight`` and passes through a marker primitive.
    The operation is treated as *diagonal* in the hidden-state space.

    Args:
        weight: Weight parameter.
        fn: Element-wise function. Defaults to identity.

    Returns:
        ``fn(weight)``.
    """
    y = fn(weight)
    y_v, y_u = u.split_mantissa_unit(y)
    r = etp_elemwise_p.bind(y_v)
    return u.maybe_decimal(r * y_u)


def conv(
    x,
    kernel,
    bias=None,
    *,
    strides: Sequence[int] = (1,),
    padding: str = 'SAME',
    lhs_dilation: Optional[Sequence[int]] = None,
    rhs_dilation: Optional[Sequence[int]] = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    dimension_numbers: Any = None,
):
    r"""ETP-aware convolution.

    Computes :math:`y = \mathrm{conv}(x, kernel) \; (+ b)`.
    Always expects a batch dimension on ``x``.

    Args:
        x: Input tensor with batch dimension.
        kernel: Convolution kernel.
        bias: Optional bias.
        strides: Window strides.
        padding: Padding mode.
        lhs_dilation: Left-hand-side dilation.
        rhs_dilation: Right-hand-side dilation.
        feature_group_count: Feature group count.
        batch_group_count: Batch group count.
        dimension_numbers: Convolution dimension numbers.

    Returns:
        Convolution output.
    """
    conv_kwargs = dict(
        strides=tuple(strides),
        padding=padding,
        lhs_dilation=tuple(lhs_dilation) if lhs_dilation is not None else None,
        rhs_dilation=tuple(rhs_dilation) if rhs_dilation is not None else None,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        dimension_numbers=dimension_numbers,
    )
    x_v, x_u = u.split_mantissa_unit(x)
    kernel_v, kernel_u = u.split_mantissa_unit(kernel)
    unit = x_u * kernel_u
    if bias is not None:
        bias_v = u.Quantity(bias).to_decimal(unit)
        r = etp_conv_p.bind(x_v, kernel_v, bias_v, has_bias=True, **conv_kwargs)
    else:
        r = etp_conv_p.bind(x_v, kernel_v, has_bias=False, **conv_kwargs)
    return u.maybe_decimal(r * x_u * kernel_u)


def sparse_matmul(x, weight_data, *, sparse_mat, bias=None):
    r"""ETP-aware sparse matrix multiplication.

    Computes :math:`y = x \mathbin{@} \mathrm{sparse}(w) \; (+ b)`.

    Auto-dispatches batched/unbatched based on ``x.ndim``.

    Args:
        x: Input array.
        weight_data: Sparse matrix data (non-zero values).
        sparse_mat: The sparse matrix structure (``brainunit.sparse.SparseMatrix``).
        bias: Optional bias.

    Returns:
        Output array.
    """
    p = etp_sp_mm_p if x.ndim >= 2 else etp_sp_mv_p
    x_v, x_u = u.split_mantissa_unit(x)
    w_v, w_u = u.split_mantissa_unit(weight_data)
    unit = x_u * w_u
    if bias is not None:
        bias_v = u.Quantity(bias).to_decimal(unit)
        r = p.bind(x_v, w_v, bias_v, sparse_mat=sparse_mat, has_bias=True)
    else:
        r = p.bind(x_v, w_v, sparse_mat=sparse_mat, has_bias=False)
    return u.maybe_decimal(r * x_u * w_u)


def lora_matmul(x, B, A, *, alpha=1.0, bias=None):
    r"""ETP-aware LoRA (Low-Rank Adaptation) matrix multiplication.

    Computes :math:`y = \alpha \cdot x \mathbin{@} B \mathbin{@} A \; (+ b)`.

    Auto-dispatches batched/unbatched based on ``x.ndim``.

    Args:
        x: Input array.
        B: Low-rank matrix B, shape ``(in_features, rank)``.
        A: Low-rank matrix A, shape ``(rank, out_features)``.
        alpha: Scaling factor.
        bias: Optional bias.

    Returns:
        Output array.
    """
    p = etp_lora_mm_p if x.ndim >= 2 else etp_lora_mv_p
    x_v, x_u = u.split_mantissa_unit(x)
    B_v, B_u = u.split_mantissa_unit(B)
    A_v, A_u = u.split_mantissa_unit(A)
    unit = x_u * B_u * A_u
    if bias is not None:
        bias_v = u.Quantity(bias).to_decimal(unit)
        r = p.bind(x_v, B_v, A_v, bias_v, alpha=alpha, has_bias=True)
    else:
        r = p.bind(x_v, B_v, A_v, alpha=alpha, has_bias=False)
    return u.maybe_decimal(r * x_u * B_u * A_u)
