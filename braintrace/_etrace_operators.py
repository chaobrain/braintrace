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
``ETP_RULES_YW_TO_W``   : trace propagation (D-RTRL)
``ETP_RULES_XY_TO_DW``  : weight gradient (D-RTRL + ES-D-RTRL)
``ETP_RULES_INIT_DRTRL``  : D-RTRL trace initialization (parameter-dim)
``ETP_RULES_INIT_PP``     : pp_prop df trace initialization (IO-dim)

User API
--------
``matmul(x, w, bias)``       — auto-dispatches mm/mv based on x.ndim
``element_wise(w, fn)``      — element-wise marker
``conv(x, kernel, bias, ..)`` — convolution
``sparse_matmul(x, w, ..)``  — sparse matmul
``lora_matmul(x, B, A, ..)`` — LoRA matmul
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Optional, Sequence, Any

import jax
import jax.numpy as jnp
import saiunit as u
from jax.core import ShapedArray
from jax.interpreters import mlir, batching, ad

from braintrace._compatible_imports import Primitive

__all__ = [
    # ETP primitive class & registration
    'ETPPrimitive',
    'ETPPrimitiveSpec',
    'register_primitive',
    'register_primitive_spec',
    'get_primitive_spec',

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

ETP_RULES_YW_TO_W: Dict[Primitive, Callable] = {}
r"""D-RTRL trace propagation: ``(hidden_dim, trace, **params) → trace``."""

ETP_RULES_XY_TO_DW: Dict[Primitive, Callable] = {}
r"""Weight gradient: ``(x, hidden_dim, w, **params) → dw``."""

ETP_RULES_INIT_DRTRL: Dict[Primitive, Callable] = {}
r"""D-RTRL trace initialization: ``(x_var, y_var, weight, num_hidden_state) → zeros``.

Returns parameter-dimension trace state. Shape depends on the weight
(e.g. ``(batch, *w_shape, ns)`` for batched matmul).
"""

ETP_RULES_INIT_PP: Dict[Primitive, Callable] = {}
r"""pp_prop (IO-dim) df trace initialization: ``(x_var, y_var, weight, num_hidden_state) → zeros``.

Returns IO-dimension df trace state. Shape is typically
``(*y_shape, num_hidden_state)``.
"""


def is_etp_primitive(primitive):
    """Check whether a JAX primitive is an ETP primitive."""
    return primitive in ETP_PRIMITIVES


GRADIENT_ENABLED_PRIMITIVES: set = set()  # populated as primitives are registered


def is_etp_enable_gradient_primitive(primitive):
    """Check whether an ETP primitive should be evaluated (rather than skipped)
    when encountered inside a ``pjit`` equation during Jaxpr traversal.

    Primitives that mark identity-like operations (e.g. ``etp_elemwise_p``) must
    be evaluated so that downstream consumers see the correct value flow. Other
    ETP primitives are structural markers whose value is supplied separately.
    """
    return primitive in GRADIENT_ENABLED_PRIMITIVES


BATCHED_PRIMITIVES: set = set()  # populated as primitives are registered


def is_batched_primitive(primitive):
    """Check whether an ETP primitive operates on batched data."""
    return primitive in BATCHED_PRIMITIVES


# ======================================================================
# ETPPrimitive wrapper & registration helper
# ======================================================================

class ETPPrimitive(Primitive):
    """A JAX ``Primitive`` subclass with built-in ETP rule registration methods.

    Returned by :func:`register_primitive`.  Supports all standard JAX
    primitive operations (``bind``, ``def_impl``, etc.) and adds
    convenience methods for registering ETP-specific rules.

    Example::

        my_p = register_primitive('etp_my_op', _my_impl, batched=True)
        my_p.register_yw_to_w(my_yw_to_w_fn)
        my_p.register_xy_to_dw(my_xy_to_dw_fn)
        my_p.register_init_drtrl(my_init_drtrl_fn)
        my_p.register_init_pp(my_init_pp_fn)
    """

    def register_yw_to_w(self, fn: Callable):
        """Register a D-RTRL trace propagation rule.

        Signature: ``(hidden_dim, trace, **params) -> trace``.
        """
        ETP_RULES_YW_TO_W[self] = fn

    def register_xy_to_dw(self, fn: Callable):
        """Register a weight-gradient rule.

        Signature: ``(x, hidden_dim, w, **params) -> dw``.
        """
        ETP_RULES_XY_TO_DW[self] = fn

    def register_init_drtrl(self, fn: Callable):
        """Register a D-RTRL trace initialization rule.

        Signature: ``(x_var, y_var, weight_var, num_hidden_state) -> zeros``.
        """
        ETP_RULES_INIT_DRTRL[self] = fn

    def register_init_pp(self, fn: Callable):
        """Register a pp_prop (IO-dim) df trace initialization rule.

        Signature: ``(x_var, y_var, weight_var, num_hidden_state) -> zeros``.
        """
        ETP_RULES_INIT_PP[self] = fn

    def register_etp_rules(
        self,
        *,
        yw_to_w: Callable = None,
        xy_to_dw: Callable = None,
        init_drtrl: Callable = None,
        init_pp: Callable = None,
    ):
        """Register multiple ETP rules in one call.

        Only non-``None`` arguments are registered.
        """
        if yw_to_w is not None:
            ETP_RULES_YW_TO_W[self] = yw_to_w
        if xy_to_dw is not None:
            ETP_RULES_XY_TO_DW[self] = xy_to_dw
        if init_drtrl is not None:
            ETP_RULES_INIT_DRTRL[self] = init_drtrl
        if init_pp is not None:
            ETP_RULES_INIT_PP[self] = init_pp


def register_primitive(name, impl_fn, *, batched=False, gradient_enabled=False):
    """Create an ETP primitive with all JAX rules auto-derived from *impl_fn*.

    Registered automatically:

    - **impl** — eager execution
    - **abstract_eval** — via ``jax.eval_shape(impl)``
    - **lowering** — via ``mlir.lower_fun(impl)``
    - **JVP** — via ``jax.jvp(impl)``
    - **transpose** — derived by JAX from the JVP
    - **batching** — via ``jax.vmap(impl)``

    Only ETP-specific rules (``yw_to_w``, ``xy_to_dw``, ``init_drtrl``,
    ``init_pp``) need hand-writing — use the returned
    :class:`ETPPrimitive`'s ``register_*`` methods.

    Args:
        name: Primitive name (e.g., ``'etp_mm'``).
        impl_fn: Implementation function.
        batched: Whether this primitive operates on batched data.
        gradient_enabled: If ``True``, this primitive is evaluated (rather
            than skipped) when the compiler encounters it inside a ``pjit``
            equation. Identity-like primitives such as ``etp_elemwise_p``
            should set this flag so their value flows to downstream consumers.

    Returns:
        :class:`ETPPrimitive`: The registered primitive.
    """
    p = ETPPrimitive(name)
    ETP_PRIMITIVES.add(p)
    if batched:
        BATCHED_PRIMITIVES.add(p)
    if gradient_enabled:
        GRADIENT_ENABLED_PRIMITIVES.add(p)

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
# ETPPrimitiveSpec — declarative, plug-in-style primitive registration
# ======================================================================

ETP_PRIMITIVE_SPECS: Dict[Primitive, 'ETPPrimitiveSpec'] = {}


@dataclass(frozen=True)
class ETPPrimitiveSpec:
    """Declarative specification of an ETP primitive.

    A single record carrying every datum the compiler and runtime need to
    identify the primitive, locate its weight and input invars, and dispatch
    the four ETP rules. Register via :func:`register_primitive_spec` to
    create the primitive and populate every registry in one call.

    Attributes:
        name: Primitive name (e.g. ``'etp_mm'``).
        impl: Implementation function. All standard JAX rules (abstract_eval,
            lowering, JVP, transpose, batching) are auto-derived from this.
        batched: Whether the primitive operates on batched inputs.
        gradient_enabled: If True, the compiler *may* traverse this primitive
            when walking ``y -> h`` (identity-like ops such as
            ``etp_elemwise_p``). If False (default for any trainable op),
            the primitive acts as a tail boundary — a preceding ETP weight
            whose only path to ``h`` passes through this primitive is
            excluded from ETP.
        weight_invar_index: Position in ``eqn.invars`` of the weight the
            compiler should trace back to a ``ParamState``. For ``mm/mv/conv``
            this is 1; for ``elemwise`` it is 0; for LoRA it is 1 (the B
            factor — the originating ``ParamState`` holds both B and A).
        x_invar_index: Position of the input ``x`` in ``eqn.invars``, or
            ``None`` for primitives that have no external input (currently
            only ``etp_elemwise_p``).
        yw_to_w: D-RTRL trace propagation rule.
        xy_to_dw: Weight-gradient rule.
        init_drtrl: D-RTRL parameter-dimension trace initialiser.
        init_pp: pp_prop IO-dimension df trace initialiser.
    """

    name: str
    impl: Callable
    yw_to_w: Callable
    xy_to_dw: Callable
    init_drtrl: Callable
    init_pp: Callable
    weight_invar_index: int
    x_invar_index: Optional[int] = 0
    y_outvar_index: int = 0
    batched: bool = False
    gradient_enabled: bool = False


def register_primitive_spec(spec: ETPPrimitiveSpec) -> 'ETPPrimitive':
    """Create an ETP primitive and install it + all four ETP rules from *spec*.

    Returns the created :class:`ETPPrimitive`. Also records ``spec`` in
    :data:`ETP_PRIMITIVE_SPECS` so the compiler can query the primitive's
    invar layout without hard-coding identity checks.
    """
    p = register_primitive(
        spec.name,
        spec.impl,
        batched=spec.batched,
        gradient_enabled=spec.gradient_enabled,
    )
    p.register_etp_rules(
        yw_to_w=spec.yw_to_w,
        xy_to_dw=spec.xy_to_dw,
        init_drtrl=spec.init_drtrl,
        init_pp=spec.init_pp,
    )
    ETP_PRIMITIVE_SPECS[p] = spec
    return p


def get_primitive_spec(primitive: Primitive) -> Optional[ETPPrimitiveSpec]:
    """Return the :class:`ETPPrimitiveSpec` for *primitive*, or ``None`` if
    the primitive was registered through the legacy ``register_primitive`` +
    ``register_*`` API without a spec."""
    return ETP_PRIMITIVE_SPECS.get(primitive)


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
    _, vjp_fn = jax.vjp(lambda w_: u.get_mantissa(x @ w_), w)
    return u.get_mantissa(vjp_fn(hidden_dim)[0])


def _mm_init_drtrl(x_var, y_var, weight_var, num_hidden_state):
    batch = x_var.aval.shape[0]
    w_shape = weight_var.aval.shape
    return jnp.zeros((batch, *w_shape, num_hidden_state))


def _mm_init_pp(x_var, y_var, weight_var, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_mm_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_mm',
        impl=_etp_matmul_impl,
        yw_to_w=_mm_yw_to_w,
        xy_to_dw=_mm_xy_to_dw,
        init_drtrl=_mm_init_drtrl,
        init_pp=_mm_init_pp,
        weight_invar_index=1,
        x_invar_index=0,
        batched=True,
    )
)


# --- mv (unbatched) ---


def _mv_yw_to_w(hidden_dim, trace, *, has_bias=False):
    r"""Unbatched: hidden_dim (out,), trace (in, out, n_state)."""
    return trace * jnp.expand_dims(hidden_dim, axis=0)


def _mv_xy_to_dw(x, hidden_dim, w, *, has_bias=False):
    r"""VJP of ``y = x @ w``."""
    _, vjp_fn = jax.vjp(lambda w_: u.get_mantissa(x @ w_), w)
    return u.get_mantissa(vjp_fn(hidden_dim)[0])


def _mv_init_drtrl(x_var, y_var, weight_var, num_hidden_state):
    w_shape = weight_var.aval.shape
    return jnp.zeros((*w_shape, num_hidden_state))


def _mv_init_pp(x_var, y_var, weight_var, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_mv_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_mv',
        impl=_etp_matmul_impl,
        yw_to_w=_mv_yw_to_w,
        xy_to_dw=_mv_xy_to_dw,
        init_drtrl=_mv_init_drtrl,
        init_pp=_mv_init_pp,
        weight_invar_index=1,
        x_invar_index=0,
        batched=False,
    )
)


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


def _elemwise_init_drtrl(x_var, y_var, weight_var, num_hidden_state):
    y_shape = y_var.aval.shape
    return jnp.zeros((*y_shape, num_hidden_state))


def _elemwise_init_pp(x_var, y_var, weight_var, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_elemwise_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_elemwise',
        impl=_etp_elemwise_impl,
        yw_to_w=_elemwise_yw_to_w,
        xy_to_dw=_elemwise_xy_to_dw,
        init_drtrl=_elemwise_init_drtrl,
        init_pp=_elemwise_init_pp,
        weight_invar_index=0,
        x_invar_index=None,
        batched=False,
        gradient_enabled=True,
    )
)


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
        return u.get_mantissa(jax.lax.conv_general_dilated(x, w_, **conv_kw))

    _, vjp_fn = jax.vjp(_fwd, w)
    return u.get_mantissa(vjp_fn(hidden_dim)[0])


def _conv_init_drtrl(x_var, y_var, weight_var, num_hidden_state):
    batch = x_var.aval.shape[0]
    kernel_shape = weight_var.aval.shape
    return jnp.zeros((batch, *kernel_shape, num_hidden_state))


def _conv_init_pp(x_var, y_var, weight_var, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_conv_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_conv',
        impl=_etp_conv_impl,
        yw_to_w=_conv_yw_to_w,
        xy_to_dw=_conv_xy_to_dw,
        init_drtrl=_conv_init_drtrl,
        init_pp=_conv_init_pp,
        weight_invar_index=1,
        x_invar_index=0,
        batched=True,
    )
)


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
    _, vjp_fn = jax.vjp(lambda w_: u.get_mantissa(x @ sparse_mat.with_data(w_)), w)
    return u.get_mantissa(vjp_fn(hidden_dim)[0])


def _sp_mm_init_drtrl(x_var, y_var, weight_var, num_hidden_state):
    batch = x_var.aval.shape[0]
    nnz = weight_var.aval.shape[0]
    return jnp.zeros((batch, nnz, num_hidden_state))


def _sp_mm_init_pp(x_var, y_var, weight_var, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


def _sp_mv_init_drtrl(x_var, y_var, weight_var, num_hidden_state):
    nnz = weight_var.aval.shape[0]
    return jnp.zeros((nnz, num_hidden_state))


def _sp_mv_init_pp(x_var, y_var, weight_var, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_sp_mm_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_sp_mm',
        impl=_etp_sp_matmul_impl,
        yw_to_w=_sp_mm_yw_to_w,
        xy_to_dw=_sp_xy_to_dw,
        init_drtrl=_sp_mm_init_drtrl,
        init_pp=_sp_mm_init_pp,
        weight_invar_index=1,
        x_invar_index=0,
        batched=True,
    )
)

etp_sp_mv_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_sp_mv',
        impl=_etp_sp_matmul_impl,
        yw_to_w=_sp_mv_yw_to_w,
        xy_to_dw=_sp_xy_to_dw,
        init_drtrl=_sp_mv_init_drtrl,
        init_pp=_sp_mv_init_pp,
        weight_invar_index=1,
        x_invar_index=0,
        batched=False,
    )
)


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
        return u.get_mantissa(alpha * (x @ B @ A))

    _, vjp_fn = jax.vjp(_fwd, w_B, w_A)
    dB, dA = vjp_fn(hidden_dim)
    return {'B': u.get_mantissa(dB), 'A': u.get_mantissa(dA)}


def _lora_mm_init_drtrl(x_var, y_var, weight_var, num_hidden_state):
    # weight_var is the B invar of the primitive; A's shape is (rank, out).
    batch = x_var.aval.shape[0]
    B_shape = weight_var.aval.shape
    rank = B_shape[1]
    out = y_var.aval.shape[-1]
    A_shape = (rank, out)
    return {
        'B': jnp.zeros((batch, *B_shape, num_hidden_state)),
        'A': jnp.zeros((batch, *A_shape, num_hidden_state)),
    }


def _lora_mm_init_pp(x_var, y_var, weight_var, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


def _lora_mv_init_drtrl(x_var, y_var, weight_var, num_hidden_state):
    B_shape = weight_var.aval.shape
    rank = B_shape[1]
    out = y_var.aval.shape[-1]
    A_shape = (rank, out)
    return {
        'B': jnp.zeros((*B_shape, num_hidden_state)),
        'A': jnp.zeros((*A_shape, num_hidden_state)),
    }


def _lora_mv_init_pp(x_var, y_var, weight_var, num_hidden_state):
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_lora_mm_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_lora_mm',
        impl=_etp_lora_impl,
        yw_to_w=_lora_mm_yw_to_w,
        xy_to_dw=_lora_xy_to_dw,
        init_drtrl=_lora_mm_init_drtrl,
        init_pp=_lora_mm_init_pp,
        weight_invar_index=1,
        x_invar_index=0,
        batched=True,
    )
)

etp_lora_mv_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_lora_mv',
        impl=_etp_lora_impl,
        yw_to_w=_lora_mv_yw_to_w,
        xy_to_dw=_lora_xy_to_dw,
        init_drtrl=_lora_mv_init_drtrl,
        init_pp=_lora_mv_init_pp,
        weight_invar_index=1,
        x_invar_index=0,
        batched=False,
    )
)


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
        sparse_mat: The sparse matrix structure (``saiunit.sparse.SparseMatrix``).
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
