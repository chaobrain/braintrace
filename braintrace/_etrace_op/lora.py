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

**Dict rule API (N-trainable-input refactor)**

Both primitives declare ``trainable_invars_fn``, which returns
``{'lora_b': 1, 'lora_a': 2}`` when ``has_bias=False`` and
``{'lora_b': 1, 'lora_a': 2, 'bias': 3}`` when ``has_bias=True``.
Keys ``'lora_b'`` / ``'lora_a'`` match the pytree leaf names in
``braintrace.nn.LoRALinear``'s merged ``ParamState``.
"""

import jax
import jax.numpy as jnp
import saiunit as u

from ._spec import ETPPrimitiveSpec, register_primitive_spec

__all__ = [
    'etp_lora_mm_p',
    'etp_lora_mv_p',
    'lora_matmul',
]


def _etp_lora_impl(*args, alpha=1.0, has_bias=False):
    x, B, A = args[0], args[1], args[2]
    y = alpha * (x @ B @ A)
    if has_bias:
        y = y + args[3]
    return y


def _lora_trainable_invars(params):
    """Return ``{key: invar_index}`` for LoRA's trainable inputs."""
    base = {'lora_b': 1, 'lora_a': 2}
    if params.get('has_bias', False):
        base['bias'] = 3
    return base


def _lora_mm_yw_to_w(hidden_dim, trace, *, alpha=1.0, has_bias=False):
    r"""LoRA batched trace propagation.

    ``hidden_dim`` is called in two contexts by the D-RTRL executor:
    (a) trace update: shape ``(..., out)``, e.g. ``(batch, out)`` after ns-vmap;
    (b) gradient solve: shape ``(out,)`` after the additional batch-vmap.

    In both cases we want to broadcast ``hidden_dim`` across the ``rank``
    axis of ``trace['lora_a']`` whose shape is ``(..., rank, out)``.
    ``jnp.expand_dims(hidden_dim, axis=-2)`` inserts a singleton at axis
    -2 so it aligns correctly:
        (out,)        → (1, out)   broadcasts with (rank, out)      ✓
        (batch, out)  → (batch, 1, out) broadcasts with (batch, rank, out) ✓

    Only propagate through A (B frozen during trace propagation).
    """
    trace_A = trace['lora_a'] * jnp.expand_dims(hidden_dim, axis=-2)
    out = {'lora_b': trace['lora_b'], 'lora_a': trace_A}
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _lora_mv_yw_to_w(hidden_dim, trace, *, alpha=1.0, has_bias=False):
    r"""LoRA unbatched: trace is ``{'lora_b': (in, rank, ns), 'lora_a': (rank, out, ns)}``."""
    trace_A = trace['lora_a'] * jnp.expand_dims(hidden_dim, axis=0)
    out = {'lora_b': trace['lora_b'], 'lora_a': trace_A}
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _lora_xy_to_dw(x, hidden_dim, weights, *, alpha=1.0, has_bias=False):
    r"""VJP of ``y = alpha * x @ B @ A (+ bias)``, returning a dict keyed
    by ``'lora_b'``, ``'lora_a'``, and optionally ``'bias'``."""

    def _fwd(w):
        y = alpha * (x @ w['lora_b'] @ w['lora_a'])
        if has_bias:
            y = y + w['bias']
        return u.get_mantissa(y)

    _, vjp_fn = jax.vjp(_fwd, weights)
    return jax.tree.map(u.get_mantissa, vjp_fn(hidden_dim)[0])


def _lora_mm_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    """Return zero-filled ``Dict[str, Array]`` for D-RTRL (batched LoRA)."""
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
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


def _lora_mv_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    """Return zero-filled ``Dict[str, Array]`` for D-RTRL (unbatched LoRA)."""
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
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_lora_mm_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_lora_mm',
        impl=_etp_lora_impl,
        yw_to_w=_lora_mm_yw_to_w,
        xy_to_dw=_lora_xy_to_dw,
        init_drtrl=_lora_mm_init_drtrl,
        init_pp=_lora_mm_init_pp,
        trainable_invars_fn=_lora_trainable_invars,
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
        trainable_invars_fn=_lora_trainable_invars,
        x_invar_index=0,
        batched=False,
    )
)


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
