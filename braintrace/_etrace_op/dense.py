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

**Dict rule API (N-trainable-input refactor)**

Both primitives now declare ``trainable_invars_fn``, which returns
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

from ._spec import ETPPrimitiveSpec, register_primitive_spec

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
    r"""Batched: hidden_dim (batch, out), trace Dict[str, Array].

    For the weight:  trace['weight'] (batch, in, out, n_state)
                     hidden_dim (batch, out) -> expand axis=1 -> (batch, 1, out, ...)
    For the bias:    trace['bias']   (batch, out, n_state)
                     hidden_dim (batch, out) -> multiply element-wise
    """
    out = {'weight': trace['weight'] * jnp.expand_dims(hidden_dim, axis=1)}
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _mm_xy_to_dw(x, hidden_dim, weights, *, has_bias=False):
    r"""VJP of ``y = x @ w (+ b)`` in one pass, returning a dict."""
    def _fwd(w_dict):
        y = x @ w_dict['weight']
        if has_bias:
            y = y + w_dict['bias']
        return u.get_mantissa(y)

    _, vjp_fn = jax.vjp(_fwd, weights)
    return jax.tree.map(u.get_mantissa, vjp_fn(hidden_dim)[0])


def _mm_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    """Return zero-filled Dict[str, Array] for the D-RTRL parameter-dim trace."""
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
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_mm_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_mm',
        impl=_etp_matmul_impl,
        yw_to_w=_mm_yw_to_w,
        xy_to_dw=_mm_xy_to_dw,
        init_drtrl=_mm_init_drtrl,
        init_pp=_mm_init_pp,
        trainable_invars_fn=_mm_trainable_invars,
        x_invar_index=0,
        batched=True,
    )
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
    r"""Unbatched: hidden_dim (out,), trace Dict[str, Array].

    For the weight:  trace['weight'] (in, out, n_state)
                     hidden_dim (out,) -> expand axis=0 -> (1, out, ...)
    For the bias:    trace['bias']   (out, n_state)
                     hidden_dim (out,) -> multiply element-wise
    """
    out = {'weight': trace['weight'] * jnp.expand_dims(hidden_dim, axis=0)}
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _mv_xy_to_dw(x, hidden_dim, weights, *, has_bias=False):
    r"""VJP of ``y = x @ w (+ b)`` in one pass, returning a dict."""
    def _fwd(w_dict):
        y = x @ w_dict['weight']
        if has_bias:
            y = y + w_dict['bias']
        return u.get_mantissa(y)

    _, vjp_fn = jax.vjp(_fwd, weights)
    return jax.tree.map(u.get_mantissa, vjp_fn(hidden_dim)[0])


def _mv_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    """Return zero-filled Dict[str, Array] for the D-RTRL parameter-dim trace."""
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
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_mv_p = register_primitive_spec(
    ETPPrimitiveSpec(
        name='etp_mv',
        impl=_etp_matmul_impl,
        yw_to_w=_mv_yw_to_w,
        xy_to_dw=_mv_xy_to_dw,
        init_drtrl=_mv_init_drtrl,
        init_pp=_mv_init_pp,
        trainable_invars_fn=_mv_trainable_invars,
        x_invar_index=0,
        batched=False,
    )
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
