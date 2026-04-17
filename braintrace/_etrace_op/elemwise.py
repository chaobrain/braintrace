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
"""

import jax.numpy as jnp
import saiunit as u

from ._spec import ETPPrimitiveSpec, register_primitive_spec

__all__ = [
    'etp_elemwise_p',
    'element_wise',
]


def _etp_elemwise_impl(y):
    return y


def _elemwise_yw_to_w(hidden_dim, trace):
    r"""Element-wise multiply."""
    return trace * hidden_dim


def _elemwise_xy_to_dw(x, hidden_dim, w):
    r"""Identity marker — gradient is just ``hidden_dim``.

    The chain rule through ``fn`` is handled by JAX on the ops *before*
    the primitive binds.
    """
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
