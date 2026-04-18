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

r""":class:`ETPPrimitive` and :func:`register_primitive`.

Each ETP primitive is a JAX :class:`~jax.core.Primitive` subclass with
four ETP-specific rule slots (``yw_to_w``, ``xy_to_dw``, ``init_drtrl``,
``init_pp``). All standard JAX rules — ``impl``, ``abstract_eval``,
MLIR lowering, JVP, transpose, batching — are auto-derived from a single
implementation function via :func:`register_primitive`.
"""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir

from braintrace._compatible_imports import Primitive
from ._registries import (
    BATCHED_PRIMITIVES,
    ETP_PRIMITIVES,
    ETP_RULES_INIT_DRTRL,
    ETP_RULES_INIT_PP,
    ETP_RULES_XY_TO_DW,
    ETP_RULES_YW_TO_W,
    GRADIENT_ENABLED_PRIMITIVES,
)

__all__ = [
    'ETPPrimitive',
    'register_primitive',
]


class ETPPrimitive(Primitive):
    """A JAX ``Primitive`` with ETP rule registration helpers.

    Returned by :func:`register_primitive`. Supports every standard JAX
    primitive operation (``bind``, ``def_impl``, ...) and adds five
    convenience methods for installing ETP-specific rules into the global
    registries.

    Example::

        my_p = register_primitive('etp_my_op', _my_impl, batched=True)
        my_p.register_yw_to_w(my_yw_to_w_fn)
        my_p.register_xy_to_dw(my_xy_to_dw_fn)
        my_p.register_init_drtrl(my_init_drtrl_fn)
        my_p.register_init_pp(my_init_pp_fn)
    """

    def register_yw_to_w(self, fn: Callable):
        """Install a D-RTRL trace propagation rule.

        Signature: ``(hidden_dim, trace, **params) -> trace``.
        """
        ETP_RULES_YW_TO_W[self] = fn

    def register_xy_to_dw(self, fn: Callable):
        """Install a weight-gradient rule.

        Signature: ``(x, hidden_dim, w, **params) -> dw``.
        """
        ETP_RULES_XY_TO_DW[self] = fn

    def register_init_drtrl(self, fn: Callable):
        """Install a D-RTRL trace initialiser.

        Signature: ``(x_var, y_var, weight_var, num_hidden_state) -> zeros``.
        """
        ETP_RULES_INIT_DRTRL[self] = fn

    def register_init_pp(self, fn: Callable):
        """Install a pp_prop (IO-dim) df trace initialiser.

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
        """Install multiple ETP rules in one call. Skips any ``None`` argument."""
        if yw_to_w is not None:
            ETP_RULES_YW_TO_W[self] = yw_to_w
        if xy_to_dw is not None:
            ETP_RULES_XY_TO_DW[self] = xy_to_dw
        if init_drtrl is not None:
            ETP_RULES_INIT_DRTRL[self] = init_drtrl
        if init_pp is not None:
            ETP_RULES_INIT_PP[self] = init_pp


def register_primitive(name, impl_fn, *, batched=False, gradient_enabled=False):
    """Create an :class:`ETPPrimitive` with all JAX rules auto-derived.

    The following rules are installed automatically:

    - **impl** — eager execution
    - **abstract_eval** — via ``jax.eval_shape(impl)``
    - **lowering** — via ``mlir.lower_fun(impl)``
    - **JVP** — via ``jax.jvp(impl)``
    - **transpose** — derived by JAX from the JVP
    - **batching** — via ``jax.vmap(impl)``

    Only the four ETP-specific rules need hand-writing — call the returned
    primitive's ``register_*`` methods.

    Args:
        name: Primitive name (e.g. ``'etp_mm'``).
        impl_fn: Implementation function.
        batched: Whether this primitive operates on batched inputs.
        gradient_enabled: If True, the compiler will *evaluate* this primitive
            when walking ``y -> h`` (identity-like ops such as
            ``etp_elemwise_p``).

    Returns:
        :class:`ETPPrimitive`: the registered primitive.
    """
    p = ETPPrimitive(name)
    ETP_PRIMITIVES.add(p)
    if batched:
        BATCHED_PRIMITIVES.add(p)
    if gradient_enabled:
        GRADIENT_ENABLED_PRIMITIVES.add(p)

    p.def_impl(impl_fn)

    @p.def_abstract_eval
    def _abstract(*args, **params):
        shapes = tuple(ShapedArray(a.shape, a.dtype) for a in args)
        out = jax.eval_shape(partial(impl_fn, **params), *shapes)
        return ShapedArray(out.shape, out.dtype)

    mlir.register_lowering(
        p, mlir.lower_fun(impl_fn, multiple_results=False),
    )

    def _jvp(primals, tangents, **params):
        tans = tuple(
            jnp.zeros(pr.shape, pr.dtype) if isinstance(t, ad.Zero) else t
            for pr, t in zip(primals, tangents)
        )
        return jax.jvp(partial(impl_fn, **params), primals, tans)

    ad.primitive_jvps[p] = _jvp

    def _batching(args, dims, **params):
        return jax.vmap(partial(impl_fn, **params), in_axes=dims)(*args), 0

    batching.primitive_batchers[p] = _batching

    return p
