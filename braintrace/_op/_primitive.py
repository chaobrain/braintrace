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
four ETP-specific rule slots (``dt_to_t``, ``xy_to_dw``, ``init_drtrl``,
``init_pp``). All standard JAX rules — ``impl``, ``abstract_eval``,
MLIR lowering, JVP, transpose, batching — are auto-derived from a single
implementation function via :func:`register_primitive`.
"""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir

from braintrace._compatible_imports import Primitive
from ._registries import (
    BATCHED_PRIMITIVES,
    ETP_FAST_PATH_RULES,
    ETP_PRIMITIVES,
    ETP_RULES_INIT_DRTRL,
    ETP_RULES_INIT_PP,
    ETP_RULES_PP_X_REPR,
    ETP_RULES_XY_TO_DW,
    ETP_RULES_DT_TO_T,
    ETP_TRAINABLE_INVARS_FNS,
    ETP_X_INVAR_INDICES,
    ETP_Y_OUTVAR_INDICES,
    FastPathRules,
    GRADIENT_ENABLED_PRIMITIVES,
    get_batched_counterpart,
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

    See Also
    --------
    register_primitive : Factory that creates and returns an instance.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintrace
        >>>
        >>> # Register a primitive whose forward delegates to a standard op.
        >>> def my_impl(x, w):
        ...     return x @ w
        >>> my_p = braintrace.register_primitive('etp_demo_mm', my_impl, batched=True)
        >>> y = my_p.bind(jnp.ones((2, 3)), jnp.ones((3, 4)))
        >>> print(y.shape)
        (2, 4)
    """

    def register_dt_to_t(self, fn: Callable[..., Any]) -> None:
        """Install a D-RTRL trace propagation rule.

        Parameters
        ----------
        fn : Callable
            Rule with signature ``(hidden_dim, trace, **params) -> trace``.
        """
        ETP_RULES_DT_TO_T[self] = fn

    def register_xy_to_dw(self, fn: Callable[..., Any]) -> None:
        """Install a weight-gradient rule.

        Parameters
        ----------
        fn : Callable
            Rule with signature ``(x, hidden_dim, w, **params) -> dw``.
        """
        ETP_RULES_XY_TO_DW[self] = fn

    def register_init_drtrl(self, fn: Callable[..., Any]) -> None:
        """Install a D-RTRL trace initialiser.

        Parameters
        ----------
        fn : Callable
            Rule with signature
            ``(x_var, y_var, weight_var, num_hidden_state) -> zeros``.
        """
        ETP_RULES_INIT_DRTRL[self] = fn

    def register_init_pp(self, fn: Callable[..., Any]) -> None:
        """Install a pp_prop (IO-dim) df trace initialiser.

        Parameters
        ----------
        fn : Callable
            Rule with signature
            ``(x_var, y_var, weight_var, num_hidden_state) -> zeros``.
        """
        ETP_RULES_INIT_PP[self] = fn

    def register_etp_rules(
        self,
        *,
        dt_to_t: Callable[..., Any] | None = None,
        xy_to_dw: Callable[..., Any] | None = None,
        init_drtrl: Callable[..., Any] | None = None,
        init_pp: Callable[..., Any] | None = None,
        fast_path: FastPathRules | None = None,
        pp_x_repr: Callable[..., Any] | None = None,
    ) -> None:
        """Install multiple ETP rules in one call.

        Any argument left as ``None`` is skipped.

        Parameters
        ----------
        dt_to_t : Callable, optional
            D-RTRL trace propagation rule. Default ``None``.
        xy_to_dw : Callable, optional
            Weight-gradient rule. Default ``None``.
        init_drtrl : Callable, optional
            D-RTRL trace initialiser. Default ``None``.
        init_pp : Callable, optional
            pp_prop (IO-dim) df trace initialiser. Default ``None``.
        fast_path : FastPathRules, optional
            Closed-form param-dim D-RTRL fast-path kernel bundle (instant /
            recurrent / solve kernels plus the ``applicable`` gate). Registered
            into :data:`ETP_FAST_PATH_RULES`. Supplied only by primitives with
            an elementwise ``dt_to_t`` rule (mm / mv / elemwise); ``None``
            leaves the primitive without a fast path. Default ``None``.
        pp_x_repr : Callable, optional
            IO-dim x-trace representation rule
            ``(x, weight_avals) -> x_repr``. Registered into
            :data:`ETP_RULES_PP_X_REPR`. Supply it when the raw ``x`` is not
            the operand the op is linear in (e.g. ``etp_emb_p`` filters the
            one-hot encoding of its integer indices); ``None`` leaves the
            IO-dim trace filtering the raw ``x``. Default ``None``.
        """
        if dt_to_t is not None:
            ETP_RULES_DT_TO_T[self] = dt_to_t
        if xy_to_dw is not None:
            ETP_RULES_XY_TO_DW[self] = xy_to_dw
        if init_drtrl is not None:
            ETP_RULES_INIT_DRTRL[self] = init_drtrl
        if init_pp is not None:
            ETP_RULES_INIT_PP[self] = init_pp
        if fast_path is not None:
            ETP_FAST_PATH_RULES[self] = fast_path
        if pp_x_repr is not None:
            ETP_RULES_PP_X_REPR[self] = pp_x_repr


def register_primitive(
    name: str,
    impl_fn: Callable[..., Any],
    *,
    batched: bool = False,
    gradient_enabled: bool = False,
    trainable_invars_fn: Callable[..., dict[str, int]] | None = None,
    x_invar_index: int | None = 0,
    y_outvar_index: int = 0,
) -> ETPPrimitive:
    """Create an :class:`ETPPrimitive` with all JAX rules auto-derived.

    Only the four ETP-specific rules need hand-writing — call the returned
    primitive's ``register_*`` methods.

    Parameters
    ----------
    name : str
        Primitive name (e.g. ``'etp_mm'``).
    impl_fn : Callable
        Implementation function.
    batched : bool, optional
        Whether this primitive operates on batched inputs. Default ``False``.
    gradient_enabled : bool, optional
        If ``True``, the compiler will *evaluate* this primitive when walking
        ``y -> h`` (identity-like ops such as ``etp_elemwise_p``). Default
        ``False``.
    trainable_invars_fn : Callable or None, optional
        Function ``eqn.params -> {key: invar_index}`` declaring the
        primitive's full trainable-input layout. Used by the compiler and
        executors to support N-trainable-input primitives (e.g.
        ``{weight, bias}`` for Linear, ``{B, A, bias}`` for LoRA). If
        ``None``, the compiler falls back to the single-weight
        ``{'weight': 1}`` layout. Default ``None``.
    x_invar_index : int or None, optional
        Position of the input ``x`` in ``eqn.invars``, or ``None`` for
        primitives with no external input (currently only ``etp_elemwise_p``).
        Default ``0``.
    y_outvar_index : int, optional
        Position of the output ``y`` in ``eqn.outvars``. Default ``0``.

    Returns
    -------
    ETPPrimitive
        The registered primitive.

    Notes
    -----
    The following standard JAX rules are installed automatically:

    - **impl** — eager execution.
    - **abstract_eval** — via ``jax.eval_shape(impl)``.
    - **lowering** — via ``mlir.lower_fun(impl)``.
    - **JVP** — via ``jax.jvp(impl)``.
    - **transpose** — derived by JAX from the JVP.
    - **batching** — identity-preserving: when only ``x`` is mapped and a
      batched counterpart is registered
      (:func:`~braintrace._op._registries.register_batched_counterpart`),
      the counterpart primitive is bound with the batch axis leading;
      otherwise falls back to ``jax.vmap(impl)`` with a ``UserWarning``
      (the decomposed weight drops out of eligibility-trace compilation).
    """
    p = ETPPrimitive(name)
    ETP_PRIMITIVES.add(p)
    if batched:
        BATCHED_PRIMITIVES.add(p)
    if gradient_enabled:
        GRADIENT_ENABLED_PRIMITIVES.add(p)
    if trainable_invars_fn is not None:
        ETP_TRAINABLE_INVARS_FNS[p] = trainable_invars_fn
    ETP_X_INVAR_INDICES[p] = x_invar_index
    ETP_Y_OUTVAR_INDICES[p] = y_outvar_index

    p.def_impl(impl_fn)

    @p.def_abstract_eval
    def _abstract(*args: Any, **params: Any) -> Any:
        shapes = tuple(ShapedArray(a.shape, a.dtype) for a in args)
        out = jax.eval_shape(partial(impl_fn, **params), *shapes)
        return ShapedArray(out.shape, out.dtype)

    mlir.register_lowering(
        p, mlir.lower_fun(impl_fn, multiple_results=False),
    )

    def _jvp(primals: Any, tangents: Any, **params: Any) -> Any:
        # ``ad.Zero`` carries the mathematically-correct tangent aval on
        # ``t.aval``: for inexact (float/complex) primals that's the
        # primal's own shape/dtype, but for int/bool primals JAX's tangent
        # space is the zero-sized ``float0`` dtype. Materializing zeros as
        # ``jnp.zeros(pr.shape, pr.dtype)`` (the primal's dtype) is wrong for
        # int/bool primals and raises inside ``jax.jvp``. Delegate to JAX's
        # own ``instantiate_zeros``, which reads ``t.aval`` and is therefore
        # correct for both cases.
        tans = tuple(
            ad.instantiate_zeros(t) if isinstance(t, ad.Zero) else t
            for t in tangents
        )
        return jax.jvp(partial(impl_fn, **params), primals, tans)

    ad.primitive_jvps[p] = _jvp

    def _batching(args: Any, dims: Any, **params: Any) -> Any:
        # Identity-preserving promotion: when only ``x`` carries the batch
        # dim and a batched counterpart is registered, re-bind that
        # counterpart so the ETP primitive stays visible to the etrace
        # compiler. Otherwise decompose (value-correct) and warn: the
        # decomposed weight cannot participate in eligibility traces.
        counterpart = get_batched_counterpart(p)
        x_idx = ETP_X_INVAR_INDICES.get(p)
        if (
            counterpart is not None
            and x_idx is not None
            and dims[x_idx] is not None
            and all(d is None for i, d in enumerate(dims) if i != x_idx)
        ):
            x = args[x_idx]
            if dims[x_idx] != 0:
                x = jnp.moveaxis(x, dims[x_idx], 0)
            new_args = tuple(
                x if i == x_idx else a for i, a in enumerate(args)
            )
            return counterpart.bind(*new_args, **params), 0
        warnings.warn(
            f'ETP primitive {name!r} was decomposed into standard JAX ops '
            f'under vmap (batch dims {tuple(dims)}). Its trainable '
            f'parameters will NOT be recognized by the eligibility-trace '
            f'compiler if this trace is compiled for online learning. '
            f'Map over the data input only (weights unbatched), or call '
            f'the batched op directly.',
            UserWarning,
            stacklevel=2,
        )
        return jax.vmap(partial(impl_fn, **params), in_axes=dims)(*args), 0

    batching.primitive_batchers[p] = _batching

    return p
