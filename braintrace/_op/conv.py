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

r"""Convolution ETP primitive (``etp_conv_p``).

Always expects a batch dimension on the input. The full keyword surface
of ``jax.lax.conv_general_dilated`` is preserved; the wrapper splits and
recombines brainunit quantities for the input and kernel.

**Forward operation**

.. math::

    y_{b, \mathbf{s}, k}
      = \sum_{\mathbf{u}, c} x_{b, \mathbf{s}+\mathbf{u}, c}\,
                              K_{\mathbf{u}, c, k}
        \;+\; b_k

where :math:`\mathbf{s}` runs over spatial output positions,
:math:`\mathbf{u}` over the kernel spatial window, :math:`c` is the input
channel, :math:`k` the output channel, and the kernel layout follows the
``dimension_numbers`` supplied at bind-time. Because the bias is a single
value per output channel shared across every spatial position, its
Jacobian is fundamentally different from the kernel's.

**Role of each ETP rule**

Let :math:`\mathbf{D}_f^t = \partial h / \partial y` (one cotangent per
output element). The conv primitive implements:

* ``xy_to_dw`` — **param-shaped** instantaneous Jacobian, consumed by the
  IO-dim (ES-D-RTRL) algorithm at solve time. For the *kernel*, uses the
  conv VJP to produce the full weight Jacobian
  :math:`\partial h / \partial K` (requires :math:`x`). For the *bias*,
  stores the per-position cotangent
  :math:`\partial h / \partial b_k = (\partial h / \partial y)_{b,\mathbf{s},k}`
  **without** summing over spatial positions (the sum happens at the
  consumer's contraction step).

* ``instant_drtrl`` / ``dt_to_t`` / ``solve_drtrl`` — the param-dim
  D-RTRL trace machinery. Because the kernel is *spatially shared*
  (every output position reads the same :math:`K`) while the D-RTRL
  discount :math:`\mathbf{D}^t` acts per output element, no kernel-shaped
  (spatially pre-summed) trace can be exact: it multiplies a sum by a sum
  where a sum of products is required. The exact trace keeps the spatial
  output axes — for a diagonal hidden state :math:`h_{b,\mathbf{s},k}`,

  .. math::

      \epsilon^t_{b,\mathbf{s},\mathbf{u},c,k}
        = D^t_{b,\mathbf{s},k}\, \epsilon^{t-1}_{b,\mathbf{s},\mathbf{u},c,k}
          + (\mathbf{D}_f^t)_{b,\mathbf{s},k}\,
            \mathrm{patch}^t_{b,\mathbf{s},\mathbf{u},c},

  where :math:`\mathrm{patch}^t` is the receptive-field window of
  :math:`x^t` extracted by :func:`_conv_extract_patches`.

  - ``instant_drtrl`` adds the per-position term
    :math:`(\mathbf{D}_f^t)_{b,\mathbf{s},k}\,\mathrm{patch}^t_{b,\mathbf{s},\mathbf{u},c}`
    for the kernel, and the per-position cotangent (with the ``bias_fn``
    chain) for the bias.
  - ``dt_to_t`` (recurrence only) multiplies every trace slot by the
    per-position factor :math:`D^t_{b,\mathbf{s},k}` — no spatial sums
    anywhere.
  - ``solve_drtrl`` contracts the learning signal with the trace,
    performing the deferred spatial sum:
    :math:`\nabla K_{\mathbf{u},c,k} = \sum_{\mathbf{s}}
    g_{\mathbf{s},k}\, \epsilon_{\mathbf{s},\mathbf{u},c,k}` and
    :math:`\nabla b_k = \sum_{\mathbf{s}} g_{\mathbf{s},k}\,
    \epsilon_{b,\mathbf{s},k}`.

* ``init_drtrl`` — allocates the per-position kernel trace
  :math:`\boldsymbol{\epsilon}_K` of shape
  ``(batch, *spatial_out, *kernel_shape, n_state)`` plus the per-position
  :math:`\boldsymbol{\epsilon}_b` with output shape. This costs
  :math:`O(B\, S\, |K|)` memory (``S`` = spatial output size) — the price
  of exactness for a spatially shared kernel; for large convolutions
  prefer the IO-dim algorithm (``pp_prop`` / ``IODimVjpAlgorithm``).
  Grouped convolutions (``feature_group_count`` / ``batch_group_count``
  != 1) are rejected with ``NotImplementedError``.

* ``init_pp`` — allocates the pp-prop output-shaped df trace; the
  :math:`\boldsymbol{\epsilon}_x` factor in ES-D-RTRL is the full
  batched input tensor held by the executor.

**Transform hooks**

The primitive accepts two optional transform hooks in its ``eqn.params``:
``kernel_fn`` (computes ``y = conv(x, kernel_fn(kernel))`` — note the
kernel-facing name, not ``weight_fn``) and ``bias_fn`` (adds
``bias_fn(b)``). The forward impl, :func:`_conv_xy_to_dw` and
:func:`_conv_instant_drtrl` apply them; the eligibility trace and gradient
are always taken w.r.t. the **raw** kernel / bias, so the transform
Jacobian :math:`f'` enters *only* through the instantaneous rules via
:func:`jax.vjp` (``kernel_fn`` through a kernel VJP — applied per output
position in ``instant_drtrl`` — ``bias_fn`` as an elementwise per-channel
diagonal factor). The ``dt_to_t`` / ``solve_drtrl`` rules and the trace
initialisers are transform-free and stay exact (they operate on
:math:`\partial h / \partial K_{\text{raw}}`).

This primitive has **no fast path** — it always uses the generic rule path,
which threads :math:`f'` correctly when a transform hook is present.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
import brainunit as u

from ._primitive import register_primitive
from ._registries import ETP_RULES_INSTANT_DRTRL, ETP_RULES_SOLVE_DRTRL
from braintrace._typing import ArrayLike, WeightFn

__all__ = [
    'etp_conv_p',
    'conv',
]


def _etp_conv_impl(
    *args: Any,
    has_bias: bool = False,
    strides: Sequence[int] = (1,),
    padding: str = 'SAME',
    lhs_dilation: Optional[Sequence[int]] = None,
    rhs_dilation: Optional[Sequence[int]] = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    dimension_numbers: Any = None,
    kernel_fn: WeightFn | None = None,
    bias_fn: WeightFn | None = None,
) -> Any:
    x, kernel = args[0], args[1]
    if kernel_fn is not None:
        kernel = kernel_fn(kernel)
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
        b = args[2]
        if bias_fn is not None:
            b = bias_fn(b)
        if b.ndim == 1:
            # Canonical per-output-channel bias vector: broadcast it along
            # the layout's channel axis. The channel axis is NOT always
            # trailing -- the default (``dimension_numbers=None``) layout is
            # NCH-style, with the channel axis at position 1, so a naive
            # ``y + b`` broadcasts against the trailing spatial axis instead
            # (raising when sizes differ, silently corrupting the output
            # when a spatial size happens to equal ``out_channels``).
            # Bias arrays with rank > 1 are assumed to already be pre-shaped
            # by the caller for direct broadcasting and are left untouched.
            _, channel_axis, _, _ = _conv_layout(
                {'strides': strides, 'dimension_numbers': dimension_numbers}
            )
            shape = [1] * y.ndim
            shape[channel_axis] = b.shape[0]
            b = b.reshape(shape)
        y = y + b
    return y


def _conv_trainable_invars(params: dict[str, Any]) -> dict[str, int]:
    """Return ``{key: invar_index}`` depending on ``has_bias``."""
    base = {'weight': 1}
    if params.get('has_bias', False):
        base['bias'] = 2
    return base


def _conv_layout(params: dict[str, Any]) -> tuple[int, int, int, int]:
    """Return ``(n_spatial, channel_axis, batch_axis, kernel_out_axis)``.

    ``n_spatial``:       spatial rank (1, 2, or 3).
    ``channel_axis``:    position of the output-channel axis in the OUTPUT
                         tensor (``y`` / ``hidden_dim`` in batched form).
    ``batch_axis``:      position of the batch axis in the OUTPUT tensor.
    ``kernel_out_axis``: position of the out-channel dimension in the KERNEL
                         tensor (shape of ``w_trace`` *without* any batch
                         prefix, i.e. as stored in the weight array).

    Sources used (in priority order):

    1. ``params['dimension_numbers']`` — when a ``ConvDimensionNumbers``
       namedtuple is present, ``out_spec[0]`` is the batch position and
       ``out_spec[1]`` is the channel position in the output; ``rhs_spec[0]``
       is the out-channel position in the kernel.
    2. ``params['strides']`` — ``len(strides)`` gives ``n_spatial``.
    3. When ``dimension_numbers`` is ``None`` JAX defaults to ``iota``
       (``(0,1,2,...)``) which maps to NCHW / NCH for the output (batch=0,
       channel=1) and OIHW / OIH for the kernel (out-channel=0).

    Notes on ``ConvDimensionNumbers``::

        ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)

    ``out_spec[0]``  → position of N (batch)   in the output
    ``out_spec[1]``  → position of C (channel) in the output
    ``rhs_spec[0]``  → position of out-channel  in the kernel (logical order)

    Example: ``('NHWC', 'HWIO', 'NHWC')`` gives ``batch_axis=0``,
    ``channel_axis=3``, ``kernel_out_axis=2`` (index of 'O' in 'HWIO').
    """
    n_spatial = len(params.get('strides', (1,)))
    dn = params.get('dimension_numbers', None)
    if dn is None:
        # JAX default: iota = (0,1,2,...) → NCHW/NCH output, OIHW/OIH kernel.
        batch_axis = 0
        channel_axis = 1
        kernel_out_axis = 0  # out-channel at axis 0 of kernel (OIHW-style)
    elif isinstance(dn, tuple) and len(dn) == 3 and isinstance(dn[2], str):
        # String-tuple form e.g. ('NHWC', 'HWIO', 'NHWC').
        out_spec_str = dn[2]
        batch_axis = out_spec_str.index('N')
        channel_axis = out_spec_str.index('C')
        rhs_spec_str = dn[1]
        kernel_out_axis = rhs_spec_str.index('O')
    else:
        # ConvDimensionNumbers namedtuple.
        out_spec = dn.out_spec
        batch_axis = out_spec[0]
        channel_axis = out_spec[1]
        rhs_spec = dn.rhs_spec
        kernel_out_axis = rhs_spec[0]  # logical out-channel position in kernel
    return n_spatial, channel_axis, batch_axis, kernel_out_axis


def _conv_dt_to_t(hidden_dim: Any, trace: dict[str, Any], **params: Any) -> dict[str, Any]:
    r"""Apply the recurrence factor :math:`D^t = \partial h^t / \partial y`
    to the per-position conv trace — a pure elementwise multiply.

    **Role in D-RTRL.** Implements the :math:`\mathbf{D}^t
    \boldsymbol{\epsilon}^{t-1}` term of the trace recurrence. The trace
    keeps the spatial output axes (see :func:`_conv_init_drtrl`), so the
    recurrence is exact per position:

    .. math::

        \epsilon^t_{b, \mathbf{s}, \mathbf{u}, c, k}
          \;=\; D^t_{b, \mathbf{s}, k}\,
                \epsilon^{t-1}_{b, \mathbf{s}, \mathbf{u}, c, k}, \qquad
        \epsilon^t_{b_{\text{ias}},\, b, \mathbf{s}, k}
          \;=\; D^t_{b, \mathbf{s}, k}\,
                \epsilon^{t-1}_{b_{\text{ias}},\, b, \mathbf{s}, k}.

    No spatial sums appear anywhere — the spatial contraction with the
    learning signal is performed once, at solve time, by
    :func:`_conv_solve_drtrl`. (The old kernel-shaped trace summed
    :math:`D^t` over :math:`\mathbf{s}` here, turning the required sum of
    products into a product of sums — wrong for the kernel even at
    :math:`T = 1` and corrupting the bias recurrence for :math:`T \geq 2`.)

    **Shapes (recurrence context; the solve path uses**
    :func:`_conv_solve_drtrl` **instead):**

        ``hidden_dim      : (batch, *y_layout)`` — the per-position
        :math:`D^t` factor, laid out like the conv output ``y``,
        ``trace['weight'] : (batch, *spatial_out, *kernel_shape)``,
        ``trace['bias']   : (batch, *y_layout)``.

    **Layout awareness.** The batch / channel / spatial positions of
    ``hidden_dim`` and the kernel's out-channel axis are derived from
    ``dimension_numbers`` / ``strides`` (NHWC / HWIO / NCHW / OIHW etc.);
    ``hidden_dim`` is transposed to ``(batch, *spatial_out, out_ch)`` and
    broadcast over the kernel axes with ``out_ch`` aligned to the kernel's
    out-channel axis.
    """
    has_bias = params.get('has_bias', False)
    n_spatial, channel_axis, batch_axis, kernel_out_axis = _conv_layout(params)
    w_trace = trace['weight']

    # hidden_dim: (batch, *y_layout) -> (batch, *spatial_out, out_ch), with
    # spatial axes in y-array order (matching the trace's spatial prefix).
    spatial_axes = tuple(
        sorted(set(range(hidden_dim.ndim)) - {batch_axis, channel_axis})
    )
    hd = jnp.transpose(hidden_dim, (batch_axis, *spatial_axes, channel_axis))

    # Broadcast over the kernel block with out_ch aligned to kernel_out_axis.
    kernel_rank = w_trace.ndim - 1 - n_spatial  # minus batch and spatial_out
    target_shape = hd.shape[:1 + n_spatial] + tuple(
        hd.shape[-1] if j == kernel_out_axis else 1 for j in range(kernel_rank)
    )
    out = {'weight': w_trace * hd.reshape(target_shape)}

    if has_bias:
        # Per-position multiply — the trace and D^t share the y layout.
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _conv_xy_to_dw(x: Any, hidden_dim: Any, weights: dict[str, Any], **params: Any) -> dict[str, Any]:
    r"""Instantaneous conv Jacobian :math:`\partial h / \partial (K, b)`.

    **Role in D-RTRL.** Produces the
    :math:`\operatorname{diag}(\mathbf{D}_f^t) \otimes \mathbf{x}^t` term
    for the conv primitive. The derivative pieces are

    .. math::

        \frac{\partial y_{b, \mathbf{s}, k}}{\partial K_{\mathbf{u}, c, k'}}
          = \delta_{k k'}\, x_{b, \mathbf{s}+\mathbf{u}, c}, \qquad
        \frac{\partial y_{b, \mathbf{s}, k}}{\partial b_{k'}}
          = \delta_{k k'},

    so pulling back a hidden cotangent ``hidden_dim`` gives

    .. math::

        \left.\frac{\partial h}{\partial K}\right|_t
          \;=\; \text{VJP}_K\bigl(\mathrm{conv}(x, K)\bigr)(\partial h/\partial y), \qquad
        \left.\frac{\partial h}{\partial b_k}\right|_{t, \mathbf{s}}
          \;=\; (\partial h/\partial y)_{b, \mathbf{s}, k}.

    **Kernel path.** Uses ``jax.vjp`` of ``jax.lax.conv_general_dilated``
    — the kernel genuinely depends on :math:`x`, so the full conv VJP is
    required. The remap ``strides → window_strides`` matches the
    low-level API.

    **Bias path.** The bias appears additively with no spatial coupling,
    so its instantaneous Jacobian is the cotangent itself at each
    spatial position. We store this per-position (no spatial sum here);
    the sum is performed inside :func:`_conv_dt_to_t` during trace
    propagation — this keeps the bias-trace shape identical to
    :math:`\partial h / \partial y` so the :math:`\mathbf{D}^t \boldsymbol{\epsilon}^{t-1}`
    recurrence can be applied element-by-element.

    **Unbatched detection.** The D-RTRL executor vmaps over the batch
    axis, so ``x`` may arrive as ``ndim == n_spatial + 1`` (no batch).
    We prepend a leading axis before calling ``conv_general_dilated``,
    then strip it on the way out implicitly via the VJP.
    """
    has_bias = params.get('has_bias', False)
    kernel_fn = params.get('kernel_fn', None)
    bias_fn = params.get('bias_fn', None)
    # Build conv_general_dilated kwargs; remap 'strides' -> 'window_strides';
    # drop the ETP-only params that conv does not understand.
    conv_kw = {}
    for k, v in params.items():
        if k in ('has_bias', 'kernel_fn', 'bias_fn'):
            continue
        if k == 'strides':
            conv_kw['window_strides'] = v
        else:
            conv_kw[k] = v

    # The batched D-RTRL executor vmaps over the batch dimension, so x may
    # arrive here without a leading batch axis.
    # Unbatched detection: a batched input has ndim == n_spatial + 2
    # (batch + spatial + channel / or the permuted equivalent), while an
    # unbatched input has ndim == n_spatial + 1.
    n_spatial = len(params.get('strides', (1,)))
    unbatched = (x.ndim == n_spatial + 1)
    if unbatched:
        x_in = x[None]
        hd_in = hidden_dim[None]
    else:
        x_in = x
        hd_in = hidden_dim

    # Kernel gradient via VJP (needs x); apply kernel_fn inside so jax.vjp
    # auto-composes f' for the kernel gradient.
    def _fwd_w(w: Any) -> Any:
        if kernel_fn is not None:
            w = kernel_fn(w)
        return u.get_mantissa(
            jax.lax.conv_general_dilated(x_in, w, **conv_kw)
        )

    _, vjp_fn = jax.vjp(_fwd_w, weights['weight'])
    dw = u.get_mantissa(vjp_fn(hd_in)[0])
    out = {'weight': dw}

    if has_bias:
        # Bias gradient = hidden_dim (cotangent at each output position).
        # No spatial summation — the trace stores per-position ∂h/∂b.
        bias_grad = hidden_dim
        if bias_fn is not None:
            b = weights['bias']
            _, b_vjp = jax.vjp(bias_fn, b)
            db = u.get_mantissa(b_vjp(jnp.ones_like(b))[0])  # bias_fn'(b), shape (out_ch,)
            _, channel_axis, _, _ = _conv_layout(params)
            batched_rank = n_spatial + 2  # (batch, *spatial, channel) up to permutation
            axes_right_of_channel = batched_rank - 1 - channel_axis
            ax = bias_grad.ndim - 1 - axes_right_of_channel
            shape = [1] * bias_grad.ndim
            shape[ax] = db.shape[0]
            bias_grad = bias_grad * db.reshape(shape)
        out['bias'] = bias_grad

    return out


def _conv_extract_patches(x: Any, kernel_shape: Sequence[int],
                          params: dict[str, Any]) -> Any:
    r"""Extract per-output-position input patches for the conv primitive.

    Wraps :func:`jax.lax.conv_general_dilated_patches` and unpacks its
    packed channel axis into an explicit ``(c_in, *kernel_spatial)`` block
    so that

    .. math::

        y_{b, \mathbf{s}, k}
          \;=\; \sum_{\mathbf{u}, c}
                \mathrm{patch}_{b, \mathbf{s}, c, \mathbf{u}}\,
                K_{\mathbf{u}, c, k}

    reproduces the forward convolution exactly. The packed-channel
    convention (channel-major: index ``c * prod(kernel_spatial) +
    flat(u)``) is pinned *numerically* by
    ``conv_test.py::TestConvPatchExtraction`` rather than trusted from the
    docs.

    Parameters
    ----------
    x : Array
        Batched input tensor, laid out per ``params['dimension_numbers']``.
    kernel_shape : Sequence[int]
        Shape of the (raw) kernel array; the spatial filter shape is read
        from it through the rhs spec.
    params : dict
        Conv eqn params (``strides`` / ``padding`` / ``lhs_dilation`` /
        ``rhs_dilation`` / ``dimension_numbers``). Group counts must be 1
        (enforced upstream by :func:`_conv_init_drtrl`).

    Returns
    -------
    Array
        Patches of shape ``(batch, *spatial_out, c_in, *kernel_spatial)``,
        with ``spatial_out`` in the output tensor's axis order and
        ``kernel_spatial`` in the canonical (strides) order.
    """
    dn = jax.lax.conv_dimension_numbers(
        x.shape, tuple(kernel_shape), params.get('dimension_numbers', None)
    )
    filter_shape = tuple(kernel_shape[a] for a in dn.rhs_spec[2:])
    patches = jax.lax.conv_general_dilated_patches(
        lhs=x,
        filter_shape=filter_shape,
        window_strides=tuple(params.get('strides', (1,))),
        padding=params.get('padding', 'SAME'),
        lhs_dilation=params.get('lhs_dilation', None),
        rhs_dilation=params.get('rhs_dilation', None),
        dimension_numbers=dn,
    )
    # patches follows the output layout with a packed channel axis of size
    # c_in * prod(filter_shape); bring it to (batch, *spatial_out, packed).
    batch_ax, ch_ax = dn.out_spec[0], dn.out_spec[1]
    spatial_axes = tuple(sorted(set(range(patches.ndim)) - {batch_ax, ch_ax}))
    patches = jnp.transpose(patches, (batch_ax, *spatial_axes, ch_ax))
    c_in = x.shape[dn.lhs_spec[1]]
    return patches.reshape(patches.shape[:-1] + (c_in,) + filter_shape)


def _conv_instant_bias(hidden_dim: Any, weights: dict[str, Any],
                       params: dict[str, Any]) -> Any:
    r"""Per-position instantaneous bias term with the ``bias_fn`` chain.

    Mirrors the bias path of :func:`_conv_xy_to_dw` byte-for-byte in
    behaviour: the instantaneous bias Jacobian at each output position is
    the cotangent itself, optionally multiplied by the elementwise
    ``bias_fn`` Jacobian diagonal along the layout's channel axis. The
    channel position is located relative to the trailing axes, so the
    helper accepts both batched and batch-free cotangents.
    """
    bias_fn = params.get('bias_fn', None)
    bias_grad = hidden_dim
    if bias_fn is not None:
        b = weights['bias']
        _, b_vjp = jax.vjp(bias_fn, b)
        db = u.get_mantissa(b_vjp(jnp.ones_like(b))[0])  # bias_fn'(b), shape (out_ch,)
        n_spatial, channel_axis, _, _ = _conv_layout(params)
        batched_rank = n_spatial + 2  # (batch, *spatial, channel) up to permutation
        axes_right_of_channel = batched_rank - 1 - channel_axis
        ax = bias_grad.ndim - 1 - axes_right_of_channel
        shape = [1] * bias_grad.ndim
        shape[ax] = db.shape[0]
        bias_grad = bias_grad * db.reshape(shape)
    return bias_grad


def _conv_instant_drtrl(x: Any, hidden_dim: Any, weights: dict[str, Any],
                        **params: Any) -> dict[str, Any]:
    r"""Per-position instantaneous term for param-dim D-RTRL.

    **Role.** Supplies the :math:`\operatorname{diag}(\mathbf{D}_f^t)
    \otimes \mathbf{x}^t` term added to :math:`\mathbf{D}^t
    \boldsymbol{\epsilon}^{t-1}` each step, in the *trace* structure of
    :func:`_conv_init_drtrl` (spatial output axes retained) rather than
    the parameter structure produced by :func:`_conv_xy_to_dw` (which
    pre-sums over spatial positions — exact for the IO-dim solve but
    incompatible with the per-position D-RTRL recurrence):

    .. math::

        \left(\frac{\partial h}{\partial K}\right)_{\mathbf{s}, \mathbf{u}, c, k}
          \;=\; (\mathbf{D}_f^t)_{\mathbf{s}, k}\,
                \mathrm{patch}^t_{\mathbf{s}, \mathbf{u}, c}, \qquad
        \left(\frac{\partial h}{\partial b}\right)_{\mathbf{s}, k}
          \;=\; (\mathbf{D}_f^t)_{\mathbf{s}, k},

    where :math:`\mathrm{patch}^t` is the receptive-field window of
    :math:`x^t` from :func:`_conv_extract_patches`.

    **Transform hooks.** ``kernel_fn``'s Jacobian enters exactly as in
    :func:`_conv_xy_to_dw` — through :func:`jax.vjp` of ``kernel_fn`` —
    applied to each per-position kernel-shaped cotangent (the VJP is
    linear in the cotangent, so the per-position application sums to the
    legacy behaviour). ``bias_fn`` enters through
    :func:`_conv_instant_bias`, unchanged from the ``xy_to_dw`` bias path.

    **Shapes.** The algorithm vmaps over the batch axis and the trailing
    ``num_state`` axis, so this rule sees batch-free slices:
    ``x : (*x_layout)``, ``hidden_dim : (*y_layout)``, returning
    ``{'weight': (*spatial_out, *kernel_shape)[, 'bias': (*y_layout)]}``.
    A batched call (``x.ndim == n_spatial + 2``) is also supported,
    returning batch-prefixed arrays.
    """
    has_bias = params.get('has_bias', False)
    kernel_fn = params.get('kernel_fn', None)
    kernel = u.get_mantissa(weights['weight'])
    n_spatial = len(params.get('strides', (1,)))

    unbatched = (x.ndim == n_spatial + 1)
    x_in = x[None] if unbatched else x
    hd_in = hidden_dim[None] if unbatched else hidden_dim

    dn = jax.lax.conv_dimension_numbers(
        x_in.shape, kernel.shape, params.get('dimension_numbers', None)
    )
    # (batch, *spatial_out, c_in, *kernel_spatial)
    patches = _conv_extract_patches(x_in, kernel.shape, params)

    # df -> (batch, *spatial_out, out_ch), spatial in output-tensor order
    # (matching the trace's spatial prefix).
    batch_ax, ch_ax = dn.out_spec[0], dn.out_spec[1]
    spatial_axes = tuple(sorted(set(range(hd_in.ndim)) - {batch_ax, ch_ax}))
    df_t = jnp.transpose(hd_in, (batch_ax, *spatial_axes, ch_ax))

    # Outer product over the kernel block: (batch, *s, c_in, *u, out_ch).
    lead = 1 + n_spatial  # batch + spatial_out prefix
    df_b = df_t.reshape(
        df_t.shape[:lead] + (1,) * (1 + n_spatial) + df_t.shape[-1:]
    )
    eff = patches[..., None] * df_b

    # Rearrange the kernel block (c_in, *u, out_ch) into the raw kernel
    # layout given by the rhs spec (OIH / HWIO / ...).
    rhs_spec = dn.rhs_spec
    src = {rhs_spec[1]: lead, rhs_spec[0]: lead + 1 + n_spatial}
    for m, ax in enumerate(rhs_spec[2:]):
        src[ax] = lead + 1 + m
    perm = tuple(range(lead)) + tuple(src[j] for j in range(kernel.ndim))
    inst = jnp.transpose(eff, perm)  # (batch, *spatial_out, *kernel_shape)

    if kernel_fn is not None:
        # Chain kernel_fn' via the same VJP _conv_xy_to_dw composes, applied
        # per output position (linear in the cotangent, hence exact).
        _, vjp_fn = jax.vjp(kernel_fn, kernel)
        flat = inst.reshape((-1,) + tuple(kernel.shape))
        flat = jax.vmap(lambda ct: u.get_mantissa(vjp_fn(ct)[0]))(flat)
        inst = flat.reshape(inst.shape)

    if unbatched:
        inst = inst[0]
    out = {'weight': inst}

    if has_bias:
        out['bias'] = _conv_instant_bias(hidden_dim, weights, params)
    return out


def _conv_solve_drtrl(dg_hidden: Any, trace: dict[str, Any],
                      weights: dict[str, Any], **params: Any) -> dict[str, Any]:
    r"""Solve-time weight gradients from the per-position conv trace.

    **Role.** Contracts the learning signal :math:`g = \partial \mathcal{L}
    / \partial h` with the eligibility trace, performing the spatial sum
    that the per-position recurrence deferred:

    .. math::

        \nabla K_{\mathbf{u}, c, k}
          \;=\; \sum_{\mathbf{s}} g_{\mathbf{s}, k}\,
                \epsilon_{\mathbf{s}, \mathbf{u}, c, k}, \qquad
        \nabla b_k
          \;=\; \sum_{\mathbf{s}} g_{\mathbf{s}, k}\,
                \epsilon_{b,\, \mathbf{s}, k}.

    The transform Jacobians (``kernel_fn'`` / ``bias_fn'``) are **not**
    re-applied here — they already entered the trace through
    :func:`_conv_instant_drtrl`, mirroring the legacy ``xy_to_dw``
    behaviour; ``weights`` is unused (kept for the registry signature).

    **Shapes.** The algorithm vmaps over the batch axis and the trailing
    ``num_state`` axis, so this rule sees batch-free, state-free slices:
    ``dg_hidden : (*y_layout)``,
    ``trace['weight'] : (*spatial_out, *kernel_shape)``,
    ``trace['bias'] : (*y_layout)``. Returns param-shaped gradients
    ``{'weight': (*kernel_shape)[, 'bias': (out_ch,)]}``.
    """
    has_bias = params.get('has_bias', False)
    n_spatial, channel_axis_b, batch_axis_b, kernel_out_axis = _conv_layout(params)

    # Batch-free renumbering (the algorithm vmapped the batch axis away).
    remaining = sorted(set(range(n_spatial + 2)) - {batch_axis_b})
    ch_axis = remaining.index(channel_axis_b)
    spatial_axes = tuple(i for i in range(len(remaining)) if i != ch_axis)

    dg_t = jnp.transpose(dg_hidden, (*spatial_axes, ch_axis))  # (*spatial_out, out_ch)
    w_trace = trace['weight']                                  # (*spatial_out, *kernel)
    kernel_rank = w_trace.ndim - n_spatial
    target_shape = dg_t.shape[:n_spatial] + tuple(
        dg_t.shape[-1] if j == kernel_out_axis else 1 for j in range(kernel_rank)
    )
    out = {
        'weight': jnp.sum(
            w_trace * dg_t.reshape(target_shape), axis=tuple(range(n_spatial))
        )
    }
    if has_bias:
        out['bias'] = jnp.sum(trace['bias'] * dg_hidden, axis=spatial_axes)
    return out


def _conv_init_drtrl(x_var: Any, y_var: Any, weight_vars: dict[str, Any],
                     num_hidden_state: int, *, eqn_params: dict[str, Any]) -> dict[str, Any]:
    r"""Initialise the per-position conv D-RTRL trace.

    .. math::

        \boldsymbol{\epsilon}_K \in
          \mathbb{R}^{B \times \text{(spatial out)} \times \text{(kernel dims)}
          \times n_{\text{state}}}, \qquad
        \boldsymbol{\epsilon}_b \in
          \mathbb{R}^{B \times \text{(spatial out)} \times O \times n_{\text{state}}}.

    The kernel trace keeps one kernel-shaped slot **per spatial output
    position**: the kernel is spatially shared while the D-RTRL discount
    :math:`\mathbf{D}^t` acts per output element, so a spatially pre-summed
    (kernel-shaped) trace cannot follow the recurrence exactly. The
    spatial-output axes come from ``y_var``'s shape via :func:`_conv_layout`
    (in y-array order); the kernel axes follow the raw weight layout. This
    costs :math:`O(B\, S\, |K|\, n_{\text{state}})` memory — for large
    convolutions prefer the IO-dim algorithm (``pp_prop`` /
    ``IODimVjpAlgorithm``), whose conv trace stays output-shaped.

    The bias trace keeps spatial dims so :func:`_conv_dt_to_t` can apply
    the per-position :math:`D^t` factor elementwise; the spatial sum
    implementing :math:`\sum_{\mathbf{s}} g_{\mathbf{s}} \partial
    h/\partial b_k` is performed at solve time by
    :func:`_conv_solve_drtrl`.

    Zero-initialised (matches :math:`\boldsymbol{\epsilon}^0 = \mathbf{0}`).

    The trace dtype is derived from the participating ``x``/``y``/weight
    avals via :func:`jax.numpy.result_type` rather than left to ``jnp.zeros``'
    default (which silently follows the global x64 flag instead of the
    operands' actual dtype).

    Parameters
    ----------
    x_var, y_var : jax variables
        The equation's input / output variables (batched shapes).
    weight_vars : dict
        Trainable variables keyed by name (``'weight'`` and optionally
        ``'bias'``).
    num_hidden_state : int
        Number of hidden states in the group (trailing trace axis).
    eqn_params : dict
        The conv equation's params — required to derive the output layout
        (``dimension_numbers`` / ``strides``) and to reject grouped
        convolutions.

    Raises
    ------
    NotImplementedError
        If ``feature_group_count != 1`` or ``batch_group_count != 1`` —
        per-position patch extraction for grouped convolutions is not
        supported; use ``pp_prop`` (``IODimVjpAlgorithm``) instead.
    """
    if (eqn_params.get('feature_group_count', 1) != 1
            or eqn_params.get('batch_group_count', 1) != 1):
        raise NotImplementedError(
            'param-dim D-RTRL (ParamDimVjpAlgorithm / D_RTRL) does not support '
            'grouped convolutions (feature_group_count != 1 or '
            'batch_group_count != 1): the exact per-position kernel trace '
            'requires ungrouped input patches. Use the IO-dim algorithm '
            '(pp_prop / IODimVjpAlgorithm) for grouped convolutions.'
        )
    batch = x_var.aval.shape[0]
    dtype = jnp.result_type(
        x_var.aval.dtype, y_var.aval.dtype,
        *(v.aval.dtype for v in weight_vars.values()),
    )
    n_spatial, channel_axis, batch_axis, _ = _conv_layout(eqn_params)
    y_shape = y_var.aval.shape
    spatial_out_axes = sorted(set(range(len(y_shape))) - {batch_axis, channel_axis})
    spatial_out = tuple(y_shape[a] for a in spatial_out_axes)
    out = {
        'weight': jnp.zeros(
            (batch, *spatial_out, *weight_vars['weight'].aval.shape, num_hidden_state),
            dtype=dtype,
        )
    }
    if 'bias' in weight_vars:
        # y_var.aval.shape = (batch, *spatial, out_ch); strip the batch dim.
        out['bias'] = jnp.zeros(
            (batch, *y_shape[1:], num_hidden_state), dtype=dtype
        )
    return out


def _conv_init_pp(x_var: Any, y_var: Any, weight_vars: dict[str, Any],
                  num_hidden_state: int) -> Any:
    r"""Initialise conv pp-prop / ES-D-RTRL df trace.

    .. math::

        \boldsymbol{\epsilon}_f \in \mathbb{R}^{B \times \text{(spatial)} \times O \times n_{\text{state}}}.

    Output-shaped like :math:`y`. The matching :math:`\boldsymbol{\epsilon}_x`
    in ES-D-RTRL is the raw batched input tensor held by the executor's
    x-trace; :func:`_conv_xy_to_dw` combines the two via conv VJP at
    solve-time.
    """
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


etp_conv_p = register_primitive(
    'etp_conv',
    _etp_conv_impl,
    batched=True,
    trainable_invars_fn=_conv_trainable_invars,
    x_invar_index=0,
)
etp_conv_p.register_etp_rules(
    dt_to_t=_conv_dt_to_t,
    xy_to_dw=_conv_xy_to_dw,
    init_drtrl=_conv_init_drtrl,
    init_pp=_conv_init_pp,
)
# Param-dim D-RTRL overrides: the per-position kernel trace (spatial output
# axes retained) differs from the parameter structure, so the instantaneous
# term and the solve-time contraction cannot be expressed by xy_to_dw /
# dt_to_t alone. IO-dim (ES-D-RTRL) keeps using xy_to_dw.
ETP_RULES_INSTANT_DRTRL[etp_conv_p] = _conv_instant_drtrl
ETP_RULES_SOLVE_DRTRL[etp_conv_p] = _conv_solve_drtrl


def conv(
    x: ArrayLike,
    kernel: ArrayLike,
    bias: ArrayLike | None = None,
    *,
    strides: Sequence[int] = (1,),
    padding: str = 'SAME',
    lhs_dilation: Optional[Sequence[int]] = None,
    rhs_dilation: Optional[Sequence[int]] = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    dimension_numbers: Any = None,
    kernel_fn: WeightFn | None = None,
    bias_fn: WeightFn | None = None,
) -> ArrayLike:
    r"""ETP-aware convolution.

    Computes :math:`y = \mathrm{conv}(x, kernel) \; (+ b)` by routing the
    kernel (and optional bias) through an ETP primitive so they participate
    in eligibility-trace computation. The full keyword surface of
    :func:`jax.lax.conv_general_dilated` is preserved. Always expects a
    batch dimension on ``x``.

    Parameters
    ----------
    x : ArrayLike
        Input tensor with a leading batch dimension.
    kernel : ArrayLike
        Convolution kernel, with layout governed by ``dimension_numbers``.
    bias : Array, optional
        Bias added to the convolution output. A 1-D array of shape
        ``(out_channels,)`` is automatically reshaped to broadcast along the
        output's channel axis as determined by ``dimension_numbers`` (so it is
        correct for channel-first layouts such as the default NCH/NCHW as well
        as channel-last ones). An array of rank > 1 is added as-is and must
        already be broadcast-compatible with the layout-dependent output shape
        (this is how ``braintrace.nn.Conv1d/2d/3d`` pass their pre-shaped
        bias). Default ``None``.
    strides : Sequence[int], optional
        Window strides. Default ``(1,)``.
    padding : str, optional
        Padding mode (e.g. ``'SAME'`` or ``'VALID'``). Default ``'SAME'``.
    lhs_dilation : Sequence[int] or None, optional
        Left-hand-side (input) dilation factors. Default ``None``.
    rhs_dilation : Sequence[int] or None, optional
        Right-hand-side (kernel) dilation factors. Default ``None``.
    feature_group_count : int, optional
        Number of feature groups. Default ``1``.
    batch_group_count : int, optional
        Number of batch groups. Default ``1``.
    dimension_numbers : Any, optional
        Convolution dimension numbers (e.g. ``('NHWC', 'HWIO', 'NHWC')``).
        Default ``None``, which uses the JAX default layout.
    kernel_fn : callable or None, optional
        Optional transform applied to the kernel *inside* the primitive before
        the convolution, e.g. ``lambda w: w ** 2``. The derivative
        ``kernel_fn'`` is composed automatically via ``jax.vjp`` in the
        ``xy_to_dw`` rule so the eligibility trace is correct.
        The transform operates on the unitless mantissa; physical units are
        split off before and recombined after.
        Default ``None`` (identity, bit-identical to the pre-transform behaviour).
    bias_fn : callable or None, optional
        Optional transform applied to the bias *inside* the primitive before
        adding it to the output. Because the bias trace is per-position (the
        spatial summation is deferred to the solve-time contraction), the
        derivative ``bias_fn'(b)`` is applied as an explicit
        per-output-channel factor in the instantaneous rules (``xy_to_dw``
        and the D-RTRL ``instant`` rule).
        **``bias_fn`` must be an elementwise (per-channel) map**; the bias
        gradient is recovered as a per-channel Jacobian-diagonal factor via
        ``jax.vjp(bias_fn, b)(ones)`` and is therefore exact only when the
        Jacobian is diagonal.  Non-elementwise bias transforms (e.g. a
        per-channel softmax that couples channels) are not supported.  By
        contrast, ``kernel_fn`` is unrestricted — it goes through a full
        ``jax.vjp`` over the entire kernel.
        The transform operates on the unitless mantissa; physical units are
        split off before and recombined after.
        Default ``None`` (identity).

    Returns
    -------
    ArrayLike
        Convolution output tensor.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>>
        >>> brainstate.environ.set(precision=64)
        >>> # 1-D conv, NCH input and OIH kernel (JAX defaults)
        >>> x = brainstate.random.randn(8, 3, 16)
        >>> kernel = brainstate.random.randn(4, 3, 5)
        >>> y = braintrace.conv(x, kernel, strides=(1,), padding='SAME')
        >>> print(y.shape)
        (8, 4, 16)
        >>>
        >>> # Apply a kernel transform (squares each weight before conv)
        >>> y2 = braintrace.conv(x, kernel, strides=(1,), padding='SAME',
        ...                      kernel_fn=lambda w: w ** 2)
        >>> print(y2.shape)
        (8, 4, 16)
    """
    conv_kwargs = dict(
        strides=tuple(strides),
        padding=padding,
        lhs_dilation=tuple(lhs_dilation) if lhs_dilation is not None else None,
        rhs_dilation=tuple(rhs_dilation) if rhs_dilation is not None else None,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        dimension_numbers=dimension_numbers,
        kernel_fn=kernel_fn,
        bias_fn=bias_fn,
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
