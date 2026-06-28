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

* ``xy_to_dw`` — for the *kernel*, uses the conv VJP to produce the full
  weight Jacobian :math:`\partial h / \partial K` (requires :math:`x`).
  For the *bias*, stores the per-position cotangent
  :math:`\partial h / \partial b_k = (\partial h / \partial y)_{b,\mathbf{s},k}`
  **without** summing over spatial positions. The spatial summation is
  deferred to ``yw_to_w`` during trace propagation — because the bias is
  spatially shared, the "true" bias gradient requires summing the
  cotangent along spatial axes, but doing the sum *inside* the trace
  rather than at the instantaneous step keeps the linear algebra
  consistent with the D-RTRL recurrence (the trace must accumulate
  :math:`\mathbf{D}^t \boldsymbol{\epsilon}^{t-1}` with the *same*
  spatial shape the executor feeds in).

* ``yw_to_w`` — applies :math:`\partial h / \partial y` to the trace.
  For the kernel: reduces ``hidden_dim`` over spatial axes
  (the kernel is spatially shared) and broadcasts the result across the
  kernel's spatial dims. For the bias: elementwise multiply with
  ``hidden_dim`` then sum over spatial axes — this is the deferred
  spatial reduction that implements :math:`\sum_{\mathbf{s}} \partial h/\partial b_k`.

* ``init_drtrl`` — allocates weight-shaped :math:`\boldsymbol{\epsilon}_K`
  plus per-position :math:`\boldsymbol{\epsilon}_b` with output-shape
  (kept spatial so the trace can accumulate before the deferred sum).

* ``init_pp`` — allocates the pp-prop output-shaped df trace; the
  :math:`\boldsymbol{\epsilon}_x` factor in ES-D-RTRL is the full
  batched input tensor held by the executor.
"""

from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
import brainunit as u

from ._primitive import register_primitive

__all__ = [
    'etp_conv_p',
    'conv',
]


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


def _conv_trainable_invars(params):
    """Return ``{key: invar_index}`` depending on ``has_bias``."""
    base = {'weight': 1}
    if params.get('has_bias', False):
        base['bias'] = 2
    return base


def _conv_layout(params):
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


def _conv_yw_to_w(hidden_dim, trace, **params):
    r"""Propagate :math:`\partial h / \partial y` through the conv trace.

    **Role in D-RTRL.** Implements the :math:`y \to (K, b)` chain factor
    in the :math:`\mathbf{D}^t \boldsymbol{\epsilon}^{t-1}` term. Unlike
    dense matmul, the kernel is *spatially shared*: every output position
    reads the same :math:`K`. Differentiating the forward equation gives

    .. math::

        \frac{\partial y_{b, \mathbf{s}, k}}{\partial K_{\mathbf{u}, c, k'}}
          \;=\; \delta_{k k'}\, x_{b,\, \mathbf{s}+\mathbf{u},\, c}, \qquad
        \frac{\partial y_{b, \mathbf{s}, k}}{\partial b_{k'}}
          \;=\; \delta_{k k'}.

    Pulling back a hidden cotangent :math:`g = \partial h / \partial y`
    therefore contracts :math:`\mathbf{s}` for the kernel and the bias
    (they are shared along spatial axes):

    .. math::

        \frac{\partial h}{\partial K_{\mathbf{u}, c, k}}
          \;=\; \sum_{\mathbf{s}} g_{b, \mathbf{s}, k}\, x_{b, \mathbf{s}+\mathbf{u}, c}, \qquad
        \frac{\partial h}{\partial b_k}
          \;=\; \sum_{\mathbf{s}} g_{b, \mathbf{s}, k}.

    Applied to a trace :math:`\boldsymbol{\epsilon}^{t-1}`, only the
    out-channel axis :math:`k` survives on the hidden-dim side of the
    product; spatial axes are summed. The kernel's spatial axes remain
    free (the trace stores one slot per kernel position), so after
    reducing :math:`g` over :math:`\mathbf{s}` we broadcast the resulting
    per-out-channel vector over the kernel's spatial and in-channel axes.

    **Two execution contexts (detected from ``hidden_dim.ndim``):**

    1. trace-update path (batch retained):
       ``hidden_dim   : (batch, *spatial_out, out_ch)`` (or permuted),
       ``trace['weight'] : (batch, *kernel_dims)`` (batch prefix present),
       ``trace['bias']   : (batch, *spatial_out, out_ch)``.

    2. gradient-solve path (outer batch-vmap strips batch):
       ``hidden_dim   : (*spatial_out, out_ch)`` (batch-free),
       ``trace['weight'] : (*kernel_dims)``,
       ``trace['bias']   : (*spatial_out, out_ch)``.

    **Bias gradient is deferred.** ``xy_to_dw`` stores the per-position
    cotangent as ``trace['bias']`` (no spatial sum). Here we complete the
    bias Jacobian by multiplying by :math:`g` and summing spatial axes —
    that deferral keeps the D-RTRL trace recurrence consistent
    (each :math:`\boldsymbol{\epsilon}^{t-1}` leaf has the same shape as
    :math:`(\partial h/\partial y)^{t-1}` before the reduction).

    **Layout awareness.** Spatial axes and the kernel out-channel axis
    are derived from ``dimension_numbers`` / ``strides``, handling NHWC /
    HWIO / NCHW / OIHW etc. The 0-D ``hidden_dim`` degenerate case is
    handled directly by elementwise multiply.
    """
    has_bias = params.get('has_bias', False)
    w_trace = trace['weight']

    if hidden_dim.ndim == 0:
        # Scalar (degenerate) case: multiply all trace entries elementwise.
        out = {'weight': w_trace * hidden_dim}
        if has_bias:
            out['bias'] = jnp.sum(trace['bias'] * hidden_dim)
        return out

    # ── Determine layout from params ──────────────────────────────────────────
    # Detect which call context we are in from hidden_dim rank:
    #   scan context:  hidden_dim.ndim == n_spatial + 2  (batch + spatial + ch)
    #   grad context:  hidden_dim.ndim == n_spatial + 1  (spatial + ch only)
    n_spatial, channel_axis_batched, batch_axis_batched, kernel_out_axis = _conv_layout(params)
    has_batch_prefix = (hidden_dim.ndim == n_spatial + 2)

    # Compute spatial axes in hidden_dim (same permutation as y output).
    if has_batch_prefix:
        # Axes in full output: {batch_axis_batched, channel_axis_batched} excluded.
        spatial_axes_hd = tuple(
            sorted(set(range(hidden_dim.ndim)) - {batch_axis_batched, channel_axis_batched})
        )
        # channel_axis in hidden_dim (same as in y).
        ch_axis_hd = channel_axis_batched
    else:
        # Batch axis is stripped.  Remaining axes: spatial + channel.
        # The original channel_axis_batched and batch_axis_batched are for the
        # batched layout.  After stripping the batch axis, remaining axes are
        # renumbered: remove batch_axis_batched from the set.
        all_axes = set(range(n_spatial + 2))
        remaining = sorted(all_axes - {batch_axis_batched})
        # remaining[i] is the original axis index; map to new (shifted) index.
        ch_axis_hd = remaining.index(channel_axis_batched)
        spatial_axes_hd = tuple(i for i in range(len(remaining)) if i != ch_axis_hd)

    # ── Weight: reduce hidden_dim over spatial axes, then broadcast ───────────
    # Target shape after reduction: only batch (if present) and ch_axis_hd survive.
    hd_reduced = jnp.sum(hidden_dim, axis=spatial_axes_hd) if spatial_axes_hd else hidden_dim
    # hd_reduced shape: (*batch_prefix, out_ch)  [only batch and ch axes survive]

    # Determine where out_ch sits in w_trace:
    #   scan context: w_trace has batch prefix at axis 0, so kernel_out_axis shifts by 1.
    #   grad context: w_trace has no batch prefix, use kernel_out_axis directly.
    w_out_axis = kernel_out_axis + 1 if has_batch_prefix else kernel_out_axis

    # Build broadcast shape: all-ones except batch (axis 0 when present) and out_ch axis.
    target_shape = [1] * w_trace.ndim
    if has_batch_prefix:
        target_shape[0] = w_trace.shape[0]  # batch size
    target_shape[w_out_axis] = w_trace.shape[w_out_axis]  # out_ch size
    hd_for_weight = jnp.reshape(hd_reduced, target_shape)

    out = {'weight': w_trace * hd_for_weight}

    # ── Bias: trace['bias'] has y-output shape; sum over spatial axes ─────────
    if has_bias:
        b_trace = trace['bias']
        # b_trace has same ndim as hidden_dim (same layout).
        b_sum_axes = spatial_axes_hd
        out['bias'] = jnp.sum(b_trace * hidden_dim, axis=b_sum_axes) if b_sum_axes else b_trace * hidden_dim
    return out


def _conv_xy_to_dw(x, hidden_dim, weights, **params):
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
    the sum is performed inside :func:`_conv_yw_to_w` during trace
    propagation — this keeps the bias-trace shape identical to
    :math:`\partial h / \partial y` so the :math:`\mathbf{D}^t \boldsymbol{\epsilon}^{t-1}`
    recurrence can be applied element-by-element.

    **Unbatched detection.** The D-RTRL executor vmaps over the batch
    axis, so ``x`` may arrive as ``ndim == n_spatial + 1`` (no batch).
    We prepend a leading axis before calling ``conv_general_dilated``,
    then strip it on the way out implicitly via the VJP.
    """
    has_bias = params.get('has_bias', False)
    # Build conv_general_dilated kwargs; remap 'strides' -> 'window_strides'.
    conv_kw = {}
    for k, v in params.items():
        if k == 'has_bias':
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

    # Kernel gradient via VJP (needs x).
    def _fwd_w(w):
        return u.get_mantissa(
            jax.lax.conv_general_dilated(x_in, w, **conv_kw)
        )

    _, vjp_fn = jax.vjp(_fwd_w, weights['weight'])
    dw = u.get_mantissa(vjp_fn(hd_in)[0])
    out = {'weight': dw}

    if has_bias:
        # Bias gradient = hidden_dim (cotangent at each output position).
        # No spatial summation — the trace stores per-position ∂h/∂b.
        out['bias'] = hidden_dim

    return out


def _conv_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    r"""Initialise conv D-RTRL weight-shaped trace.

    .. math::

        \boldsymbol{\epsilon}_K \in
          \mathbb{R}^{B \times \text{(kernel dims)} \times n_{\text{state}}}, \qquad
        \boldsymbol{\epsilon}_b \in
          \mathbb{R}^{B \times \text{(spatial out)} \times O \times n_{\text{state}}}.

    The bias trace intentionally keeps spatial dims so that
    :func:`_conv_yw_to_w` can apply the :math:`\partial h / \partial y`
    cotangent elementwise before collapsing them. The spatial sum
    implementing :math:`\sum_{\mathbf{s}} \partial h/\partial b_k` is
    performed on the trace-update side, not at the ``xy_to_dw`` step.

    Zero-initialised (matches :math:`\boldsymbol{\epsilon}^0 = \mathbf{0}`).
    """
    batch = x_var.aval.shape[0]
    out = {
        'weight': jnp.zeros(
            (batch, *weight_vars['weight'].aval.shape, num_hidden_state)
        )
    }
    if 'bias' in weight_vars:
        # y_var.aval.shape = (batch, *spatial, out_ch); strip the batch dim.
        out['bias'] = jnp.zeros(
            (batch, *y_var.aval.shape[1:], num_hidden_state)
        )
    return out


def _conv_init_pp(x_var, y_var, weight_vars, num_hidden_state):
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
    yw_to_w=_conv_yw_to_w,
    xy_to_dw=_conv_xy_to_dw,
    init_drtrl=_conv_init_drtrl,
    init_pp=_conv_init_pp,
)


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
    bias : ArrayLike or None, optional
        Per-output-channel bias. Default ``None``.
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
