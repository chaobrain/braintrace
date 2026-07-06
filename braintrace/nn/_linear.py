# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Callable, Optional, Union

import brainstate
import braintools
import brainunit as u

from braintrace._op import grouped_matmul, matmul, sparse_matmul, lora_matmul
from braintrace._typing import ArrayLike

__all__ = [
    'Linear',
    'GroupedLinear',
    'SignedWLinear',
    'ScaledWSLinear',
    'SparseLinear',
    'LoRA',
]


def _fold_leading_axes(x: ArrayLike) -> tuple[ArrayLike, Callable[[ArrayLike], ArrayLike]]:
    """Fold all leading axes of ``x`` into a single batch axis.

    The rank-guarded ETP ops (:func:`braintrace.matmul`,
    :func:`braintrace.lora_matmul`) accept only rank-1/rank-2 inputs, while
    the nn layers document an ``(..., in_size)`` contract.  This helper
    bridges the two: for ``x.ndim > 2`` it returns the rank-2 view
    ``x.reshape(-1, in_size)`` plus an ``unfold`` callable that restores the
    original leading axes on the op output; for ``x.ndim <= 2`` it returns
    ``x`` unchanged and an identity ``unfold``, so the rank-1/rank-2 paths
    are untouched.

    ``reshape`` goes through ``brainunit`` so physical-unit quantities keep
    their units on both the fold and the unfold.

    Parameters
    ----------
    x : ArrayLike
        Input array (or unit-carrying quantity) of shape ``(..., in_size)``.

    Returns
    -------
    tuple
        ``(x2, unfold)`` where ``x2`` has shape ``(prod(leading), in_size)``
        (or is ``x`` itself when ``x.ndim <= 2``) and ``unfold`` maps an
        output of shape ``(prod(leading), out_size)`` back to
        ``(*leading, out_size)``.
    """
    if x.ndim <= 2:  # type: ignore[union-attr]  # scalars never reach here: callers feed (..., in_size)
        return x, lambda y: y
    lead_shape = x.shape[:-1]  # type: ignore[union-attr]
    x2 = u.math.reshape(x, (-1, x.shape[-1]))  # type: ignore[union-attr]

    def unfold(y: ArrayLike) -> ArrayLike:
        return u.math.reshape(y, (*lead_shape, y.shape[-1]))  # type: ignore[union-attr]

    return x2, unfold


class Linear(brainstate.nn.Linear):
    __module__ = 'braintrace.nn'
    __doc__ = (brainstate.nn.Linear.__doc__ or '').replace('brainstate', 'braintrace')

    def update(self, x: ArrayLike) -> ArrayLike:
        """Apply the linear transform through the ETP ``matmul`` primitive.

        Routing the matrix multiplication through :func:`braintrace.matmul`
        (instead of a plain JAX dot) is what makes ``weight`` eligible for
        online-learning trace computation.

        Parameters
        ----------
        x : ArrayLike
            Input array, of shape ``(..., in_size)``.

        Returns
        -------
        ArrayLike
            The transformed output, of shape ``(..., out_size)``.
        """
        w = self.weight.value['weight']
        b = self.weight.value.get('bias')
        x2, unfold = _fold_leading_axes(x)
        if self.w_mask is not None:
            mask = self.w_mask
            return unfold(matmul(x2, w, b, weight_fn=lambda ww: ww * mask))
        return unfold(matmul(x2, w, b))


class SignedWLinear(brainstate.nn.SignedWLinear):
    __module__ = 'braintrace.nn'
    __doc__ = (brainstate.nn.SignedWLinear.__doc__ or '').replace('brainstate', 'braintrace')

    def update(self, x: ArrayLike) -> ArrayLike:
        """Apply the sign-constrained linear transform through ETP ``matmul``.

        The stored weight magnitudes are made non-negative and then given a
        fixed sign before being routed through :func:`braintrace.matmul`, so
        the weight participates in online-learning trace computation.

        Parameters
        ----------
        x : ArrayLike
            Input array, of shape ``(..., in_size)``.

        Returns
        -------
        ArrayLike
            The transformed output, of shape ``(..., out_size)``.
        """
        w = self.weight.value
        sign = self.w_sign
        x2, unfold = _fold_leading_axes(x)
        if sign is not None:
            return unfold(matmul(x2, w, weight_fn=lambda ww: u.math.abs(ww) * sign))
        return unfold(matmul(x2, w, weight_fn=lambda ww: u.math.abs(ww)))


class ScaledWSLinear(brainstate.nn.ScaledWSLinear):
    __module__ = 'braintrace.nn'
    __doc__ = (brainstate.nn.ScaledWSLinear.__doc__ or '').replace('brainstate', 'braintrace')

    def update(self, x: ArrayLike) -> ArrayLike:
        """Apply the weight-standardized linear transform through ETP ``matmul``.

        Weight standardization (and the optional mask) are applied inside
        ``weight_fn``, which closes over the current ``eps`` value only.
        Routing the transform through :func:`braintrace.matmul` with
        ``weight_fn=`` causes the ETP compiler to track the gradient w.r.t. the
        raw ``weight`` leaf exactly (the standardization Jacobian is recovered
        via ``jax.vjp``).

        Note on post-ops: ``gain`` and ``bias`` are applied OUTSIDE the
        matmul primitive as post-operations, so the eligibility trace tracks
        only the standardized ``weight`` leaf.  ``gain`` and ``bias`` are
        therefore non-temporal for the eligibility trace — in genuine online
        training their trace-based gradient is partial.  (The multi-step VJP
        *oracle* path used in tests, which autodiffs through the full rollout,
        still recovers them exactly; only the online eligibility-trace gradient
        is non-temporal.)

        Parameters
        ----------
        x : ArrayLike
            Input array, of shape ``(..., in_size)``.

        Returns
        -------
        ArrayLike
            The transformed output, of shape ``(..., out_size)``.
        """
        params = self.weight.value
        eps = self.eps
        gain = params.get('gain', None)
        mask = self.w_mask

        # ``gain`` is a trainable ParamState leaf that is *non-temporal* for the
        # eligibility trace: it reaches the hidden state only through the
        # standardized weight map, so its online gradient will not match BPTT
        # exactly.  To avoid a JAX tracer-leak (JAX forbids closing over traced
        # state values in static primitive parameters), we apply weight
        # standardization *without* gain inside ``weight_fn`` and multiply by
        # gain *after* the matmul.  This is mathematically equivalent because
        #   x @ (std(w) * gain) == (x @ std(w)) * gain
        # when gain has shape (1, out_size), and the ETP trace's hidden_dim
        # already carries the gain factor through the hidden-to-output Jacobian.
        def _wfn(ww: ArrayLike) -> ArrayLike:
            w = brainstate.nn.weight_standardization(ww, eps, None)
            if mask is not None:
                w = w * mask
            return w

        b = params.get('bias', None)
        # Do NOT pass bias into matmul; add it after gain so that bias is not
        # scaled by gain.  Correct forward: (x @ std(w)*mask) * gain + b.
        x2, unfold = _fold_leading_axes(x)
        result = unfold(matmul(x2, params['weight'], weight_fn=_wfn))
        if gain is not None:
            result = result * gain
        if b is not None:
            result = result + b
        return result


class SparseLinear(brainstate.nn.SparseLinear):
    __module__ = 'braintrace.nn'
    __doc__ = (brainstate.nn.SparseLinear.__doc__ or '').replace('brainstate', 'braintrace')

    def update(self, x: ArrayLike) -> ArrayLike:
        """Apply the sparse linear transform through the ETP ``sparse_matmul``.

        The dense data of the sparse weight is routed through
        :func:`braintrace.sparse_matmul`, so it participates in
        online-learning trace computation.

        Parameters
        ----------
        x : ArrayLike
            Input array, of shape ``(..., in_size)``.

        Returns
        -------
        ArrayLike
            The transformed output, of shape ``(..., out_size)``.
        """
        weight = self.weight.value['weight']
        bias = self.weight.value.get('bias', None)
        return sparse_matmul(x, weight, sparse_mat=self.spar_mat, bias=bias)


class LoRA(brainstate.nn.LoRA):
    r"""A standalone LoRA layer.

    LoRA (Low-Rank Adaptation) injects two low-rank factors into a layer
    so a large pre-trained model can be fine-tuned with far fewer
    parameters. This subclass preserves the upstream
    :class:`brainstate.nn.LoRA` constructor and replaces only the forward
    pass so that the multiplication is routed through
    :func:`braintrace.lora_matmul` and therefore participates in
    eligibility-trace computation.

    The layer adds a low-rank component :math:`\frac{1}{r} B A` to the
    base weight, where :math:`B` and :math:`A` are learnable factors of
    rank :math:`r`:

    .. math::

        W_{\mathrm{LoRA}} = W_{\text{orig}} + \frac{1}{r} B A

    The scaling factor is fixed to ``1 / lora_rank``.

    Parameters
    ----------
    in_features : int
        Number of input features.
    lora_rank : int
        Rank of the LoRA decomposition.
    out_features : int
        Number of output features.
    base_module : brainstate.nn.Module or None, optional
        Optional base layer that is called on ``x`` and added to the LoRA
        branch. Default ``None``.
    kernel_init : Callable or ArrayLike, optional
        Initializer used for **both** ``lora_a`` (in×rank) and ``lora_b``
        (rank×out). Default is ``LecunNormal()``. To get the classic
        "LoRA-zero" initialisation use ``init.ZeroInit()``.
    param_type : type, optional
        ``ParamState`` subclass used to wrap the weights. Default is
        ``brainstate.ParamState``.
    in_size : int or Sequence[int], optional
        Optional explicit input size override. Default ``None``.

    Attributes
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    base_module : brainstate.nn.Module or None
        The optional base layer added to the LoRA branch.
    weight : ParamState
        ``ParamState`` whose value is a dict with two keys: ``'lora_a'``
        of shape ``(in_features, lora_rank)`` and ``'lora_b'`` of shape
        ``(lora_rank, out_features)``.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>>
        >>> # Create a standalone LoRA layer
        >>> brainstate.environ.set(precision=64)
        >>> layer = braintrace.nn.LoRA(in_features=3, lora_rank=2, out_features=4)
        >>> x = brainstate.random.randn(16, 3)
        >>> y = layer(x)
        >>> print(y.shape)
        (16, 4)
        >>>
        >>> # Wrap around an existing linear layer
        >>> linear = brainstate.nn.Linear(3, 4)
        >>> wrapper = braintrace.nn.LoRA(3, 2, 4, base_module=linear)
        >>> assert wrapper.base_module is linear
        >>> y = wrapper(x)
        >>> print(y.shape)
        (16, 4)
    """
    __module__ = 'braintrace.nn'

    def update(self, x: ArrayLike) -> ArrayLike:
        r"""Apply the low-rank adaptation through the ETP ``lora_matmul``.

        Computes :math:`y = \frac{1}{r}\, x\, \mathbf{A}\, \mathbf{B}` via
        :func:`braintrace.lora_matmul`, where :math:`\mathbf{A}` is the
        input-facing factor ``lora_a`` of shape ``(in, rank)`` and
        :math:`\mathbf{B}` is the output-facing factor ``lora_b`` of shape
        ``(rank, out)`` (so the LoRA factors participate in online-learning
        trace computation), and adds the optional base-module output.

        Parameters
        ----------
        x : ArrayLike
            Input array, of shape ``(..., in_features)``.

        Returns
        -------
        ArrayLike
            The adapted output, of shape ``(..., out_features)``.
        """
        param = self.weight.value
        lora_rank = param['lora_b'].shape[0]
        # ``lora_a`` is the input-facing ``(in, rank)`` factor and must be
        # applied before ``lora_b`` the ``(rank, out)`` factor:
        # ``y = alpha * x @ lora_a @ lora_b``.
        x2, unfold = _fold_leading_axes(x)
        out = unfold(lora_matmul(x2, param['lora_a'], param['lora_b'], alpha=1.0 / lora_rank))
        if self.base_module is not None:
            if not callable(self.base_module):
                raise ValueError('`self.base_module` must be callable.')
            out += self.base_module(x)
        return out

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Route the forward pass through the ETP-aware :meth:`update`.

        The upstream :class:`brainstate.nn.LoRA` defines its own ``__call__``
        that computes the low-rank product directly via plain matmuls,
        bypassing ``update``. Overriding it here ensures ``layer(x)``
        dispatches to the ETP-routed :meth:`update`, so the LoRA factors
        participate in eligibility-trace computation.

        See Also
        --------
        update : The ETP-routed forward pass; documents parameters and returns.
        """
        return self.update(x)


class GroupedLinear(brainstate.nn.Module):
    r"""Block-diagonal (grouped) linear layer backed by ETP :func:`braintrace.grouped_matmul`.

    Applies ``num_groups`` independent ``in_features → out_features`` linear
    maps — equivalent to one dense layer with a block-diagonal weight matrix,
    but with a ``num_groups×`` smaller D-RTRL eligibility trace.

    Parameters
    ----------
    num_groups : int
        Number of independent blocks ``G``.
    in_features : int
        Input features per block ``K``.
    out_features : int
        Output features per block ``N``.
    w_init : Callable or ArrayLike, optional
        Weight initializer for the ``(G, K, N)`` block weights.
    b_init : Callable or ArrayLike or None, optional
        Bias initializer for the ``(G, N)`` per-block bias; ``None`` disables
        the bias.
    name : str, optional
        Module name.
    param_type : type, optional
        The ParamState subclass holding the parameter dict.

    Notes
    -----
    The underlying op is rank-guarded to ``x.ndim ∈ {2, 3}``; this layer
    bridges the documented ``(..., G, K)`` contract by folding extra leading
    axes into a single batch axis and unfolding the output.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintrace
        >>> layer = braintrace.nn.GroupedLinear(4, 8, 8)
        >>> y = layer(jnp.ones((16, 4, 8)))
        >>> print(y.shape)
        (16, 4, 8)
    """
    __module__ = 'braintrace.nn'

    def __init__(
        self,
        num_groups: int,
        in_features: int,
        out_features: int,
        w_init: Union[ArrayLike, Callable] = braintools.init.KaimingNormal(),
        b_init: Optional[Union[ArrayLike, Callable]] = braintools.init.ZeroInit(),
        name: str | None = None,
        param_type: type = brainstate.ParamState,
    ) -> None:
        super().__init__(name=name)  # type: ignore[call-arg]  # brainstate hides Module.__init__ from type checkers
        self.num_groups = num_groups
        self.in_features = in_features
        self.out_features = out_features
        self.in_size = (num_groups, in_features)
        self.out_size = (num_groups, out_features)
        params = {
            'weight': braintools.init.param(
                w_init, (num_groups, in_features, out_features), allow_none=False)
        }
        if b_init is not None:
            params['bias'] = braintools.init.param(
                b_init, (num_groups, out_features), allow_none=False)
        self.weight = param_type(params)

    def update(self, x: ArrayLike) -> ArrayLike:
        """Apply the grouped linear transform through ETP ``grouped_matmul``.

        Parameters
        ----------
        x : ArrayLike
            Input array, of shape ``(..., num_groups, in_features)``.

        Returns
        -------
        ArrayLike
            The transformed output, of shape ``(..., num_groups, out_features)``.
        """
        params = self.weight.value
        if x.ndim <= 3:  # type: ignore[union-attr]  # callers feed (..., G, K)
            return grouped_matmul(x, params['weight'], params.get('bias'))
        # the rank-guarded op accepts only (G, K) / (batch, G, K); fold extra
        # leading axes into one batch axis, unfold on the output (reshape via
        # brainunit so quantities keep their units)
        lead_shape = x.shape[:-2]  # type: ignore[union-attr]
        x2 = u.math.reshape(x, (-1, *x.shape[-2:]))  # type: ignore[union-attr]
        y = grouped_matmul(x2, params['weight'], params.get('bias'))
        return u.math.reshape(y, (*lead_shape, *y.shape[-2:]))  # type: ignore[union-attr]
