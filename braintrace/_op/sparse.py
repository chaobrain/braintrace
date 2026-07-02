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

r"""Sparse-matmul ETP primitives.

``etp_sp_mm_p`` (batched) and ``etp_sp_mv_p`` (unbatched). The sparse
structure is supplied as a static parameter (``sparse_mat``); only the
non-zero values flow through the primitive as the ``weight_data`` invar.
The structure object must be a :class:`brainevent.DataRepresentation`,
which provides the ETP online-learning protocol: ``with_data`` (substitute
new data into the structure), ``yw_to_w_transposed`` (apply the transposed
sparse pattern to a trace) and ``yw_to_w`` (the non-transposed counterpart).

**Forward operation**

Let :math:`W = \mathrm{sparse}(w_{\text{data}})` denote the dense matrix
obtained by placing the vector :math:`w_{\text{data}} \in \mathbb{R}^{nnz}`
into the fixed sparse pattern stored in ``sparse_mat``. The forward op
is just dense matmul over the materialised representation:

.. math::

    y = x\, W \;(+ b), \qquad
    W = \mathrm{sparse}(w_{\text{data}}).

Only the nnz non-zero entries are trainable; the structural zeros are
frozen.

**Role of each ETP rule**

* ``xy_to_dw(x, hidden_dim, weights)`` — pullback of :math:`y = x\,\mathrm{sparse}(w) + b`
  by :math:`\jax.vjp`. Sparse-aware: the VJP natively restricts the
  Jacobian to the nnz-entries, returning :math:`\partial h/\partial w_{\text{data}} \in \mathbb{R}^{nnz}`.
  This is the instantaneous
  :math:`\operatorname{diag}(\mathbf{D}_f^t)\otimes \mathbf{x}^t` term
  for D-RTRL, projected onto the sparse support.

* ``yw_to_w(hidden_dim, trace)`` — propagation of
  :math:`\mathbf{D}^t \boldsymbol{\epsilon}^{t-1}`. For the weight data,
  delegates to ``sparse_mat.yw_to_w_transposed``: this contracts
  ``hidden_dim`` along ``out`` and restricts to the sparse pattern in a
  single kernel call — equivalent to computing the dense
  :math:`(\partial h/\partial y) \cdot \mathrm{scatter}^{\top}` but only
  touching the nnz entries.

* ``init_drtrl`` — nnz-dimensional trace
  :math:`\boldsymbol{\epsilon}_w \in \mathbb{R}^{nnz \times n_{\text{state}}}`
  (plus bias trace) instead of :math:`I \times O`; this is the whole
  point of ``etp_sp_*`` — ETP memory scales with ``nnz`` not
  :math:`I \cdot O`.

* ``init_pp`` — output-shaped df trace, identical to the dense case
  (pp-prop factorises :math:`\boldsymbol{\epsilon} \approx \boldsymbol{\epsilon}_f \otimes \boldsymbol{\epsilon}_x`
  and the :math:`\boldsymbol{\epsilon}_f` side is output-shaped
  regardless of how :math:`W` is stored).

**Dict rule API (N-trainable-input refactor)**

Both primitives declare ``trainable_invars_fn``, which returns
``{'weight': 1}`` when ``has_bias=False`` and ``{'weight': 1, 'bias': 2}``
when ``has_bias=True``. The four ETP rules accept / return
``Dict[str, Array]`` instead of bare arrays so the executor can route
gradients to *both* weight and bias ``ParamState`` objects in one pass.

When ``has_bias=False`` the ``'bias'`` key is simply absent from every
dict, so the legacy (no-bias) code path is unchanged in behaviour.

**Transform hooks**

Both primitives accept two optional elementwise transform hooks in their
``eqn.params``: ``weight_fn`` (computes ``y = x @ sparse(weight_fn(w_data))``)
and ``bias_fn`` (adds ``bias_fn(b)``). The forward impl and
:func:`_sp_xy_to_dw` apply them; the eligibility trace and gradient are
always taken w.r.t. the **raw** weight data / bias, so the transform
Jacobian :math:`f'` enters *only* through ``xy_to_dw`` via :func:`jax.vjp`.
The ``yw_to_w`` rule and the trace initialisers are transform-free and stay
exact (they operate on :math:`\partial h / \partial w_{\text{raw}}`).

These primitives have **no fast path** — they always use the generic rule
path, which threads :math:`f'` correctly when a transform hook is present.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import brainunit as u
import brainevent

from ._primitive import register_primitive
from ._registries import register_batched_counterpart
from braintrace._typing import ArrayLike, WeightFn

__all__ = [
    'etp_sp_mm_p',
    'etp_sp_mv_p',
    'sparse_matmul',
]


class _HashableSparseMat:
    r"""Identity-hashable wrapper around a validated ``brainevent.DataRepresentation``.

    ``brainevent.DataRepresentation`` subclasses (e.g. :class:`brainevent.CSR`)
    define ``__eq__`` (structural/dense comparison) without a matching
    ``__hash__``, so binding one directly as a jaxpr equation parameter raises
    ``TypeError: parameters to jaxpr equations must have __hash__`` under
    JAX >= 0.7 — the documented ``sparse_matmul`` usage otherwise fails
    outright (audit finding H1).

    This wrapper carries the already-validated structure opaquely and relies
    on the inherited ``object.__hash__``/``object.__eq__`` (identity), which
    is exactly what a static jaxpr parameter needs: the *same* Python sparse
    object compares equal to itself (cache/jit-trace reuse), while two
    distinct sparse objects are correctly unequal, even if their dense forms
    happen to coincide.

    Attributes
    ----------
    mat : brainevent.DataRepresentation
        The wrapped, pre-validated sparse structure.
    """

    __slots__ = ('mat',)

    def __init__(self, mat: brainevent.DataRepresentation) -> None:
        self.mat = mat


def _unwrap_sparse_mat(sparse_mat: Any) -> Any:
    """Return the underlying sparse structure, unwrapping ``_HashableSparseMat``.

    ``sparse_matmul`` binds a ``_HashableSparseMat`` wrapper (see above) so the
    equation parameter is hashable. Every consumer of ``params['sparse_mat']``
    (the forward impl -- and therefore also the auto-derived abstract-eval,
    lowering, JVP and batching rules -- plus the ``yw_to_w``/``xy_to_dw`` ETP
    rules) must unwrap it before calling the sparse-matrix protocol methods.

    Direct (non-``sparse_matmul``) call sites -- e.g. the rule-level unit
    tests in this module and in ``op_rule_oracle_test.py`` -- pass an
    already-unwrapped ``DataRepresentation`` (or test stub) straight to the
    impl/rule functions; those are returned unchanged.
    """
    if isinstance(sparse_mat, _HashableSparseMat):
        return sparse_mat.mat
    return sparse_mat


def _etp_sp_matmul_impl(*args: Any,
                        sparse_mat: brainevent.DataRepresentation | None = None,
                        has_bias: bool = False,
                        weight_fn: WeightFn | None = None,
                        bias_fn: WeightFn | None = None) -> Any:
    x, weight_data = args[0], args[1]
    mat = _unwrap_sparse_mat(sparse_mat)
    if weight_fn is not None:
        weight_data = weight_fn(weight_data)
    w = mat.with_data(weight_data)  # type: ignore[union-attr]  # sparse_mat is always supplied at bind time
    y = x @ w
    if has_bias:
        b = args[2]
        if bias_fn is not None:
            b = bias_fn(b)
        y = y + b
    return y


# ---------------------------------------------------------------------------
# trainable_invars_fn — shared by both mm and mv
# ---------------------------------------------------------------------------

def _sp_trainable_invars(params: dict[str, Any]) -> dict[str, int]:
    """Return ``{key: invar_index}`` depending on ``has_bias``."""
    base = {'weight': 1}
    if params.get('has_bias', False):
        base['bias'] = 2
    return base


# ---------------------------------------------------------------------------
# etp_sp_mm_p — batched
# ---------------------------------------------------------------------------

def _sp_mm_yw_to_w(hidden_dim: Any, trace: dict[str, Any], *,
                   sparse_mat: brainevent.DataRepresentation | None = None,
                   has_bias: bool = False, weight_fn: WeightFn | None = None,
                   bias_fn: WeightFn | None = None) -> dict[str, Any]:
    r"""Batched sparse ``yw_to_w`` — propagate :math:`\partial h / \partial y`
    through the nnz-shaped D-RTRL trace.

    **Role in D-RTRL.** Implements the :math:`y \to w_{\text{data}}` chain
    factor inside :math:`\mathbf{D}^t \boldsymbol{\epsilon}^{t-1}`. For
    the dense-equivalent :math:`y_j = \sum_i x_i W_{ij}` we would write
    :math:`\partial y_j / \partial W_{ik} = \delta_{jk} x_i`. Restricted
    to the sparse support, only positions with
    :math:`(i, j) \in \mathrm{pattern}` are kept; ``yw_to_w_transposed``
    performs the contraction and scatter-restrict in one sparse kernel:

    .. math::

        \epsilon^{t}_{w, b, p} \;=\;
          \sum_j (\partial h / \partial y)_{b, j}\,
                 \epsilon^{t-1}_{w, b, p}\,
                 \mathbb{1}[\mathrm{col}(p) = j],

    for each nnz index :math:`p`.

    **Bias**: :math:`y_j = \dots + b_j` ⇒
    :math:`\partial y_j / \partial b_k = \delta_{jk}`, so the bias-trace
    propagation is the familiar elementwise product — just like dense
    matmul.

    **Shapes.**
        scan context: ``hidden_dim : (batch, out)``,
                      ``trace['weight'] : (batch, nnz)``,
                      ``trace['bias']   : (batch, out)``.
        solve context: batch axis dropped by the outer vmap.

    **Batching (audit C3).** The scan-context call above hands this rule a
    2-D ``hidden_dim``/``trace['weight']`` pair straight from the online
    trace-recurrence update (no outer vmap — unlike the solve context).
    ``brainevent``'s ``yw_to_w_transposed`` kernel (``csrmv_yw2y_p_call``)
    only accepts 1-D operands, so when ``hidden_dim.ndim == 2`` this rule
    ``jax.vmap``\ s the sparse call over the leading batch axis of both
    operands instead of handing it the batched arrays directly.
    """
    mat = _unwrap_sparse_mat(sparse_mat)
    weight_trace = trace['weight']
    if hidden_dim.ndim == 2:
        # (batch, out), (batch, nnz) -> vmap the 1-D-only brainevent kernel
        # over the leading batch axis.
        weight_out = jax.vmap(mat.yw_to_w_transposed)(hidden_dim, weight_trace)
    else:
        weight_out = mat.yw_to_w_transposed(hidden_dim, weight_trace)  # type: ignore[union-attr]  # sparse_mat is always supplied at bind time
    out = {'weight': weight_out}
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _sp_xy_to_dw(x: Any, hidden_dim: Any, weights: dict[str, Any], *,
                 sparse_mat: brainevent.DataRepresentation | None = None,
                 has_bias: bool = False, weight_fn: WeightFn | None = None,
                 bias_fn: WeightFn | None = None) -> dict[str, Any]:
    r"""Sparse instantaneous Jacobian :math:`\partial h / \partial w_{\text{data}}`,
    and :math:`\partial h / \partial b`.

    **Role.** Gives the
    :math:`\operatorname{diag}(\mathbf{D}_f^t)\otimes \mathbf{x}^t` term
    of D-RTRL (and the solve-time factor of ES-D-RTRL) restricted to the
    sparse support. The chain rule gives

    .. math::

        \frac{\partial h}{\partial w_p}
          \;=\; x_{\mathrm{row}(p)} \cdot
                \Bigl(\frac{\partial h}{\partial y}\Bigr)_{\mathrm{col}(p)},

    for each nnz index :math:`p`. :func:`jax.vjp` of
    ``sparse_mat.with_data`` returns exactly this nnz-shaped gradient —
    the zeros outside the pattern are never materialised.

    When ``weight_fn`` is provided, the transform is applied inside ``_fwd``
    so that ``jax.vjp`` auto-composes the chain rule: the gradient is taken
    w.r.t. the **raw** data, not the transformed data. Likewise for
    ``bias_fn``.

    Bias gradient: identical to dense,
    :math:`\partial h / \partial b = \partial h / \partial y`.

    Both weight and bias pullbacks are fused into one ``jax.vjp`` over a
    dict-valued forward function.
    """
    mat = _unwrap_sparse_mat(sparse_mat)

    def _fwd(w_dict: dict[str, Any]) -> Any:
        wd = w_dict['weight']
        if weight_fn is not None:
            wd = weight_fn(wd)
        y = x @ mat.with_data(wd)  # type: ignore[union-attr]  # sparse_mat is always supplied at bind time
        if has_bias:
            b = w_dict['bias']
            if bias_fn is not None:
                b = bias_fn(b)
            y = y + b
        return u.get_mantissa(y)

    _, vjp_fn = jax.vjp(_fwd, weights)
    return jax.tree.map(u.get_mantissa, vjp_fn(hidden_dim)[0])


def _sp_mm_init_drtrl(x_var: Any, y_var: Any, weight_vars: dict[str, Any],
                      num_hidden_state: int) -> dict[str, Any]:
    r"""Initialise batched sparse D-RTRL trace.

    The memory advantage of sparse vs dense lives here:

    .. math::

        \boldsymbol{\epsilon}_w \in \mathbb{R}^{B \times nnz \times n_{\text{state}}}, \qquad
        \boldsymbol{\epsilon}_b \in \mathbb{R}^{B \times O \times n_{\text{state}}}.

    ``nnz`` can be orders of magnitude smaller than :math:`I \cdot O`
    for typical connectivity matrices. Zero-initialised.

    The trace dtype is derived from the participating ``x``/``y``/weight
    avals via :func:`jax.numpy.result_type` rather than left to ``jnp.zeros``'
    default (which silently follows the global x64 flag instead of the
    operands' actual dtype).
    """
    batch = x_var.aval.shape[0]
    nnz = weight_vars['weight'].aval.shape[0]
    dtype = jnp.result_type(
        x_var.aval.dtype, y_var.aval.dtype,
        *(v.aval.dtype for v in weight_vars.values()),
    )
    out = {'weight': jnp.zeros((batch, nnz, num_hidden_state), dtype=dtype)}
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros(
            (batch, *weight_vars['bias'].aval.shape, num_hidden_state), dtype=dtype
        )
    return out


def _sp_mm_init_pp(x_var: Any, y_var: Any, weight_vars: dict[str, Any],
                   num_hidden_state: int) -> Any:
    r"""Initialise batched sparse pp-prop / ES-D-RTRL df trace.

    .. math::

        \boldsymbol{\epsilon}_f \in \mathbb{R}^{B \times O \times n_{\text{state}}}.

    Output-shaped — same as the dense case. The :math:`\boldsymbol{\epsilon}_x`
    factor is the raw dense input :math:`x`, held by the executor.
    """
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


# ---------------------------------------------------------------------------
# etp_sp_mv_p — unbatched
# ---------------------------------------------------------------------------

def _sp_mv_yw_to_w(hidden_dim: Any, trace: dict[str, Any], *,
                   sparse_mat: brainevent.DataRepresentation | None = None,
                   has_bias: bool = False, weight_fn: WeightFn | None = None,
                   bias_fn: WeightFn | None = None) -> dict[str, Any]:
    r"""Unbatched sparse ``yw_to_w`` — identical algebra to the batched case
    with no batch axis.

    Propagates :math:`\partial h / \partial y` through the sparse pattern:

    .. math::

        \epsilon^t_{w, p} \;=\;
          \sum_j (\partial h / \partial y)_j\,
                 \epsilon^{t-1}_{w, p}\,
                 \mathbb{1}[\mathrm{col}(p) = j], \qquad
        \epsilon^t_{b, k} \;=\; (\partial h / \partial y)_k\, \epsilon^{t-1}_{b, k}.

    Shapes:  ``hidden_dim : (out,)``,
             ``trace['weight'] : (nnz,)``,
             ``trace['bias']   : (out,)``.
    """
    mat = _unwrap_sparse_mat(sparse_mat)
    out = {'weight': mat.yw_to_w_transposed(hidden_dim, trace['weight'])}  # type: ignore[union-attr]  # sparse_mat is always supplied at bind time
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _sp_mv_init_drtrl(x_var: Any, y_var: Any, weight_vars: dict[str, Any],
                      num_hidden_state: int) -> dict[str, Any]:
    r"""Initialise unbatched sparse D-RTRL trace.

    .. math::

        \boldsymbol{\epsilon}_w \in \mathbb{R}^{nnz \times n_{\text{state}}}, \qquad
        \boldsymbol{\epsilon}_b \in \mathbb{R}^{O \times n_{\text{state}}}.

    Zero-initialised.

    The trace dtype is derived from the participating ``x``/``y``/weight
    avals via :func:`jax.numpy.result_type` rather than left to ``jnp.zeros``'
    default (which silently follows the global x64 flag instead of the
    operands' actual dtype).
    """
    nnz = weight_vars['weight'].aval.shape[0]
    dtype = jnp.result_type(
        x_var.aval.dtype, y_var.aval.dtype,
        *(v.aval.dtype for v in weight_vars.values()),
    )
    out = {'weight': jnp.zeros((nnz, num_hidden_state), dtype=dtype)}
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros(
            (*weight_vars['bias'].aval.shape, num_hidden_state), dtype=dtype
        )
    return out


def _sp_mv_init_pp(x_var: Any, y_var: Any, weight_vars: dict[str, Any],
                   num_hidden_state: int) -> Any:
    r"""Initialise unbatched sparse pp-prop / ES-D-RTRL df trace.

    .. math::

        \boldsymbol{\epsilon}_f \in \mathbb{R}^{O \times n_{\text{state}}}.
    """
    return jnp.zeros((*y_var.aval.shape, num_hidden_state), dtype=y_var.aval.dtype)


# ---------------------------------------------------------------------------
# Primitive registration
# ---------------------------------------------------------------------------

etp_sp_mm_p = register_primitive(
    'etp_sp_mm',
    _etp_sp_matmul_impl,
    batched=True,
    trainable_invars_fn=_sp_trainable_invars,
    x_invar_index=0,
)
etp_sp_mm_p.register_etp_rules(
    yw_to_w=_sp_mm_yw_to_w,
    xy_to_dw=_sp_xy_to_dw,
    init_drtrl=_sp_mm_init_drtrl,
    init_pp=_sp_mm_init_pp,
)

etp_sp_mv_p = register_primitive(
    'etp_sp_mv',
    _etp_sp_matmul_impl,
    batched=False,
    trainable_invars_fn=_sp_trainable_invars,
    x_invar_index=0,
)
etp_sp_mv_p.register_etp_rules(
    yw_to_w=_sp_mv_yw_to_w,
    xy_to_dw=_sp_xy_to_dw,
    init_drtrl=_sp_mv_init_drtrl,
    init_pp=_sp_mv_init_pp,
)
register_batched_counterpart(etp_sp_mv_p, etp_sp_mm_p)


def sparse_matmul(
    x: ArrayLike,
    weight: ArrayLike,
    *,
    sparse_mat: brainevent.DataRepresentation,
    bias: ArrayLike | None = None,
    weight_fn: WeightFn | None = None,
    bias_fn: WeightFn | None = None,
) -> ArrayLike:
    r"""ETP-aware sparse matrix multiplication.

    Computes :math:`y = x \mathbin{@} \mathrm{sparse}(f(w)) \; (+ g(b))`, where
    only the non-zero entries (``weight``) of the fixed sparse pattern
    are trainable and participate in eligibility-trace computation.
    Auto-dispatches batched/unbatched based on ``x.ndim``.

    Parameters
    ----------
    x : ArrayLike
        Input array.
    weight : ArrayLike
        Sparse-matrix data, i.e. the non-zero values, shape ``(nnz,)``.
    sparse_mat : brainevent.DataRepresentation
        Sparse-matrix structure (e.g. a :class:`brainevent.CSR`). Must be a
        :class:`brainevent.DataRepresentation`, which implements the ETP
        online-learning protocol: ``with_data`` (substitute new data into the
        structure), ``yw_to_w_transposed`` (apply the transposed sparse
        pattern to a trace) and ``yw_to_w`` (its non-transposed counterpart).
        Passing any other object raises :class:`TypeError`.
    bias : ArrayLike or None, optional
        Bias vector. Default ``None``.
    weight_fn : callable or None, optional
        Elementwise transform applied to the non-zero ``weight`` data before
        the matmul.  ``None`` means identity (no transform).  When provided,
        the transform is applied *inside* the primitive so that
        ``xy_to_dw`` auto-composes the derivative via ``jax.vjp``, returning
        the gradient w.r.t. the **raw** data.
        The transform operates on the unitless mantissa; physical units are
        split off before and recombined after.
    bias_fn : callable or None, optional
        Elementwise transform applied to ``bias`` before it is added to the
        output.  ``None`` means identity.  The derivative is composed by the
        same ``jax.vjp`` call as ``weight_fn``.
        The transform operates on the unitless mantissa; physical units are
        split off before and recombined after.

    Returns
    -------
    ArrayLike
        Output array.

    Raises
    ------
    TypeError
        If ``sparse_mat`` is not a :class:`brainevent.DataRepresentation`.
    """
    if not isinstance(sparse_mat, brainevent.DataRepresentation):
        raise TypeError(
            'sparse_mat must be a brainevent.DataRepresentation providing the '
            'with_data, yw_to_w_transposed and yw_to_w online-learning protocol '
            f'methods, got {type(sparse_mat).__name__!r}.'
        )
    p = etp_sp_mm_p if x.ndim >= 2 else etp_sp_mv_p  # type: ignore[union-attr]  # x is an array here; ArrayLike also admits scalars without .ndim
    x_v, x_u = u.split_mantissa_unit(x)
    w_v, w_u = u.split_mantissa_unit(weight)
    unit = x_u * w_u
    # Bind an identity-hashable wrapper, not the DataRepresentation itself:
    # brainevent structures define __eq__ without __hash__, which JAX >= 0.7
    # rejects as a jaxpr equation parameter (audit finding H1).
    wrapped_mat = _HashableSparseMat(sparse_mat)
    if bias is not None:
        bias_v = u.Quantity(bias).to_decimal(unit)
        r = p.bind(x_v, w_v, bias_v, sparse_mat=wrapped_mat, has_bias=True,
                   weight_fn=weight_fn, bias_fn=bias_fn)
    else:
        r = p.bind(x_v, w_v, sparse_mat=wrapped_mat, has_bias=False,
                   weight_fn=weight_fn, bias_fn=bias_fn)
    return u.maybe_decimal(r * x_u * w_u)
