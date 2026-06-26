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
The structure object must implement ``with_data`` (substitute new data
into the structure) and ``yw_to_w_transposed`` (apply the transposed
sparse pattern to a trace).

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
"""

import jax
import jax.numpy as jnp
import brainunit as u

from ._primitive import register_primitive

__all__ = [
    'etp_sp_mm_p',
    'etp_sp_mv_p',
    'sparse_matmul',
]


def _etp_sp_matmul_impl(*args, sparse_mat=None, has_bias=False):
    x, weight_data = args[0], args[1]
    w = sparse_mat.with_data(weight_data)
    y = x @ w
    if has_bias:
        y = y + args[2]
    return y


# ---------------------------------------------------------------------------
# trainable_invars_fn — shared by both mm and mv
# ---------------------------------------------------------------------------

def _sp_trainable_invars(params):
    """Return ``{key: invar_index}`` depending on ``has_bias``."""
    base = {'weight': 1}
    if params.get('has_bias', False):
        base['bias'] = 2
    return base


# ---------------------------------------------------------------------------
# etp_sp_mm_p — batched
# ---------------------------------------------------------------------------

def _sp_mm_yw_to_w(hidden_dim, trace, *, sparse_mat=None, has_bias=False):
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
    """
    out = {'weight': sparse_mat.yw_to_w_transposed(hidden_dim, trace['weight'])}
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _sp_xy_to_dw(x, hidden_dim, weights, *, sparse_mat=None, has_bias=False):
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

    Bias gradient: identical to dense,
    :math:`\partial h / \partial b = \partial h / \partial y`.

    Both weight and bias pullbacks are fused into one ``jax.vjp`` over a
    dict-valued forward function.
    """

    def _fwd(w_dict):
        y = x @ sparse_mat.with_data(w_dict['weight'])
        if has_bias:
            y = y + w_dict['bias']
        return u.get_mantissa(y)

    _, vjp_fn = jax.vjp(_fwd, weights)
    return jax.tree.map(u.get_mantissa, vjp_fn(hidden_dim)[0])


def _sp_mm_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    r"""Initialise batched sparse D-RTRL trace.

    The memory advantage of sparse vs dense lives here:

    .. math::

        \boldsymbol{\epsilon}_w \in \mathbb{R}^{B \times nnz \times n_{\text{state}}}, \qquad
        \boldsymbol{\epsilon}_b \in \mathbb{R}^{B \times O \times n_{\text{state}}}.

    ``nnz`` can be orders of magnitude smaller than :math:`I \cdot O`
    for typical connectivity matrices. Zero-initialised.
    """
    batch = x_var.aval.shape[0]
    nnz = weight_vars['weight'].aval.shape[0]
    out = {'weight': jnp.zeros((batch, nnz, num_hidden_state))}
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros(
            (batch, *weight_vars['bias'].aval.shape, num_hidden_state)
        )
    return out


def _sp_mm_init_pp(x_var, y_var, weight_vars, num_hidden_state):
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

def _sp_mv_yw_to_w(hidden_dim, trace, *, sparse_mat=None, has_bias=False):
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
    out = {'weight': sparse_mat.yw_to_w_transposed(hidden_dim, trace['weight'])}
    if has_bias:
        out['bias'] = trace['bias'] * hidden_dim
    return out


def _sp_mv_init_drtrl(x_var, y_var, weight_vars, num_hidden_state):
    r"""Initialise unbatched sparse D-RTRL trace.

    .. math::

        \boldsymbol{\epsilon}_w \in \mathbb{R}^{nnz \times n_{\text{state}}}, \qquad
        \boldsymbol{\epsilon}_b \in \mathbb{R}^{O \times n_{\text{state}}}.

    Zero-initialised.
    """
    nnz = weight_vars['weight'].aval.shape[0]
    out = {'weight': jnp.zeros((nnz, num_hidden_state))}
    if 'bias' in weight_vars:
        out['bias'] = jnp.zeros(
            (*weight_vars['bias'].aval.shape, num_hidden_state)
        )
    return out


def _sp_mv_init_pp(x_var, y_var, weight_vars, num_hidden_state):
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


def sparse_matmul(x, weight_data, *, sparse_mat, bias=None):
    r"""ETP-aware sparse matrix multiplication.

    Computes :math:`y = x \mathbin{@} \mathrm{sparse}(w) \; (+ b)`, where
    only the non-zero entries (``weight_data``) of the fixed sparse pattern
    are trainable and participate in eligibility-trace computation.
    Auto-dispatches batched/unbatched based on ``x.ndim``.

    Parameters
    ----------
    x : ArrayLike
        Input array.
    weight_data : ArrayLike
        Sparse-matrix data, i.e. the non-zero values, shape ``(nnz,)``.
    sparse_mat : object
        Sparse-matrix structure (e.g. a ``brainunit.sparse`` matrix object).
        Must expose ``with_data`` (substitute new data into the structure)
        and ``yw_to_w_transposed`` (apply the transposed sparse pattern to
        a trace).
    bias : ArrayLike or None, optional
        Bias vector. Default ``None``.

    Returns
    -------
    ArrayLike
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
