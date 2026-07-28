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

"""SnAp-n — Sparse n-step Approximation (Menick et al., 2021).

SnAp-n interpolates between the cheap and the exact end of the
``recurrence_scope`` axis by *sparsifying the influence matrix*
:math:`\\mathbf{J} = \\partial \\mathbf{h} / \\partial \\boldsymbol{\\theta}`
to the sparsity pattern that the instantaneous term acquires after being
propagated ``n - 1`` further times through the transition operator
:math:`\\mathbf{D}`.

The scale is a genuine interpolation, and both of its endpoints already existed
in this codebase under different names:

- ``n = 1`` keeps only the influence a parameter has on the hidden units it
  *directly* drives. That is exactly ``recurrence_scope='coupled'`` — the
  per-position block-diagonal Jacobian that :class:`~braintrace.OSTLRecurrent`
  uses — so ``SnAp(model, n=1)`` canonicalises to that coordinate rather than
  introducing a second spelling of it.
- ``n`` at or above the hidden group's *position-graph diameter* saturates: the
  neighbourhood of every position is the whole group, and the rule becomes full
  within-group RTRL.

``recurrence_scope='diagonal'`` sits *below* this scale rather than on it: it
deletes the recurrent mixing primitive before differentiating (the e-prop /
RFLO regime), which is not any ``n``.

See :class:`SnAp` for the formulation, the limitations, and an example.
"""

from __future__ import annotations

from typing import Any, Optional

import brainstate

from braintrace._compiler import DEFAULT_MAX_JACOBIAN_ELEMENTS
from .axes import ETraceConfig
from .param_dim_vjp import ParamDimVjpAlgorithm

__all__ = ['SnAp']


class SnAp(ParamDimVjpAlgorithm):
    r"""Sparse n-step Approximation of RTRL.

    SnAp-:math:`n` [1]_ sparsifies the full influence matrix carried by RTRL
    [2]_. In the unapproximated construction, RTRL carries
    :math:`\mathbf{J}^t_{q,\theta} = \partial h^t_q / \partial \theta` and rolls
    it with

    .. math::

        \mathbf{J}^t = \mathbf{D}^t \mathbf{J}^{t-1} + \mathbf{J}_f^t ,
        \qquad
        \mathbf{D}^t = \frac{\partial \mathbf{h}^t}{\partial \mathbf{h}^{t-1}},
        \qquad
        (\mathbf{J}_f^t)_{q,\theta} = \frac{\partial h^t_q}{\partial \theta}
        \bigg|_{\mathbf{h}^{t-1}} ,

    which costs :math:`O(P \cdot |\theta|)` memory. SnAp-:math:`n` keeps the
    same recursion but *masks* :math:`\mathbf{J}` to the sparsity pattern

    .. math::

        \mathcal{S}_n \;=\; \operatorname{nz}\!\Bigl(
          \textstyle\bigvee_{k < n} \mathbf{A}^k \Bigr),

    where :math:`\mathbf{A}` is the one-step position-adjacency of the hidden
    group (position :math:`p` influences position :math:`q` in one step). Only
    the retained entries are stored and rolled, so the trace's trailing state
    axis widens from :math:`S` (states per position) to :math:`M = K \cdot S`,
    with :math:`K = |\mathcal{N}_n(p)|` the largest retained neighbourhood.

    Parameters
    ----------
    model : brainstate.nn.Module
        The model whose weights are trained online.
    n : int, default 2
        The SnAp order — how many propagation steps of the instantaneous term
        are kept. Must be an integer ``>= 1``. ``n = 1`` canonicalises to
        ``recurrence_scope='coupled'`` (:class:`~braintrace.OSTLRecurrent`'s
        coordinate); any ``n`` at or above a group's diameter saturates to full
        within-group RTRL. There is deliberately no "infinity" spelling:
        saturation is a property of the model, not of the vocabulary.
    name : str, optional
        Name of the algorithm instance. Forwarded verbatim.
    vjp_method : {'single-step', 'multi-step'}, default 'single-step'
        Execution option, forwarded verbatim. The finite-window oracle
        (``chunked_online_param_gradients``) needs ``'multi-step'``.
    fast_solve : bool, default True
        Execution option, forwarded verbatim: whether to use the per-primitive
        fast contraction instead of the legacy vmap path.
    snap_max_jacobian_elements : int, optional
        Ceiling on each hidden group's widened block Jacobian,
        ``P * (K * S) ** 2`` elements. Raising it is how a deliberately large
        neighbourhood is admitted; the default rejects roughly half a gigabyte
        per operator. Forwarded to the compiler.
    **kwargs : Any
        Additional options forwarded to
        :class:`~braintrace.ParamDimVjpAlgorithm`, including ``trace_dtype``,
        ``chunked_trace``, ``control_flow``, ``config``, and
        ``random_feedback_key``.

    Attributes
    ----------
    n : int
        The requested order, as passed (before canonicalisation).

    Notes
    -----
    **The pattern is computed, not assumed.** The compiler analyses the hidden
    group's transition jaxpr and derives :math:`\mathbf{A}` from the recurrent
    mixing primitive it finds. Two primitive families yield a *precise* pattern:
    dense (``etp_mm`` / ``etp_mv``, all-to-all) and sparse (``etp_sp_mm`` /
    ``etp_sp_mv``, the structural pattern of ``sparse_mat``, transposed). For
    **every other** primitive, more than one mixing equation, any control flow
    around it, or a non-position-preserving tail, the analysis falls back to the
    all-to-all pattern and emits a
    ``DiagnosticKind.SNAP_PATTERN_CONSERVATIVE`` diagnostic. A conservative
    pattern is always *correct* — it retains a superset of the true influence —
    but it costs saturated memory, so the diagnostic is worth reading.

    **The scale is within-group.** :math:`\mathcal{N}_n(p)` never leaves the
    hidden group that owns :math:`p`; cross-group influence is not represented
    by the per-parameter trace at any coordinate. On a model with exactly one
    hidden group and an elementwise ``y -> hidden`` tail, saturation therefore
    equals BPTT; on a multi-group model it equals full RTRL *within* each group,
    which is not BPTT.

    **Memory.** The trace grows linearly in :math:`K`; the widened transition
    operator :math:`\mathbf{D}_g` costs :math:`P \cdot K^2 \cdot S^2` on top,
    which is the term that actually bounds usable :math:`n` on large groups.

    References
    ----------
    .. [1] Menick, J., Elsen, E., Evci, U., Osindero, S., Simonyan, K., &
       Graves, A. (2021). "Practical Real Time Recurrent Learning with Sparse
       Non-Linear Recurrence" (SnAp). *ICLR 2021*. arXiv:2006.07232
    .. [2] Williams, R. J., & Zipser, D. (1989). "A Learning Algorithm for
       Continually Running Fully Recurrent Neural Networks" (RTRL). *Neural
       Computation*, 1(2), 270-280. https://doi.org/10.1162/neco.1989.1.2.270

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>>
        >>> class Net(brainstate.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.cell = braintrace.nn.ValinaRNNCell(1, 20, activation='tanh')
        ...         self.out = braintrace.nn.Linear(20, 1)
        ...     def update(self, x):
        ...         return x >> self.cell >> self.out
        >>>
        >>> model = Net()
        >>> x0 = brainstate.random.randn(1)
        >>> learner = braintrace.compile(model, braintrace.SnAp, x0, n=2)
        >>> y = learner(x0)
    """

    __module__ = 'braintrace'

    def __init__(
        self,
        model: brainstate.nn.Module,
        n: int = 2,
        name: Optional[str] = None,
        vjp_method: str = 'single-step',
        fast_solve: bool = True,
        snap_max_jacobian_elements: int = DEFAULT_MAX_JACOBIAN_ELEMENTS,
        **kwargs: Any,
    ) -> None:
        # The preset's constructor is the boundary where the paper's parameter
        # name becomes an axis coordinate. `n=1` canonicalises back to
        # `recurrence_scope='coupled'` inside ETraceConfig -- one coordinate,
        # one spelling -- so `SnAp(model, n=1)` *is* OSTLRecurrent's rule by
        # construction rather than by an `if` at the use site. Validation of
        # `n` (int, >= 1, bool rejected) also lives in ETraceConfig, so the
        # error message is identical however the coordinate is reached.
        config = ETraceConfig(recurrence_scope='sparse_n', sparse_n=n)
        super().__init__(
            model,
            name=name,
            vjp_method=vjp_method,
            fast_solve=fast_solve,
            config=config,
            snap_max_jacobian_elements=snap_max_jacobian_elements,
            **kwargs,
        )
        self.n = n
        # Provenance for the anchor check. The coordinate alone cannot carry it:
        # at `n = 1` the coordinate *is* `coupled`, which is legal on models
        # with no anchored primitive, but a caller who asked for SnAp must not
        # be handed `coupled` in its place. See
        # ``ETraceVjpAlgorithm._assert_relations_are_snap_anchored``.
        self._requested_snap_order = n
