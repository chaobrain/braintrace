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

"""OSTL — Online Spatio-Temporal Learning (Bohnstingl et al. 2023).

The paper defines two regimes that differ in whether the recurrent (hidden-to-
hidden) Jacobian ``H`` is retained. Each regime is its own class so that it
inherits the full, separately-tested machinery of an existing VJP algorithm —
``compile_graph``, ``update``, ``reset_state``, ``get_etrace_of`` — without
branching:

- :class:`OSTLRecurrent` ('with-H') keeps ``H`` and is RTRL-exact for a single
  recurrent layer *whose hidden-to-hidden Jacobian is block-diagonal* (i.e. the
  only path from one hidden unit to another is through a traced ETP recurrent
  weight, not e.g. a hand-written mixing term) — per-parameter D-RTRL trace,
  O(P·H). Genuine cross-hidden-unit coupling outside the traced weight is
  dropped, so the rule is then only an approximation to BPTT.
- :class:`OSTLFeedforward` ('without-H') drops ``H``; the temporal term vanishes
  and the update reduces to pp_prop with negligible decay (feedforward SNN).

Reference
---------
Bohnstingl, T., Woźniak, S., Pantazi, A., & Eleftheriou, E. (2023). "Online
Spatio-Temporal Learning in Deep Neural Networks." *IEEE Transactions on Neural
Networks and Learning Systems*, 34(11), 8894-8908.
https://doi.org/10.1109/TNNLS.2022.3153985 (arXiv:2007.12723)
"""

from __future__ import annotations

from typing import Any, Optional

import brainstate

from .param_dim_vjp import ParamDimVjpAlgorithm
from .pp_prop import pp_prop

__all__ = ['OSTLRecurrent', 'OSTLFeedforward']


class OSTLRecurrent(ParamDimVjpAlgorithm):
    r"""OSTL 'with-H' regime — RTRL-exact single-layer factorization.

    OSTL derives an online rule by cleanly separating the gradient into a
    *temporal* eligibility trace and a *spatial* learning signal. The 'with-H'
    regime retains the hidden-to-hidden Jacobian, so the trace carries the full
    temporal term:

    .. math::

        \boldsymbol{\epsilon}^t = \mathbf{D}^t\,\boldsymbol{\epsilon}^{t-1}
        + \operatorname{diag}(\mathbf{D}_f^t)\otimes \mathbf{x}^t ,
        \qquad
        \nabla_{\boldsymbol{\theta}}\mathcal{L}
        = \sum_t \frac{\partial \mathcal{L}^t}{\partial \mathbf{h}^t}
          \circ \boldsymbol{\epsilon}^t ,

    where :math:`\mathbf{D}^t` is the hidden-to-hidden Jacobian, :math:`\mathbf{D}_f^t`
    the state-to-output Jacobian, and :math:`\mathbf{x}^t` the presynaptic input.
    This is exactly the per-parameter D-RTRL trace (memory :math:`O(P\cdot H)`),
    so the class delegates entirely to :class:`~braintrace.ParamDimVjpAlgorithm`.

    **Accuracy caveat.** The D-RTRL trace machinery underlying this class only
    ever retains the per-position *block-diagonal* of :math:`\mathbf{D}^t`
    (:func:`HiddenGroup.diagonal_jacobian`, via ``block_diagonal_last_dim``):
    cross-hidden-unit terms :math:`\partial h^t_p / \partial h^{t-1}_q` for
    :math:`p \ne q` are retained only insofar as they flow through a *traced
    ETP weight* (e.g. a recurrent :func:`~braintrace.matmul`); any other
    hidden-to-hidden mixing (e.g. a hand-written convolution/roll/mixing term
    not expressed as an ETP op) is *not* captured. Consequently the rule is
    gradient-equivalent to BPTT only when the hidden-to-hidden Jacobian is
    (effectively) block-diagonal in this sense — for a single recurrent layer
    whose only cross-unit coupling is the traced ETP recurrent weight, the two
    coincide to machine precision. If some other part of the model couples
    hidden units directly (bypassing the traced weight), the two diverge; see
    ``TestOSTLRecurrentVsBPTT`` in ``ostl_test.py`` for a worked example of both
    regimes.

    Parameters
    ----------
    model : brainstate.nn.Module
        The recurrent SNN whose weights are trained online.
    name, vjp_method, fast_solve : optional
        Forwarded verbatim to :class:`~braintrace.ParamDimVjpAlgorithm`.

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
        >>> # one call: initialise states, build the trace graph, return a learner
        >>> learner = braintrace.compile(model, braintrace.OSTLRecurrent, x0)
        >>> y = learner(x0)

    References
    ----------
    .. [1] Bohnstingl, T., Woźniak, S., Pantazi, A., & Eleftheriou, E. (2023).
       "Online Spatio-Temporal Learning in Deep Neural Networks." *IEEE
       Transactions on Neural Networks and Learning Systems*, 34(11), 8894-8908.
       https://doi.org/10.1109/TNNLS.2022.3153985 (arXiv:2007.12723)
    """

    __module__ = 'braintrace'

    #: Identifies the OSTL regime this class implements.
    regime = 'with-H'

    #: 'with-H' keeps the full recurrent (hidden-to-hidden) Jacobian, so the
    #: hidden-group transition must trace recurrent ETP mixing primitives and
    #: extract the true per-position block-diagonal (bounded) Jacobian.
    _include_recurrent_mixing = True


class OSTLFeedforward(pp_prop):
    r"""OSTL 'without-H' regime — feedforward / no recurrent Jacobian.

    The 'without-H' regime drops the hidden-to-hidden Jacobian
    :math:`\mathbf{D}^t`, so the temporal term of the eligibility trace
    vanishes and only the instantaneous (spatial) contribution survives:

    .. math::

        \boldsymbol{\epsilon}^t \approx \operatorname{diag}(\mathbf{D}_f^t)
        \otimes \mathbf{x}^t ,
        \qquad
        \nabla_{\boldsymbol{\theta}}\mathcal{L}
        = \sum_t \frac{\partial \mathcal{L}^t}{\partial \mathbf{h}^t}
          \circ \boldsymbol{\epsilon}^t .

    This is the appropriate (and exact) approximation for feed-forward SNNs. It
    is realized by delegating to :class:`~braintrace.pp_prop` (the input-output
    factorized trace) with a *negligible* decay, so the trace does not
    accumulate across time.

    Parameters
    ----------
    model : brainstate.nn.Module
        The SNN whose weights are trained online.
    decay_or_rank : float or int, default 1e-6
        Exponential-smoothing factor of the IO-dim trace. The tiny default makes
        the temporal contribution negligible, matching the 'without-H' regime. A
        float must lie in (0, 1); an int is read as an approximation rank.
    name, vjp_method, fast_solve : optional
        Forwarded verbatim to :class:`~braintrace.pp_prop`.

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
        >>> # one call: initialise states, build the trace graph, return a learner
        >>> learner = braintrace.compile(model, braintrace.OSTLFeedforward, x0)
        >>> y = learner(x0)

    References
    ----------
    .. [1] Bohnstingl, T., Woźniak, S., Pantazi, A., & Eleftheriou, E. (2023).
       "Online Spatio-Temporal Learning in Deep Neural Networks." *IEEE
       Transactions on Neural Networks and Learning Systems*, 34(11), 8894-8908.
       https://doi.org/10.1109/TNNLS.2022.3153985 (arXiv:2007.12723)
    """

    __module__ = 'braintrace'

    #: Identifies the OSTL regime this class implements.
    regime = 'without-H'

    def __init__(
        self,
        model: brainstate.nn.Module,
        decay_or_rank: float | int = 1e-6,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, decay_or_rank=decay_or_rank, name=name, **kwargs)
