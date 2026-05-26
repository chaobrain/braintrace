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
  recurrent layer (per-parameter D-RTRL trace, O(P·H)).
- :class:`OSTLFeedforward` ('without-H') drops ``H``; the temporal term vanishes
  and the update reduces to pp_prop with negligible decay (feedforward SNN).

:func:`OSTL` is a thin factory selecting between the two by ``regime`` keyword,
preserving the historical ``OSTL(model, regime=...)`` call site.

Reference
---------
Bohnstingl, T., Woźniak, S., Pantazi, A., & Eleftheriou, E. (2023). "Online
Spatio-Temporal Learning in Deep Neural Networks." *IEEE Transactions on Neural
Networks and Learning Systems*, 34(11), 8894-8908.
https://doi.org/10.1109/TNNLS.2022.3153985 (arXiv:2007.12723)
"""

from typing import Optional

import brainstate

from .param_dim_vjp import ParamDimVjpAlgorithm
from .pp_prop import pp_prop

__all__ = ['OSTL', 'OSTLRecurrent', 'OSTLFeedforward']


class OSTLRecurrent(ParamDimVjpAlgorithm):
    r"""OSTL 'with-H' regime — RTRL-exact single-layer factorization.

    OSTL derives an online rule by cleanly separating the gradient into a
    *temporal* eligibility trace and a *spatial* learning signal. The 'with-H'
    regime retains the hidden-to-hidden Jacobian, so the trace carries the full
    temporal term and the rule is gradient-equivalent to BPTT for a single
    recurrent layer:

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

    Parameters
    ----------
    model : brainstate.nn.Module
        The recurrent SNN whose weights are trained online.
    name, vjp_method, fast_solve, normalize_matrix_spectrum : optional
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
        >>> brainstate.nn.init_all_states(model)
        >>> learner = braintrace.OSTLRecurrent(model)   # or braintrace.OSTL(model, regime='with-H')
        >>> x0 = brainstate.random.randn(1)
        >>> learner.compile_graph(x0)
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
        >>> brainstate.nn.init_all_states(model)
        >>> learner = braintrace.OSTLFeedforward(model)  # or braintrace.OSTL(model, regime='without-H')
        >>> x0 = brainstate.random.randn(1)
        >>> learner.compile_graph(x0)
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
        **kwargs,
    ):
        super().__init__(model, decay_or_rank=decay_or_rank, name=name, **kwargs)


def OSTL(
    model: brainstate.nn.Module,
    regime: str = 'with-H',
    name: Optional[str] = None,
    **kwargs,
):
    """Construct the OSTL algorithm for the selected regime.

    A convenience dispatcher over :class:`OSTLRecurrent` ('with-H') and
    :class:`OSTLFeedforward` ('without-H'). Prefer constructing those classes
    directly when the regime is known at the call site.

    Parameters
    ----------
    model : brainstate.nn.Module
    regime : {'with-H', 'without-H'}, default 'with-H'
        'with-H' keeps the recurrent Jacobian (RTRL-exact, per-parameter trace).
        'without-H' drops it (feedforward SNN, IO-dim trace with tiny decay).
    name : optional name forwarded to the chosen class.
    **kwargs : forwarded to the chosen class constructor.

    Returns
    -------
    OSTLRecurrent or OSTLFeedforward
    """
    if regime == 'with-H':
        return OSTLRecurrent(model, name=name, **kwargs)
    if regime == 'without-H':
        return OSTLFeedforward(model, name=name, **kwargs)
    raise ValueError(f"regime must be 'with-H' or 'without-H'; got {regime!r}")
