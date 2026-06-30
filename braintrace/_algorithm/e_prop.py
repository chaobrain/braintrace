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

"""E-Prop — Eligibility Propagation (Bellec et al., 2020).

E-prop factorizes the BPTT gradient of a recurrent SNN into a *local*
eligibility trace and a *learning signal* broadcast from the readout. This
module builds on ``D_RTRL``'s per-parameter trace and adds the two ingredients
that make the rule biologically plausible:

- An optional κ-filter on each HiddenGroup's learning signal
  (:math:`\\bar L = F_\\kappa(L)`), matching the paper's readout-side low-pass.
- An optional random-feedback variant (``feedback='random'``) that replaces the
  readout's symmetric gradient with a fixed random projection, removing the
  weight-transport requirement.

See :class:`EProp` for the mathematical formulation, references, and an example.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import brainstate
import jax
import jax.numpy as jnp

from ._common import FixedRandomFeedback, KappaFilter
from .param_dim_vjp import ParamDimVjpAlgorithm

__all__ = ['EProp']


class EProp(ParamDimVjpAlgorithm):
    r"""Eligibility Propagation (e-prop) for recurrent spiking networks.

    E-prop approximates the gradient of a loss :math:`\mathcal{L}` with respect
    to a recurrent weight :math:`W_{ji}` by the product of a *local* eligibility
    trace and a *global* learning signal, dropping the temporally non-local
    terms of BPTT:

    .. math::

        \frac{d\mathcal{L}}{dW_{ji}}
        = \sum_t L_j^t \, \bar{e}_{ji}^t ,

    where

    .. math::

        e_{ji}^t = \frac{\partial h_j^t}{\partial W_{ji}}
                 \approx D_j^t \, e_{ji}^{t-1}
                 + \big[\operatorname{diag}(D_{f,j}^t)\big]\, x_i^t ,
        \qquad
        \bar{e}_{ji}^t = \kappa\,\bar{e}_{ji}^{t-1} + e_{ji}^t .

    Here :math:`h_j^t` is the hidden state of neuron :math:`j` at time
    :math:`t`, :math:`x_i^t` the presynaptic input, :math:`D_j^t` the
    hidden-to-hidden (recurrent) Jacobian diagonal, :math:`D_{f,j}^t` the
    state-to-output Jacobian, and :math:`\kappa \in [0, 1)` the readout-side
    low-pass factor. The learning signal is

    .. math::

        L_j^t =
        \begin{cases}
          \dfrac{\partial \mathcal{L}}{\partial h_j^t}
            & \text{(symmetric feedback, standard backprop through readout)} \\[2ex]
          \big(B\,e^t\big)_j
            & \text{(random feedback: a fixed random projection } B\text{)} .
        \end{cases}

    **How it works.** The eligibility trace :math:`e_{ji}^t` is exactly the
    per-parameter trace maintained by :class:`~braintrace.D_RTRL`; it depends
    only on quantities local to the synapse and is updated forward in time. The
    learning signal :math:`L_j^t` is broadcast from the readout. E-prop is
    therefore *online* (no backward pass through time) and uses memory linear in
    the number of parameters. With ``kappa_filter_decay > 0`` the learning
    signal is additionally low-pass filtered; with ``feedback='random'`` the
    symmetric readout gradient is replaced by a frozen random matrix, removing
    the biologically implausible weight-transport requirement.

    Parameters
    ----------
    model : brainstate.nn.Module
        The recurrent SNN whose weights are trained online.
    feedback : {'symmetric', 'random'}, default 'symmetric'
        ``'symmetric'`` uses reverse-AD's :math:`\partial \mathcal{L}/\partial h`
        (standard backprop through the readout). ``'random'`` replaces the
        readout gradient with a frozen random projection (requires
        ``random_feedback_key``).
    kappa_filter_decay : float in [0, 1), default 0.0
        Readout-side low-pass factor :math:`\kappa`. If ``> 0``, each
        HiddenGroup's learning signal is filtered each step
        (:math:`\bar L^t = (1-\kappa)L^t + \kappa\bar L^{t-1}`). ``0`` disables
        filtering.
    random_feedback_key : jax.random.PRNGKey, optional
        Seed for the random-feedback matrices. Required when
        ``feedback='random'``; ignored otherwise.
    name : str, optional
        Name of the algorithm instance.
    vjp_method, fast_solve
        Forwarded verbatim to :class:`~braintrace.D_RTRL`.

    Raises
    ------
    ValueError
        If ``feedback`` is not one of ``{'symmetric', 'random'}``, or if
        ``feedback='random'`` is given without ``random_feedback_key``.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>>
        >>> class RSNN(brainstate.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.cell = braintrace.nn.ValinaRNNCell(1, 20, activation='tanh')
        ...         self.out = braintrace.nn.Linear(20, 1)
        ...     def update(self, x):
        ...         return x >> self.cell >> self.out
        >>>
        >>> model = RSNN()
        >>> x0 = brainstate.random.randn(1)
        >>> # one call: initialise states, build the trace graph, return a learner
        >>> learner = braintrace.compile(model, braintrace.EProp, x0, kappa_filter_decay=0.9)
        >>> y = learner(x0)             # forward pass + eligibility-trace update

    References
    ----------
    .. [1] Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D.,
       Legenstein, R., & Maass, W. (2020). "A solution to the learning dilemma
       for recurrent networks of spiking neurons." *Nature Communications*,
       11, 3625. https://doi.org/10.1038/s41467-020-17236-y
    """

    __module__ = 'braintrace'

    def __init__(
        self,
        model: brainstate.nn.Module,
        feedback: str = 'symmetric',
        kappa_filter_decay: float = 0.0,
        random_feedback_key: Optional[jax.Array] = None,
        name: Optional[str] = None,
        vjp_method: str = 'single-step',
        fast_solve: bool = True,
        **kwargs: Any,
    ) -> None:
        if feedback not in ('symmetric', 'random'):
            raise ValueError(
                f"feedback must be 'symmetric' or 'random'; got {feedback!r}"
            )
        if feedback == 'random' and random_feedback_key is None:
            raise ValueError(
                "feedback='random' requires random_feedback_key=<PRNGKey>"
            )
        super().__init__(
            model,
            name=name,
            vjp_method=vjp_method,
            fast_solve=fast_solve,
            **kwargs,
        )
        self.feedback = feedback
        self.kappa_filter_decay = float(kappa_filter_decay)
        self._random_feedback_key = random_feedback_key
        self._kappa_filters: Dict[int, KappaFilter] = {}
        self._random_feedback: Dict[int, FixedRandomFeedback] = {}

    def init_etrace_state(self, *args: Any, **kwargs: Any) -> None:
        super().init_etrace_state(*args, **kwargs)
        self._kappa_filters = {}
        self._random_feedback = {}
        if self.kappa_filter_decay > 0.0:
            for rel in self.graph.hidden_param_op_relations:
                for group in rel.hidden_groups:
                    gid = group.index
                    if gid in self._kappa_filters:
                        continue
                    zeros = jnp.zeros(group.varshape, dtype=jnp.float32)
                    self._kappa_filters[gid] = KappaFilter(
                        zeros, self.kappa_filter_decay
                    )
        if self.feedback == 'random':
            key = self._random_feedback_key
            assert key is not None  # constructor enforces a key when feedback='random'
            for rel in self.graph.hidden_param_op_relations:
                for group in rel.hidden_groups:
                    gid = group.index
                    if gid in self._random_feedback:
                        continue
                    key, sub = jax.random.split(key)
                    n_layer = int(group.varshape[-1])
                    # n_target == n_layer — square projection over reverse-AD signal.
                    self._random_feedback[gid] = FixedRandomFeedback(
                        n_target=n_layer, n_layer=n_layer, key=sub, init_scale=0.1
                    )

    def reset_state(self, batch_size: Optional[int] = None, **kwargs: Any) -> None:
        super().reset_state(batch_size=batch_size, **kwargs)
        for flt in self._kappa_filters.values():
            flt.reset_state(batch_size=batch_size)

    def _compute_learning_signal(self, dl_autodiff: Any, args: Any) -> Any:
        signals = list(dl_autodiff)
        if self.feedback == 'random' and self._random_feedback:
            # dl_autodiff[g].shape == (*varshape, num_state). Project over the
            # trailing n_layer axis (-2), preserving num_state on the tail.
            def _project(B: Any, s: Any) -> Any:
                return jnp.einsum('...lj,lk->...kj', s, B)

            signals = [
                _project(self._random_feedback[gid].B, s)
                if gid in self._random_feedback else s
                for gid, s in enumerate(signals)
            ]
        if self._kappa_filters:
            # KappaFilter state carries varshape, but signal has an extra
            # trailing num_state axis. Collapse num_state for filter purposes;
            # broadcast the filtered value back.
            def _filter(flt: Any, s: Any) -> Any:
                # collapse num_state tail: sum over last axis produces shape (*varshape,)
                collapsed = s.sum(axis=-1)
                filtered = flt.update(collapsed)
                return jnp.expand_dims(filtered, axis=-1).astype(s.dtype)

            signals = [
                _filter(self._kappa_filters[gid], s)
                if gid in self._kappa_filters else s
                for gid, s in enumerate(signals)
            ]
        return signals
