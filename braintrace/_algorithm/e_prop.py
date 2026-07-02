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

- An optional κ-filter on each weight's *eligibility trace*
  (:math:`\\bar e = F_\\kappa(e)`), matching the paper's low-pass eligibility
  filter. The trailing per-hidden-state axis is filtered elementwise, so
  multi-state HiddenGroups (``num_state > 1``) never mix across states.
- An optional random-feedback variant (``feedback='random'``) that replaces the
  readout's symmetric gradient with a fixed random projection, removing the
  weight-transport requirement.

See :class:`EProp` for the mathematical formulation, references, and an example.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import brainstate
import jax
import jax.numpy as jnp

from braintrace._typing import ETraceWG_Key, Path, PyTree
from ._common import FixedRandomFeedback, _reset_state_in_a_dict
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
          \ell_j^t = \dfrac{\partial \mathcal{L}}{\partial h_j^t}
            & \text{(symmetric feedback, standard backprop through readout)} \\[2ex]
          \big(B\,\hat\ell^t\big)_j,
          \quad \hat\ell^t = \dfrac{\ell^t}{\lVert \ell^t \rVert + \varepsilon}
            & \text{(random feedback: a fixed random projection } B\text{)} .
        \end{cases}

    **How it works.** The eligibility trace :math:`e_{ji}^t` is exactly the
    per-parameter trace maintained by :class:`~braintrace.D_RTRL`; it depends
    only on quantities local to the synapse and is updated forward in time. The
    learning signal :math:`L_j^t` is broadcast from the readout. E-prop is
    therefore *online* (no backward pass through time) and uses memory linear in
    the number of parameters. With ``kappa_filter_decay > 0`` each weight's
    *eligibility trace* is additionally low-pass filtered (elementwise over the
    trailing per-hidden-state axis, so multi-state HiddenGroups are filtered
    per state with no cross-state mixing); with ``feedback='random'`` the
    symmetric readout gradient :math:`\ell^t` is L2-normalized (removing its
    dependence on the *magnitude* of the real readout weights, since
    reverse-AD only ever exposes :math:`\ell^t = W_\mathrm{out}^\top \delta^t`,
    never the pre-readout error :math:`\delta^t` itself) and then projected
    through a frozen random matrix :math:`B`, removing the biologically
    implausible weight-transport requirement up to that residual scale
    dependence.

    Parameters
    ----------
    model : brainstate.nn.Module
        The recurrent SNN whose weights are trained online.
    feedback : {'symmetric', 'random'}, default 'symmetric'
        ``'symmetric'`` uses reverse-AD's :math:`\partial \mathcal{L}/\partial h`
        (standard backprop through the readout). ``'random'`` replaces the
        readout gradient with a frozen random projection of its L2-normalized
        direction (requires ``random_feedback_key``). The projection matrix is
        square (hidden-dim × hidden-dim): E-prop's hooks only see
        :math:`\partial\mathcal L/\partial h`, which has no visibility into a
        separate readout layer's width, so ``feedback='random'`` assumes a
        single, direct readout whose output dimensionality equals the
        HiddenGroup's own width.
    kappa_filter_decay : float in [0, 1), default 0.0
        Eligibility-trace low-pass factor :math:`\kappa` (see
        :math:`\bar{e}_{ji}^t` above). If ``> 0``, each trainable weight's raw
        eligibility trace is filtered every step, per hidden-state channel.
        ``0`` disables filtering (the algorithm then reduces exactly to
        :class:`~braintrace.D_RTRL`).
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
        # Keyed exactly like ``self.etrace_bwg`` (ETraceWG_Key == (id(y_var),
        # group.index)): one filter state per trainable weight / HiddenGroup
        # relation, holding a PyTree that mirrors the raw trace's own
        # structure and shape (including the trailing num_state axis).
        self._trace_filters: Dict[ETraceWG_Key, brainstate.State] = {}
        self._random_feedback: Dict[int, FixedRandomFeedback] = {}

    def init_etrace_state(self, *args: Any, **kwargs: Any) -> None:
        super().init_etrace_state(*args, **kwargs)
        self._trace_filters = {}
        self._random_feedback = {}
        if self.kappa_filter_decay > 0.0:
            # One filter per raw-trace key, initialised to zeros with the
            # exact PyTree structure/shape of that trace (batch axis, weight
            # shape, and the trailing num_state axis all included) -- no
            # reduction, so per-state channels never mix.
            for trace_key, trace_state in self.etrace_bwg.items():
                self._trace_filters[trace_key] = brainstate.ShortTermState(
                    jax.tree.map(jnp.zeros_like, trace_state.value)
                )
        if self.feedback == 'random':
            rf_key = self._random_feedback_key
            assert rf_key is not None  # constructor enforces a key when feedback='random'
            # Collect the (group id -> width) pairs needed first so the
            # random draws below are made under a single seeded scope, using
            # only `brainstate.random` (never `jax.random` directly).
            groups_needed: Dict[int, int] = {}
            for rel in self.graph.hidden_param_op_relations:
                for group in rel.hidden_groups:
                    groups_needed.setdefault(group.index, int(group.varshape[-1]))
            with brainstate.random.seed_context(rf_key):
                for gid, n_layer in groups_needed.items():
                    # n_target == n_layer — square projection over reverse-AD
                    # signal (see the class docstring's single-readout note).
                    self._random_feedback[gid] = FixedRandomFeedback(
                        n_target=n_layer,
                        n_layer=n_layer,
                        key=brainstate.random.split_key(),
                        init_scale=0.1,
                    )

    def reset_state(self, batch_size: Optional[int] = None, **kwargs: Any) -> None:
        super().reset_state(batch_size=batch_size, **kwargs)
        _reset_state_in_a_dict(self._trace_filters, batch_size)

    def _compute_learning_signal(self, dl_autodiff: Any, args: Any) -> Any:
        signals = list(dl_autodiff)
        if self.feedback == 'random' and self._random_feedback:
            # dl_autodiff[g].shape == (*varshape, num_state); varshape[-1] is
            # the n_layer (hidden-width) axis, i.e. axis -2 of the full array.
            # `s` is proportional to the *real* readout weights (reverse-AD
            # only ever exposes W_out^T @ delta, never delta itself), so a
            # linear projection through any fixed B cannot remove that
            # dependency. L2-normalizing `s` per num_state channel over the
            # n_layer axis strips the magnitude dependence on W_out (while
            # keeping direction and the per-state axis untouched) before
            # projecting through the frozen random matrix.
            def _project(B: Any, s: Any) -> Any:
                norm = jnp.sqrt(jnp.sum(jnp.square(s), axis=-2, keepdims=True))
                s_normalized = s / (norm + 1e-8)
                return jnp.einsum('...lj,lk->...kj', s_normalized, B)

            signals = [
                _project(self._random_feedback[gid].B, s)
                if gid in self._random_feedback else s
                for gid, s in enumerate(signals)
            ]
        return signals

    def _solve_weight_gradients(
        self,
        running_index: int,
        etrace_h2w_at_t: Dict[ETraceWG_Key, PyTree],
        dl_to_hidden_groups: Sequence[jax.Array],
        weight_vals: Dict[Path, PyTree],
        dl_to_nonetws_at_t: Dict[Path, PyTree],
        dl_to_etws_at_t: Optional[Dict[Path, PyTree]],
    ) -> Any:
        """Low-pass filter the raw eligibility trace, then delegate to D-RTRL.

        Implements :math:`\\bar e_{ji}^t = \\kappa\\,\\bar e_{ji}^{t-1} + e_{ji}^t`
        per weight-key, applied elementwise (``jax.tree.map`` over the trace's
        own PyTree, including its trailing num_state axis) so multi-state
        HiddenGroups are filtered independently per state -- never summed
        across states and broadcast back.
        """
        if self._trace_filters:
            kappa = self.kappa_filter_decay
            filtered: Dict[ETraceWG_Key, PyTree] = {}
            for key, trace in etrace_h2w_at_t.items():
                flt = self._trace_filters.get(key)
                if flt is None:
                    filtered[key] = trace
                    continue
                new_val = jax.tree.map(
                    lambda prev, e: kappa * prev + e, flt.value, trace
                )
                flt.value = new_val
                filtered[key] = new_val
            etrace_h2w_at_t = filtered
        return super()._solve_weight_gradients(
            running_index,
            etrace_h2w_at_t,
            dl_to_hidden_groups,
            weight_vals,
            dl_to_nonetws_at_t,
            dl_to_etws_at_t,
        )
