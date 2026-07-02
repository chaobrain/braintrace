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

"""OTTT — Online Training Through Time (Xiao et al., 2022).

OTTT is derived from BPTT but discards the hidden-to-hidden recurrent Jacobian,
so it keeps only a leaky presynaptic eligibility trace
:math:`\\hat a^t \\leftarrow \\lambda\\,\\hat a^{t-1} + x^t` and forms the weight
gradient at each step as the outer product of that trace with the (instantaneous)
learning signal, :math:`\\Delta W = \\hat a^t \\otimes (L \\cdot \\sigma'(u))`.
This yields constant training memory, independent of the number of time steps.

See :class:`OTTT` for the mathematical formulation, references, and an example.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import brainstate
import jax.numpy as jnp

from braintrace._op import is_batched_primitive
from ._common import PresynapticTrace, _route_grads_by_path, _update_dict
from .vjp_base import ETraceVjpAlgorithm

__all__ = ['OTTT']


class OTTT(ETraceVjpAlgorithm):
    r"""Online Training Through Time for spiking neural networks.

    OTTT tracks only a leaky **presynaptic trace** and forms the weight gradient
    each step as an outer product with the local learning signal:

    .. math::

        \hat{a}^t =
        \begin{cases}
          \lambda\,\hat{a}^{t-1} + x^t & \text{(mode='A', accumulated)} \\
          x^t                           & \text{(mode='O', instantaneous)}
        \end{cases}

    .. math::

        \nabla_{W}\mathcal{L}^t
        = \hat{a}^t \otimes
          \Big( \frac{\partial \mathcal{L}^t}{\partial s^t}\,\sigma'(u^t) \Big)
        \;=\; \hat{a}^t \otimes L^t ,

    where :math:`x^t` is the presynaptic input, :math:`u^t` the membrane
    potential, :math:`s^t = \sigma(u^t)` the (surrogate) spike, :math:`\sigma'`
    the surrogate-gradient function, :math:`\lambda \in (0, 1)` the membrane
    leak, and :math:`L^t` the learning signal already propagated through the
    spike nonlinearity.

    **How it works.** Starting from BPTT, OTTT keeps the spatial credit
    assignment but **drops the hidden-to-hidden recurrent Jacobian**. The only
    state it carries forward in time is the rank-1 presynaptic trace
    :math:`\hat{a}^t`, so the per-step gradient is the outer product of that
    trace with the instantaneous learning signal. Training memory is therefore
    :math:`O(B \cdot I)` per layer and **independent of the sequence length** —
    the cheapest of the algorithms here, at the cost of ignoring longer-range
    temporal credit.

    Parameters
    ----------
    model : brainstate.nn.Module
        The SNN whose weights are trained online.
    mode : {'A', 'O'}, default 'A'
        ``'A'`` accumulates the presynaptic trace over time
        (:math:`\hat a \leftarrow \lambda\,\hat a + x`). ``'O'`` uses the
        instantaneous presynaptic spike only (:math:`\hat a := x^t`).
    leak : float
        Presynaptic leak :math:`\lambda \in (0, 1)`. **Required** — it must be
        supplied explicitly and is never inferred from the model (see
        *Limitations*). Mathematically :math:`\lambda` is the membrane leak of
        the *postsynaptic* neuron whose trace is being accumulated.
    name : str, optional
        Name of the algorithm instance.
    vjp_method : str, optional
        Forwarded to the base algorithm. Only ``'single-step'`` is supported by
        OTTT v1; multi-step inputs raise :class:`NotImplementedError`.

    Limitations
    -----------
    - **The leak must be supplied by the user.** OTTT does *not* try to read
      :math:`\lambda` off the model's neuron states. A previous version walked
      ``model.states()`` and took the first state exposing a ``leak`` attribute,
      but on heterogeneous or multi-population models that silently picks an
      arbitrary (often wrong) value — e.g. the leak of the *presynaptic* layer,
      a readout filter, or whichever population happens to be enumerated first.
      Since :math:`\lambda` is, by the derivation, the membrane leak of the
      postsynaptic neuron of each trained connection, the framework cannot
      guess it safely. A single network with different leaks per layer therefore
      cannot be trained correctly with one global ``leak`` and is unsupported.
    - **Single-state hidden groups only.** Each trained connection must project
      into a :class:`HiddenGroup` with ``num_state == 1``. The weight gradient
      contracts the learning signal ``L`` (shape ``(*varshape, num_state)``)
      down to ``(*varshape,)``; collapsing a ``num_state > 1`` tail (e.g. an
      ALIF neuron carrying both membrane potential and an adaptation variable)
      has no theoretical justification — the trace is a single leaky scalar and
      cannot disentangle per-state credit — so OTTT raises at compile time
      instead of silently summing across states.
    - **Single-step inputs only** (OTTT v1); multi-step inputs raise
      :class:`NotImplementedError`.

    Raises
    ------
    ValueError
        If ``mode`` is not ``'A'`` or ``'O'``, if ``leak`` is not in
        :math:`(0, 1)`, or (at :meth:`compile_graph`) if a trained connection
        projects into a hidden group with ``num_state > 1``.

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
        >>> # ``leak`` is the postsynaptic membrane leak and must be passed
        >>> # explicitly; it is never inferred from the model. ``compile`` does the
        >>> # state init + graph build in one call.
        >>> learner = braintrace.compile(model, braintrace.OTTT, x0, mode='A', leak=0.9)
        >>> y = learner(x0)

    References
    ----------
    .. [1] Xiao, M., Meng, Q., Zhang, Z., He, D., & Lin, Z. (2022). "Online
       Training Through Time for Spiking Neural Networks." *Advances in Neural
       Information Processing Systems (NeurIPS)* 35.
       https://arxiv.org/abs/2210.04195
    """

    __module__ = 'braintrace'

    def __init__(
        self,
        model: brainstate.nn.Module,
        mode: str = 'A',
        *,
        leak: float,
        name: Optional[str] = None,
        vjp_method: str = 'single-step',
    ) -> None:
        if mode not in ('A', 'O'):
            raise ValueError(f"mode must be 'A' or 'O'; got {mode!r}")
        if not (0.0 < float(leak) < 1.0):
            raise ValueError(f'leak must be in (0, 1); got {leak}')
        super().__init__(model, name=name, vjp_method=vjp_method)
        self.mode = mode
        self.leak = float(leak)
        self._pre_traces: Dict[int, PresynapticTrace] = {}

    def init_etrace_state(self, *args: Any, **kwargs: Any) -> None:
        self._pre_traces = {}
        for rel in self.graph.hidden_param_op_relations:
            for group in rel.hidden_groups:
                if group.num_state > 1:
                    raise ValueError(
                        f'OTTT only supports hidden groups with num_state == 1, '
                        f'but a trained connection projects into a group with '
                        f'num_state == {group.num_state}. Collapsing the learning '
                        f'signal across multiple hidden states (e.g. an ALIF '
                        f'neuron with membrane potential plus an adaptation '
                        f'variable) has no theoretical basis for OTTT; the leaky '
                        f'scalar presynaptic trace cannot assign per-state credit.'
                    )
            rid = id(rel.x_var)
            if rid in self._pre_traces:
                continue
            assert rel.x_var is not None  # non-elemwise primitives always have an x_var
            shape = rel.x_var.aval.shape
            dtype = rel.x_var.aval.dtype
            self._pre_traces[rid] = PresynapticTrace(
                jnp.zeros(shape, dtype=dtype), leak=self.leak
            )

    def reset_state(self, batch_size: Optional[int] = None, **kwargs: Any) -> None:
        self.running_index.value = 0
        for t in self._pre_traces.values():
            t.reset_state(batch_size=batch_size)

    def _get_etrace_data(self) -> Any:
        return {rid: t.value for rid, t in self._pre_traces.items()}

    def _assign_etrace_data(self, vals: Any) -> None:
        for rid, v in vals.items():
            self._pre_traces[rid].value = v

    def _update_etrace_data(
        self, running_index: Any, hist_vals: Any,
        hid2weight_jac: Any, hid2hid_jac: Any, weight_vals: Any, input_is_multi_step: Any,
    ) -> Any:
        """``â ← λ·â + x_t`` (mode='A') or ``â := x_t`` (mode='O').

        Ignores ``hid2hid_jac`` — OTTT's core approximation.
        """
        if input_is_multi_step:
            raise NotImplementedError('OTTT v1 supports single-step only')
        xs_at_t = hid2weight_jac[0]

        new_vals = {}
        for rid, old in hist_vals.items():
            x_t = xs_at_t[rid]
            if self.mode == 'A':
                new_vals[rid] = self.leak * old + x_t
            else:
                new_vals[rid] = x_t
        return new_vals

    def _solve_weight_gradients(
        self, running_index: Any, etrace_at_t: Any, dl_to_hidden_groups: Any,
        weight_vals: Any, dl_to_nonetws_at_t: Any, dl_to_etws_at_t: Any,
    ) -> Any:
        """``ΔW = outer(â, L)`` where ``L`` is the (already σ'-propagated) signal."""
        dG = {path: None for path in self.param_states}
        for rel in self.graph.hidden_param_op_relations:
            a_hat = etrace_at_t[id(rel.x_var)]
            for group in rel.hidden_groups:
                L = dl_to_hidden_groups[group.index]
                # L shape = (*varshape, num_state); num_state == 1 is enforced at
                # compile time (see init_etrace_state), so this drops the singleton
                # tail rather than summing across genuinely distinct hidden states.
                L_proj = L.sum(axis=-1)
                if is_batched_primitive(rel.primitive):
                    # a_hat: (batch, in), L_proj: (batch, out). ΔW: (in, out)
                    dw = jnp.einsum('bi,bo->io', a_hat, L_proj)
                else:
                    dw = jnp.einsum('i,o->io', a_hat, L_proj)
                _route_grads_by_path(rel, {'weight': dw}, weight_vals, dG)

        for path, dg in dl_to_nonetws_at_t.items():
            _update_dict(dG, path, dg)
        if dl_to_etws_at_t is not None:
            for path, dg in dl_to_etws_at_t.items():
                _update_dict(dG, path, dg, error_when_no_key=True)
        return dG
