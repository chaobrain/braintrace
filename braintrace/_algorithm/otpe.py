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

"""OTPE — Online Training with Postsynaptic Estimates (Summe et al., 2023).

OTPE replaces RTRL's full Jacobian with a leaky-additive per-parameter
accumulator :math:`\\hat R \\leftarrow \\lambda\\,\\hat R + \\partial s/\\partial
\\theta_\\text{local}` that estimates how a parameter's influence persists in the
postsynaptic membrane across several time steps — temporal structure that the
single-step approximations OTTT and OSTL drop. Cross-layer coupling is handled
inside ``_solve_weight_gradients`` without relaxing the compiler's "no W→W→h"
invariant.

See :class:`OTPE` for the mathematical formulation, references, and an example.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import brainstate
import jax.numpy as jnp

from braintrace._op import is_batched_primitive
from braintrace._misc import etrace_df_key
from ._common import _route_grads_by_path, _update_dict
from .vjp_base import ETraceVjpAlgorithm

__all__ = ['OTPE']


class OTPE(ETraceVjpAlgorithm):
    r"""Online Training with Postsynaptic Estimates for spiking networks.

    OTPE maintains a leaky, additive estimate :math:`\hat R^t` of each
    parameter's accumulated influence on the postsynaptic state, then contracts
    it with the learning signal :math:`L^t` to obtain the weight gradient:

    .. math::

        \hat R^t = \lambda\,\hat R^{t-1}
                   + \frac{\partial s^t}{\partial \theta}
                 = \lambda\,\hat R^{t-1}
                   + x^t \otimes \operatorname{diag}(D_f^t) ,
        \qquad
        \nabla_{W}\mathcal{L}^t = L^t \cdot \hat R^t ,

    where :math:`x^t` is the presynaptic input, :math:`D_f^t` the
    state-to-output Jacobian (surrogate gradient of the spike), :math:`\lambda
    \in (0, 1)` the membrane leak, and :math:`L^t = \partial \mathcal{L}^t /
    \partial s^t` the learning signal. The contraction runs over the output
    dimension, leaving a gradient with the weight's shape.

    In the low-rank ``'approx'`` mode (**F-OTPE**) the estimate is factorized as
    an outer product, reducing memory from :math:`O(I\cdot O)` to :math:`O(I+O)`
    per layer:

    .. math::

        \hat R^t \approx \hat z_\text{in}^t \otimes \bar g_\text{out}^t ,
        \quad
        \hat z_\text{in}^t = \lambda\,\hat z_\text{in}^{t-1} + x^t ,
        \quad
        \bar g_\text{out}^t = \lambda\,\bar g_\text{out}^{t-1}
                              + \operatorname{diag}(D_f^t) ,

    with gradient :math:`\nabla_{W}\mathcal{L}^t
    = \hat z_\text{in}^t \otimes (L^t \cdot \bar g_\text{out}^t)`.

    **How it works.** Unlike OTTT/OSTL, which assign temporal credit only within
    the current layer's output, OTPE keeps a per-parameter trace that decays
    with the membrane leak, approximating the *entire* temporal effect of a
    weight on downstream activity while staying local to each layer. This
    improves gradient alignment with BPTT in deep feed-forward SNNs at modest
    extra cost.

    Parameters
    ----------
    model : brainstate.nn.Module
        The SNN whose weights are trained online.
    mode : {'full', 'approx'}, default 'full'
        ``'full'`` keeps the full ``(batch, I, O)`` estimate :math:`\hat R` per
        layer. ``'approx'`` (F-OTPE) factorizes it as an outer product for
        :math:`O(I+O)` memory; emits a :class:`UserWarning` when the network has
        more than one HiddenGroup, because the factorization bias compounds with
        depth.
    leak : float
        Decay factor :math:`\lambda \in (0, 1)`. **Required** — it must be
        supplied explicitly and is never inferred from the model. :math:`\lambda`
        is the membrane leak of the *postsynaptic* neuron whose influence is
        being accumulated; auto-inferring it from ``model.states()`` silently
        picks an arbitrary (often wrong) value on heterogeneous or
        multi-population models, so the framework will not guess it.
    trace_clip_abs : float, optional
        Elementwise clip applied to :math:`\hat R` each step (full mode only).
        ``None`` disables clipping.
    name : str, optional
        Name of the algorithm instance.
    vjp_method : str, optional
        Forwarded to the base algorithm. Only ``'single-step'`` is supported by
        OTPE v1; multi-step inputs raise :class:`NotImplementedError`.

    Limitations
    -----------
    OTPE's published derivation is **narrower than OTTT's**, and this
    implementation is a *general operator* that will happily run far outside
    that proven regime. The estimate :math:`\hat R` is built on the assumption
    that the only temporal coupling of the postsynaptic state is the scalar
    membrane leak, :math:`\partial U^t / \partial U^{t-1} = \lambda` — exactly
    the leaky integrate-and-fire (LIF) recurrence. On top of that scalar-leak
    assumption (inherited from OTTT), OTPE adds three further restrictions:

    1. **A single global time constant.** One scalar :math:`\lambda` is shared by
       every traced connection. Heterogeneous leaks across neurons or layers
       break the estimate; ``leak`` is therefore a user-supplied global constant
       and is never inferred from the model (see the ``leak`` parameter).
    2. **Feed-forward only.** The trace omits the hidden-to-hidden Jacobian, so
       it is the *postsynaptic estimate* for feed-forward SNNs. Applying it to a
       recurrent network silently drops the recurrent temporal credit.
    3. **Single-hidden-layer exactness.** The estimate is gradient-exact for one
       hidden layer; with depth the per-layer factorization accumulates bias.

    The low-rank ``'approx'`` mode (**F-OTPE**) layers an additional
    outer-product approximation on top, which is itself justified only under the
    same linear-leak assumption; its bias compounds with network depth (hence
    the :class:`UserWarning` for multi-group networks).

    Concretely, ``braintrace`` exposes OTPE as a generic ETP operator: it accepts
    arbitrary ETP weights and hidden states, multi-layer stacks, recurrent
    connectivity, and even non-spiking cells (e.g. a ``tanh`` RNN). All of these
    *run* mechanically, but **the moment the model deviates from a feed-forward
    LIF network with a single global scalar leak, the computed gradient leaves
    the regime in which OTPE is proven correct** and should be treated as a
    heuristic approximation rather than a faithful gradient estimate. The one
    structural case that is rejected outright is a multi-state hidden group
    (``num_state > 1``, e.g. ALIF with an adaptation variable): the leaky scalar
    estimate cannot assign per-state credit, so :meth:`compile_graph` raises
    rather than silently summing across states.

    Raises
    ------
    ValueError
        If ``mode`` is not ``'full'`` or ``'approx'``, if ``leak`` is not in
        :math:`(0, 1)`, if a weight-to-hidden relation reaches more than one
        HiddenGroup (OTPE v1 requires one-hop per-layer relations), or (at
        :meth:`compile_graph`) if a trained connection projects into a hidden
        group with ``num_state > 1``.

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
        >>> learner = braintrace.compile(model, braintrace.OTPE, x0, mode='full', leak=0.9)
        >>> y = learner(x0)

    References
    ----------
    .. [1] Summe, T. M., Schaefer, C. J. S., & Joshi, S. (2023). "Estimating
       Post-Synaptic Effects for Online Training of Feed-Forward SNNs."
       *arXiv preprint* arXiv:2311.16151. https://arxiv.org/abs/2311.16151
    """

    __module__ = 'braintrace'

    def __init__(
        self,
        model: brainstate.nn.Module,
        mode: str = 'full',
        *,
        leak: float,
        name: Optional[str] = None,
        vjp_method: str = 'single-step',
        trace_clip_abs: Optional[float] = None,
    ) -> None:
        if mode not in ('full', 'approx'):
            raise ValueError(f"mode must be 'full' or 'approx'; got {mode!r}")
        if not (0.0 < float(leak) < 1.0):
            raise ValueError(f'leak must be in (0, 1); got {leak}')
        super().__init__(model, name=name, vjp_method=vjp_method)
        self.mode = mode
        self.leak = float(leak)
        self.trace_clip_abs = trace_clip_abs
        self._R_hat: Dict[int, brainstate.ShortTermState] = {}
        self._R_hat_x: Dict[int, brainstate.ShortTermState] = {}
        self._R_hat_g: Dict[int, brainstate.ShortTermState] = {}

    def compile_graph(self, *args: Any) -> None:
        super().compile_graph(*args)
        if self.mode == 'approx':
            n_groups = len(self.graph.hidden_groups)
            if n_groups > 1:
                warnings.warn(
                    "OTPE(mode='approx') bias compounds with network depth; "
                    "consider F-OTPE or mode='full'.",
                    UserWarning,
                )
        # Invariant: each relation maps to exactly one HiddenGroup in OTPE v1.
        for rel in self.graph.hidden_param_op_relations:
            if len(rel.hidden_groups) != 1:
                raise ValueError(
                    f'OTPE requires per-layer one-hop weight-to-hidden relations; '
                    f'found relation reaching {len(rel.hidden_groups)} groups.'
                )
            # OTPE's derivation assumes a single scalar membrane state per neuron
            # (the LIF case). A hidden group bundling several states (e.g. ALIF's
            # membrane potential plus adaptation variable) cannot be handled: the
            # leaky scalar estimate cannot assign per-state credit, and collapsing
            # the num_state axis with a sum has no theoretical basis (see the
            # *Limitations* section of the class docstring).
            group = rel.hidden_groups[0]
            if group.num_state > 1:
                raise ValueError(
                    f'OTPE only supports hidden groups with num_state == 1 '
                    f'(single-state LIF-like neurons), but a trained connection '
                    f'projects into a group with num_state == {group.num_state}. '
                    f'Multi-state neurons (e.g. ALIF with an adaptation variable) '
                    f'are outside the regime where OTPE is derived.'
                )

    def init_etrace_state(self, *args: Any, **kwargs: Any) -> None:
        self._R_hat = {}
        self._R_hat_x = {}
        self._R_hat_g = {}
        for rel in self.graph.hidden_param_op_relations:
            rid = id(rel.y_var)
            assert rel.x_var is not None  # non-elemwise primitives always have an x_var
            in_shape = rel.x_var.aval.shape
            out_shape = rel.y_var.aval.shape
            if self.mode == 'full':
                weight_key = next(iter(rel.trainable_vars))
                weight_var = rel.trainable_vars[weight_key]
                weight_shape = weight_var.aval.shape
                if is_batched_primitive(rel.primitive):
                    shape = (in_shape[0], *weight_shape)
                else:
                    shape = weight_shape
                self._R_hat[rid] = brainstate.ShortTermState(
                    jnp.zeros(shape, dtype=jnp.float32)
                )
            else:
                self._R_hat_x[rid] = brainstate.ShortTermState(
                    jnp.zeros(in_shape, dtype=jnp.float32)
                )
                self._R_hat_g[rid] = brainstate.ShortTermState(
                    jnp.zeros(out_shape, dtype=jnp.float32)
                )

    def reset_state(self, batch_size: Optional[int] = None, **kwargs: Any) -> None:
        self.running_index.value = 0

        def _rezero(state: Any) -> None:
            shape = state.value.shape
            new_shape = (batch_size, *shape[1:]) if batch_size is not None else shape
            state.value = jnp.zeros(new_shape, dtype=state.value.dtype)

        for store in (self._R_hat, self._R_hat_x, self._R_hat_g):
            for r in store.values():
                _rezero(r)

    def _get_etrace_data(self) -> Any:
        if self.mode == 'full':
            return {rid: r.value for rid, r in self._R_hat.items()}
        return (
            {rid: r.value for rid, r in self._R_hat_x.items()},
            {rid: r.value for rid, r in self._R_hat_g.items()},
        )

    def _assign_etrace_data(self, vals: Any) -> None:
        if self.mode == 'full':
            for rid, v in vals.items():
                self._R_hat[rid].value = v
        else:
            vals_x, vals_g = vals
            for rid, v in vals_x.items():
                self._R_hat_x[rid].value = v
            for rid, v in vals_g.items():
                self._R_hat_g[rid].value = v

    def _update_etrace_data(
        self, running_index: Any, hist_vals: Any,
        hid2weight_jac: Any, hid2hid_jac: Any, weight_vals: Any, input_is_multi_step: Any,
    ) -> Any:
        """``R_hat ← λ·R_hat + ∂s/∂θ_local``. Ignores ``hid2hid_jac``."""
        if input_is_multi_step:
            raise NotImplementedError('OTPE v1 supports single-step only')
        xs = hid2weight_jac[0]
        dfs = hid2weight_jac[1]

        if self.mode == 'full':
            new_R = {}
            for rel in self.graph.hidden_param_op_relations:
                rid = id(rel.y_var)
                group = rel.hidden_groups[0]
                x = xs[id(rel.x_var)]
                df = dfs[etrace_df_key(rel.y_var, group.index)]
                df_proj = df.sum(axis=-1)
                if is_batched_primitive(rel.primitive):
                    local = jnp.einsum('bi,bo->bio', x, df_proj)
                else:
                    local = jnp.einsum('i,o->io', x, df_proj)
                updated = self.leak * hist_vals[rid] + local
                if self.trace_clip_abs is not None:
                    updated = jnp.clip(
                        updated, -self.trace_clip_abs, self.trace_clip_abs
                    )
                new_R[rid] = updated
            return new_R
        else:
            new_Rx = {}
            new_Rg = {}
            hist_x, hist_g = hist_vals
            for rel in self.graph.hidden_param_op_relations:
                rid = id(rel.y_var)
                group = rel.hidden_groups[0]
                x = xs[id(rel.x_var)]
                df = dfs[etrace_df_key(rel.y_var, group.index)].sum(axis=-1)
                new_Rx[rid] = self.leak * hist_x[rid] + x
                new_Rg[rid] = self.leak * hist_g[rid] + df
            return (new_Rx, new_Rg)

    def _solve_weight_gradients(
        self, running_index: Any, etrace_at_t: Any, dl_to_hidden_groups: Any,
        weight_vals: Any, dl_to_nonetws_at_t: Any, dl_to_etws_at_t: Any,
    ) -> Any:
        dG = {path: None for path in self.param_states}
        if self.mode == 'full':
            for rel in self.graph.hidden_param_op_relations:
                rid = id(rel.y_var)
                R = etrace_at_t[rid]
                group = rel.hidden_groups[0]
                L = dl_to_hidden_groups[group.index].sum(axis=-1)
                if is_batched_primitive(rel.primitive):
                    dw = jnp.einsum('bo,bio->io', L, R)
                else:
                    dw = jnp.einsum('o,io->io', L, R)
                _route_grads_by_path(rel, {'weight': dw}, weight_vals, dG)
        else:
            Rx_map, Rg_map = etrace_at_t
            for rel in self.graph.hidden_param_op_relations:
                rid = id(rel.y_var)
                group = rel.hidden_groups[0]
                L = dl_to_hidden_groups[group.index].sum(axis=-1)
                Rx = Rx_map[rid]
                Rg = Rg_map[rid]
                if is_batched_primitive(rel.primitive):
                    dw = jnp.einsum('bi,bo->io', Rx, L * Rg)
                else:
                    dw = jnp.einsum('i,o->io', Rx, L * Rg)
                _route_grads_by_path(rel, {'weight': dw}, weight_vals, dG)

        for path, dg in dl_to_nonetws_at_t.items():
            _update_dict(dG, path, dg)
        if dl_to_etws_at_t is not None:
            for path, dg in dl_to_etws_at_t.items():
                _update_dict(dG, path, dg, error_when_no_key=True)
        return dG
