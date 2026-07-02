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

from braintrace._op import etp_mm_p, etp_mv_p, is_batched_primitive
from braintrace._misc import etrace_df_key
from ._common import _route_grads_by_path, _update_dict
from .vjp_base import ETraceVjpAlgorithm

__all__ = ['OTPE']

# OTPE's per-parameter leaky trace only reduces correctly for dense matmul
# relations. LoRA, sparse, conv, and element-wise (no x_var) relations are
# excluded -- see the compile-time guard in `init_etrace_state`.
_SUPPORTED_PRIMITIVES = frozenset({etp_mm_p, etp_mv_p})


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
        OTPE v1 -- its trace update and weight-gradient formulas are derived
        one step at a time. ``vjp_method='multi-step'`` is rejected with
        :class:`ValueError` at construction; multi-step *inputs* (i.e. calling
        the compiled learner with :class:`~braintrace.MultiStepData`) raise
        :class:`NotImplementedError` instead, at call time.

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
        :math:`(0, 1)`, if ``vjp_method`` is not ``'single-step'``, if a
        weight-to-hidden relation reaches more than one HiddenGroup (OTPE v1
        requires one-hop per-layer relations), or (at :meth:`compile_graph`)
        if a trained connection projects into a hidden group with
        ``num_state > 1``.
    NotImplementedError
        At :meth:`compile_graph`, if a trained connection is routed through a
        primitive other than dense ``matmul`` (batched or unbatched) -- e.g.
        LoRA, sparse, or convolutional relations, whose weight-gradient chain
        rule does not reduce to OTPE's per-parameter leaky trace.

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
        if vjp_method != 'single-step':
            raise ValueError(
                f"OTPE v1 only supports vjp_method='single-step': its trace "
                f"update and weight-gradient formulas are derived one step at "
                f"a time and have no multi-step form; got "
                f"vjp_method={vjp_method!r}."
            )
        super().__init__(model, name=name, vjp_method=vjp_method)
        self.mode = mode
        self.leak = float(leak)
        self.trace_clip_abs = trace_clip_abs
        self._R_hat: Dict[int, brainstate.ShortTermState] = {}
        self._R_hat_x: Dict[int, brainstate.ShortTermState] = {}
        self._R_hat_g: Dict[int, brainstate.ShortTermState] = {}
        self._R_hat_bias: Dict[int, brainstate.ShortTermState] = {}
        self._rid_is_batched: Dict[int, bool] = {}

    def compile_graph(self, *args: Any) -> None:
        # `super().compile_graph()` builds the graph, calls `init_etrace_state`,
        # and -- only once both succeed -- sets `self.is_compiled = True`. The
        # validation below runs *after* that flag is already True, so any
        # failure here must explicitly reset it (finding M6); otherwise a
        # failed compile would leave `is_compiled` stuck True and silently
        # short-circuit a later, valid `compile_graph` call (the base class's
        # `if not self.is_compiled:` guard would treat it as already compiled).
        super().compile_graph(*args)
        try:
            if self.mode == 'approx':
                n_groups = len(self.graph.hidden_groups)
                if n_groups > 1:
                    warnings.warn(
                        "OTPE(mode='approx') (F-OTPE) adds an extra outer-product "
                        "approximation whose bias compounds with network depth; "
                        "mode='full' reduces (but does not eliminate) this "
                        "depth-dependent bias -- see the Limitations section of "
                        "the class docstring.",
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
        except Exception:
            self.is_compiled = False
            raise

    def init_etrace_state(self, *args: Any, **kwargs: Any) -> None:
        self._R_hat = {}
        self._R_hat_x = {}
        self._R_hat_g = {}
        self._R_hat_bias = {}
        self._rid_is_batched = {}
        for rel in self.graph.hidden_param_op_relations:
            if rel.primitive not in _SUPPORTED_PRIMITIVES:
                raise NotImplementedError(
                    f'OTPE only supports dense matmul relations (etp_mm/etp_mv); '
                    f'got a trained connection routed through primitive '
                    f'{rel.primitive.name!r}. LoRA, sparse, convolutional, and '
                    f'element-wise relations do not reduce to OTPE\'s '
                    f'per-parameter leaky trace and are unsupported.'
                )
            if rel.x_var is None:
                # Unreachable given the primitive guard above (both etp_mm and
                # etp_mv always carry an x_var); kept as an explicit, message-
                # bearing guard rather than a bare assert (see finding N3).
                raise ValueError(
                    f'OTPE requires a relation with an explicit presynaptic '
                    f'input (x_var), but the relation for primitive '
                    f'{rel.primitive.name!r} has none.'
                )
            rid = id(rel.y_var)
            in_shape = rel.x_var.aval.shape
            out_shape = rel.y_var.aval.shape
            batched = is_batched_primitive(rel.primitive)
            self._rid_is_batched[rid] = batched
            if self.mode == 'full':
                weight_key = next(iter(rel.trainable_vars))
                weight_var = rel.trainable_vars[weight_key]
                weight_shape = weight_var.aval.shape
                if batched:
                    shape = (in_shape[0], *weight_shape)
                else:
                    shape = weight_shape
                self._R_hat[rid] = brainstate.ShortTermState(
                    jnp.zeros(shape, dtype=jnp.float32)
                )
                if 'bias' in rel.trainable_vars:
                    self._R_hat_bias[rid] = brainstate.ShortTermState(
                        jnp.zeros(out_shape, dtype=jnp.float32)
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

        def _rezero(rid: int, state: Any) -> None:
            shape = state.value.shape
            if batch_size is not None and self._rid_is_batched.get(rid, False):
                new_shape = (batch_size, *shape[1:])
            else:
                # Unbatched relations (etp_mv) have no batch axis in their
                # trace; `batch_size` does not apply to them, and reusing the
                # stored shape as-is (rather than assuming shape[0] is a batch
                # axis) avoids corrupting it -- see finding M4.
                new_shape = shape
            state.value = jnp.zeros(new_shape, dtype=state.value.dtype)

        for store in (self._R_hat, self._R_hat_x, self._R_hat_g, self._R_hat_bias):
            for rid, r in store.items():
                _rezero(rid, r)

    def _get_etrace_data(self) -> Any:
        if self.mode == 'full':
            return (
                {rid: r.value for rid, r in self._R_hat.items()},
                {rid: r.value for rid, r in self._R_hat_bias.items()},
            )
        return (
            {rid: r.value for rid, r in self._R_hat_x.items()},
            {rid: r.value for rid, r in self._R_hat_g.items()},
        )

    def _assign_etrace_data(self, vals: Any) -> None:
        if self.mode == 'full':
            vals_R, vals_bias = vals
            for rid, v in vals_R.items():
                self._R_hat[rid].value = v
            for rid, v in vals_bias.items():
                self._R_hat_bias[rid].value = v
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
        """``R_hat ← λ·R_hat + ∂s/∂θ_local``. Ignores ``hid2hid_jac``.

        The bias companion trace ``R_hat_bias`` follows the same recursion
        with the presynaptic input dropped (bias's local Jacobian has no ``x``
        factor): ``R_hat_bias ← λ·R_hat_bias + df``, identical in form to
        ``_R_hat_g`` (which the ``'approx'`` mode already maintains for its own
        outer-product factorization).
        """
        if input_is_multi_step:
            raise NotImplementedError('OTPE v1 supports single-step only')
        xs = hid2weight_jac[0]
        dfs = hid2weight_jac[1]

        if self.mode == 'full':
            hist_R, hist_bias = hist_vals
            new_R = {}
            new_bias = {}
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
                updated = self.leak * hist_R[rid] + local
                if self.trace_clip_abs is not None:
                    updated = jnp.clip(
                        updated, -self.trace_clip_abs, self.trace_clip_abs
                    )
                new_R[rid] = updated
                if rid in hist_bias:
                    new_bias[rid] = self.leak * hist_bias[rid] + df_proj
            return (new_R, new_bias)
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
            R_map, Rbias_map = etrace_at_t
            for rel in self.graph.hidden_param_op_relations:
                rid = id(rel.y_var)
                R = R_map[rid]
                group = rel.hidden_groups[0]
                L = dl_to_hidden_groups[group.index].sum(axis=-1)
                per_key = {}
                if is_batched_primitive(rel.primitive):
                    per_key['weight'] = jnp.einsum('bo,bio->io', L, R)
                    if 'bias' in rel.trainable_vars:
                        per_key['bias'] = jnp.einsum('bo,bo->o', L, Rbias_map[rid])
                else:
                    per_key['weight'] = jnp.einsum('o,io->io', L, R)
                    if 'bias' in rel.trainable_vars:
                        per_key['bias'] = L * Rbias_map[rid]
                _route_grads_by_path(rel, per_key, weight_vals, dG)
        else:
            Rx_map, Rg_map = etrace_at_t
            for rel in self.graph.hidden_param_op_relations:
                rid = id(rel.y_var)
                group = rel.hidden_groups[0]
                L = dl_to_hidden_groups[group.index].sum(axis=-1)
                Rx = Rx_map[rid]
                Rg = Rg_map[rid]
                Lg = L * Rg
                per_key = {}
                if is_batched_primitive(rel.primitive):
                    per_key['weight'] = jnp.einsum('bi,bo->io', Rx, Lg)
                    if 'bias' in rel.trainable_vars:
                        # Bias has no "in" dimension: sum the per-batch-row
                        # contributions directly (matches the 'b' contraction
                        # the weight einsum performs above).
                        per_key['bias'] = Lg.sum(axis=0)
                else:
                    per_key['weight'] = jnp.einsum('i,o->io', Rx, Lg)
                    if 'bias' in rel.trainable_vars:
                        per_key['bias'] = Lg
                _route_grads_by_path(rel, per_key, weight_vals, dG)

        for path, dg in dl_to_nonetws_at_t.items():
            _update_dict(dG, path, dg)
        if dl_to_etws_at_t is not None:
            for path, dg in dl_to_etws_at_t.items():
                _update_dict(dG, path, dg, error_when_no_key=True)
        return dG
