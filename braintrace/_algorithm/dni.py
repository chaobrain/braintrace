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

"""DNI -- Decoupled Neural Interfaces / synthetic gradients (Jaderberg et al., 2017).

A truncated window throws away the cotangent that would have arrived at its exit
from the future. For the ETP parameters that loss is already made good: the
eligibility trace carries their cross-window credit forward, which is what a trace
is *for*. For every **plain** parameter -- an input projection, a readout, anything
not routed through an ETP primitive -- it is simply lost, and no trace exists to
recover it.

DNI learns to predict that missing cotangent. A small synthesiser ``M`` maps the
window-exit hidden state to an estimate of ``dL_future/dh^exit``, the estimate is
injected ahead of the window's own reverse pass, and the sum over windows
telescopes to the exact gradient. See :class:`DNI` for the scope of that claim
and :func:`train_synthetic_gradient` for the recipe that fits ``M``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

from braintrace._compiler import ControlFlowPolicy, DEFAULT_MAX_JACOBIAN_ELEMENTS
from braintrace._typing import Path
from .axes import ETraceConfig
from .param_dim_vjp import ParamDimVjpAlgorithm

__all__ = ['SyntheticGradient', 'DNI', 'train_synthetic_gradient']


class SyntheticGradient(brainstate.nn.Module):
    r"""A per-hidden-group linear synthesiser of the future cotangent.

    One affine map per hidden group, taking that group's concatenated hidden
    value ``(*varshape, num_state)`` to a cotangent of the same shape. Linear
    with a bias, as in the paper: the target is itself a gradient, so a linear
    predictor is a reasonable hypothesis class and keeps the auxiliary problem
    convex.

    The parameters are **not** ETP routed. They are ordinary
    :class:`brainstate.ParamState`\ s that never appear in a
    ``hidden_param_op_relation``, because the synthesiser is not part of the
    model's recurrence: it observes ``h^exit`` and predicts, it does not
    participate in producing ``h``. :class:`DNI` keeps them out of the compiled
    graph by construction -- it is handed the *values* functionally rather than
    closing over the states.

    The final layer is zero-initialised, so a freshly constructed synthesiser
    predicts exactly zero and the learner starts bit-identical to the plain
    truncated rule. That is a deliberate property: it makes "DNI is off" and "DNI
    is untrained" the same run, so B1's no-op criterion is checkable.

    Parameters
    ----------
    group_shapes : dict of int to tuple
        ``group index -> (*varshape, num_state)``, from the compiled graph.
    hidden_width : int, optional
        Unused; kept for signature stability.
    scale : float, optional
        Standard deviation of the input-layer draw. Default ``0.0`` -- see the
        zero-initialisation note above; a non-zero value makes the synthesiser
        live from the start, which the B1 negative control needs.
    seed : int, optional
        Seed for the draws.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> synth = braintrace.SyntheticGradient({0: (1, 4, 1)})
        >>> values = synth.param_values()
        >>> est = synth.apply(values, {0: jnp.zeros((1, 4, 1))})
        >>> est[0].shape
        (1, 4, 1)
    """

    __module__ = 'braintrace'

    def __init__(
        self,
        group_shapes: Dict[int, tuple],
        hidden_width: Optional[int] = None,
        scale: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.group_shapes = {int(k): tuple(v) for k, v in group_shapes.items()}
        self.weights: Dict[int, brainstate.ParamState] = {}
        self.biases: Dict[int, brainstate.ParamState] = {}
        with brainstate.random.seed_context(seed):
            for gid, shape in self.group_shapes.items():
                width = int(shape[-2]) * int(shape[-1])
                w = scale * brainstate.random.randn(width, width)
                self.weights[gid] = brainstate.ParamState(
                    jnp.asarray(w, dtype=jnp.float32))
                self.biases[gid] = brainstate.ParamState(
                    jnp.zeros((width,), dtype=jnp.float32))

    def param_values(self) -> Dict[str, Any]:
        """The synthesiser's parameter *values*, for the functional call.

        Returns
        -------
        dict
            ``{'w': {gid: array}, 'b': {gid: array}}``.
        """
        return {
            'w': {gid: st.value for gid, st in self.weights.items()},
            'b': {gid: st.value for gid, st in self.biases.items()},
        }

    def states_dict(self) -> Dict[tuple, brainstate.ParamState]:
        """The synthesiser's :class:`brainstate.ParamState`\\ s, keyed for an optimiser."""
        out: Dict[tuple, brainstate.ParamState] = {}
        for gid, st in self.weights.items():
            out[('synth_w', gid)] = st
        for gid, st in self.biases.items():
            out[('synth_b', gid)] = st
        return out

    def apply(
        self,
        param_values: Dict[str, Any],
        group_hiddens: Dict[int, Any],
    ) -> Dict[int, jax.Array]:
        """Predict each group's future cotangent, functionally.

        Parameters
        ----------
        param_values : dict
            As returned by :meth:`param_values`. Passed explicitly rather than
            read off ``self`` so that a caller inside ``jax.custom_vjp`` never
            captures a tracer it might later be asked to differentiate.
        group_hiddens : dict of int to array
            ``group index -> concatenated hidden value``.

        Returns
        -------
        dict of int to jax.Array
            ``group index -> estimated cotangent``, shaped like the input.
        """
        out = {}
        for gid, h in group_hiddens.items():
            shape = tuple(u.math.shape(h))
            flat = jnp.reshape(u.get_mantissa(h), shape[:-2] + (-1,))
            est = flat @ param_values['w'][gid] + param_values['b'][gid]
            out[gid] = jnp.reshape(est, shape)
        return out


class DNI(ParamDimVjpAlgorithm):
    r"""Decoupled Neural Interfaces: a learned estimate of the truncated future.

    The coordinate is :class:`~braintrace.D_RTRL`'s with
    ``learning_signal='bootstrapped'``, and unlike the other two P4 presets it is
    **multi-step**: the whole point is a window with an exit, and a one-step
    window has almost no truncated future to estimate.

    What DNI fixes, precisely. Index windows ``[a_k, b_k)`` with ``b_k = a_{k+1}``
    and let ``l_t`` be the loss of the step that writes ``h^{t+1}``. The injected
    estimate is

    .. math::

        g_k \;\approx\; \frac{\partial \sum_{t \ge b_k} l_t}{\partial h^{b_k}}

    -- strictly future, half-open, so the exit step's own loss lies *inside*
    window ``k`` and is not counted twice.

    It reaches the plain parameters, the inputs and the other states, where the
    sum over windows then telescopes to the exact gradient. It deliberately does
    **not** reach the ETP parameters or the boundary learning signal: their
    cross-window credit is already carried by the eligibility trace, and adding
    the estimate there would count the same path a second time. So DNI does not
    make the ETP gradients better or worse -- with a synthesiser attached they are
    bit-identical to the plain run -- it gives the *plain* parameters the credit
    the trace already gives the ETP ones.

    Parameters
    ----------
    model : brainstate.nn.Module
        The one-step model.
    synthesizer : SyntheticGradient, optional
        The estimator. May be attached later via :meth:`attach_synthesizer`; a
        window that runs without one raises rather than silently degrading to the
        truncated rule.
    name : str, optional
        Node name.
    vjp_method : str, optional
        Must be ``'multi-step'`` (the default).
    fast_solve : bool, optional
        Whether registered closed-form kernels may be used. Default ``True``.
    trace_dtype : DTypeLike, optional
        Reduced trace precision.
    chunked_trace : bool, optional
        Whether to roll the trace in chunks. Default ``True``.
    control_flow : ControlFlowPolicy, optional
        Control-flow canonicalization policy.
    snap_max_jacobian_elements : int, optional
        Passed through; unused at ``recurrence_scope='diagonal'``.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate, braintrace, jax.numpy as jnp
        >>> model = ...                                    # doctest: +SKIP
        >>> learner = braintrace.DNI(model)                # doctest: +SKIP
        >>> learner.compile_graph(braintrace.MultiStepData(xs))   # doctest: +SKIP
        >>> learner.init_etrace_state()                    # doctest: +SKIP
        >>> learner.attach_synthesizer(
        ...     braintrace.SyntheticGradient(learner.group_signal_shapes()))  # doctest: +SKIP

    Notes
    -----
    The synthesiser's parameter *values* are threaded into the traced computation
    as an explicit argument rather than closed over, so ``jax.custom_vjp`` never
    captures a tracer it might be asked to differentiate, and the estimate is
    wrapped in ``stop_gradient`` so the online loss cannot train the synthesiser
    through the wrong path. Fit it with :func:`train_synthetic_gradient`.

    See Also
    --------
    braintrace.SyntheticGradient : the estimator.
    train_synthetic_gradient : the fitting recipe.
    braintrace.D_RTRL : the same coordinate with ``learning_signal='symmetric'``.
    """

    __module__ = 'braintrace'

    #: D_RTRL's coordinate with a bootstrapped exit cotangent.
    _default_config = ETraceConfig(
        trace_factorization='per_param',
        temporal_recursion='jacobian',
        recurrence_scope='diagonal',
        learning_signal='bootstrapped',
    )

    def __init__(
        self,
        model: brainstate.nn.Module,
        synthesizer: Optional[SyntheticGradient] = None,
        name: Optional[str] = None,
        vjp_method: str = 'multi-step',
        fast_solve: bool = True,
        trace_dtype: Any = None,
        chunked_trace: bool = True,
        control_flow: Optional[ControlFlowPolicy] = None,
        snap_max_jacobian_elements: int = DEFAULT_MAX_JACOBIAN_ELEMENTS,
    ) -> None:
        super().__init__(
            model,
            name=name,
            vjp_method=vjp_method,
            fast_solve=fast_solve,
            trace_dtype=trace_dtype,
            chunked_trace=chunked_trace,
            control_flow=control_flow,
            config=self._default_config,
            snap_max_jacobian_elements=snap_max_jacobian_elements,
        )
        self.synthesizer = synthesizer

    def attach_synthesizer(self, synthesizer: SyntheticGradient) -> None:
        """Attach (or replace) the estimator after construction.

        Parameters
        ----------
        synthesizer : SyntheticGradient
            The estimator. Its group shapes must match
            :meth:`group_signal_shapes`.
        """
        self.synthesizer = synthesizer

    def group_signal_shapes(self) -> Dict[int, tuple]:
        """``group index -> (*varshape, num_state)`` for the compiled graph.

        Returns
        -------
        dict of int to tuple
            The shapes a :class:`SyntheticGradient` must emit.
        """
        self._assert_compiled()
        return {
            group.index: tuple(group.varshape) + (group.num_state,)
            for group in self.graph.hidden_groups
        }

    def _get_update_aux(self) -> Any:
        """The synthesiser's parameter values, threaded in as a real argument.

        Returns
        -------
        dict or None
            ``{'w': ..., 'b': ...}``.

        Raises
        ------
        RuntimeError
            If no synthesiser is attached. Falling back to a zero estimate would
            silently compute the plain truncated rule under a name that promises
            otherwise.
        """
        if self.synthesizer is None:
            raise RuntimeError(
                "learning_signal='bootstrapped' was configured but no "
                'synthesizer is attached, so the exit cotangent would silently '
                'stay zero -- the plain truncated rule under a different name. '
                'Pass one to the constructor or call '
                '`attach_synthesizer(SyntheticGradient(learner.group_signal_shapes()))`.'
            )
        return self.synthesizer.param_values()

    def _inject_exit_cotangent(
        self, exit_hiddens: Dict[Path, Any], aux: Any
    ) -> Optional[Dict[Path, Any]]:
        """Map ``h^exit`` through the synthesiser and back to per-path cotangents.

        Parameters
        ----------
        exit_hiddens : dict
            Hidden-path-keyed window-exit values.
        aux : dict
            The synthesiser's parameter values from :meth:`_get_update_aux`.

        Returns
        -------
        dict or None
            Hidden-path-keyed cotangents.
        """
        if aux is None or self.synthesizer is None:
            return None

        groups = self.graph.hidden_groups
        group_hiddens = {
            group.index: group.concat_hidden(
                [u.get_mantissa(exit_hiddens[path]) for path in group.hidden_paths])
            for group in groups
        }
        estimates = self.synthesizer.apply(aux, group_hiddens)

        out: Dict[Path, Any] = {}
        for group in groups:
            # `stop_gradient` here, not in the synthesiser: the estimate must be
            # a constant as far as the *online* loss is concerned, or that loss
            # would train the synthesiser through the injection path instead of
            # through its own regression objective.
            est = jax.lax.stop_gradient(estimates[group.index])
            for path, part in zip(group.hidden_paths, group.split_hidden(est)):
                out[path] = part
        return out


def train_synthetic_gradient(
    learner: DNI,
    inputs,
    *,
    chunk_size: int = 1,
    loss_fn: Optional[Callable] = None,
    optimizer=None,
    lr: float = 1e-2,
    epochs: int = 1,
    reset: bool = True,
) -> list:
    r"""Fit the synthesiser against the learner's own returned hidden cotangent.

    The regression target for ``M(h^{a_k})`` is ``dL_{>= a_k}/dh^{a_k}``, and the
    learner already produces exactly that: with the hidden states in the
    differentiation set, ``brainstate.transform.grad`` returns the window's
    hidden cotangent, which (by the second pass of
    ``_update_fn_bwd``) already carries the future term. So no new side channel is
    needed -- the target comes out of the public API.

    Two properties make the fit honest, and both are enforced here:

    * the **model** parameters do not move -- only the synthesiser's are handed
      to the optimiser;
    * the **target is detached**, so the regression cannot reshape the model's
      gradients to make itself easy to predict.

    The auxiliary optimiser is left to the caller, as it is in the paper.

    Parameters
    ----------
    learner : DNI
        A compiled learner with a synthesiser attached.
    inputs : array
        A ``(T, ...)`` sequence, consumed one window at a time.
    chunk_size : int, optional
        Steps per window. This **must match the window size the learner will be
        driven with**: the synthesiser predicts the future at a window boundary,
        and boundaries move when the window size does. Training on one-step
        windows and then deploying on longer ones fits the wrong target -- a
        much shorter future -- and the result can easily be worse than no
        synthesiser at all. Default ``1``.
    loss_fn : callable, optional
        ``loss_fn(output) -> scalar``. This **must be the objective the learner
        will actually be trained on**, and it is the second half of the same trap
        ``chunk_size`` documents. The synthesiser predicts
        ``dL_{>= b}/dh^b`` -- a derivative *of this loss*. Fit it against the
        default sum-of-squares and then descend on, say,
        ``((out - target) ** 2).mean()``, and the injected cotangent is the
        gradient of a different function at a different scale: not an
        approximation of the future credit but noise with the shape of one.
        Measured on the delayed-reward fixture, a mismatched ``loss_fn`` left the
        run *worse* than leaving DNI off entirely. Default: sum of squares.
    optimizer : braintools.optim.Optimizer, optional
        Already registered against ``learner.synthesizer.states_dict()``. If
        ``None``, a plain SGD step with learning rate ``lr`` is applied.
    lr : float, optional
        Learning rate for the built-in SGD step. Ignored when ``optimizer`` is
        given. Default ``1e-2``.
    epochs : int, optional
        Passes over ``inputs``.
    reset : bool, optional
        Whether to re-initialise the model states and the trace before each
        epoch. Default ``True``.

    Returns
    -------
    list of float
        The mean squared prediction error per epoch.
    """
    if learner.synthesizer is None:
        raise RuntimeError(
            'train_synthetic_gradient needs a synthesizer attached to the '
            'learner; call `learner.attach_synthesizer(...)` first.')
    loss_fn = loss_fn or (lambda out: (out ** 2).sum())
    synth_states = learner.synthesizer.states_dict()
    hidden_states = dict(learner.hidden_states)
    history = []

    for _ in range(epochs):
        if reset:
            brainstate.nn.init_all_states(
                learner.graph_executor.model, batch_size=1)
            learner.init_etrace_state()
        errors = []
        for start in range(0, inputs.shape[0], chunk_size):
            window = inputs[start:start + chunk_size]
            # The entry hiddens are what the synthesiser sees; snapshot before
            # the window advances them.
            entry = {path: st.value for path, st in hidden_states.items()}
            grads = brainstate.transform.grad(
                lambda seq: loss_fn(learner(_as_window(seq, chunk_size))),
                hidden_states)(window)
            # Detached: the regression must not be able to reshape the model's
            # gradients into something easier to predict.
            target = jax.tree.map(jax.lax.stop_gradient, grads)

            groups = learner.graph.hidden_groups
            group_entry = {
                g.index: g.concat_hidden(
                    [u.get_mantissa(entry[p]) for p in g.hidden_paths])
                for g in groups
            }
            group_target = {
                g.index: g.concat_hidden(
                    [u.get_mantissa(target[p]) for p in g.hidden_paths])
                for g in groups
            }

            def regression(values):
                pred = learner.synthesizer.apply(values, group_entry)
                return sum(jnp.mean((pred[gi] - group_target[gi]) ** 2)
                           for gi in group_entry)

            g_synth = brainstate.transform.grad(
                lambda: regression(learner.synthesizer.param_values()),
                synth_states)()
            errors.append(float(regression(learner.synthesizer.param_values())))

            if optimizer is None:
                for key, st in synth_states.items():
                    st.value = st.value - lr * g_synth[key]
            else:
                optimizer.update(g_synth)
        history.append(float(sum(errors) / max(len(errors), 1)))
    return history


def _as_window(seq, chunk_size: int):
    """Wrap a window for the learner, matching how it will be driven."""
    import braintrace
    if chunk_size == 1:
        return seq[0]
    return braintrace.MultiStepData(seq)
