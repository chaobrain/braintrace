# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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


from __future__ import annotations

from typing import Callable, Dict, Tuple, Any, List, Optional, Sequence

import brainstate
import jax
import jax.numpy as jnp
import brainunit as u

from braintrace._input_data import has_multistep_data
from braintrace._state_management import assign_state_values_v2
from braintrace._typing import (
    Path,
    PyTree,
    Outputs,
    WeightID,
    WeightVals,
    HiddenVals,
    StateVals,
    ETraceVals,
    Hid2WeightJacobian,
    dG_Inputs,
    dG_Weight,
    dG_Hidden,
    dG_State,
)
from braintrace._compiler import ControlFlowPolicy, DEFAULT_MAX_JACOBIAN_ELEMENTS
from braintrace._op import is_snap_anchored
from ._common import FixedRandomFeedback
from .axes import ETraceConfig
from .base import ETraceAlgorithm
from .vjp_graph_executor import ETraceVjpGraphExecutor

__all__ = [
    'ETraceVjpAlgorithm',  # the base class for the eligibility trace algorithm with the VJP gradient computation
]


def _add_future_for_plain_paths(base: Any, future: Any, etp_paths: Any) -> Any:
    """Add a second-pass parameter-gradient tree, skipping ETP-routed paths.

    The plain parameters need the future cotangent -- it is the credit their
    truncated window threw away, and adding it makes the sum over windows
    telescope. The ETP-routed ones must not get it: their cross-window credit is
    already carried by the eligibility trace, so adding it here would count the
    same path twice.

    Parameters
    ----------
    base : dict or None
        Path-keyed gradients from the first pass.
    future : dict or None
        Path-keyed gradients from the injected second pass, same structure.
    etp_paths : set
        Paths whose credit the trace already carries.

    Raises
    ------
    KeyError
        If ``future`` carries a path ``base`` does not. Both trees are unflattened
        from the *same* ``in_tree``, so their key sets are identical by
        construction and this cannot happen. It is checked rather than skipped
        because the failure mode would be silent: a dropped path loses exactly the
        cross-window credit this function exists to deliver, and every one of
        DNI's invariance tests would still pass -- which is how F-34 survived its
        first implementation.

    Returns
    -------
    dict or None
        ``base`` with the plain-path entries of ``future`` added in.
    """
    if not base or not future:
        return base
    missing = set(future).difference(base)
    if missing:
        raise KeyError(
            f'the second backward pass produced parameter gradients for paths '
            f'the first pass did not: {sorted(missing)}. Both trees come from '
            f'the same `in_tree`, so this indicates the two passes have drifted '
            f'apart; adding them by path would silently drop these.')
    out = dict(base)
    for path, val in future.items():
        if path in etp_paths:
            continue
        out[path] = jax.tree.map(lambda x, y: x + y, out[path], val,
                                 is_leaf=u.math.is_quantity)
    return out


def _static_dtype(x: Any) -> Any:
    """The dtype of ``x`` without materializing or converting it.

    Read straight off the array/tracer when it has one, so that JAX's
    ``float0`` cotangent dtype -- which does not survive ``jnp.result_type`` --
    comes through intact, and so that nothing is moved to a device.

    Parameters
    ----------
    x : Any
        An array, tracer, ``Quantity`` mantissa or Python scalar.

    Returns
    -------
    numpy.dtype
        The dtype of ``x``.
    """
    dtype = getattr(x, 'dtype', None)
    if dtype is not None:
        return dtype
    return jnp.result_type(x)


def _expected_hidden_shape(state: brainstate.HiddenState) -> Tuple[int, ...]:
    """The shape a cotangent for ``state`` must have.

    A plain :class:`brainstate.HiddenState` carries one state per position, so
    its cotangent is ``varshape``-shaped; a
    :class:`brainstate.HiddenGroupState` stacks ``num_state`` of them on a
    trailing axis, which is exactly the axis
    :meth:`~braintrace.HiddenGroup.concat_hidden` concatenates along.

    Parameters
    ----------
    state : brainstate.HiddenState
        The hidden state whose cotangent shape is wanted.

    Returns
    -------
    tuple of int
        The required cotangent shape.
    """
    varshape = tuple(state.varshape)
    if isinstance(state, brainstate.HiddenGroupState):
        return varshape + (int(state.num_state),)
    return varshape


def _check_hidden_gradient_correspondence(
    hidden_groups: Sequence[Any],
    path_to_cotangent: Dict[Path, Any],
    *,
    source: str,
) -> None:
    """Check that cotangent *i* really belongs to hidden state *i*.

    The backward pass hands back a collection of hidden-state cotangents which
    is then re-ordered onto the hidden groups by path and concatenated. Nothing
    downstream re-derives which hidden state a given cotangent came from, so a
    mis-ordered or mis-shaped collection produces a **wrong gradient rather than
    an error**. This function is the contract that makes that impossible.

    Three things are checked, for every hidden group and every position in it:

    1. **Totality** -- every path a group needs is present in
       ``path_to_cotangent``.
    2. **No strays** -- ``path_to_cotangent`` carries no path that no group
       claims, which would mean the cotangent collection and the compiled graph
       describe different models.
    3. **Correspondence** -- the cotangent at each path has the shape and dtype
       of the hidden state at that path (see :func:`_expected_hidden_shape`),
       compared after :func:`brainunit.get_mantissa`, exactly as the consumers
       strip units before concatenating.

    Parameters
    ----------
    hidden_groups : sequence of HiddenGroup
        The compiled hidden groups the cotangents will be routed onto.
    path_to_cotangent : dict
        Mapping from hidden-state path to that state's cotangent.
    source : str
        Human-readable name of the branch that produced ``path_to_cotangent``,
        used in the error messages so the reader is not left guessing which of
        the two VJP modes failed.

    Raises
    ------
    ValueError
        If any of the three conditions above does not hold. The message names
        the hidden group, the position within it, the path, and the expected
        versus actual shape/dtype.

    Notes
    -----
    Deliberately ``if ... raise`` and not ``assert``: ``python -O`` /
    ``PYTHONOPTIMIZE=1`` strips ``assert`` statements, and this guard has to
    hold on an optimised interpreter too.

    Every comparison is on Python-level metadata -- dict keys, static shapes,
    dtypes -- and never on array *data*. Nothing here is traced into the
    compiled program, so it adds no XLA operation and forces no device sync,
    even though it runs once per backward pass.
    """
    needed: Dict[Path, Tuple[int, int, Any]] = {}
    for group in hidden_groups:
        for position, (path, state) in enumerate(zip(group.hidden_paths, group.hidden_states)):
            needed.setdefault(path, (group.index, position, state))

    missing = [path for path in needed if path not in path_to_cotangent]
    if missing:
        group_index, position, _ = needed[missing[0]]
        raise ValueError(
            f'The {source} did not provide a gradient for every hidden state. '
            f'Hidden group {group_index} needs {missing[0]} at position '
            f'{position}, and {len(missing)} needed path(s) are absent: '
            f'{sorted(map(str, missing))}. The gradients provided are for '
            f'{sorted(map(str, path_to_cotangent))}.'
        )

    strays = [path for path in path_to_cotangent if path not in needed]
    if strays:
        raise ValueError(
            f'The {source} provided gradients for {len(strays)} hidden state '
            f'path(s) that no hidden group claims: {sorted(map(str, strays))}. '
            f'The hidden groups cover {sorted(map(str, needed))}. The compiled '
            f'graph and the backward pass disagree about the model\'s hidden '
            f'states; recompile the graph.'
        )

    for path, (group_index, position, state) in needed.items():
        cotangent = u.get_mantissa(path_to_cotangent[path])
        want_shape = _expected_hidden_shape(state)
        got_shape = tuple(u.math.shape(cotangent))
        if got_shape != want_shape:
            raise ValueError(
                f'The {source} produced a gradient of shape {got_shape} for the '
                f'hidden state {path}, which is at position {position} of hidden '
                f'group {group_index} and has shape {want_shape}. A gradient '
                f'cannot belong to a hidden state of a different shape, so the '
                f'gradients and the hidden states are not in correspondence.'
            )
        want_dtype = _static_dtype(u.get_mantissa(state.value))
        got_dtype = _static_dtype(cotangent)
        # `float0` is JAX's cotangent dtype for a non-differentiable leaf; it is
        # a correct cotangent for an integer or boolean hidden state and must not
        # be read as a mismatch.
        if got_dtype != want_dtype and got_dtype != jax.dtypes.float0:
            raise ValueError(
                f'The {source} produced a gradient of dtype {got_dtype} for the '
                f'hidden state {path}, which is at position {position} of hidden '
                f'group {group_index} and has dtype {want_dtype}. A gradient '
                f'cannot belong to a hidden state of a different dtype, so the '
                f'gradients and the hidden states are not in correspondence.'
            )


def expand_modulator_to_group(
    modulator: Any,
    group_shape: Tuple[int, ...],
    *,
    group_index: int,
    hidden_paths: Any = None,
) -> Any:
    r"""Expand a modulatory signal to one hidden group's signal shape.

    Implements the ``learning_signal='modulatory'`` broadcasting contract. It is
    deliberately *not* bare NumPy broadcasting: NumPy aligns trailing axes, so a
    ``(1, n_rec)`` modulator does **not** broadcast to a ``(1, n_rec, 1)``
    signal -- it would raise, or worse, silently align ``n_rec`` against the
    state axis when ``num_state == n_rec``. The contract, matching AGENTS.md's
    SNN learning-signal trailing-axis rule:

    * a scalar is broadcast to the whole group shape;
    * a modulator shaped exactly like the group's ``varshape`` gains a trailing
      size-1 state axis first, then broadcasts;
    * anything else must broadcast against ``(*varshape, num_state)`` as given.

    The expansion is driven only by shapes, never by group index or group count.
    A scalar reward is therefore valid for any model whatever its HiddenGroup
    count, which is the property that keeps this axis general (roadmap risk 5).

    Parameters
    ----------
    modulator : array_like or Quantity
        The user-supplied signal. A ``list`` or ``tuple`` is refused: see Raises.
    group_shape : tuple of int
        The target ``(*varshape, num_state)``.
    group_index : int
        Index of the hidden group, used only in error messages.
    hidden_paths : optional
        The group's hidden paths, used only in error messages.

    Returns
    -------
    jax.Array or Quantity
        An array of exactly ``group_shape``.

    Raises
    ------
    TypeError
        If ``modulator`` is a list or tuple. A length-``n_groups`` sequence is
        the binding that made OSTTP non-general, and there is deliberately no
        spelling for it.
    ValueError
        If the modulator cannot be broadcast, naming the group and both shapes.
    """
    if isinstance(modulator, (list, tuple)):
        raise TypeError(
            f'The modulatory signal must be ONE array (or scalar) expanded to '
            f'every hidden group, but a {type(modulator).__name__} of length '
            f'{len(modulator)} was given. There is deliberately no per-group '
            f'sequence spelling: binding the signal to the hidden-group '
            f'decomposition is what made OSTTP non-general, since the group '
            f'count is a property of the compiled graph rather than of the '
            f'task. A scalar reward is valid for any model, whatever its group '
            f'count. To vary the signal across units, pass one array shaped '
            f'like the group (or its varshape) instead.'
        )

    m = modulator if isinstance(modulator, u.Quantity) else u.math.asarray(modulator)
    group_shape = tuple(group_shape)

    if m.ndim > 0 and tuple(m.shape) == group_shape[:-1]:
        # varshape -> varshape + (1,): the trailing state axis.
        m = u.math.reshape(m, tuple(m.shape) + (1,))

    try:
        return u.math.broadcast_to(m, group_shape)
    except Exception as e:
        raise ValueError(
            f'The modulatory signal of shape {tuple(u.math.shape(modulator))} '
            f'cannot be expanded to hidden group {group_index} '
            f'(hidden_paths={hidden_paths}), whose signal shape is '
            f'{group_shape}. Pass a scalar, an array shaped like the group '
            f'varshape {group_shape[:-1]}, or an array that broadcasts against '
            f'{group_shape}.'
        ) from e


class ETraceVjpAlgorithm(ETraceAlgorithm):
    r"""Provide VJP-based eligibility-trace gradient computation.

    The term ``VJP`` comes from two aspects. First, this module is designed to be
    compatible with JAX's VJP mechanism, so the gradient is computed according to the
    reverse-mode differentiation interface, like ``jax.grad``, ``jax.vjp``, or
    ``jax.jacrev``. The true update function is defined as a custom VJP function
    ``._true_update_fun()``, which receives the inputs, the hidden states, other states,
    and etrace variables at the last time step, and returns the outputs, the hidden
    states, other states, and etrace variables at the current time step. Second, the
    algorithm computes the spatial gradient :math:`\partial L^t / \partial H^t` using the
    standard back-propagation algorithm, which enhances the accuracy and the stability of
    the gradient computation.

    Parameters
    ----------
    model : brainstate.nn.Module
        The model function, which receives the input arguments and returns the model output.
    name : str, optional
        The name of the etrace algorithm.
    vjp_method : str, optional
        The method for computing the VJP. It should be either ``"single-step"`` or
        ``"multi-step"``. Default is ``"single-step"``.

        - ``"single-step"``: The VJP is computed at the current time step, i.e.,
          :math:`\partial L^t/\partial h^t`.
        - ``"multi-step"``: The VJP is computed at multiple time steps, i.e.,
          :math:`\partial L^t/\partial h^{t-k}`, where :math:`k` is determined by the
          data input.
    control_flow : ControlFlowPolicy, optional
        Policy governing control-flow canonicalization (cond if-conversion,
        scan unrolling, structured scan descent, ...) during graph
        compilation. ``None`` (default) uses
        ``ControlFlowPolicy()``.
    config : ETraceConfig, optional
        Learning-rule coordinates. ``None`` uses the subclass's preset
        coordinate.
    random_feedback_key : jax.Array, optional
        Key used to initialize fixed random-feedback projections when
        ``config.learning_signal='random_feedback'``.
    snap_max_jacobian_elements : int, optional
        Maximum number of elements permitted in each SnAp widened block
        Jacobian. The default is ``16777216``.

    Notes
    -----
    For each subclass (or the instance of an etrace algorithm), the following methods
    define the custom VJP rule:

    - ``._update()``: update the eligibility trace states and return the outputs, hidden
      states, other states, and etrace data.
    - ``._update_fwd()``: the forward pass of the custom VJP rule.
    - ``._update_bwd()``: the backward pass of the custom VJP rule.

    This class provides a default implementation for the ``._update()``,
    ``._update_fwd()``, and ``._update_bwd()`` methods. To implement a new etrace
    algorithm, users just need to override the following methods:

    - ``._solve_weight_gradients()``: solve the gradients of the learnable weights / parameters.
    - ``._update_etrace_data()``: update the eligibility trace data.
    - ``._assign_etrace_data()``: assign the eligibility trace data to the states.
    - ``._get_etrace_data()``: get the eligibility trace data.
    """

    __module__ = 'braintrace'
    graph_executor: ETraceVjpGraphExecutor

    #: The learning-rule coordinate this algorithm implements. Subclasses set a
    #: default; callers may pass one explicitly. See :class:`ETraceConfig`.
    config: ETraceConfig

    #: Default coordinate when the caller passes no ``config``. Subclasses
    #: override this with their own preset coordinate.
    _default_config: ETraceConfig = ETraceConfig()

    #: ``group index -> fixed random projection``, allocated by
    #: :meth:`init_etrace_state` when ``learning_signal='random_feedback'``.
    _random_feedback: Dict[int, FixedRandomFeedback]

    #: The standing modulatory signal for ``learning_signal='modulatory'``,
    #: assignable between calls. ``update(..., modulator=m)`` takes precedence
    #: for that one call. There is no fallback to ``symmetric``: a missing
    #: modulator raises, because silently computing a different learning rule
    #: than the configured one is worse than failing.
    modulator: Any = None

    #: Per-call modulator from the ``update(..., modulator=...)`` keyword. Set at
    #: the top of :meth:`update` and cleared in its ``finally``, so an exception
    #: mid-update cannot leak a stale signal into the next call.
    _modulator_this_call: Any = None

    def __init__(
        self,
        model: brainstate.nn.Module,
        name: Optional[str] = None,
        vjp_method: str = 'single-step',
        control_flow: Optional[ControlFlowPolicy] = None,
        config: Optional[ETraceConfig] = None,
        random_feedback_key: Optional[jax.Array] = None,
        snap_max_jacobian_elements: int = DEFAULT_MAX_JACOBIAN_ELEMENTS,
    ):

        # the VJP method
        assert vjp_method in ('single-step', 'multi-step'), (
            'The VJP method should be either "single-step" or "multi-step". '
            f'While we got {vjp_method}. '
        )
        self.vjp_method = vjp_method

        # the learning-rule coordinate
        if config is None:
            config = self._default_config
        elif not isinstance(config, ETraceConfig):
            raise TypeError(
                f'config must be an ETraceConfig, got {type(config).__name__}.')
        self.config = config
        self._random_feedback_key = random_feedback_key
        self._random_feedback = {}
        if (config.learning_signal == 'random_feedback'
                and random_feedback_key is None):
            raise ValueError(
                "learning_signal='random_feedback' needs a "
                '`random_feedback_key` to draw the fixed projection matrices '
                'from, so the run is reproducible. Pass one, e.g. '
                'random_feedback_key=brainstate.random.split_key().'
            )
        if (config.learning_signal == 'modulatory'
                and vjp_method != 'single-step'):
            raise ValueError(
                "learning_signal='modulatory' requires "
                "vjp_method='single-step', but got "
                f"vjp_method={vjp_method!r}. Under multi-step, "
                '`_solve_weight_gradients` adds `dl_to_etws_at_t` -- the '
                'within-window reverse-AD gradient of the ETP parameters -- on '
                'top of the trace contraction. Replacing the *boundary* signal '
                'would therefore leave that in-window half unmodulated, giving '
                'a hybrid whose gradient is part three-factor and part plain '
                'loss gradient: not the rule this axis names. Single-step makes '
                'every ETP contribution flow through the replaced signal, and '
                'makes the modulator per step, which is what a neuromodulator '
                'is.'
            )
        if (config.learning_signal == 'bootstrapped'
                and type(self)._inject_exit_cotangent
                is ETraceVjpAlgorithm._inject_exit_cotangent):
            raise NotImplementedError(
                "learning_signal='bootstrapped' needs a subclass that overrides "
                '`_inject_exit_cotangent` to supply the estimate of the future '
                f'hidden cotangent, and {type(self).__name__} does not. Left '
                'unchecked this is not an error but something worse: the default '
                'hook returns None, so the second backward pass never runs and '
                'the rule silently degrades to exactly `symmetric` -- a '
                'configuration that reads as DNI and behaves as if the axis were '
                'never set. Use `braintrace.DNI`, or override the hook.'
            )

        # graph
        graph_executor = ETraceVjpGraphExecutor(
            model,
            vjp_method=vjp_method,
            include_recurrent_mixing=config.include_recurrent_mixing,
            sparse_n=config.sparse_n,
            snap_max_jacobian_elements=snap_max_jacobian_elements,
            control_flow=control_flow,
            # A rank-1 estimator only pays for itself against the *full*
            # within-group recursion (matrix rule 11 pins the scope), so the
            # random-projection engine needs the whole transition Jacobian
            # rather than its per-position blocks. One constructor flag, no
            # compiler metadata -- unlike ``sparse_n``, which needs
            # ``group.snap`` built at compile time.
            full_jacobian=config.trace_factorization == 'random_projection',
        )

        # super initialization
        super().__init__(model=model, name=name, graph_executor=graph_executor)

        # the update rule
        self._true_update_fun = jax.custom_vjp(self._update_fn)
        self._true_update_fun.defvjp(
            fwd=self._update_fn_fwd,
            bwd=self._update_fn_bwd
        )

    def _assert_compiled(self) -> None:
        if not self.is_compiled:
            raise ValueError('The etrace algorithm has not been compiled. Please call `compile_graph()` first. ')

    # ------------------------------------------------------------------ #
    # axis: recurrence_scope
    # ------------------------------------------------------------------ #

    def compile_graph(self, *args: Any) -> None:
        super().compile_graph(*args)
        if self.is_compiled:
            self._assert_recurrence_scope_is_honoured()
            self._assert_relations_are_snap_anchored()

    def _assert_recurrence_scope_is_honoured(self) -> None:
        """Raise if a non-diagonal ``recurrence_scope`` cannot be delivered.

        ``_compiler/scan_descent.py`` analyses a descended scan body with
        ``include_recurrent_mixing=False`` unconditionally: the per-substep
        trace fold consumes diagonal Jacobians, so recurrent ETP mixing traced
        into a body transition would have no consumer. That is defensible while
        the flag is private, but ``recurrence_scope`` is a public axis — asking
        for ``'coupled'`` and silently getting ``'diagonal'`` inside the scan is
        exactly the silent-degradation failure the axis decomposition exists to
        remove.

        ``'sparse_n'`` inherits the same limitation, and for a second reason on
        top of the first: a descended group's per-substep Jacobian is folded
        before the trace ever sees it, so there is no single transition jaxpr
        for the position analysis to read. ``_attach_snap_pattern`` therefore
        skips descended groups outright, which would leave the requested order
        silently unapplied.
        """
        scope = self.config.recurrence_scope
        if scope not in ('coupled', 'sparse_n'):
            return
        # A descended *group* can exist without any descended relation (all body
        # weights routed through plain ops), so check both — as
        # ``ETraceAlgorithm.compile_graph`` does for its own scan-descent gate.
        relations = sum(
            r.control_flow_context is not None
            for r in self.graph.hidden_param_op_relations
        )
        groups = sum(g.descent is not None for g in self.graph.hidden_groups)
        if relations or groups:
            raise NotImplementedError(
                f"recurrence_scope={scope!r} is not honoured inside a "
                f'descended scan ({relations} ETP relation(s) and {groups} '
                f'hidden group(s) were discovered inside a scan body, whose '
                f'transition is always analysed with diagonal recurrence). Use '
                f"recurrence_scope='diagonal', or keep the recurrent weights "
                f"outside the scan body, or set "
                f"ControlFlowPolicy(scan_descent='off')."
            )

    def _assert_relations_are_snap_anchored(self) -> None:
        """Raise if ``sparse_n`` meets a primitive with no well-defined anchor.

        SnAp-n widens each trace slot into ``(neighbour, state)`` pairs relative
        to *the one hidden position the slot's instantaneous term lands on*. A
        primitive that spreads one weight entry across several hidden positions
        — ``etp_einsum`` with a shared axis, ``etp_embedding``'s gathered row —
        has no such position, so the widened representation is not merely
        approximate for it, it is undefined.

        Whether a primitive has an anchor is a property of the primitive, so it
        is declared in the operator protocol (``register_etp_rules(
        snap_anchor=...)``) and *default-deny*: a newly added primitive is
        rejected here until it says otherwise, rather than silently producing a
        gradient nobody derived.

        The check also fires for ``SnAp(n=1)``, whose coordinate canonicalises
        to ``'coupled'``. That is deliberate. ``coupled`` itself needs no
        anchor — it takes the per-position block diagonal of the full Jacobian
        and never asks where an instantaneous term lands — so plain
        ``recurrence_scope='coupled'`` stays legal on every model. But SnAp-1 is
        *defined* as the instantaneous nonzero pattern of ``∂h/∂θ``, and on a
        primitive that spreads one weight entry across positions that pattern is
        not a single position: ``coupled`` then drops cross-position
        instantaneous terms which true SnAp-1 retains. Accepting the request
        silently would hand back a rule the caller did not ask for, so the
        preset carries its provenance (``_requested_snap_order``) and is checked
        on it rather than on the canonicalised scope.
        """
        requested = getattr(self, '_requested_snap_order', None)
        if self.config.recurrence_scope != 'sparse_n' and requested is None:
            return
        order = self.config.sparse_n if requested is None else requested
        offenders = sorted({
            relation.primitive.name
            for relation in self.graph.hidden_param_op_relations
            if not is_snap_anchored(relation.primitive, relation.eqn_params)
        })
        if offenders:
            raise NotImplementedError(
                f"recurrence_scope='sparse_n' (sparse_n="
                f'{order}) requires every eligibility-trace '
                f'relation to anchor on a single hidden position, but '
                f'{", ".join(offenders)} does not declare a SnAp anchor: one '
                f'weight entry of it reaches several hidden positions, so a '
                f'widened trace slot has no position to be a neighbourhood of. '
                f"Use recurrence_scope='coupled' or 'diagonal' for this model."
            )

    # ------------------------------------------------------------------ #
    # axis: temporal_recursion
    # ------------------------------------------------------------------ #

    @property
    def _substitutes_hidden_jacobians(self) -> bool:
        """Whether :attr:`config` asks for anything other than the raw ``D``."""
        recursion = (
            self.config.recursion_f if self.config.is_factorized
            else self.config.temporal_recursion
        )
        return recursion != 'jacobian'

    def _substitute_hidden_jacobians(
        self,
        jacobians: Sequence[jax.Array],
    ) -> Sequence[jax.Array]:
        """Realise ``temporal_recursion`` by replacing the transition operator.

        The whole axis reduces to *which array the trace roll multiplies by*, so
        it is implemented once here rather than per primitive: the fast path,
        the legacy nested-vmap path, the chunk-factorised roll
        (``suffix_products`` of ``lam * I`` is ``lam**k * I``), the
        descended-scan substep fold and both engines all consume the substituted
        array without knowing it was substituted.

        Parameters
        ----------
        jacobians : sequence of jax.Array
            The executor's per-hidden-group hidden-to-hidden Jacobians. Reached
            two ways: as the executor's return value on the non-fused path, and
            as the stepper's per-step argument on the fused one (see
            :meth:`_wrap_etrace_stepper`).

        Returns
        -------
        sequence of jax.Array
            The substituted Jacobians, or the input unchanged under
            ``temporal_recursion='jacobian'``.

        Notes
        -----
        The substitution differs per engine, and the difference is load-bearing.
        Under ``per_param`` the roll is ``eps <- D @ eps + instant``, so a leak
        must be carried by the substituted array itself (``lam * I``). Under
        ``io_factorized`` the f-side applies ``D`` first and *then* smooths by
        ``alpha_f``, so substituting ``lam * I`` there would yield
        ``alpha_f * lam`` — an accidental ``alpha_f ** 2``. The f-side leak
        already lives in ``alpha_f``, so its substitution is the bare identity.

        Replacements are built from ``D.shape``, never from a fixed rank: the
        leading axes vary by path — ``(*varshape, S, S)`` single-step,
        ``(T, *varshape, S, S)`` stacked multi-step, and ``(L, *varshape, S, S)``
        for descended-scan groups.
        """
        if not self._substitutes_hidden_jacobians:
            return jacobians
        recursion = (
            self.config.recursion_f if self.config.is_factorized
            else self.config.temporal_recursion
        )
        if recursion == 'none':
            return [jnp.zeros_like(d) for d in jacobians]
        # 'scalar_leak'
        coefficient = 1.0 if self.config.is_factorized else self.config.decay
        return [
            jnp.broadcast_to(
                jnp.asarray(coefficient, dtype=d.dtype)
                * jnp.eye(d.shape[-1], dtype=d.dtype),
                jnp.shape(d),
            )
            for d in jacobians
        ]

    def _wrap_etrace_stepper(self, weight_vals: WeightVals) -> Optional[Callable]:
        """Build the fused stepper with the Jacobian substitution installed.

        The trace roll fuses into the executor's over-time scan when the input
        is multi-step, in which case :meth:`_update_etrace_data` is never called
        and the substitution has to happen inside the stepper instead. This
        wrapper is the *only* place that happens; the engines' internal calls to
        their own steppers stay unwrapped, so every path substitutes exactly
        once. Do not wrap :meth:`_make_etrace_stepper` itself.
        """
        stepper = self._make_etrace_stepper(weight_vals)
        if stepper is None or not self._substitutes_hidden_jacobians:
            return stepper

        def substituted_stepper(carry: Any, step_inputs: Any) -> Any:
            xs, dfs, hid2hid_jac = step_inputs
            return stepper(
                carry, (xs, dfs, self._substitute_hidden_jacobians(hid2hid_jac))
            )

        return substituted_stepper

    def update(self, *args: Any, modulator: Any = None) -> Any:
        r"""
        Update the model states and the eligibility trace.

        The input arguments ``args`` here support very complex data structures, including
        the combination of :py:class:`SingleStepData` and :py:class:`MultiStepData`.

        - :py:class:`SingleStepData`: indicating the data at the single time step,
          :math:`x_t`.
        - :py:class:`MultiStepData`: indicating the data at multiple time steps,
          :math:`[x_{t-k}, ..., x_t]`.

        Parameters
        ----------
        *args
            The input arguments.
        modulator : array_like or Quantity, optional
            The per-call modulatory signal for ``learning_signal='modulatory'``,
            taking precedence over the ``modulator`` attribute for this call
            only. It is **not** forwarded to the model's forward call; it reaches
            the rule through ``_get_update_aux``. Ignored on every other
            ``learning_signal``.

        Returns
        -------
        Any
            The model output.

        Notes
        -----
        Suppose all inputs have the shape of ``(10,)``.

        If the input arguments are given by:

        .. code-block:: python

            x = [jnp.ones((10,)), jnp.zeros((10,))]

        Then, two input arguments are considered as the :py:class:`SingleStepData`.

        If the input arguments are given by:

        .. code-block:: python

            x = [braintrace.SingleStepData(jnp.ones((10,))),
                 braintrace.SingleStepData(jnp.zeros((10,)))]

        This is the same as the previous case, they are all considered as the input at the current time step.

        If the input arguments are given by:

        .. code-block:: python

            x = [braintrace.MultiStepData(jnp.ones((5, 10)),
                 jnp.zeros((10,)))]

        or,

        .. code-block:: python

            x = [braintrace.MultiStepData(jnp.ones((5, 10)),
                 braintrace.SingleStepData(jnp.zeros((10,)))]

        Then, the first input argument is considered as the :py:class:`MultiStepData`, and its data will
        be fed into the model within five consecutive steps, and the second input argument will be fed
        into the model at each time of this five consecutive steps.
        """
        # The per-call modulator is stashed rather than threaded, because
        # `update()` is the public entry point of a long chain that ends in
        # `_get_update_aux`. Cleared in the `finally` so that an exception raised
        # anywhere below -- a shape error in the model, a failed expansion --
        # cannot leave a stale signal to be silently reused by the next call.
        self._modulator_this_call = modulator
        try:
            return self._update_impl(*args)
        finally:
            self._modulator_this_call = None

    def _update_impl(self, *args: Any) -> Any:
        """The body of :meth:`update`. See there for the semantics."""

        # ----------------------------------------------------------------------------------------------
        #
        # This method is the main function to
        #
        # - update the model
        # - update the eligibility trace states
        # - compute the weight gradients
        #
        # The key here is that we change the object-oriented attributes as the function arguments.
        # Therefore, the function arguments are the states of the current time step, and the function
        # returns the states of the next time step.
        #
        # Particularly, the model calls the "_true_update_fun()" function to update the states.
        #
        # ----------------------------------------------------------------------------------------------

        #
        # This function need to process the following multiple cases:
        #
        # 1. if vjp_method = 'single-step', input = SingleStepData, then output is single step
        #
        # 2. if vjp_method = 'single-step', input = MultiStepData, then output is multiple step data
        #
        # 3. if vjp_method = 'multi-step', input = SingleStepData, then output is single step
        #
        # 4. if vjp_method = 'multi-step', input = MultiStepData, then output is multiple step data
        #

        # check the compilation
        self._assert_compiled()

        # state values
        weight_vals = {
            key: st.value
            for key, st in self.param_states.items()
        }
        hidden_vals = {
            key: st.value
            for key, st in self.hidden_states.items()
        }
        other_vals = {
            key: st.value
            for key, st in self.other_states.items()
        }
        # etrace data
        last_etrace_vals = self._get_etrace_data()

        # Optional per-call auxiliary data (e.g. a neuromodulatory signal) that
        # a subclass needs inside `_compute_learning_signal` but that must not be
        # forwarded to the model itself. Read synchronously here -- before the
        # custom_vjp machinery runs -- so it becomes a genuine argument of
        # `_true_update_fun` rather than an instance-attribute side channel:
        # outer transforms (e.g. `brainstate.transform.grad`) may stage the
        # forward trace and only invoke the fwd/bwd rules after this `update()`
        # call has already returned, by which point any such stash would
        # already be gone.
        aux = self._get_update_aux()

        # update all states
        #
        # [KEY] The key here is that we change the object-oriented attributes as the function arguments.
        #       Therefore, the function arguments are the states of the current time step, and the function
        #       returns the states of the next time step.
        #
        # out: is always multiple step
        (
            out,
            hidden_vals,
            other_vals,
            new_etrace_vals
        ) = self._true_update_fun(
            args,
            weight_vals,
            hidden_vals,
            other_vals,
            last_etrace_vals,
            self.running_index.value,
            aux,
        )

        # assign/restore the weight values back
        #
        # [KEY] assuming the weight values are not changed
        #       This is a key assumption in the RTRL algorithm.
        #       This is very important for the implementation.
        assign_state_values_v2(self.param_states, weight_vals, write=False)

        # assign the new hidden and state values
        assign_state_values_v2(self.hidden_states, hidden_vals)
        assign_state_values_v2(self.other_states, other_vals)

        #
        # assign the new etrace values
        #
        # "self._assign_etrace_data()" is a protocol method that should be implemented in the subclass.
        # It's logic may be different for different etrace algorithms.
        #
        self._assign_etrace_data(new_etrace_vals)  # call the protocol method

        # update the running index
        running_index = self.running_index.value + 1
        self.running_index.value = jax.lax.stop_gradient(jnp.where(running_index >= 0, running_index, 0))

        # return the model output
        return out

    def _update_fn(
        self,
        args: Any,
        weight_vals: WeightVals,
        hidden_vals: HiddenVals,
        oth_state_vals: StateVals,
        etrace_vals: ETraceVals,
        running_index: Any,
        aux: Any = None,
    ) -> Tuple[Outputs, HiddenVals, StateVals, ETraceVals]:
        """
        The main function to update the [model] and the [eligibility trace] states.

        Particularly, ``self.graph.solve_h2w_h2h_jacobian()`` is called to:
          - compute the model output, the hidden states, and the other states
          - compute the hidden-to-weight Jacobian and the hidden-to-hidden Jacobian

        Then, ``self._update_etrace_data`` is called to:
          - update the eligibility trace data

        Moreover, this function returns:
          - the model output
          - the updated hidden states
          - the updated other states
          - the updated eligibility trace states

        Note that the weight values are assumed not changed in this function.

        The ``aux`` argument (see :meth:`_get_update_aux`) is unused on this
        plain (non-differentiating) path; it only matters on the
        ``_update_fn_fwd``/``_update_fn_bwd`` path, where it is threaded to
        ``_compute_learning_signal``.
        """
        input_is_multi_step = has_multistep_data(*args)

        # state value assignment
        assign_state_values_v2(self.param_states, weight_vals, write=False)
        assign_state_values_v2(self.hidden_states, hidden_vals, write=False)
        assign_state_values_v2(self.other_states, oth_state_vals, write=False)

        # When the trace roll can be fused into the executor's over-time scan
        # (multi-step input + a fusable subclass), hand the per-step stepper down
        # so the executor updates the trace in-loop and returns the final trace,
        # avoiding a second scan over stacked Jacobians.
        etrace_stepper = self._wrap_etrace_stepper(weight_vals) if input_is_multi_step else None

        # necessary jacobian information of the weights
        (
            out,
            hidden_vals,
            oth_state_vals,
            hid2weight_jac_single_or_multi_steps,
            hid2hid_jac_single_or_multi_steps,
            final_etrace,
        ) = self.graph_executor.solve_h2w_h2h_jacobian(
            *args,
            etrace_stepper=etrace_stepper,
            init_etrace=etrace_vals if etrace_stepper is not None else None,
        )

        if final_etrace is not None:
            # fused path: the executor already rolled the eligibility trace in-loop.
            etrace_vals = final_etrace
        else:
            # eligibility trace update
            #
            # "self._update_etrace_data()" is a protocol method that should be implemented in the subclass.
            # It's logic may be different for different etrace algorithms.
            #
            etrace_vals = self._update_etrace_data(
                running_index,
                etrace_vals,
                hid2weight_jac_single_or_multi_steps,
                # `temporal_recursion`: the non-fused half of the substitution.
                self._substitute_hidden_jacobians(hid2hid_jac_single_or_multi_steps),
                weight_vals,
                input_is_multi_step,
            )

        # returns
        return out, hidden_vals, oth_state_vals, etrace_vals

    def _update_fn_fwd(
        self,
        args: Any,
        weight_vals: WeightVals,
        hidden_vals: HiddenVals,
        othstate_vals: StateVals,
        etrace_vals: ETraceVals,
        running_index: int,
        aux: Any = None,
    ) -> Tuple[Tuple[Outputs, HiddenVals, StateVals, ETraceVals], Any]:
        """
        The forward function to update the [model] and the [eligibility trace] states when computing
        the VJP gradients.

        Particularly, ``self.graph.solve_h2w_h2h_jacobian_and_l2h_vjp()`` is called to:

        - compute the model output, the hidden states, and the other states
        - compute the hidden-to-weight Jacobian and the hidden-to-hidden Jacobian
        - compute the loss-to-hidden or loss-to-weight Jacobian

        Then, ``self._update_etrace_data`` is called to:

        - update the eligibility trace data

        The forward function returns two parts of data:

        - The first part is the functional returns (same as "self._update()" function):
              * the model output
              * the updated hidden states
              * the updated other states
              * the updated eligibility trace states

        - The second part is the data used for backward gradient computation:
              * the residuals of the model
              * the eligibility trace data at the current/last time step
              * the weight id to its value mapping
              * the running index
        """
        input_is_multi_step = has_multistep_data(*args)

        # state value assignment
        assign_state_values_v2(self.param_states, weight_vals, write=False)
        assign_state_values_v2(self.hidden_states, hidden_vals, write=False)
        assign_state_values_v2(self.other_states, othstate_vals, write=False)

        # As in ``_update_fn``: when fusable + multi-step, push the stepper down so
        # the executor rolls the trace inside the same scan that builds the VJP
        # residual. The trace carry is detached (stop_gradient), so it never enters
        # the residual jaxpr.
        etrace_stepper = self._wrap_etrace_stepper(weight_vals) if input_is_multi_step else None

        # necessary gradients of the weights
        (
            out,
            hiddens,
            oth_states,
            hid2weight_jac_single_or_multi_steps,
            hid2hid_jac_single_or_multi_steps,
            residuals,
            final_etrace,
        ) = self.graph_executor.solve_h2w_h2h_l2h_jacobian(
            *args,
            etrace_stepper=etrace_stepper,
            init_etrace=etrace_vals if etrace_stepper is not None else None,
        )

        if final_etrace is not None:
            # fused path: the executor already rolled the eligibility trace in-loop.
            new_etrace_vals = final_etrace
        else:
            # eligibility trace update
            #
            # "self._update_etrace_data()" is a protocol method that should be implemented in the subclass.
            # It's logic may be different for different etrace algorithms.
            #
            new_etrace_vals = self._update_etrace_data(
                running_index,
                etrace_vals,
                hid2weight_jac_single_or_multi_steps,
                # `temporal_recursion`: the non-fused half of the substitution.
                self._substitute_hidden_jacobians(hid2hid_jac_single_or_multi_steps),
                weight_vals,
                input_is_multi_step
            )

        # returns
        old_etrace_vals = etrace_vals
        fwd_out = (out, hiddens, oth_states, new_etrace_vals)
        fwd_res = (
            residuals,
            (
                old_etrace_vals
                if self.graph_executor.is_multi_step_vjp else
                new_etrace_vals
            ),
            weight_vals,
            running_index,
            args,  # threaded to _update_fn_bwd for the learning-signal hook
            aux,  # per-call auxiliary data (see _get_update_aux), also threaded to the hook
            # `learning_signal='bootstrapped'`: the synthetic cotangent for the
            # window-exit hidden values. Computed here because this is the only
            # place `h^exit` exists, and stashed in the residuals rather than
            # recomputed in the backward pass.
            #
            # Gated on the axis, not merely on the hook returning None. A subclass
            # that overrides the hook would otherwise inject under *any*
            # `learning_signal`, including `symmetric` -- so the axis would
            # describe the rule without controlling it.
            (self._inject_exit_cotangent(hiddens, aux)
             if self.config.learning_signal == 'bootstrapped' else None),
        )
        return fwd_out, fwd_res

    def _update_fn_bwd(
        self,
        fwd_res: Any,
        grads: Any,
    ) -> Tuple[dG_Inputs, dG_Weight, dG_Hidden, dG_State, None, None, None]:
        """
        The backward function to compute the VJP gradients when the learning signal is arrived at
        this time step.

        There are three steps:

        1. Interpret the forward results (eligibility trace) and top-down gradients (learning signal)
        2. Compute the gradients of input arguments
           (maybe necessary, but it can be optimized away but the XLA compiler)
        3. Compute the gradients of the weights

        """

        # [1] Interpret the fwd results
        #
        (
            residuals,  # the residuals of the VJP computation, for computing the gradients of input arguments
            etrace_vals_at_t_or_t_minus_1,  # the eligibility trace data at the current or last time step
            weight_vals,  # the weight id to its value mapping
            running_index,  # the running index
            args,  # original update(*args) tuple, used by _compute_learning_signal
            aux,  # per-call auxiliary data from _get_update_aux, also used by _compute_learning_signal
            exit_cotangent,  # `bootstrapped`: synthetic cotangent at h^exit, or None
        ) = fwd_res

        (
            jaxpr,
            in_tree,
            out_tree,
            consts
        ) = residuals

        # [2] Interpret the top-down gradient signals
        #
        # Since
        #
        #     dg_out, dg_hiddens, dg_others, dg_etrace = grads
        #
        # we need to remove the "dg_etrace" iterm from the gradients for matching
        # the jaxpr vjp gradients.
        #
        grad_flat, grad_tree = jax.tree.flatten((grads[:-1],))

        # [3] Compute the gradients of the input arguments
        #     It may be unnecessary, but it can be optimized away by the XLA compiler after it is computed.
        #
        # The input argument gradients are computed through the normal back-propagation algorithm.
        #
        if out_tree != grad_tree:
            raise TypeError(
                f'Gradient tree should be the same as the function output tree. '
                f'While we got: \n'
                f'out_tree  = {out_tree}\n!=\n'
                f'grad_tree = {grad_tree}'
            )
        cts_out = jax.core.eval_jaxpr(jaxpr, consts, *grad_flat)

        #
        # We compute:
        #
        #   - the gradients of input arguments,
        #     maybe necessary to propagate the gradients to the last layer
        #
        #   - the gradients of the hidden states at the last time step,
        #     maybe unnecessary but can be optimized away by the XLA compiler
        #
        #   - the gradients of the non-etrace parameters, defined by "NonTempParam"
        #
        #   - the gradients of the other states
        #
        #   - the gradients of the loss-to-hidden at the current time step
        #

        # the `_jaxpr_compute_model_with_vjp()` in `ETraceGraphExecutor`
        (
            dg_args,
            dg_last_hiddens,
            dg_non_etrace_params,
            dg_etrace_params,
            dg_oth_states,
            dg_hid_perturb_or_dl2h
        ) = jax.tree.unflatten(in_tree, cts_out)

        #
        # get the gradients of the hidden states at the last time step
        #
        #
        # Every guard below is `if ... raise`, never `assert`: `python -O` strips
        # `assert` statements, and a mis-routed cotangent here produces a *wrong
        # gradient rather than an error*, so the check has to survive an
        # optimised interpreter. All of them read Python-level metadata only
        # (lengths, dict keys, static shapes, dtypes), so none of it reaches the
        # compiled program.
        #
        if self.graph_executor.is_single_step_vjp:
            if len(dg_etrace_params) != 0:
                # Under `vjp_method='single-step'` the ETP weights are updated by
                # the RTRL recursion, so the transposed jaxpr must not also hand
                # back gradients for them.
                raise ValueError(
                    f'Under vjp_method=\'single-step\' the ETP weight gradients '
                    f'come from the eligibility-trace recursion, so the backward '
                    f'pass must not return any. It returned '
                    f'{len(dg_etrace_params)} for '
                    f'{sorted(map(str, dg_etrace_params))}. The compiled graph '
                    f'and the graph executor disagree; recompile the graph.'
                )
            hidden_perturb = self.graph.hidden_perturb
            if hidden_perturb is None:
                raise ValueError(
                    'vjp_method=\'single-step\' reads the hidden-state gradients '
                    'off the hidden perturbation variables, but the compiled '
                    'graph carries no hidden perturbation. Call `compile_graph()` '
                    'on this algorithm before running the backward pass.'
                )
            if len(hidden_perturb.perturb_vars) != len(dg_hid_perturb_or_dl2h):
                raise ValueError(
                    f'The backward pass returned {len(dg_hid_perturb_or_dl2h)} '
                    f'hidden-perturbation gradient(s) for '
                    f'{len(hidden_perturb.perturb_vars)} perturbation variable(s) '
                    f'{list(hidden_perturb.perturb_hidden_paths)}. The two are '
                    f'matched by position, so a length disagreement re-attributes '
                    f'every gradient past the mismatch.'
                )
            _check_hidden_gradient_correspondence(
                self.graph.hidden_groups,
                dict(zip(hidden_perturb.perturb_hidden_paths, dg_hid_perturb_or_dl2h)),
                source='single-step hidden perturbation',
            )
            dl2h_at_t_or_t_minus_1 = hidden_perturb.perturb_data_to_hidden_group_data(
                dg_hid_perturb_or_dl2h, self.graph.hidden_groups,
            )

        else:
            if set(dg_last_hiddens.keys()) != set(self.hidden_states.keys()):
                raise ValueError(
                    f'The hidden states should be the same. Bug got \n'
                    f'{set(dg_last_hiddens.keys())}\n'
                    f'!=\n'
                    f'{set(self.hidden_states.keys())}'
                )
            _check_hidden_gradient_correspondence(
                self.graph.hidden_groups,
                dg_last_hiddens,
                source='multi-step last-hidden gradients',
            )
            dl2h_at_t_or_t_minus_1 = [
                group.concat_hidden(
                    [
                        # dimensionless processing
                        u.get_mantissa(dg_last_hiddens[path])
                        for path in group.hidden_paths
                    ]
                )
                for group in self.graph.hidden_groups
            ]

        #
        # Hook: subclasses may replace the reverse-AD learning signal with an
        # alternative (e.g. the κ-filtered signal in EProp).
        #
        # `aux` (see `_get_update_aux`) is appended to `args` here, not inside
        # `args` itself, so it never reaches the model's own forward call --
        # only this hook sees it. Algorithms that don't use it (the default
        # `_get_update_aux` returns None) are unaffected: the base
        # implementation and EProp's override both ignore `args` entirely.
        dl2h_at_t_or_t_minus_1 = self._compute_learning_signal(
            dl2h_at_t_or_t_minus_1, (*args, aux)
        )

        #
        # [3b] `learning_signal='bootstrapped'`: the second linear pass.
        #
        # `eval_jaxpr` on the residual jaxpr is linear in the cotangents, so the
        # synthetic future gradient can be pushed through the *same* transposed
        # jaxpr on its own and the results added -- one extra evaluation per
        # window, and none at all when no synthesiser is active.
        #
        # Which consumers it reaches is the crux, and getting it wrong
        # double-counts silently:
        #
        #   * plain (non-ETP) parameters, inputs, other states -- ADD it. Their
        #     cross-window credit is truncated at the window edge, and adding the
        #     estimate makes the sum over windows telescope to the exact gradient.
        #   * ETP parameters and the boundary learning signal -- do NOT. Their
        #     cross-window credit is *already* carried by the eligibility trace:
        #     an occurrence inside this window that reaches a later window's loss
        #     is counted there, because that window's trace contains it. Adding
        #     the estimate here would count the same path twice. This is what an
        #     eligibility trace is for; DNI's job is to give the *plain*
        #     parameters the cross-window credit the trace already gives the ETP
        #     ones.
        #   * the returned hidden cotangent -- ADD it. It is both the previous
        #     window's incoming cotangent and the synthesiser's own regression
        #     target, and both want the future term.
        #
        # Note the ordering: the learning signal above is derived from pass 1
        # only, which is why this block sits *after* it rather than folding into
        # `dg_last_hiddens` before.
        #
        dg_last_hiddens_future = None
        if exit_cotangent is not None:
            injected = self._exit_cotangent_grads(grads[:-1], exit_cotangent)
            flat_injected, injected_tree = jax.tree.flatten((injected,))
            # `PyTreeDef` compares by value at runtime; mypy's jax stubs do not
            # model its `__ne__`.
            if injected_tree != grad_tree:  # type: ignore[operator]
                raise TypeError(
                    f'The injected exit cotangent must have the same tree '
                    f'structure as the incoming gradients, so it can go through '
                    f'the same transposed jaxpr. Got\n'
                    f'{injected_tree}\n!=\n{grad_tree}'
                )
            cts_future = jax.core.eval_jaxpr(jaxpr, consts, *flat_injected)
            (
                dg_args_future,
                dg_last_hiddens_future,
                dg_non_etrace_params_future,
                dg_etrace_params_future,    # plain paths only: see below
                dg_oth_states_future,
                _dg_hid_perturb_future,     # ditto; the signal is pass 1 only
            ) = jax.tree.unflatten(in_tree, cts_future)

            def _add_leaf(x: Any, y: Any) -> Any:
                # An integer or boolean input has no derivative, and JAX gives it
                # a `float0` cotangent -- a zero-sized dtype that `+` is not
                # defined for. Both passes produce the same `float0` placeholder,
                # so keeping either is correct; adding them raises.
                if jax.dtypes.result_type(x) == jax.dtypes.float0:
                    return x
                return x + y

            def _add(a: Any, b: Any) -> Any:
                return jax.tree.map(_add_leaf, a, b,
                                    is_leaf=u.math.is_quantity)

            dg_args = _add(dg_args, dg_args_future)
            dg_oth_states = _add(dg_oth_states, dg_oth_states_future)
            # The ETP/plain split cannot be read off *which dict* a parameter
            # arrives in: under multi-step the reverse pass delivers the
            # within-window gradient of every trainable parameter -- plain ones
            # included -- through `dg_etrace_params`, leaving
            # `dg_non_etrace_params` empty. The authority on which parameters
            # have a trace carrying their cross-window credit is the compiled
            # graph, so ask it by path.
            etp_paths = self._etp_routed_paths()
            dg_non_etrace_params = _add_future_for_plain_paths(
                dg_non_etrace_params, dg_non_etrace_params_future, etp_paths)
            # Added *before* `_solve_weight_gradients`, which folds these into
            # its result.
            dg_etrace_params = _add_future_for_plain_paths(
                dg_etrace_params, dg_etrace_params_future, etp_paths)

        #
        # [4] Compute the gradients of the weights
        #
        # the gradients of the weights are computed through the RTRL algorithm.
        #
        # "self._solve_weight_gradients()" is a protocol method that should be implemented in the subclass.
        # It's logic may be different for different etrace algorithms.
        #
        dg_weights = self._solve_weight_gradients(
            running_index,
            etrace_vals_at_t_or_t_minus_1,
            dl2h_at_t_or_t_minus_1,
            weight_vals,
            dg_non_etrace_params,
            dg_etrace_params,
        )

        # The returned hidden cotangent carries the future term too: it is what
        # the *previous* window will receive, and what a synthesiser regresses on.
        if dg_last_hiddens_future is not None:
            dg_last_hiddens = jax.tree.map(
                lambda x, y: x + y, dg_last_hiddens, dg_last_hiddens_future,
                is_leaf=u.math.is_quantity)

        # Note that there are no gradients flowing through the etrace data, the
        # running index, or the auxiliary data.
        dg_etrace = None
        dg_running_index = None
        dg_aux = None

        return (
            dg_args,
            dg_weights,
            dg_last_hiddens,
            dg_oth_states,
            dg_etrace,
            dg_running_index,
            dg_aux,
        )

    def _etp_routed_paths(self) -> set:
        """The parameter paths whose cross-window credit an eligibility trace carries.

        Read off the compiled graph rather than inferred from which gradient
        dictionary a parameter turns up in, because that grouping does not mean
        what its name suggests: under multi-step, the within-window reverse pass
        delivers the gradient of *every* trainable parameter through
        ``dg_etrace_params``, plain ones included.

        Returns
        -------
        set
            Paths appearing in some ``hidden_param_op_relation``.
        """
        paths: set = set()
        for relation in self.graph.hidden_param_op_relations:
            paths.update(relation.trainable_paths.values())
        return paths

    def _inject_exit_cotangent(self, exit_hiddens: Dict[Path, Any], aux: Any) -> Any:
        """Override hook. The synthetic cotangent to add at the window-exit hidden values.

        Implements the *injection* half of ``learning_signal='bootstrapped'``.
        Called inside `_update_fn_fwd`, which is the only place the window-exit
        hidden values exist; the result is stashed in the residuals and consumed
        by the second linear pass in `_update_fn_bwd`.

        This is deliberately **not** part of `_compute_learning_signal`. Replacing
        a boundary learning signal and adding an exit cotangent are different
        operations, on different tensors, at different points of the window:
        the boundary signal is per hidden *group* and is consumed only by the
        trace contraction, while the exit cotangent is per hidden *path* and has
        to be propagated by the window's own reverse pass to reach the plain
        parameters at all.

        Implementations must apply `jax.lax.stop_gradient` to their estimate, or
        the online loss will train the synthesiser through the wrong path.

        Parameters
        ----------
        exit_hiddens : Any
            Hidden-path-keyed window-exit values, `h^exit`.
        aux : Any
            Per-call auxiliary data from `_get_update_aux`.

        Returns
        -------
        dict or None
            A hidden-path-keyed mapping of cotangents shaped like the
            corresponding entry of `exit_hiddens`, or `None` for "no injection"
            (the default, and the zero-cost path: no second pass runs).
        """
        return None

    def _exit_cotangent_grads(self, grads_wo_etrace: tuple, exit_cotangent: Any) -> tuple:
        """Build the pass-2 cotangent tuple: zero everywhere but the exit hiddens.

        Parameters
        ----------
        grads_wo_etrace : tuple
            The incoming `(dg_out, dg_hiddens, dg_oth_states)` triple, used
            purely as a shape/unit/dtype template so the result flattens to the
            same tree the transposed jaxpr expects.
        exit_cotangent : Any
            The hidden-path-keyed synthetic cotangents.

        Returns
        -------
        tuple
            A triple with the same structure as `grads_wo_etrace`.

        Raises
        ------
        ValueError
            If a synthetic cotangent's shape does not match the hidden cotangent
            it replaces, or if it is keyed by a path that is not a hidden state.
        """
        dg_out, dg_hiddens, dg_oth = grads_wo_etrace

        unknown = set(exit_cotangent).difference(dg_hiddens)
        if unknown:
            raise ValueError(
                f'The synthetic gradient is keyed by {sorted(unknown)}, which '
                f'are not hidden states of this model. Expected a subset of '
                f'{sorted(dg_hiddens)}.'
            )

        injected_hiddens = {}
        for path, template in dg_hiddens.items():
            synthetic = exit_cotangent.get(path)
            if synthetic is None:
                injected_hiddens[path] = u.math.zeros_like(template)
                continue
            want, got = u.math.shape(template), u.math.shape(synthetic)
            if want != got:
                raise ValueError(
                    f'The synthetic gradient for hidden path {path} has shape '
                    f'{got}, but the hidden cotangent it is added to has shape '
                    f'{want}. The synthesiser must emit one cotangent per hidden '
                    f'state, shaped like that state.'
                )
            unit = u.get_unit(template)
            if isinstance(synthetic, u.Quantity) and not unit.is_unitless:
                # *Convert*, do not relabel. Taking the mantissa of a `1 V`
                # estimate and reattaching the template's `mV` would inject
                # `1 mV` -- a thousandfold error that no shape or dtype check
                # sees. `in_unit` also raises on incompatible dimensions, which a
                # relabel would have silently accepted.
                try:
                    synthetic = synthetic.in_unit(unit)
                except Exception as e:
                    raise ValueError(
                        f'The synthetic gradient for hidden path {path} has unit '
                        f'{u.get_unit(synthetic)}, which cannot be converted to '
                        f'{unit}, the unit of the hidden cotangent it is added '
                        f'to.'
                    ) from e
            mantissa = jnp.asarray(u.get_mantissa(synthetic),
                                   dtype=u.get_mantissa(template).dtype)
            injected_hiddens[path] = (
                mantissa if unit.is_unitless else mantissa * unit)

        return (
            jax.tree.map(u.math.zeros_like, dg_out),
            injected_hiddens,
            jax.tree.map(u.math.zeros_like, dg_oth),
        )

    def _get_update_aux(self) -> Any:
        """Override hook. Return per-call auxiliary data threaded to `_compute_learning_signal`.

        Called synchronously at the start of `update()`, before the custom_vjp
        forward/backward machinery runs, and passed to `_true_update_fun` as a
        genuine argument (see `_update_fn`/`_update_fn_fwd`/`_update_fn_bwd`).
        This makes the value a real data dependency of the traced computation.

        This is deliberately *not* read lazily from an instance attribute
        inside `_update_fn_fwd`/`_update_fn_bwd`/`_compute_learning_signal`:
        outer transforms (e.g. `brainstate.transform.grad`) may stage the
        forward trace and only invoke the custom_vjp fwd/bwd rules after the
        `update()` call that set such a stash has already returned, so a
        lazy read would silently observe a cleared/stale value.

        Default returns `None` (unused). Subclasses that need auxiliary data
        not already part of the model's own forward arguments (e.g. a reward or
        neuromodulatory signal, which must not be forwarded to the model call)
        should override this and read whatever they stashed on `self` during
        their own `update()` override, at this exact call site.

        Returns
        -------
        Any
            Any pytree (or `None`). Appended to the `args` tuple seen by
            `_compute_learning_signal` (see below); never forwarded to the
            model's forward call.

        Raises
        ------
        RuntimeError
            If `learning_signal='modulatory'` but no modulator was supplied.
            Falling back to the symmetric signal there would silently compute a
            different learning rule.
        """
        if self.config.learning_signal != 'modulatory':
            return None
        # The keyword wins over the standing attribute, for this call only.
        modulator = (self._modulator_this_call if self._modulator_this_call
                     is not None else self.modulator)
        if modulator is None:
            raise RuntimeError(
                "learning_signal='modulatory' was configured but no modulator "
                'was supplied, so the signal would silently fall back to '
                'symmetric -- a different learning rule. Pass one per call as '
                '`update(*inputs, modulator=m)`, or set the standing attribute '
                '`learner.modulator = m`.'
            )
        # Validated here, synchronously, against every group's declared signal
        # shape. The expansion that actually feeds the rule happens in
        # `_compute_learning_signal`, which runs inside the custom_vjp *backward*
        # pass -- so without this pre-flight a malformed modulator would pass a
        # forward-only `update()` silently and only fail once the caller got
        # around to differentiating, with a traceback pointing into JAX internals
        # rather than at the offending call.
        # Descended groups included. This loop used to skip them, on the belief
        # that a descended group's signal carries a leading substep axis and so
        # is not `(*varshape, num_state)`. Measured, it is: `scan_descent` folds
        # the per-substep Jacobians inside the body, and `_compute_learning_signal`
        # sees one array per *group*, of exactly the group shape, whether or not
        # the group descended. The skip therefore bought nothing and cost the
        # pre-flight: on a descended model a malformed modulator was accepted by
        # a forward-only `update()` and only failed once the caller differentiated.
        for group in self.graph.hidden_groups:
            expand_modulator_to_group(
                modulator,
                tuple(group.varshape) + (group.num_state,),
                group_index=group.index,
                hidden_paths=group.hidden_paths,
            )
        return modulator

    # ------------------------------------------------------------------ #
    # axis: learning_signal
    # ------------------------------------------------------------------ #

    def init_etrace_state(self, *args: Any, **kwargs: Any) -> None:
        """Allocate the axis-side state.

        Concrete here, unlike :meth:`ETraceAlgorithm.init_etrace_state`, which
        raises: the lifted axes own state of their own, and every engine must
        reach it. Engines override this to build their traces and then call
        ``super().init_etrace_state(...)`` **as the last statement**.

        Parameters
        ----------
        *args, **kwargs
            The example inputs, forwarded from :meth:`compile_graph`. Unused
            here; engines size their traces from the compiled graph.
        """
        self._random_feedback = {}
        if self.config.learning_signal != 'random_feedback':
            return
        # Collect the (group index -> width) pairs first so every draw happens
        # inside one seeded scope, using `brainstate.random` throughout.
        widths: Dict[int, int] = {}
        for relation in self.graph.hidden_param_op_relations:
            for group in relation.hidden_groups:
                widths.setdefault(group.index, int(group.varshape[-1]))
        with brainstate.random.seed_context(self._random_feedback_key):
            for group_index, width in widths.items():
                # n_target == n_layer: a square projection over the reverse-AD
                # signal. See F-28 for why the readout width is not visible here.
                self._random_feedback[group_index] = FixedRandomFeedback(
                    n_target=width,
                    n_layer=width,
                    key=brainstate.random.split_key(),
                    init_scale=0.1,
                )

    def _compute_learning_signal(
        self,
        dl_to_hidden_from_autodiff: Sequence[jax.Array],
        args: tuple,
    ) -> Sequence[jax.Array]:
        """Return the learning signal used by `_solve_weight_gradients`.

        Implements the ``learning_signal`` axis. ``'symmetric'`` returns the
        reverse-AD gradient unchanged; ``'random_feedback'`` projects it through
        a frozen random matrix (feedback alignment); ``'modulatory'`` *replaces*
        it with the user-supplied signal from :meth:`_get_update_aux`, expanded
        per group by :func:`expand_modulator_to_group`. Because the hook only sees
        per-hidden-group signals, which carry no trace-factorization structure,
        this works identically on both engines — the IO-dim engine gains random
        feedback from the lift for free.

        ``'bootstrapped'`` is *not* handled here: adding a synthetic cotangent at
        the window exit is a different operation, on a different tensor, at a
        different time than replacing a boundary signal. See
        :meth:`_inject_exit_cotangent`.

        Parameters
        ----------
        dl_to_hidden_from_autodiff : Sequence[jax.Array]
            Sequence of per-hidden-group gradients produced by reverse-AD inside
            `_update_fn_bwd`.
        args : tuple
            The `*args` tuple passed to the most recent `update()` call, with the
            per-call auxiliary data from `_get_update_aux` appended as the
            trailing element. Subclasses that override `_get_update_aux` can pull
            their auxiliary tensor from there; subclasses that don't (the default
            `None`) can ignore `args` entirely, as this implementation does.

        Returns
        -------
        Sequence[jax.Array]
            Sequence of per-hidden-group gradient arrays, one per HiddenGroup. Must
            match the shape and length of ``dl_to_hidden_from_autodiff``.

        Raises
        ------
        RuntimeError
            If ``learning_signal='random_feedback'`` but no projection was
            allocated. Returning the symmetric signal there would quietly compute
            a *different rule* than the one configured.
        """
        # `bootstrapped` leaves the boundary signal exactly alone -- its whole
        # effect is the extra cotangent injected at the window *exit*, which
        # reaches the plain parameters through the window's own reverse pass. It
        # must not fall through to the projection branch below.
        if self.config.learning_signal in ('symmetric', 'bootstrapped'):
            return dl_to_hidden_from_autodiff

        if self.config.learning_signal == 'modulatory':
            # *Replace*, not multiply. Multiplying (`m * dL/dh`) would be a
            # four-factor rule and would make the degenerate criterion --
            # "equals `symmetric` element-wise when the modulator is set to
            # `dL/dh`" -- unsatisfiable, so there would be no coordinate at
            # which the axis reduces to the rule it generalises.
            #
            # `dl_to_hidden_from_autodiff[g].shape == (*varshape_g, num_state_g)`,
            # which is the expansion target. It is read off the signal rather than
            # rebuilt from the group so that this stays a single source of truth
            # with whatever the reverse pass actually produced -- descended groups
            # included, whose signals measure the same shape as any other's.
            modulator = args[-1] if args else None
            if modulator is None:
                raise RuntimeError(
                    "learning_signal='modulatory' reached the learning signal "
                    'with no modulator. `_get_update_aux` must supply one; if '
                    'this algorithm overrides that hook, it must return the '
                    'modulator (or defer to `super()`).'
                )
            groups = self.graph.hidden_groups
            return [
                expand_modulator_to_group(
                    modulator,
                    u.math.shape(sig),
                    group_index=gid,
                    hidden_paths=(groups[gid].hidden_paths
                                  if gid < len(groups) else None),
                )
                for gid, sig in enumerate(dl_to_hidden_from_autodiff)
            ]

        signals = list(dl_to_hidden_from_autodiff)
        if not self._random_feedback:
            raise RuntimeError(
                "learning_signal='random_feedback' was configured but no "
                'feedback matrices were allocated, so the signal would '
                'silently fall back to symmetric — a different learning rule. '
                'Call `init_etrace_state()` (or `compile_graph()`, which calls '
                'it) before `update()`.'
            )

        # `dl[g].shape == (*varshape, num_state)`; varshape[-1] is the hidden
        # width, i.e. axis -2 of the full array. `s` is proportional to the
        # *real* readout weights (reverse-AD only ever exposes `W_out.T @ delta`,
        # never `delta` itself), so projecting through any fixed B cannot remove
        # that dependency. L2-normalising per num_state channel over the width
        # axis strips the magnitude dependence on W_out — keeping direction and
        # the per-state axis untouched — before the frozen projection.
        def _project(B: Any, s: Any) -> Any:
            norm = jnp.sqrt(jnp.sum(jnp.square(s), axis=-2, keepdims=True))
            return jnp.einsum('...lj,lk->...kj', s / (norm + 1e-8), B)

        return [
            _project(self._random_feedback[gid].B, s)
            if gid in self._random_feedback else s
            for gid, s in enumerate(signals)
        ]

    def _solve_weight_gradients(
        self,
        running_index: int,
        # The eligibility-trace container and the weight-value mapping are keyed
        # differently per algorithm (e.g. Path- vs WeightID-keyed), so this
        # abstract hook leaves their concrete types implementation-defined.
        etrace_h2w_at_t: Any,
        dl_to_hidden_groups: Sequence[jax.Array],
        weight_vals: Any,
        dl_to_nonetws_at_t: Dict[Path, PyTree],
        dl_to_etws_at_t: Optional[Dict[Path, PyTree]],
    ) -> Any:
        r"""
        The method to solve the weight gradients, i.e., :math:`\partial L / \partial W`.

        .. note::

            This is the protocol method that should be implemented in the subclass.


        Particularly, the weight gradients are computed through::

        .. math::

            \frac{\partial L^t}{\partial W} = \frac{\partial L^t}{\partial h^t} \frac{\partial h^t}{\partial W}

        Or,

        .. math::

            \frac{\partial L^t}{\partial W} = \frac{\partial L^{t-1}}{\partial h^{t-1}}
                                              \frac{\partial h^{t-1}}{\partial W}
                                              + \frac{\partial L^t}{\partial W^t}


        Parameters
        ----------
        running_index : int, optional
            The running index.
        etrace_h2w_at_t : Any
            The eligibility trace data (which track the hidden-to-weight Jacobian)
            that have accumulated util the time ``t``.
        dl_to_hidden_groups : Dict[HiddenOutVar, jax.Array]
            The gradients of the loss-to-hidden at the time ``t``.
        weight_vals : Dict[WeightID, PyTree]
            The weight values.
        dl_to_nonetws_at_t : List[PyTree]
            The gradients of the loss-to-non-etrace parameters at the time ``t``,
            i.e., :math:``\partial L^t / \partial W^t``.
        dl_to_etws_at_t : List[PyTree]
            The gradients of the loss-to-etrace parameters at the time ``t``,
            i.e., :math:``\partial L^t / \partial W^t``.
        """
        raise NotImplementedError

    def _make_etrace_stepper(self, weight_vals: WeightVals) -> Optional[Callable]:
        """Return a per-step eligibility-trace update callback, or ``None``.

        When a subclass can express its trace roll as a pure step function with
        signature ``(etrace_carry, (x_dict, df_dict, diag_list)) -> (new_carry, None)``,
        it should override this to return that callback (typically the same
        ``partial`` it builds inside :meth:`_update_etrace_data`). For multi-step
        input the graph executor then fuses the roll into its over-time scan,
        eliminating the separate trace scan and the stacked per-step Jacobians.

        Returning ``None`` (the default) keeps the legacy two-pass behavior: the
        executor stacks the Jacobians and :meth:`_update_etrace_data` rolls the
        trace in a second scan. Subclasses whose update cannot be written as such a
        step function (or that do not support multi-step input) leave this ``None``.

        Parameters
        ----------
        weight_vals : WeightVals
            The current parameter values, captured by the returned callback.

        Returns
        -------
        Callable or None
            The per-step stepper, or ``None`` to disable scan fusion.
        """
        return None

    def _update_etrace_data(
        self,
        running_index: Optional[int],
        # The eligibility-trace container type is implementation-defined.
        etrace_vals_util_t_1: Any,
        hid2weight_jac_single_or_multi_times: Hid2WeightJacobian,
        hid2hid_jac_single_or_multi_times: Sequence[jax.Array],
        weight_vals: WeightVals,
        input_is_multi_step: bool,
    ) -> Any:
        """
        The method to update the eligibility trace data.

        .. note::

            This is the protocol method that should be implemented in the subclass.

        Parameters
        ----------
        running_index : int, optional
            The running index.
        etrace_vals_util_t_1 : ETraceVals
            The history eligibility trace data that have accumulated util :math:`t-1`.
        hid2weight_jac_single_or_multi_times : ETraceVals
            The current eligibility trace data at the time :math:`t`.
        hid2hid_jac_single_or_multi_times : Sequence[jax.Array]
            The data for computing the hidden-to-hidden Jacobian at the time :math:`t`.
        weight_vals : Dict[WeightID, PyTree]
            The weight values.
        input_is_multi_step : bool
            Whether the Jacobian inputs span multiple time steps.

        Returns
        -------
        ETraceVals
            The updated eligibility trace data that have accumulated util :math:`t`.
        """
        raise NotImplementedError

    def _get_etrace_data(self) -> Any:
        """
        Get the eligibility trace data at the last time-step.

        .. note::

            This is the protocol method that should be implemented in the subclass.

        Returns
        -------
        ETraceVals
            The eligibility trace data.
        """
        raise NotImplementedError

    def _assign_etrace_data(self, etrace_vals: Any) -> None:
        """
        Assign the eligibility trace data to the states at the current time-step.

        .. note::

            This is the protocol method that should be implemented in the subclass.

        Parameters
        ----------
        etrace_vals : ETraceVals
            The eligibility trace data.
        """
        raise NotImplementedError
