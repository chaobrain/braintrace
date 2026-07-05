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

from __future__ import annotations

from typing import Any, Type, Union

import jax
import brainstate

from ._misc import CompilationError
from ._algorithm import (
    ETraceAlgorithm,
    D_RTRL,
    pp_prop,
    EProp,
    OSTLRecurrent,
    OSTLFeedforward,
    OTPE,
    OTTT,
    OSTTP,
)

__all__ = ['compile']

# Canonical lowercase name (+ aliases) -> algorithm class. No bare ``ostl``
# alias: the ambiguous OSTL factory was removed in 0.2.0, so callers pick
# ``ostl_recurrent`` vs ``ostl_feedforward`` explicitly.
_ALGORITHM_REGISTRY: dict[str, type[ETraceAlgorithm]] = {
    'd_rtrl': D_RTRL,
    'pp_prop': pp_prop,
    'es_d_rtrl': pp_prop,
    'esd_rtrl': pp_prop,
    'eprop': EProp,
    'e_prop': EProp,
    'ostl_recurrent': OSTLRecurrent,
    'ostl_feedforward': OSTLFeedforward,
    'otpe': OTPE,
    'ottt': OTTT,
    'osttp': OSTTP,
}


def _resolve_algorithm(
    algorithm: Union[str, Type[ETraceAlgorithm]]
) -> Type[ETraceAlgorithm]:
    """Resolve ``algorithm`` to an :class:`ETraceAlgorithm` subclass.

    Parameters
    ----------
    algorithm : type or str
        Either an :class:`ETraceAlgorithm` subclass (returned unchanged) or a
        registered string name (case-insensitive), e.g. ``'D_RTRL'``,
        ``'eprop'``, ``'ottt'``.

    Returns
    -------
    type
        The resolved :class:`ETraceAlgorithm` subclass.

    Raises
    ------
    ValueError
        If ``algorithm`` is a string that is not a registered name.
    TypeError
        If ``algorithm`` is a class that is not an ``ETraceAlgorithm`` subclass,
        or is neither a class nor a string.
    """
    if isinstance(algorithm, type):
        if issubclass(algorithm, ETraceAlgorithm):
            return algorithm
        raise TypeError(
            f'algorithm class must be a subclass of ETraceAlgorithm, got {algorithm!r}.'
        )
    if isinstance(algorithm, str):
        key = algorithm.strip().lower()
        try:
            return _ALGORITHM_REGISTRY[key]
        except KeyError:
            valid = ', '.join(sorted(_ALGORITHM_REGISTRY))
            raise ValueError(
                f'Unknown algorithm name {algorithm!r}. Valid names: {valid}. '
                f'Or pass an ETraceAlgorithm subclass directly.'
            )
    raise TypeError(
        f'algorithm must be an ETraceAlgorithm subclass or a registered string name, '
        f'got {type(algorithm)}.'
    )


def compile(
    model: brainstate.nn.Module,
    algorithm: Union[str, Type[ETraceAlgorithm]],
    *example_inputs: Any,
    batch_size: int | None = None,
    seed: int | None = None,
    verbose: int = 0,
    vmap: bool = False,
    **options: Any,
) -> ETraceAlgorithm | brainstate.nn.Vmap:
    """Define an eligibility-trace online-learning model in one call.

    This is the unified entry point. It initializes the model's states, builds
    the eligibility-trace graph, checks that the model is trainable online, and
    (optionally) prints a compilation report — returning a ready-to-``update``
    learner.

    Parameters
    ----------
    model : brainstate.nn.Module
        The recurrent / spiking model defining one-step behavior. It does **not**
        need to be pre-initialized; ``compile`` always (re)initializes its states.
    algorithm : type or str
        An :class:`ETraceAlgorithm` subclass, or a registered case-insensitive
        name, e.g. ``'D_RTRL'``, ``'es_d_rtrl'``, ``'eprop'``, ``'otpe'``,
        ``'ottt'``, ``'osttp'``, ``'ostl_recurrent'``, ``'ostl_feedforward'``.
    *example_inputs
        Example call inputs (arrays / :class:`SingleStepData` /
        :class:`MultiStepData`) matching what ``learner.update(...)`` will
        receive. At least one is required.
    batch_size : int or None, optional
        Forwarded to :func:`brainstate.nn.init_all_states`. ``None`` (default)
        initializes unbatched states. Must match the batch dimension of
        ``example_inputs``.
    seed : int or None, optional
        If given, state initialization runs inside
        :func:`brainstate.random.seed_context` for reproducibility; the global
        RNG is restored afterwards. ``None`` (default) leaves the RNG untouched.
        Weights created at model-construction time are outside this scope.
    verbose : int, optional
        Report verbosity printed at compile time: ``0`` (default) silent, ``1``
        the structural summary, ``2`` additionally compiler WARNING/ERROR
        diagnostics. Other values raise :class:`ValueError`.
    vmap : bool, optional
        When ``False`` (default) states are initialized with
        ``init_all_states(model, batch_size=batch_size)``. When ``True``, states
        are created under
        ``brainstate.transform.vmap_new_states(state_tag='new', axis_size=batch_size)``
        and the learner is wrapped in
        ``brainstate.nn.Vmap(vmap_states='new')``. In vmap mode:
        ``example_inputs`` carry the batch axis (axis 0); ``batch_size`` is
        **required** and used as the vmap ``axis_size``; the return value is a
        ``brainstate.nn.Vmap`` whose ``.module`` is the learner (use
        ``result.module.report``). Requires a model whose hidden states are all
        (re)created in ``init_all_states``; models holding construction-time
        states may raise ``brainstate.transform.BatchAxisError``.
    **options
        Forwarded to the algorithm constructor. See *Algorithm options* below.

    Returns
    -------
    ETraceAlgorithm or brainstate.nn.Vmap
        The compiled learner, carrying a :attr:`~ETraceAlgorithm.report`. Call
        ``.update(*inputs)`` to train. When ``vmap=True``, returns a
        ``brainstate.nn.Vmap`` wrapper; access the learner via ``.module``.

    Raises
    ------
    ValueError
        If ``algorithm`` is an unknown name, no ``example_inputs`` are given,
        ``verbose`` is not in ``{0, 1, 2}``, or ``vmap=True`` without
        ``batch_size``.
    TypeError
        If ``algorithm`` is neither an ``ETraceAlgorithm`` subclass nor a string,
        or a required algorithm option (see below) is missing.
    braintrace.CompilationError
        If no trainable weights are routed through ETP ops (nothing to learn
        online).

    Notes
    -----
    **Algorithm options.** ``**options`` are forwarded verbatim to the algorithm
    constructor. Required options have no default; omitting one raises
    ``TypeError``. Authoritative descriptions live on each algorithm class.

    *Common (algorithm-dependent subset):*

    - ``name`` (str) — module name.
    - ``vjp_method`` (str, default ``'single-step'``) — loss→hidden VJP unrolling.
    - ``fast_solve`` (bool, default ``True``) — fast linear solve for the
      hidden→weight Jacobian.
    - ``trace_dtype`` — eligibility-trace storage dtype.

    *Per algorithm:*

    - ``'D_RTRL'`` — ``vjp_method``, ``fast_solve``, ``trace_dtype``,
      ``chunked_trace`` (bool, default ``True``: closed-form multi-step trace
      roll via chunk factorization; see :class:`ParamDimVjpAlgorithm`),
      ``name``.
    - ``'es_d_rtrl'`` / ``'pp_prop'`` — ``decay_or_rank`` (**required**: float in
      (0, 1) ⇒ decay, or int ≥ 1 ⇒ rank), ``vjp_method``, ``fast_solve``,
      ``name``.
    - ``'eprop'`` — ``feedback`` (default ``'symmetric'``), ``kappa_filter_decay``
      (default ``0.0``), ``random_feedback_key`` (default ``None``),
      ``vjp_method``, ``fast_solve``, ``name``.
    - ``'otpe'`` — ``leak`` (**required**, keyword-only), ``mode`` (default
      ``'full'``), ``trace_clip_abs`` (default ``None``), ``vjp_method``,
      ``name``.
    - ``'ottt'`` — ``leak`` (**required**, keyword-only), ``mode`` (default
      ``'A'``), ``vjp_method``, ``name``.
    - ``'osttp'`` — ``B_list`` (**required**: per-layer feedback matrices),
      ``target_timing`` (default ``'per-step'``), ``vjp_method``, ``fast_solve``,
      ``name``.
    - ``'ostl_recurrent'`` — ``vjp_method``, ``fast_solve``, ``trace_dtype``,
      ``name``.
    - ``'ostl_feedforward'`` — ``decay_or_rank`` (default ``1e-6``), ``name``.

    Calling ``compile`` twice on the same model re-initializes its states.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>>
        >>> class RNN(brainstate.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.cell = braintrace.nn.ValinaRNNCell(3, 4, activation='tanh')
        ...         self.out = braintrace.nn.Linear(4, 1)
        ...     def update(self, x):
        ...         return x >> self.cell >> self.out
        >>>
        >>> model = RNN()
        >>> x0 = brainstate.random.randn(1, 3)   # (batch, features)
        >>> # one call: initialise states, build the trace graph, return a learner
        >>> learner = braintrace.compile(model, 'D_RTRL', x0, batch_size=1)
        >>> y = learner.update(x0)               # forward pass + eligibility-trace update
    """
    cls = _resolve_algorithm(algorithm)
    if len(example_inputs) == 0:
        raise ValueError(
            'compile() needs at least one example input to build the graph '
            'eagerly, e.g. compile(model, "D_RTRL", x0). Pass the same inputs '
            'you will give to learner.update(...).'
        )
    if verbose not in (0, 1, 2):
        raise ValueError(f'verbose must be 0, 1, or 2, got {verbose!r}.')
    if vmap and batch_size is None:
        raise ValueError(
            'compile(..., vmap=True) requires batch_size, used as the per-sample '
            'vmap axis size. Pass batch_size=<n_batch> matching the batch axis '
            '(axis 0) of example_inputs.'
        )

    if vmap:
        # Per-sample vmap scheme: example_inputs carry the batch axis (axis 0);
        # the eligibility-trace graph is built per-lane on an unbatched sample,
        # while hidden + trace states are created with the new per-sample axis.
        learner = cls(model, **options)
        unbatched = jax.tree.map(lambda a: a[0], example_inputs)

        @brainstate.transform.vmap_new_states(state_tag='new', axis_size=batch_size)
        def _init() -> None:
            brainstate.nn.init_all_states(model)
            learner.compile_graph(*unbatched)

        if seed is not None:
            with brainstate.random.seed_context(seed):
                _init()
        else:
            _init()
        result: ETraceAlgorithm | brainstate.nn.Vmap = brainstate.nn.Vmap(learner, vmap_states='new')
    else:
        # --- state initialization (always) --- #
        if seed is not None:
            with brainstate.random.seed_context(seed):
                brainstate.nn.init_all_states(model, batch_size=batch_size)
        else:
            brainstate.nn.init_all_states(model, batch_size=batch_size)
        # --- construct + compile the graph --- #
        learner = cls(model, **options)
        learner.compile_graph(*example_inputs)
        result = learner

    # --- guardrail: nothing trainable online (uses learner.graph in both modes) --- #
    # A model is trainable online iff the compiler discovered at least one
    # hidden<->parameter ETP relation. No relations means no trainable weight
    # reaches a hidden state through an ETP op — nothing to train online.
    if len(learner.graph.hidden_param_op_relations) == 0:
        raise CompilationError(
            'No trainable weights are routed through ETP ops, so the model has '
            'nothing to train online. Route trainable parameters through an ETP '
            'op (braintrace.matmul / conv / sparse_matmul / lora_matmul / '
            'element_wise) instead of a plain JAX op.'
        )

    # --- compile-time report --- #
    if verbose >= 1:
        learner.report.show(level=verbose)

    return result
