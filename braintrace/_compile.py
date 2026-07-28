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
    ETraceConfig,
    ETraceVmap,
    IODimVjpAlgorithm,
    ParamDimVjpAlgorithm,
    RandomProjectionVjpAlgorithm,
    UORO,
    ThreeFactor,
    DNI,
    D_RTRL,
    pp_prop,
    EProp,
    OSTLRecurrent,
    OSTLFeedforward,
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
    'uoro': UORO,
    'three_factor': ThreeFactor,
    'dni': DNI,
}


#: ``trace_factorization -> engine class``. The factorization axis *is* the
#: engine choice, so this is the whole of config-to-class resolution.
_FACTORIZATION_TO_ENGINE: dict[str, type[ETraceAlgorithm]] = {
    'per_param': ParamDimVjpAlgorithm,
    'io_factorized': IODimVjpAlgorithm,
    'random_projection': RandomProjectionVjpAlgorithm,
}


def _resolve_algorithm(
    algorithm: Union[str, ETraceConfig, Type[ETraceAlgorithm]]
) -> Type[ETraceAlgorithm]:
    """Resolve ``algorithm`` to an :class:`ETraceAlgorithm` subclass.

    Parameters
    ----------
    algorithm : type, str or ETraceConfig
        An :class:`ETraceAlgorithm` subclass (returned unchanged), a registered
        string name (case-insensitive), e.g. ``'D_RTRL'``, ``'eprop'``,
        ``'ostl_recurrent'``, or an :class:`ETraceConfig` whose
        ``trace_factorization`` selects the engine.

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
        or is neither a class, a string, nor an ``ETraceConfig``.
    """
    if isinstance(algorithm, ETraceConfig):
        # Unimplemented factorizations are already rejected by ETraceConfig's
        # compatibility matrix, so a missing key here is a registry bug.
        return _FACTORIZATION_TO_ENGINE[algorithm.trace_factorization]
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
        f'algorithm must be an ETraceAlgorithm subclass, a registered string '
        f'name, or an ETraceConfig, got {type(algorithm)}.'
    )


def compile(
    model: brainstate.nn.Module,
    algorithm: Union[str, ETraceConfig, Type[ETraceAlgorithm]],
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
    algorithm : type, str or ETraceConfig
        An :class:`ETraceAlgorithm` subclass, a registered case-insensitive
        name, or an :class:`ETraceConfig` naming the learning-rule coordinate
        directly. Registered names are ``'d_rtrl'``, ``'pp_prop'``,
        ``'es_d_rtrl'``, ``'esd_rtrl'``, ``'eprop'``, ``'e_prop'``,
        ``'ostl_recurrent'``, ``'ostl_feedforward'``, ``'uoro'``,
        ``'three_factor'``, and ``'dni'``. The aliases ``'es_d_rtrl'`` and
        ``'esd_rtrl'`` select :class:`pp_prop`; ``'e_prop'`` selects
        :class:`EProp`. :class:`SnAp` has no registered string name and must be
        passed as a class. A config selects the engine through its
        ``trace_factorization`` and is forwarded to the constructor, so any
        coordinate admitted by the compatibility matrix can be compiled
        without a named preset.
    *example_inputs : Any
        Example call inputs (arrays / :class:`SingleStepData` /
        :class:`MultiStepData`) matching what ``learner.update(...)`` will
        receive. At least one is required.
    batch_size : int or None, optional
        Forwarded to ``brainstate.nn.init_all_states``. ``None`` (default)
        initializes unbatched states. Must match the batch dimension of
        ``example_inputs``.
    seed : int or None, optional
        If given, state initialization runs inside
        ``brainstate.random.seed_context`` for reproducibility; the global
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
    **options : Any
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
        If ``algorithm`` is not an ``ETraceAlgorithm`` subclass, registered
        string, or :class:`ETraceConfig`; if a required algorithm option is
        missing; or if the same config is supplied both as ``algorithm`` and
        through ``config=``.
    braintrace.CompilationError
        If no trainable weights are routed through ETP ops (nothing to learn
        online).

    Notes
    -----
    **Algorithm options.** ``**options`` are forwarded verbatim to the algorithm
    constructor. Required options have no default; omitting one raises
    ``TypeError``. The class pages are the authoritative source for accepted
    options:

    - :class:`D_RTRL` for ``'d_rtrl'``.
    - :class:`pp_prop` for ``'pp_prop'``, ``'es_d_rtrl'``, and ``'esd_rtrl'``.
    - :class:`EProp` for ``'eprop'`` and ``'e_prop'``.
    - :class:`OSTLRecurrent` for ``'ostl_recurrent'``.
    - :class:`OSTLFeedforward` for ``'ostl_feedforward'``.
    - :class:`UORO` for ``'uoro'``.
    - :class:`ThreeFactor` for ``'three_factor'``.
    - :class:`DNI` for ``'dni'``.
    - :class:`SnAp` for class-only SnAp compilation.

    Passing an algorithm class supports subclasses and avoids the string
    registry. Passing :class:`ETraceConfig` selects
    :class:`ParamDimVjpAlgorithm`, :class:`IODimVjpAlgorithm`, or
    :class:`RandomProjectionVjpAlgorithm` from ``trace_factorization``. Do not
    also pass ``config=`` in that case.

    Calling ``compile`` twice on the same model re-initializes its states.

    **Axis coordinates.** Passing an :class:`ETraceConfig` in the ``algorithm``
    position compiles a coordinate that may have no preset name:

    .. code-block:: python

        # an x-side leak with an instantaneous f-side
        learner = braintrace.compile(
            model,
            braintrace.ETraceConfig(
                trace_factorization='io_factorized', decay=(0.9, 0.0)),
            x0,
        )

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>> import jax.numpy as jnp
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
        >>> # Registered string: initialize states, build the graph, return a learner.
        >>> by_name = braintrace.compile(model, 'd_rtrl', x0, batch_size=1)
        >>> y = by_name.update(x0)
        >>>
        >>> # Algorithm classes, including SnAp, can be passed directly.
        >>> by_class = braintrace.compile(
        ...     model, braintrace.D_RTRL, x0, batch_size=1)
        >>>
        >>> # A config selects an engine from its trace factorization.
        >>> config = braintrace.ETraceConfig()
        >>> by_config = braintrace.compile(
        ...     model, config, x0, batch_size=1)
        >>>
        >>> # Every learner compile returns carries the two sequence drivers.
        >>> xs = brainstate.random.randn(10, 1, 3)   # (T, batch, features)
        >>> ys = brainstate.random.randn(10, 1, 1)
        >>> def step_loss(x, y):
        ...     return jnp.mean((by_name(x) - y) ** 2)
        >>> grads, losses = by_name.etrace_grad(xs, ys, step_fn=step_loss, return_value=True)
        >>> # etrace_evolve is the same drive with no loss: it advances hidden
        >>> # state and the eligibility trace, optionally stacking the outputs.
        >>> outs = by_name.etrace_evolve(xs, return_outputs=True)
    """
    cls = _resolve_algorithm(algorithm)
    if isinstance(algorithm, ETraceConfig):
        if 'config' in options:
            raise TypeError(
                'compile() got a config both in the `algorithm` position and as '
                'a `config=` option. Pass it once.'
            )
        options['config'] = algorithm
        if cls is IODimVjpAlgorithm:
            # The engine's `decay_or_rank` stays a required argument (it is the
            # documented user-facing spelling); source it from the config's
            # already-canonical (x, f) pair rather than asking twice.
            options.setdefault('decay_or_rank', algorithm.decay)
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
        # ETraceVmap, not brainstate.nn.Vmap: the wrapper must carry
        # etrace_grad / etrace_evolve so the call site is identical in batched
        # and unbatched mode. Reaching into `.module` instead would drive the
        # *unbatched* learner and silently give per-lane-wrong results. It is
        # still a brainstate.nn.Vmap, so existing users are unaffected.
        result: ETraceAlgorithm | brainstate.nn.Vmap = ETraceVmap(learner, vmap_states='new')
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
