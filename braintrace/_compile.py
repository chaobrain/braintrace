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

from typing import Type, Union

import brainstate

from ._misc import CompilationError
from ._etrace_algorithms import (
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
    model,
    algorithm,
    *example_inputs,
    batch_size=None,
    seed=None,
    verbose=0,
    **options,
):
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
    **options
        Forwarded to the algorithm constructor. See *Algorithm options* below.

    Returns
    -------
    ETraceAlgorithm
        The compiled learner, carrying a :attr:`~ETraceAlgorithm.report`. Call
        ``.update(*inputs)`` to train.

    Raises
    ------
    ValueError
        If ``algorithm`` is an unknown name, no ``example_inputs`` are given, or
        ``verbose`` is not in ``{0, 1, 2}``.
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

    - ``'D_RTRL'`` — ``vjp_method``, ``fast_solve``, ``trace_dtype``, ``name``.
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

        >>> import braintrace, jax.numpy as jnp
        >>> model = MyRNN()
        >>> x0 = jnp.ones((1, 3))
        >>> learner = braintrace.compile(model, 'D_RTRL', x0, batch_size=1, verbose=1)
        >>> y = learner.update(x0)
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

    # --- state initialization (always) --- #
    if seed is not None:
        with brainstate.random.seed_context(seed):
            brainstate.nn.init_all_states(model, batch_size=batch_size)
    else:
        brainstate.nn.init_all_states(model, batch_size=batch_size)

    # --- construct + compile the graph --- #
    learner = cls(model, **options)
    learner.compile_graph(*example_inputs)

    # --- guardrail: nothing trainable online --- #
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

    return learner
