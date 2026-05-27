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
_ALGORITHM_REGISTRY = {
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


def compile(model, algorithm, *example_inputs, **options):
    """Construct an online-learning algorithm for ``model`` and eagerly build its
    eligibility-trace graph, returning a ready-to-``update`` learner.

    Parameters
    ----------
    model : brainstate.nn.Module
        The recurrent model. Its states must already be initialized, e.g. via
        ``brainstate.nn.init_all_states(model)``.
    algorithm : type or str
        An :class:`ETraceAlgorithm` subclass, or a registered string name
        (case-insensitive), e.g. ``'D_RTRL'``, ``'eprop'``, ``'ottt'``.
    *example_inputs
        Example call inputs (arrays / :class:`SingleStepData` /
        :class:`MultiStepData`), matching what ``learner.update(...)`` will later
        receive. Forwarded to :meth:`ETraceAlgorithm.compile_graph` to trace the
        jaxpr graph. At least one is required.
    **options
        Keyword options forwarded to the algorithm constructor, e.g.
        ``vjp_method``, ``leak``, ``fast_solve``, ``trace_dtype``, ``feedback``.

    Returns
    -------
    ETraceAlgorithm
        The compiled learner; call ``.update(*inputs)`` to train.

    Raises
    ------
    ValueError
        If ``algorithm`` is an unknown string name, or no ``example_inputs`` are
        given.
    TypeError
        If ``algorithm`` is neither an ``ETraceAlgorithm`` subclass nor a string.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> model = MyRNN()
        >>> brainstate.nn.init_all_states(model, batch_size=1)
        >>> x0 = jnp.ones((3,))
        >>> learner = braintrace.compile(model, 'D_RTRL', x0, vjp_method='multi-step')
        >>> y = learner.update(x0)
    """
    cls = _resolve_algorithm(algorithm)
    if len(example_inputs) == 0:
        raise ValueError(
            'compile() needs at least one example input to build the graph '
            'eagerly, e.g. compile(model, "D_RTRL", x0). Pass the same inputs '
            'you will give to learner.update(...).'
        )
    learner = cls(model, **options)
    learner.compile_graph(*example_inputs)
    return learner
