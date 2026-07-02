# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

"""braintrace: online learning for recurrent networks via Eligibility Trace Propagation (ETP).

``braintrace`` trains recurrent and spiking neural networks **online** — forward
in time, without backpropagation through time (BPTT). Models mark their
trainable operations with ETP user-API ops (for example :func:`matmul`,
:func:`conv`, :func:`sparse_matmul`, :func:`lora_matmul`, :func:`element_wise`)
instead of wrapping parameters in a special class. A compiler then walks the
JAX ``jaxpr``, identifies those ETP primitives, and connects each parameter to
the hidden states it influences so that eligibility traces can be propagated.

The public API is organised in four layers, with dependencies pointing strictly
downward:

1. **ETP operators** — the user-facing ops (:func:`matmul`, :func:`conv`, ...),
   the :class:`ETPPrimitive` class, and :func:`register_primitive` for adding
   new ones.
2. **Compiler** — :func:`compile_etrace_graph` and the analysis containers
   (:class:`ETraceGraph`, :class:`ModuleInfo`, :class:`HiddenGroup`,
   :class:`HiddenParamOpRelation`, :class:`HiddenPerturbation`) plus the
   diagnostics types (:class:`CompilationRecord`, :class:`DiagnosticKind`,
   :class:`DiagnosticLevel`).
3. **Graph executor** — :class:`ETraceGraphExecutor` /
   :class:`ETraceVjpGraphExecutor`, which run the forward pass and the
   hidden->weight / hidden->hidden Jacobian computations.
4. **Algorithms** — online-learning orchestrators: the exact algorithms
   :class:`D_RTRL` / :func:`pp_prop` / :class:`ES_D_RTRL`, and the SNN family
   :class:`EProp`, :class:`OSTLRecurrent`, :class:`OSTLFeedforward`,
   :class:`OTPE`, :class:`OTTT`, :class:`OSTTP`.

The :mod:`braintrace.nn` subpackage provides ready-made ETP-wired layers
(linear maps, convolutions, recurrent cells, read-outs).

Notes
-----
The convenience entry point :func:`compile` wraps a model together with an
algorithm into a single trainable object and is the recommended starting point.
The ``braintrace.MatMulOp`` / ``ETraceParam`` style names from the v0.1.x API
are deprecated shims served lazily with a :class:`DeprecationWarning`; new code
should mark parameters by routing them through ETP ops instead.

Examples
--------
.. code-block:: python

    >>> import braintrace
    >>> # the public API surface is enumerated by __all__
    >>> 'matmul' in braintrace.__all__
    True
"""


from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import nn
from ._compile import compile
from ._algorithm import (
    ETraceAlgorithm,
    EligibilityTrace,
    ETraceGraphExecutor,
    ETraceVjpAlgorithm,
    ETraceVjpGraphExecutor,
    ParamDimVjpAlgorithm,
    D_RTRL,
    pp_prop,
    ES_D_RTRL,
    IODimVjpAlgorithm,
    EProp,
    OSTLRecurrent,
    OSTLFeedforward,
    OTPE,
    OTTT,
    OSTTP,
    FixedRandomFeedback,
    KappaFilter,
    PresynapticTrace,
)
from ._compiler import (
    ControlFlowPolicy,
    ETraceGraph,
    compile_etrace_graph,
    HiddenParamOpRelation,
    find_hidden_param_op_relations_from_minfo,
    find_hidden_param_op_relations_from_module,
    HiddenGroup,
    find_hidden_groups_from_minfo,
    find_hidden_groups_from_module,
    HiddenPerturbation,
    add_hidden_perturbation_from_minfo,
    add_hidden_perturbation_in_module,
    ModuleInfo,
    extract_module_info,
    CompilationRecord,
    CompilationReport,
    DiagnosticKind,
    DiagnosticLevel,
)
from ._op import (
    ETPPrimitive,
    matmul,
    grouped_matmul,
    embedding,
    einsum,
    element_wise,
    conv,
    sparse_matmul,
    lora_matmul,
    register_primitive,
)
from ._grad_exponential import GradExpon
from ._input_data import (
    SingleStepData,
    MultiStepData,
)
from ._misc import NotSupportedError, CompilationError
from ._version import __version__, __version_info__

if TYPE_CHECKING:
    # The v0.1.x legacy shims are deprecated and served lazily via ``__getattr__``
    # below. Re-import them here so static type checkers / IDEs can still resolve
    # ``braintrace.MatMulOp`` etc.
    from ._legacy import (
        ConvOp,
        ElemWiseOp,
        ElemWiseParam,
        ETraceOp,
        ETraceParam,
        FakeElemWiseParam,
        FakeETraceParam,
        LoraOp,
        MatMulOp,
        NonTempParam,
        SpMatMulOp,
    )

__all__ = [
    # version
    '__version__',
    '__version_info__',

    # algorithms
    'ETraceAlgorithm',
    'EligibilityTrace',
    'ETraceVjpAlgorithm',
    'ETraceVjpGraphExecutor',
    'ParamDimVjpAlgorithm',
    'D_RTRL',
    'pp_prop',
    'ES_D_RTRL',
    'IODimVjpAlgorithm',

    # one-call entry point
    'compile',

    # ETP primitives (user API)
    'matmul',
    'grouped_matmul',
    'embedding',
    'einsum',
    'element_wise',
    'conv',
    'sparse_matmul',
    'lora_matmul',

    # ETP primitive class & rule registration
    'ETPPrimitive',
    'register_primitive',

    # input data
    'SingleStepData',
    'MultiStepData',

    # graph executor
    'ETraceGraphExecutor',

    # compiler
    'ControlFlowPolicy',
    'ETraceGraph',
    'compile_etrace_graph',
    'HiddenGroup',
    'find_hidden_groups_from_minfo',
    'find_hidden_groups_from_module',
    'HiddenParamOpRelation',
    'find_hidden_param_op_relations_from_minfo',
    'find_hidden_param_op_relations_from_module',
    'ModuleInfo',
    'extract_module_info',
    'HiddenPerturbation',
    'add_hidden_perturbation_from_minfo',
    'add_hidden_perturbation_in_module',

    # compiler diagnostics
    'CompilationRecord',
    'CompilationReport',
    'DiagnosticKind',
    'DiagnosticLevel',

    # gradient utilities
    'GradExpon',

    # SNN online-learning algorithms
    'EProp',
    'OSTLRecurrent',
    'OSTLFeedforward',
    'OTPE',
    'OTTT',
    'OSTTP',
    'FixedRandomFeedback',
    'KappaFilter',
    'PresynapticTrace',

    # errors
    'NotSupportedError',
    'CompilationError',

    # submodules
    'nn',
]


# --- v0.1.x legacy shims: deprecated, served lazily with an access-time warning.
# Each maps the public name -> migration replacement text. The shim classes still
# work; new code should use the primitive-based ETP user-API instead.
_DEPRECATED_LEGACY = {
    'MatMulOp': 'braintrace.matmul (with a brainstate.ParamState)',
    'ElemWiseOp': 'braintrace.element_wise',
    'ConvOp': 'braintrace.conv',
    'SpMatMulOp': 'braintrace.sparse_matmul',
    'LoraOp': 'braintrace.lora_matmul',
    'ETraceOp': 'the braintrace ETP primitive functions (matmul, conv, ...)',
    'ETraceParam': 'brainstate.ParamState together with an ETP primitive function',
    'ElemWiseParam': 'brainstate.ParamState together with braintrace.element_wise',
    'NonTempParam': 'brainstate.ParamState with plain JAX ops (keeps the weight out of the ETP graph)',
    'FakeETraceParam': 'a plain object with plain JAX ops',
    'FakeElemWiseParam': 'a plain object with plain JAX ops',
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_LEGACY:
        import warnings
        warnings.warn(
            f'braintrace.{name} is deprecated since 0.2.0 and will be removed in a '
            f'future release; use {_DEPRECATED_LEGACY[name]} instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        from . import _legacy
        return getattr(_legacy, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:
    return sorted(list(__all__) + list(_DEPRECATED_LEGACY))
