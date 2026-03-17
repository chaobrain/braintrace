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


__version__ = "0.1.2"
__versio_info__ = tuple(map(int, __version__.split('.')))

from . import nn

# algorithms
from ._etrace_algorithms import ETraceAlgorithm, EligibilityTrace
from ._etrace_vjp import (
    ETraceVjpAlgorithm,
    ETraceVjpGraphExecutor,
    ParamDimVjpAlgorithm,
    D_RTRL,
    IODimVjpAlgorithm,
    ES_D_RTRL,
    pp_prop,
    HybridDimVjpAlgorithm,
)

# concepts
from ._etrace_concepts import (
    ETraceParam,
    NonTempParam,
    ElemWiseParam,
    FakeETraceParam,
    FakeElemWiseParam,
)

# operators
from ._etrace_operators import (
    ETraceOp,
    MatMulOp,
    ElemWiseOp,
    ConvOp,
    SpMatMulOp,
    LoraOp,
    general_y2w,
    stop_param_gradients,
)

# input data
from ._etrace_input_data import SingleStepData, MultiStepData

# graph executor
from ._etrace_graph_executor import ETraceGraphExecutor

# compiler
from ._etrace_compiler_graph import ETraceGraph, compile_etrace_graph
from ._etrace_compiler_hidden_group import HiddenGroup, find_hidden_groups_from_minfo, find_hidden_groups_from_module
from ._etrace_compiler_hid_param_op import (
    HiddenParamOpRelation,
    find_hidden_param_op_relations_from_minfo,
    find_hidden_param_op_relations_from_module,
)
from ._etrace_compiler_module_info import ModuleInfo, extract_module_info
from ._etrace_compiler_hidden_pertubation import (
    HiddenPerturbation,
    add_hidden_perturbation_from_minfo,
    add_hidden_perturbation_in_module,
)

# gradient utilities
from ._grad_exponential import GradExpon

# errors
from ._misc import NotSupportedError, CompilationError

__all__ = [
    # algorithms
    'ETraceAlgorithm',
    'EligibilityTrace',
    'ETraceVjpAlgorithm',
    'ETraceVjpGraphExecutor',
    'ParamDimVjpAlgorithm',
    'D_RTRL',
    'IODimVjpAlgorithm',
    'ES_D_RTRL',
    'pp_prop',
    'HybridDimVjpAlgorithm',

    # concepts
    'ETraceParam',
    'NonTempParam',
    'ElemWiseParam',
    'FakeETraceParam',
    'FakeElemWiseParam',

    # operators
    'ETraceOp',
    'MatMulOp',
    'ElemWiseOp',
    'ConvOp',
    'SpMatMulOp',
    'LoraOp',
    'general_y2w',
    'stop_param_gradients',

    # input data
    'SingleStepData',
    'MultiStepData',

    # graph executor
    'ETraceGraphExecutor',

    # compiler
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

    # gradient utilities
    'GradExpon',

    # errors
    'NotSupportedError',
    'CompilationError',

    # submodules
    'nn',
]


def __getattr__(name):
    mapping = {
        'ETraceState': 'HiddenState',
        'ETraceGroupState': 'HiddenGroupState',
        'ETraceTreeState': 'HiddenTreeState',
    }

    if name in mapping:
        import warnings
        import brainstate

        warnings.warn(
            f"braintrace.{name} is deprecated and will be removed in a future release. "
            f"Please use brainstate.{mapping[name]} instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return getattr(brainstate, mapping[name])
    raise AttributeError(name)
