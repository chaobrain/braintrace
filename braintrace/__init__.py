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


from . import nn
from ._etrace_algorithms import (
    ETraceAlgorithm,
    EligibilityTrace,
    ETraceGraphExecutor,
    ETraceVjpAlgorithm,
    ETraceVjpGraphExecutor,
    ParamDimVjpAlgorithm,
    D_RTRL,
    IODimVjpAlgorithm,
    ES_D_RTRL,
    pp_prop,
    EProp,
    OSTL,
    OTPE,
    OTTT,
    OSTTP,
    FixedRandomFeedback,
    KappaFilter,
    PresynapticTrace,
)
from ._etrace_compiler import (
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
    DiagnosticKind,
    DiagnosticLevel,
)
from ._etrace_op import (
    ETPPrimitive,
    matmul,
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
from ._misc import NotSupportedError, CompilationError
from ._version import __version__, __version_info__

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
    'IODimVjpAlgorithm',
    'ES_D_RTRL',
    'pp_prop',

    # ETP primitives (user API)
    'matmul',
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
    'DiagnosticKind',
    'DiagnosticLevel',

    # gradient utilities
    'GradExpon',

    # SNN online-learning algorithms
    'EProp',
    'OSTL',
    'OTPE',
    'OTTT',
    'OSTTP',
    'FixedRandomFeedback',
    'KappaFilter',
    'PresynapticTrace',

    # errors
    'NotSupportedError',
    'CompilationError',

    # legacy v0.1.x shims (deprecated)
    'ETraceParam',
    'ElemWiseParam',
    'NonTempParam',
    'FakeETraceParam',
    'FakeElemWiseParam',
    'ETraceOp',
    'MatMulOp',
    'ElemWiseOp',
    'ConvOp',
    'SpMatMulOp',
    'LoraOp',

    # submodules
    'nn',
]
