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

"""Layer 2 — the ETP compiler.

Walks a model's jaxpr to find ETP primitives and connect trainable parameters
to the hidden states they influence. The names re-exported here are the
compiler's façade; a subset of them is lifted again onto the top-level
``braintrace`` namespace.
"""

from braintrace._compiler.base import (
    JaxprEvaluation,
    check_unsupported_op,
    find_element_exist_in_the_set,
    find_matched_vars,
)
from braintrace._compiler.canonicalize import (
    ControlFlowPolicy,
    DEFAULT_CONTROL_FLOW_POLICY,
    canonicalize_control_flow,
    if_convert_conds,
    unroll_inner_scans,
)
from braintrace._compiler.diagnostics import (
    CompilationRecord,
    DiagnosticKind,
    DiagnosticLevel,
    DiagnosticReporter,
    diagnostic_context,
    emit,
    get_reporter,
)
from braintrace._compiler.graph import (
    ETraceGraph,
    compile_etrace_graph,
)
from braintrace._compiler.report import (
    CompilationReport,
)
from braintrace._compiler.hid_param_op import (
    HiddenParamOpRelation,
    find_hidden_param_op_relations_from_minfo,
    find_hidden_param_op_relations_from_module,
)
from braintrace._compiler.hidden_group import (
    HiddenGroup,
    find_hidden_groups_from_minfo,
    find_hidden_groups_from_module,
    gather_learning_signal,
    widen_instant_term,
)
from braintrace._compiler.hidden_pertubation import (
    HiddenPerturbation,
    add_hidden_perturbation_from_minfo,
    add_hidden_perturbation_in_module,
)
from braintrace._compiler.module_info import (
    ModuleInfo,
    extract_module_info,
)
from braintrace._compiler.position_graph import (
    DEFAULT_MAX_JACOBIAN_ELEMENTS,
    SnapPattern,
    build_snap_pattern,
)

__all__ = [
    # jaxpr walking primitives
    'JaxprEvaluation',
    'check_unsupported_op',
    'find_element_exist_in_the_set',
    'find_matched_vars',

    # control-flow canonicalization
    'ControlFlowPolicy',
    'DEFAULT_CONTROL_FLOW_POLICY',
    'canonicalize_control_flow',
    'if_convert_conds',
    'unroll_inner_scans',

    # diagnostics
    'CompilationRecord',
    'CompilationReport',
    'DiagnosticKind',
    'DiagnosticLevel',
    'DiagnosticReporter',
    'diagnostic_context',
    'emit',
    'get_reporter',

    # the compiled graph
    'ETraceGraph',
    'compile_etrace_graph',

    # hidden <-> parameter relations
    'HiddenParamOpRelation',
    'find_hidden_param_op_relations_from_minfo',
    'find_hidden_param_op_relations_from_module',

    # hidden groups
    'HiddenGroup',
    'find_hidden_groups_from_minfo',
    'find_hidden_groups_from_module',
    'gather_learning_signal',
    'widen_instant_term',

    # hidden perturbation
    'HiddenPerturbation',
    'add_hidden_perturbation_from_minfo',
    'add_hidden_perturbation_in_module',

    # module introspection
    'ModuleInfo',
    'extract_module_info',

    # SnAp position graph
    'DEFAULT_MAX_JACOBIAN_ELEMENTS',
    'SnapPattern',
    'build_snap_pattern',
]
