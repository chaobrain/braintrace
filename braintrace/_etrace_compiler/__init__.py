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

from braintrace._etrace_compiler.base import (
    JaxprEvaluation,
    check_unsupported_op,
    find_element_exist_in_the_set,
    find_matched_vars,
)
from braintrace._etrace_compiler.graph import (
    ETraceGraph,
    compile_etrace_graph,
)
from braintrace._etrace_compiler.hid_param_op import (
    HiddenParamOpRelation,
    find_hidden_param_op_relations_from_minfo,
    find_hidden_param_op_relations_from_module,
)
from braintrace._etrace_compiler.hidden_group import (
    HiddenGroup,
    find_hidden_groups_from_minfo,
    find_hidden_groups_from_module,
)
from braintrace._etrace_compiler.hidden_pertubation import (
    HiddenPerturbation,
    add_hidden_perturbation_from_minfo,
    add_hidden_perturbation_in_module,
)
from braintrace._etrace_compiler.module_info import (
    ModuleInfo,
    extract_module_info,
)
