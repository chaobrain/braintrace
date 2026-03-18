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

"""
Eligibility Trace Propagation (ETP) — a JAX transformation for online
learning of recurrent networks.

This package provides a primitive-based alternative to the existing
``ETraceOp`` / name-matching system.  The key components are:

**Primitives & Functions** (Layer 1–2)::

    y = bt.etp.etp_matmul(x, w, bias=b)
    y = bt.etp.etp_elemwise(w, fn=jax.nn.softplus)
    y = bt.etp.etp_conv(x, kernel, bias=b, strides=(1,), padding='SAME')

**Compiler & Graph** (Layer 3)::

    graph = bt.etp.compile_etp_graph(model, sample_input)

**Algorithms** (Layer 4)::

    algo = bt.etp.ETP_DRTRL(model)
    algo.compile(sample_input)
    out = algo(x)  # forward + trace update
"""

# Primitives & rule registries
from ._primitives import (
    etp_matmul_p,
    etp_elemwise_p,
    etp_conv_p,
    ETP_PRIMITIVES,
    is_etp_primitive,
    etp_rules_xy_to_dw,
    etp_rules_xy_to_dw,
)

# User-facing functions
from ._functions import (
    matmul,
    element_wise,
    conv,
)

# Compiler
from ._compiler import (
    ETPOpRelation,
    find_etp_relations_from_jaxpr,
    find_etp_relations_from_minfo,
)

# Graph
from ._graph import (
    ETPGraph,
    compile_etp_graph,
)

# Executor
from ._executor import (
    ETPGraphExecutor,
)

# Algorithms
from ._algorithms import (
    ETP_DRTRL,
)
