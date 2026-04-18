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

"""Backwards-compatibility shims for the v0.1.x braintrace API.

All names re-exported here are deprecated in v0.2.0. Each class either:

* routes through the new ETP primitive user-API (``ETraceParam`` +
  :class:`MatMulOp` / :class:`ConvOp` / ...), or
* uses plain JAX ops so the compiler does not register a relation
  (``NonTempParam``, ``FakeETraceParam``, ``FakeElemWiseParam``).

Importing any of these names triggers a :class:`DeprecationWarning`
(once per class, per Python process).
"""

from ._ops import (
    ConvOp,
    ElemWiseOp,
    ETraceOp,
    LoraOp,
    MatMulOp,
    SpMatMulOp,
    general_y2w,
    stop_param_gradients,
)
from ._params import (
    ElemWiseParam,
    ETraceParam,
    FakeElemWiseParam,
    FakeETraceParam,
    NonTempParam,
)

__all__ = [
    # params
    'ETraceParam',
    'ElemWiseParam',
    'NonTempParam',
    'FakeETraceParam',
    'FakeElemWiseParam',
    # ops
    'ETraceOp',
    'MatMulOp',
    'ElemWiseOp',
    'ConvOp',
    'SpMatMulOp',
    'LoraOp',
    # utilities
    'general_y2w',
    'stop_param_gradients',
]
