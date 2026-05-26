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

"""Eligibility-trace online-learning algorithms.

Groups the core ETrace infrastructure (``ETraceAlgorithm``, ``EligibilityTrace``,
``ETraceGraphExecutor``), VJP-based algorithms (D-RTRL, pp_prop / ES-D-RTRL),
and paper-faithful SNN algorithms (EProp, OSTL, OTPE, OTTT, OSTTP).
"""

from ._common import FixedRandomFeedback, KappaFilter, PresynapticTrace
from .base import ETraceAlgorithm, EligibilityTrace
from .d_rtrl import D_RTRL, ParamDimVjpAlgorithm
from .e_prop import EProp
from .graph_executor import ETraceGraphExecutor
from .ostl import OSTL, OSTLFeedforward, OSTLRecurrent
from .osttp import OSTTP
from .otpe import OTPE
from .ottt import OTTT
from .pp_prop import ES_D_RTRL, IODimVjpAlgorithm, pp_prop
from .vjp_base import ETraceVjpAlgorithm
from .vjp_graph_executor import ETraceVjpGraphExecutor

__all__ = [
    # core
    'ETraceAlgorithm',
    'EligibilityTrace',
    'ETraceGraphExecutor',
    # VJP
    'ETraceVjpAlgorithm',
    'ETraceVjpGraphExecutor',
    'ParamDimVjpAlgorithm',
    'D_RTRL',
    'IODimVjpAlgorithm',
    'ES_D_RTRL',
    'pp_prop',
    # SNN
    'EProp',
    'OSTL',
    'OSTLRecurrent',
    'OSTLFeedforward',
    'OTPE',
    'OTTT',
    'OSTTP',
    'FixedRandomFeedback',
    'KappaFilter',
    'PresynapticTrace',
]
