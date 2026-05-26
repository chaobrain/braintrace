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

r"""ETP (Eligibility Trace Propagation) primitives + rule registries + user API.

This package replaces the legacy single-file ``braintrace._etrace_op``
module. The submodule layout is:

* :mod:`._registries` — global registries + flag-checking helpers
* :mod:`._primitive` — :class:`ETPPrimitive` + :func:`register_primitive`
* :mod:`.dense` — ``etp_mm_p``, ``etp_mv_p``, :func:`matmul`
* :mod:`.elemwise` — ``etp_elemwise_p``, :func:`element_wise`
* :mod:`.conv` — ``etp_conv_p``, :func:`conv`
* :mod:`.sparse` — ``etp_sp_mm_p``, ``etp_sp_mv_p``, :func:`sparse_matmul`
* :mod:`.lora` — ``etp_lora_mm_p``, ``etp_lora_mv_p``, :func:`lora_matmul`

The public surface mirrors the legacy module: every name previously
exported from ``braintrace._etrace_op`` is also available here.
"""

from ._primitive import ETPPrimitive, register_primitive
from ._registries import (
    BATCHED_PRIMITIVES,
    ETP_PRIMITIVES,
    ETP_RULES_INIT_DRTRL,
    ETP_RULES_INIT_PP,
    ETP_RULES_XY_TO_DW,
    ETP_RULES_YW_TO_W,
    ETP_TRAINABLE_INVARS_FNS,
    ETP_X_INVAR_INDICES,
    ETP_Y_OUTVAR_INDICES,
    GRADIENT_ENABLED_PRIMITIVES,
    get_trainable_invars,
    get_x_invar_index,
    get_y_outvar_index,
    is_batched_primitive,
    is_etp_enable_gradient_primitive,
    is_etp_primitive,
)
from .conv import _etp_conv_impl
from .conv import conv, etp_conv_p
from .dense import etp_mm_p, etp_mv_p, matmul
from .elemwise import element_wise, etp_elemwise_p
from .lora import etp_lora_mm_p, etp_lora_mv_p, lora_matmul
from .sparse import etp_sp_mm_p, etp_sp_mv_p, sparse_matmul

__all__ = [
    # ETP primitive class & registration
    'ETPPrimitive',
    'register_primitive',

    # registries + flag helpers
    'ETP_PRIMITIVES',
    'ETP_RULES_YW_TO_W',
    'ETP_RULES_XY_TO_DW',
    'ETP_RULES_INIT_DRTRL',
    'ETP_RULES_INIT_PP',
    'ETP_TRAINABLE_INVARS_FNS',
    'ETP_X_INVAR_INDICES',
    'ETP_Y_OUTVAR_INDICES',
    'GRADIENT_ENABLED_PRIMITIVES',
    'BATCHED_PRIMITIVES',
    'is_etp_primitive',
    'is_etp_enable_gradient_primitive',
    'is_batched_primitive',
    'get_trainable_invars',
    'get_x_invar_index',
    'get_y_outvar_index',

    # primitives
    'etp_mm_p',
    'etp_mv_p',
    'etp_elemwise_p',
    'etp_conv_p',
    'etp_sp_mm_p',
    'etp_sp_mv_p',
    'etp_lora_mm_p',
    'etp_lora_mv_p',

    # user API
    'matmul',
    'element_wise',
    'conv',
    'sparse_matmul',
    'lora_matmul',
]
