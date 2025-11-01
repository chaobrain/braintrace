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


__version__ = "0.1.0"
__version_info__ = (0, 1, 0)

from brainscale._etrace_algorithms import *
from brainscale._etrace_algorithms import __all__ as _alg_all
from brainscale._etrace_compiler_graph import *
from brainscale._etrace_compiler_graph import __all__ as _compiler_all
from brainscale._etrace_compiler_hid_param_op import *
from brainscale._etrace_compiler_hid_param_op import __all__ as _hid_param_all
from brainscale._etrace_compiler_hidden_group import *
from brainscale._etrace_compiler_hidden_group import __all__ as _hid_group_all
from brainscale._etrace_compiler_hidden_pertubation import *
from brainscale._etrace_compiler_hidden_pertubation import __all__ as _hid_pertub_all
from brainscale._etrace_compiler_module_info import *
from brainscale._etrace_compiler_module_info import __all__ as _mod_info_all
from brainscale._etrace_concepts import *
from brainscale._etrace_concepts import __all__ as _con_all
from brainscale._etrace_graph_executor import *
from brainscale._etrace_graph_executor import __all__ as _exec_all
from brainscale._etrace_input_data import *
from brainscale._etrace_input_data import __all__ as _data_all
from brainscale._etrace_operators import *
from brainscale._etrace_operators import __all__ as _op_all
from brainscale._etrace_vjp import *
from brainscale._etrace_vjp import __all__ as _vjp_all
from brainscale._grad_exponential import *
from brainscale._grad_exponential import __all__ as _grad_exp_all
from brainscale._misc import *
from brainscale._misc import __all__ as _misc_all
from . import nn

__all__ = ['nn'] + _alg_all + _compiler_all + _hid_param_all + _hid_group_all + _hid_pertub_all
__all__ += _mod_info_all + _con_all + _exec_all + _data_all + _op_all + _vjp_all
__all__ += _grad_exp_all + _misc_all

del _alg_all, _compiler_all, _hid_param_all, _hid_group_all, _hid_pertub_all
del _mod_info_all, _con_all, _exec_all, _data_all, _op_all, _vjp_all
del _grad_exp_all,
del _misc_all


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
            f"brainscale.{name} is deprecated and will be removed in a future release. "
            f"Please use brainstate.{mapping[name]} instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return getattr(brainstate, mapping[name])
    raise AttributeError(name)
