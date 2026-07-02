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

import jax

__all__ = [
    'Primitive',
    'Var',
    'JaxprEqn',
    'Jaxpr',
    'ClosedJaxpr',
    'Literal',
    'new_var',
    'new_jaxpr_eqn',
    'stop_gradient_p',
    'is_jit_primitive',
    'is_scan_primitive',
    'is_while_primitive',
    'is_cond_primitive',
]

from brainstate._compatible_import import Primitive, Var, JaxprEqn, Jaxpr, ClosedJaxpr, Literal

try:
    from jax.extend.core import new_jaxpr_eqn
except ImportError:  # older JAX exposes it on jax.core only
    from jax.core import new_jaxpr_eqn

try:
    from jax._src.ad_util import stop_gradient_p
except ImportError:  # future JAX relocation: recover the primitive by tracing
    import jax.numpy as _jnp

    stop_gradient_p = jax.make_jaxpr(jax.lax.stop_gradient)(
        _jnp.zeros((1,))
    ).jaxpr.eqns[0].primitive
    assert stop_gradient_p.name == 'stop_gradient'


def new_var(suffix, aval):
    if jax.__version_info__ < (0, 6, 2):
        return Var(suffix, aval)
    else:
        return Var(aval)


def is_jit_primitive(eqn: JaxprEqn) -> bool:
    if jax.__version_info__ < (0, 7, 0):
        return eqn.primitive.name == 'pjit'
    else:
        return eqn.primitive.name == 'jit'


def is_scan_primitive(eqn: JaxprEqn) -> bool:
    return eqn.primitive.name == 'scan'


def is_while_primitive(eqn: JaxprEqn) -> bool:
    return eqn.primitive.name == 'while'


def is_cond_primitive(eqn: JaxprEqn) -> bool:
    return eqn.primitive.name == 'cond'
