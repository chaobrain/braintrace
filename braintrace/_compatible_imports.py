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
    'wrap_init',
]

from brainstate._compatible_import import Primitive, Var, JaxprEqn, Jaxpr, ClosedJaxpr, Literal, wrap_init

try:
    from jax.extend.core import new_jaxpr_eqn
except ImportError:  # older JAX exposes it on jax.core only
    from jax.core import new_jaxpr_eqn

try:
    from jax._src.ad_util import stop_gradient_p
except ImportError:  # future JAX relocation: recover the primitive by tracing
    import jax.numpy as _jnp

    _probe_eqns = jax.make_jaxpr(jax.lax.stop_gradient)(_jnp.zeros((1,))).jaxpr.eqns
    if len(_probe_eqns) != 1 or _probe_eqns[0].primitive.name != 'stop_gradient':
        raise ImportError(
            'Could not locate the stop_gradient primitive: jax._src.ad_util no '
            'longer exposes stop_gradient_p and tracing jax.lax.stop_gradient '
            f'produced {[e.primitive.name for e in _probe_eqns]} instead of a '
            'single stop_gradient equation. Update braintrace._compatible_imports '
            'for this JAX version.'
        )
    stop_gradient_p = _probe_eqns[0].primitive


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
