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
    'scan_num_consts_carry',
    'scan_params_add_ys',
    'wrap_init',
]

from typing import Dict, Tuple

from brainstate._compatible_import import Primitive, Var, JaxprEqn, Jaxpr, ClosedJaxpr, Literal, wrap_init

try:
    from jax.extend.core import new_jaxpr_eqn
except ImportError:  # older JAX exposes it on jax.core only
    # jax.core dropped ``new_jaxpr_eqn`` in JAX 0.11; this fallback only runs on
    # older JAX, so silence mypy's static attr-defined/no-redef complaints.
    from jax.core import new_jaxpr_eqn  # type: ignore[attr-defined, no-redef]

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


def scan_num_consts_carry(eqn: JaxprEqn) -> Tuple[int, int]:
    """Return ``(num_consts, num_carry)`` for a ``scan`` equation.

    JAX < 0.11 stores ``num_consts`` / ``num_carry`` directly in the equation
    params. JAX 0.11 removed them (part of the "flattree" scan refactor) and
    instead encodes the ``(consts, carry, xs)`` input split in the ``ft_in``
    flattree; the counts are the leaf counts of its first two groups. Detection
    is capability-based (which params exist), so this works across every
    supported JAX version without a hard-coded version comparison.

    Parameters
    ----------
    eqn : JaxprEqn
        A ``scan`` equation (``is_scan_primitive(eqn)`` must hold).

    Returns
    -------
    tuple of int
        ``(num_consts, num_carry)``.
    """
    params = eqn.params
    if 'num_consts' in params:  # JAX < 0.11
        return params['num_consts'], params['num_carry']
    consts, carry, _xs = params['ft_in'].unpack()  # JAX >= 0.11
    return len(consts), len(carry)


def scan_params_add_ys(params: Dict, n_extra: int) -> Dict:
    """Return ``scan`` params describing ``n_extra`` extra trailing ``ys`` outputs.

    Used when rebuilding a scan whose body gains extra stacked ``ys`` outvars.

    On JAX < 0.11 the number of ``ys`` is implicit (``len(outvars) - num_carry``),
    so appending outvars needs no param change and ``params`` is returned
    unchanged. On JAX 0.11 the ``ft_out`` flattree — which records the
    ``(carry, ys)`` output split — must grow by ``n_extra`` leaves in its ``ys``
    (second) component; the extra leaves are appended *after* the original ys so
    the flattree leaf order matches ``[*carry, *original_ys, *extra]``.

    Parameters
    ----------
    params : dict
        The params of a ``scan`` equation (typically ``{**eqn.params, ...}``).
    n_extra : int
        Number of extra trailing ``ys`` leaves to describe.

    Returns
    -------
    dict
        Params updated for the extra ``ys`` (the same object when no change is
        needed).
    """
    if n_extra == 0 or 'ft_out' not in params:  # JAX < 0.11, or nothing to add
        return params
    from jax._src import flattree as _ft  # only reached on JAX >= 0.11
    carry_ft, ys_ft = params['ft_out'].unpack()
    new_ft_out = _ft.pack((carry_ft, _ft.pack((ys_ft, _ft.nones(n_extra)))))
    return {**params, 'ft_out': new_ft_out}
