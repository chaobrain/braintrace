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
    'jaxpr_all_invars',
    'split_jaxpr_invars',
    'jaxpr_constvars',
]

from typing import Any, Dict, List, Tuple

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


def new_var(suffix: Any, aval: Any) -> Var:
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


def jaxpr_all_invars(jaxpr: Jaxpr) -> List[Var]:
    """Return every positional input of *jaxpr*, constvars first.

    This is the argument order :func:`jax.core.eval_jaxpr` expects: it writes
    ``[*consts, *args]`` against exactly this list.

    JAX 0.11 merged ``ClosedJaxpr`` into ``Jaxpr`` and now derives the
    ``constvars`` / ``invars`` boundary from how many const *values* are
    attached, rather than storing it with the symbols. An open jaxpr built with
    symbolic constvars and no attached values therefore reports
    ``constvars == []`` and ``invars == [*constvars, *invars]``. Their
    concatenation is stable across every supported JAX version, which is why
    every split in this module is derived from it.

    Parameters
    ----------
    jaxpr : Jaxpr
        The jaxpr to inspect.

    Returns
    -------
    list of Var
        The concatenation ``[*jaxpr.constvars, *jaxpr.invars]``.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from braintrace._compatible_imports import jaxpr_all_invars
        >>> jaxpr = jax.make_jaxpr(lambda x, y: x * y)(jnp.ones(3), jnp.ones(3))
        >>> len(jaxpr_all_invars(jaxpr.jaxpr))
        2
    """
    return [*jaxpr.constvars, *jaxpr.invars]


def split_jaxpr_invars(jaxpr: Jaxpr, num_invars: int) -> Tuple[List[Var], List[Var]]:
    """Split *jaxpr*'s positional inputs into ``(constvars, invars)``.

    The compiler builds transition jaxprs whose ``invars`` are the
    differentiated inputs and whose ``constvars`` are surrounding intermediates
    bound at execution time. Since JAX 0.11 that boundary cannot be read back
    off the jaxpr (see :func:`jaxpr_all_invars`), so it is recovered from the
    invar count, which every caller already knows -- the invars are what it is
    about to feed in.

    Parameters
    ----------
    jaxpr : Jaxpr
        The jaxpr to split.
    num_invars : int
        The number of trailing positional inputs that are ``invars``. The
        leading remainder are the ``constvars``.

    Returns
    -------
    tuple of (list of Var, list of Var)
        ``(constvars, invars)``.

    Raises
    ------
    ValueError
        If *num_invars* is negative or exceeds the number of positional inputs.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from braintrace._compatible_imports import split_jaxpr_invars
        >>> jaxpr = jax.make_jaxpr(lambda x, y: x * y)(jnp.ones(3), jnp.ones(3))
        >>> constvars, invars = split_jaxpr_invars(jaxpr.jaxpr, 1)
        >>> len(constvars), len(invars)
        (1, 1)
    """
    all_invars = jaxpr_all_invars(jaxpr)
    if not 0 <= num_invars <= len(all_invars):
        raise ValueError(
            f'Cannot split a jaxpr with {len(all_invars)} positional inputs at '
            f'num_invars={num_invars}. This means the caller and the jaxpr '
            f'disagree about the transition signature; evaluating it would '
            f'misalign the eval_jaxpr argument list.'
        )
    boundary = len(all_invars) - num_invars
    return all_invars[:boundary], all_invars[boundary:]


def jaxpr_constvars(jaxpr: Jaxpr, num_invars: int) -> List[Var]:
    """Return only the ``constvars`` of *jaxpr*, given its invar count.

    Thin wrapper over :func:`split_jaxpr_invars` for the common case where the
    caller needs the const values to pass to :func:`jax.core.eval_jaxpr` and
    already holds the invars.

    Parameters
    ----------
    jaxpr : Jaxpr
        The jaxpr to inspect.
    num_invars : int
        The number of trailing positional inputs that are ``invars``.

    Returns
    -------
    list of Var
        The leading ``constvars``.

    Raises
    ------
    ValueError
        If *num_invars* is negative or exceeds the number of positional inputs.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from braintrace._compatible_imports import jaxpr_constvars
        >>> jaxpr = jax.make_jaxpr(lambda x, y: x * y)(jnp.ones(3), jnp.ones(3))
        >>> len(jaxpr_constvars(jaxpr.jaxpr, 1))
        1
    """
    constvars, _ = split_jaxpr_invars(jaxpr, num_invars)
    return constvars
