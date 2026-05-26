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

"""OSTL — Online Spatio-Temporal Learning (Bohnstingl et al. 2023).

The paper defines two regimes that differ in whether the recurrent (hidden-to-
hidden) Jacobian ``H`` is retained. Each regime is its own class so that it
inherits the full, separately-tested machinery of an existing VJP algorithm —
``compile_graph``, ``update``, ``reset_state``, ``get_etrace_of`` — without
branching:

- :class:`OSTLRecurrent` ('with-H') keeps ``H`` and is RTRL-exact for a single
  recurrent layer (per-parameter D-RTRL trace, O(P·H)).
- :class:`OSTLFeedforward` ('without-H') drops ``H``; the temporal term vanishes
  and the update reduces to pp_prop with negligible decay (feedforward SNN).

:func:`OSTL` is a thin factory selecting between the two by ``regime`` keyword,
preserving the historical ``OSTL(model, regime=...)`` call site.
"""

from typing import Optional

import brainstate

from .d_rtrl import ParamDimVjpAlgorithm
from .pp_prop import IODimVjpAlgorithm

__all__ = ['OSTL', 'OSTLRecurrent', 'OSTLFeedforward']


class OSTLRecurrent(ParamDimVjpAlgorithm):
    """OSTL 'with-H' regime — RTRL-exact single-layer factorization.

    Retains the hidden-to-hidden Jacobian, so the eligibility trace carries the
    full temporal term ``ε^t = D^t·ε^{t-1} + diag(D_f^t)⊗x^t``. Exact for a
    single recurrent layer; per-parameter trace with O(P·H) memory. Delegates
    entirely to :class:`ParamDimVjpAlgorithm` (D-RTRL).

    Parameters
    ----------
    model : brainstate.nn.Module
        The recurrent SNN whose weights are trained online.
    name, vjp_method, fast_solve, normalize_matrix_spectrum : optional
        Forwarded verbatim to :class:`ParamDimVjpAlgorithm`.
    """

    __module__ = 'braintrace'

    #: Identifies the OSTL regime this class implements.
    regime = 'with-H'


class OSTLFeedforward(IODimVjpAlgorithm):
    """OSTL 'without-H' regime — feedforward / no recurrent Jacobian.

    Drops the hidden-to-hidden Jacobian. With a negligible decay the input-
    output factorized trace stops accumulating across time, so the update is the
    purely-spatial (feedforward SNN) approximation. Delegates to
    :class:`IODimVjpAlgorithm` (pp_prop).

    Parameters
    ----------
    model : brainstate.nn.Module
        The SNN whose weights are trained online.
    decay_or_rank : float or int, default 1e-6
        Exponential-smoothing factor of the IO-dim trace. The tiny default makes
        the temporal contribution negligible, matching the 'without-H' regime. A
        float must lie in (0, 1); an int is read as an approximation rank.
    name, vjp_method, fast_solve : optional
        Forwarded verbatim to :class:`IODimVjpAlgorithm`.
    """

    __module__ = 'braintrace'

    #: Identifies the OSTL regime this class implements.
    regime = 'without-H'

    def __init__(
        self,
        model: brainstate.nn.Module,
        decay_or_rank: float | int = 1e-6,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, decay_or_rank=decay_or_rank, name=name, **kwargs)


def OSTL(
    model: brainstate.nn.Module,
    regime: str = 'with-H',
    name: Optional[str] = None,
    **kwargs,
):
    """Construct the OSTL algorithm for the selected regime.

    A convenience dispatcher over :class:`OSTLRecurrent` ('with-H') and
    :class:`OSTLFeedforward` ('without-H'). Prefer constructing those classes
    directly when the regime is known at the call site.

    Parameters
    ----------
    model : brainstate.nn.Module
    regime : {'with-H', 'without-H'}, default 'with-H'
        'with-H' keeps the recurrent Jacobian (RTRL-exact, per-parameter trace).
        'without-H' drops it (feedforward SNN, IO-dim trace with tiny decay).
    name : optional name forwarded to the chosen class.
    **kwargs : forwarded to the chosen class constructor.

    Returns
    -------
    OSTLRecurrent or OSTLFeedforward
    """
    if regime == 'with-H':
        return OSTLRecurrent(model, name=name, **kwargs)
    if regime == 'without-H':
        return OSTLFeedforward(model, name=name, **kwargs)
    raise ValueError(f"regime must be 'with-H' or 'without-H'; got {regime!r}")
