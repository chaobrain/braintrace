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

Regime 'with-H'    — RTRL-exact single-layer factorization (delegates to D_RTRL).
Regime 'without-H' — feedforward / no recurrent Jacobian (delegates to pp_prop with decay≈0).
"""

from typing import Optional

import brainstate

from .d_rtrl import ParamDimVjpAlgorithm
from .pp_prop import IODimVjpAlgorithm


__all__ = ['OSTL']


def OSTL(
    model: brainstate.nn.Module,
    regime: str = 'with-H',
    name: Optional[str] = None,
    **kwargs,
):
    """Factory returning the appropriate VJP algorithm for the selected regime.

    Using a factory (not a subclass with branching) lets each regime inherit
    everything — compile_graph, update, reset_state, get_etrace_of — from the
    existing tested algorithm classes without duplication.

    Parameters
    ----------
    model : brainstate.nn.Module
    regime : {'with-H', 'without-H'}
        'with-H' uses D_RTRL-shape traces (per-parameter, O(P·H)). Exact for
        single-recurrent-layer networks. 'without-H' drops the temporal term,
        equivalent to pp_prop with negligible decay (feedforward SNN).
    name : optional name.
    **kwargs : forwarded to the base algorithm constructor.
    """
    if regime not in ('with-H', 'without-H'):
        raise ValueError(f"regime must be 'with-H' or 'without-H'; got {regime!r}")

    if regime == 'with-H':
        algo = ParamDimVjpAlgorithm(model, name=name, **kwargs)
    else:
        decay = kwargs.pop('decay_or_rank', 1e-6)
        algo = IODimVjpAlgorithm(model, decay_or_rank=decay, name=name, **kwargs)

    algo.regime = regime
    return algo
