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

"""OSTTP — Online Spatio-Temporal Target Projection (Ortner et al. 2023).

D_RTRL trace machinery + DRTP-style target projection replacing reverse-AD's
``∂L/∂h``. Each HiddenGroup receives a signal ``B_l @ y_target`` instead of the
autodiff gradient.
"""

from typing import Optional, Sequence

import brainstate
import jax
import jax.numpy as jnp

from .d_rtrl import ParamDimVjpAlgorithm

__all__ = ['OSTTP']


class OSTTP(ParamDimVjpAlgorithm):
    """Online Spatio-Temporal Target Projection.

    Parameters
    ----------
    model : brainstate.nn.Module
    B_list : Sequence[jax.Array]
        One feedback matrix per HiddenGroup, each of shape ``(n_target, n_l)``.
        Frozen via ``stop_gradient`` at construction.
    target_timing : {'per-step', 'sequence-end'}
        'per-step' requires ``y_target`` at every ``update()`` call.
        'sequence-end' zeros the signal on intermediate steps and only applies
        the projection when ``y_target`` is supplied.
    name, vjp_method, fast_solve : forwarded to ``ParamDimVjpAlgorithm``.
    """

    __module__ = 'braintrace'

    def __init__(
        self,
        model: brainstate.nn.Module,
        B_list: Sequence[jax.Array],
        target_timing: str = 'per-step',
        name: Optional[str] = None,
        vjp_method: str = 'single-step',
        fast_solve: bool = True,
        **kwargs,
    ):
        if target_timing not in ('per-step', 'sequence-end'):
            raise ValueError(
                f"target_timing must be 'per-step' or 'sequence-end'; got {target_timing!r}"
            )
        super().__init__(
            model, name=name, vjp_method=vjp_method, fast_solve=fast_solve, **kwargs
        )
        self._B_list = tuple(jax.lax.stop_gradient(B) for B in B_list)
        self.target_timing = target_timing
        self._current_y_target: Optional[jax.Array] = None

    def compile_graph(self, *args) -> None:
        super().compile_graph(*args)
        n_groups = len(self.graph.hidden_groups)
        if len(self._B_list) != n_groups:
            raise ValueError(
                f'B_list has {len(self._B_list)} entries but model has {n_groups} '
                f'HiddenGroup(s). One B matrix per HiddenGroup is required.'
            )
        for B, group in zip(self._B_list, self.graph.hidden_groups):
            n_l = int(group.varshape[-1])
            if B.shape[1] != n_l:
                raise ValueError(
                    f'B_list[{group.index}].shape[1] == {B.shape[1]} but HiddenGroup '
                    f'{group.index} has n_l={n_l}.'
                )

    def update(self, x, y_target=None):
        """Call ``super().update(x)`` after stashing ``y_target`` for the hook."""
        if self.target_timing == 'per-step' and y_target is None:
            raise ValueError(
                "OSTTP(target_timing='per-step') requires y_target at every update() call."
            )
        self._current_y_target = y_target
        try:
            return super().update(x)
        finally:
            self._current_y_target = None

    def _compute_learning_signal(self, dl_autodiff, args):
        """Replace reverse-AD ``dL/dh`` with ``B_l @ y_target`` per HiddenGroup."""
        y_target = self._current_y_target
        if y_target is None:
            # target_timing='sequence-end' with no y_target: zero out so traces
            # accumulate without emitting a weight update this step.
            return [jnp.zeros_like(s) for s in dl_autodiff]
        out = []
        for gid, s in enumerate(dl_autodiff):
            B = self._B_list[gid]
            projected = y_target @ B  # (batch, n_l)
            # Reshape projected into the autodiff signal shape (which has a
            # trailing num_state axis appended by concat_hidden).
            # s shape == (*varshape, num_state); projected shape == (*varshape,)
            target_shape = s.shape
            expanded = projected.reshape(target_shape[:-1] + (1,))
            # Broadcast across the num_state tail.
            out.append(jnp.broadcast_to(expanded, target_shape).astype(s.dtype))
        return out
