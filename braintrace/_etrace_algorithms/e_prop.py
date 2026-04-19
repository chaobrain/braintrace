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

"""E-Prop — Eligibility Propagation (Bellec et al. 2020).

``D_RTRL``'s per-parameter trace plus:
- An optional κ-filter on each HiddenGroup's eligibility signal (ē = F_κ(L))
  matching the paper's readout-side low-pass.
- An optional random-feedback variant (feedback='random') that replaces the
  readout's symmetric gradient with a fixed random projection.
"""

from typing import Dict, Optional

import brainstate
import jax
import jax.numpy as jnp

from .d_rtrl import ParamDimVjpAlgorithm
from ._common import FixedRandomFeedback, KappaFilter


__all__ = ['EProp']


class EProp(ParamDimVjpAlgorithm):
    """Eligibility Propagation.

    Parameters
    ----------
    model : brainstate.nn.Module
    feedback : {'symmetric', 'random'}
        'symmetric' uses reverse-AD's ∂L/∂h (standard backprop through readout).
        'random' replaces the readout gradient with a frozen random projection.
    kappa_filter_decay : float in [0, 1)
        If > 0, apply an output-side low-pass to each HiddenGroup's learning
        signal each step. 0 disables (paper default for hard tasks).
    random_feedback_key : jax.random.PRNGKey, optional
        Seed for the random-feedback matrices when feedback='random'.
    name, vjp_method, fast_solve, normalize_matrix_spectrum : forwarded to D_RTRL.
    """

    __module__ = 'braintrace'

    def __init__(
        self,
        model: brainstate.nn.Module,
        feedback: str = 'symmetric',
        kappa_filter_decay: float = 0.0,
        random_feedback_key: Optional[jax.Array] = None,
        name: Optional[str] = None,
        vjp_method: str = 'single-step',
        fast_solve: bool = True,
        normalize_matrix_spectrum: bool = False,
        **kwargs,
    ):
        if feedback not in ('symmetric', 'random'):
            raise ValueError(
                f"feedback must be 'symmetric' or 'random'; got {feedback!r}"
            )
        if feedback == 'random' and random_feedback_key is None:
            raise ValueError(
                "feedback='random' requires random_feedback_key=<PRNGKey>"
            )
        super().__init__(
            model,
            name=name,
            vjp_method=vjp_method,
            fast_solve=fast_solve,
            normalize_matrix_spectrum=normalize_matrix_spectrum,
            **kwargs,
        )
        self.feedback = feedback
        self.kappa_filter_decay = float(kappa_filter_decay)
        self._random_feedback_key = random_feedback_key
        self._kappa_filters: Dict[int, KappaFilter] = {}
        self._random_feedback: Dict[int, FixedRandomFeedback] = {}

    def init_etrace_state(self, *args, **kwargs):
        super().init_etrace_state(*args, **kwargs)
        self._kappa_filters = {}
        self._random_feedback = {}
        if self.kappa_filter_decay > 0.0:
            for rel in self.graph.hidden_param_op_relations:
                for group in rel.hidden_groups:
                    gid = group.index
                    if gid in self._kappa_filters:
                        continue
                    zeros = jnp.zeros(group.varshape, dtype=jnp.float32)
                    self._kappa_filters[gid] = KappaFilter(
                        zeros, self.kappa_filter_decay
                    )
        if self.feedback == 'random':
            key = self._random_feedback_key
            for rel in self.graph.hidden_param_op_relations:
                for group in rel.hidden_groups:
                    gid = group.index
                    if gid in self._random_feedback:
                        continue
                    key, sub = jax.random.split(key)
                    n_layer = int(group.varshape[-1])
                    # n_target == n_layer — square projection over reverse-AD signal.
                    self._random_feedback[gid] = FixedRandomFeedback(
                        n_target=n_layer, n_layer=n_layer, key=sub, init_scale=0.1
                    )

    def reset_state(self, batch_size: Optional[int] = None, **kwargs):
        super().reset_state(batch_size=batch_size, **kwargs)
        for flt in self._kappa_filters.values():
            flt.reset_state(batch_size=batch_size)

    def _compute_learning_signal(self, dl_autodiff, args):
        signals = list(dl_autodiff)
        if self.feedback == 'random' and self._random_feedback:
            # dl_autodiff[g].shape == (*varshape, num_state). Project over the
            # trailing n_layer axis (-2), preserving num_state on the tail.
            def _project(B, s):
                return jnp.einsum('...lj,lk->...kj', s, B)

            signals = [
                _project(self._random_feedback[gid].B, s)
                if gid in self._random_feedback else s
                for gid, s in enumerate(signals)
            ]
        if self._kappa_filters:
            # KappaFilter state carries varshape, but signal has an extra
            # trailing num_state axis. Collapse num_state for filter purposes;
            # broadcast the filtered value back.
            def _filter(flt, s):
                # collapse num_state tail: sum over last axis produces shape (*varshape,)
                collapsed = s.sum(axis=-1)
                filtered = flt.update(collapsed)
                return jnp.expand_dims(filtered, axis=-1).astype(s.dtype)

            signals = [
                _filter(self._kappa_filters[gid], s)
                if gid in self._kappa_filters else s
                for gid, s in enumerate(signals)
            ]
        return signals
