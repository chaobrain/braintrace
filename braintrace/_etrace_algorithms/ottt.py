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

"""OTTT — Online Training Through Time (Xiao et al. 2022).

Drops both the hidden-to-hidden and hidden-to-weight Jacobians; maintains only a
presynaptic eligibility trace ``â ← λ·â + x_t`` and computes weight gradients as
``ΔW = outer(â, L · σ'(u))`` per step.
"""

from typing import Dict, Optional

import brainstate
import jax.numpy as jnp

from braintrace._etrace_op import is_batched_primitive
from ._common import PresynapticTrace, _resolve_leak
from .misc import _route_grads_by_path, _update_dict
from .vjp_base import ETraceVjpAlgorithm

__all__ = ['OTTT']


class OTTT(ETraceVjpAlgorithm):
    """Online Training Through Time.

    Parameters
    ----------
    model : brainstate.nn.Module
    mode : {'A', 'O'}
        'A' (default) accumulates â over time (â ← λ·â + x). 'O' uses the
        instantaneous presynaptic spike only (â := x_t).
    leak : float, optional
        Presynaptic leak λ. If None, discovered from the model via
        ``_resolve_leak``.
    name, vjp_method : forwarded to base.
    """

    __module__ = 'braintrace'

    def __init__(
        self,
        model: brainstate.nn.Module,
        mode: str = 'A',
        leak: Optional[float] = None,
        name: Optional[str] = None,
        vjp_method: str = 'single-step',
        **kwargs,
    ):
        if mode not in ('A', 'O'):
            raise ValueError(f"mode must be 'A' or 'O'; got {mode!r}")
        super().__init__(model, name=name, vjp_method=vjp_method)
        self.mode = mode
        self.leak = _resolve_leak(model, leak)
        self._pre_traces: Dict[int, PresynapticTrace] = {}

    def init_etrace_state(self, *args, **kwargs):
        self._pre_traces = {}
        for rel in self.graph.hidden_param_op_relations:
            rid = id(rel.x_var)
            if rid in self._pre_traces:
                continue
            shape = rel.x_var.aval.shape
            dtype = rel.x_var.aval.dtype
            self._pre_traces[rid] = PresynapticTrace(
                jnp.zeros(shape, dtype=dtype), leak=self.leak
            )

    def reset_state(self, batch_size: Optional[int] = None, **kwargs):
        self.running_index.value = 0
        for t in self._pre_traces.values():
            t.reset_state(batch_size=batch_size)

    def _get_etrace_data(self):
        return {rid: t.value for rid, t in self._pre_traces.items()}

    def _assign_etrace_data(self, vals):
        for rid, v in vals.items():
            self._pre_traces[rid].value = v

    def _update_etrace_data(
        self, running_index, hist_vals,
        hid2weight_jac, hid2hid_jac, weight_vals, input_is_multi_step,
    ):
        """``â ← λ·â + x_t`` (mode='A') or ``â := x_t`` (mode='O').

        Ignores ``hid2hid_jac`` — OTTT's core approximation.
        """
        if input_is_multi_step:
            raise NotImplementedError('OTTT v1 supports single-step only')
        xs_at_t = hid2weight_jac[0]

        new_vals = {}
        for rid, old in hist_vals.items():
            x_t = xs_at_t[rid]
            if self.mode == 'A':
                new_vals[rid] = self.leak * old + x_t
            else:
                new_vals[rid] = x_t
        return new_vals

    def _solve_weight_gradients(
        self, running_index, etrace_at_t, dl_to_hidden_groups,
        weight_vals, dl_to_nonetws_at_t, dl_to_etws_at_t,
    ):
        """``ΔW = outer(â, L)`` where ``L`` is the (already σ'-propagated) signal."""
        dG = {path: None for path in self.param_states}
        for rel in self.graph.hidden_param_op_relations:
            a_hat = etrace_at_t[id(rel.x_var)]
            for group in rel.hidden_groups:
                L = dl_to_hidden_groups[group.index]
                # L shape = (*varshape, num_state); collapse num_state tail
                L_proj = L.sum(axis=-1)
                if is_batched_primitive(rel.primitive):
                    # a_hat: (batch, in), L_proj: (batch, out). ΔW: (in, out)
                    dw = jnp.einsum('bi,bo->io', a_hat, L_proj)
                else:
                    dw = jnp.einsum('i,o->io', a_hat, L_proj)
                _route_grads_by_path(rel, {'weight': dw}, weight_vals, dG)

        for path, dg in dl_to_nonetws_at_t.items():
            _update_dict(dG, path, dg)
        if dl_to_etws_at_t is not None:
            for path, dg in dl_to_etws_at_t.items():
                _update_dict(dG, path, dg, error_when_no_key=True)
        return dG
