# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""
ETP-based online learning algorithms.

Provides ``ETP_DRTRL`` — the D-RTRL algorithm implemented on top of the
ETP primitive system.  The main entry point is:

    algo = bt.etp.ETP_DRTRL(model)
    algo.compile(sample_input)

    for x_t, y_t in sequence:
        out = algo(x_t)
        loss = loss_fn(out, y_t)
        grads = jax.grad(loss_fn)(...)   # uses eligibility traces
"""

from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

from braintrace._etrace_algorithms import EligibilityTrace
from braintrace._etrace_compiler.hidden_group import HiddenGroup
from braintrace._etrace_input_data import has_multistep_data
from braintrace._misc import etrace_param_key, etrace_df_key
from braintrace._state_managment import assign_state_values_v2
from braintrace._typing import (
    PyTree,
    Path,
    WeightVals,
    HiddenVals,
    StateVals,
    ETraceVals,
    ETraceWG_Key,
    ETraceX_Key,
    ETraceDF_Key,
    Hid2WeightJacobian,
    HiddenGroupJacobian,
    dG_Weight,
)
from ._compiler import ETPOpRelation
from ._executor import ETPGraphExecutor
from ._graph import ETPGraph
from ._primitives import (
    etp_elemwise_p,
    etp_rules_yw_to_w,
    etp_rules_xy_to_dw,
)

__all__ = [
    'ETP_DRTRL',
]


# ---------------------------------------------------------------------------
# Trace initialisation helpers
# ---------------------------------------------------------------------------

def _batched_zeros_like(batch_size, n_state, x):
    """Create zeros with shape ``(*batch?, *x.shape, n_state)``."""
    if batch_size is not None:
        return jnp.zeros((batch_size, *x.shape, n_state), dtype=x.dtype)
    return jnp.zeros((*x.shape, n_state), dtype=x.dtype)


def _sum_dim(x, axis=-1):
    return jnp.sum(x, axis=axis)


def _update_dict(d, key, val):
    if d.get(key) is None:
        d[key] = val
    else:
        d[key] = jax.tree.map(jnp.add, d[key], val)


# ---------------------------------------------------------------------------
# ETP_DRTRL
# ---------------------------------------------------------------------------

class ETP_DRTRL(brainstate.nn.Module):
    r"""
    D-RTRL algorithm using the ETP primitive system.

    This is the ETP equivalent of ``ParamDimVjpAlgorithm`` / ``D_RTRL``.

    Parameters
    ----------
    model : brainstate.nn.Module
        The recurrent model (single-step).
    mode : brainstate.mixin.Mode, optional
        Computing mode (batching, etc.).
    """
    __module__ = 'braintrace.etp'

    def __init__(
        self,
        model: brainstate.nn.Module,
        mode: Optional[brainstate.mixin.Mode] = None,
    ):
        super().__init__()
        self.model4compile = model
        self.graph_executor = ETPGraphExecutor(model)

        if mode is None:
            self.mode = brainstate.environ.get('mode', brainstate.mixin.Mode())
        else:
            self.mode = mode

        self.is_compiled = False
        self.running_index = brainstate.LongTermState(0)

        # Caches (populated by compile)
        self._param_states = None
        self._hidden_states = None
        self._other_states = None
        self.etrace_bwg: Dict[ETraceWG_Key, brainstate.State] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def graph(self) -> ETPGraph:
        return self.graph_executor.graph

    @property
    def param_states(self):
        if self._param_states is None:
            self._split_states()
        return self._param_states

    @property
    def hidden_states(self):
        if self._hidden_states is None:
            self._split_states()
        return self._hidden_states

    @property
    def other_states(self):
        if self._other_states is None:
            self._split_states()
        return self._other_states

    def _split_states(self):
        (
            self._param_states,
            self._hidden_states,
            self._other_states,
        ) = self.graph.module_info.retrieved_model_states.split(
            brainstate.ParamState, brainstate.HiddenState, ...
        )

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile(self, *args):
        """Compile the ETP graph and initialise trace states."""
        if not self.is_compiled:
            self._param_states = None
            self._hidden_states = None
            self._other_states = None

            self.graph_executor.compile_graph(*args)
            self._init_traces()
            self.is_compiled = True

    def _init_traces(self):
        """Initialise eligibility-trace states to zero."""
        y_shape_cache = {}
        batch_size = None
        for rel in self.graph.etp_op_relations:
            y_shape = rel.y_var.aval.shape
            if self.mode.has(brainstate.mixin.Batching):
                batch_size = y_shape[0]

            for group in rel.hidden_groups:
                key = etrace_param_key(rel.weight_path, rel.y_var, group.index)
                if key in self.etrace_bwg:
                    raise ValueError(f'Duplicate trace key: {key}')

                # Trace shape = weight_shape + (n_states,)
                w_val = rel.weight.value
                self.etrace_bwg[key] = EligibilityTrace(
                    jax.tree.map(
                        partial(_batched_zeros_like, batch_size, group.num_state),
                        w_val,
                    )
                )

    def reset_state(self, batch_size=None, **kwargs):
        self.running_index.value = 0
        for st in self.etrace_bwg.values():
            st.value = jax.tree.map(jnp.zeros_like, st.value)

    # ------------------------------------------------------------------
    # Forward + trace update
    # ------------------------------------------------------------------

    def __call__(self, *args):
        return self.update(*args)

    def update(self, *args):
        """
        Run one step: forward pass + eligibility-trace update.

        This is wrapped in ``custom_vjp`` so that ``jax.grad`` on the
        loss automatically uses eligibility traces for parameter
        gradients.
        """
        if not self.is_compiled:
            raise RuntimeError('Call compile() first.')

        # Collect current state values
        weight_vals = {k: st.value for k, st in self.param_states.items()}
        hidden_vals = {k: st.value for k, st in self.hidden_states.items()}
        other_vals = {k: st.value for k, st in self.other_states.items()}
        etrace_vals = {k: v.value for k, v in self.etrace_bwg.items()}

        # Forward + trace update (via custom_vjp)
        (
            out,
            new_hidden_vals,
            new_other_vals,
            new_etrace_vals,
        ) = self._true_update(
            args,
            weight_vals,
            hidden_vals,
            other_vals,
            etrace_vals,
            self.running_index.value,
        )

        # Write back
        assign_state_values_v2(self.param_states, weight_vals, write=False)
        assign_state_values_v2(self.hidden_states, new_hidden_vals)
        assign_state_values_v2(self.other_states, new_other_vals)
        for k, v in new_etrace_vals.items():
            self.etrace_bwg[k].value = v

        idx = self.running_index.value + 1
        self.running_index.value = jax.lax.stop_gradient(
            jnp.where(idx >= 0, idx, 0)
        )

        return out

    def _true_update(
        self,
        args,
        weight_vals,
        hidden_vals,
        other_vals,
        etrace_vals,
        running_index,
    ):
        """Forward pass + trace update (not yet wrapped in custom_vjp)."""

        # Assign states
        assign_state_values_v2(self.param_states, weight_vals, write=False)
        assign_state_values_v2(self.hidden_states, hidden_vals, write=False)
        assign_state_values_v2(self.other_states, other_vals, write=False)

        # Forward + Jacobians
        (
            out,
            new_hidden_vals,
            new_other_vals,
            h2w_jac,
            h2h_jacs,
        ) = self.graph_executor.solve_h2w_h2h_jacobian(*args)

        # Update traces
        new_etrace_vals = self._update_traces(
            etrace_vals, h2w_jac, h2h_jacs, weight_vals,
        )

        return out, new_hidden_vals, new_other_vals, new_etrace_vals

    def _update_traces(
        self,
        old_traces: Dict[ETraceWG_Key, PyTree],
        h2w_jac: Hid2WeightJacobian,
        h2h_jacs: HiddenGroupJacobian,
        weight_vals: Dict[Path, PyTree],
    ) -> Dict[ETraceWG_Key, PyTree]:
        """
        Core D-RTRL trace update::

            ε^t = J_hh ⊙ ε^{t-1} + ∂h^t/∂θ
        """
        etrace_xs, etrace_dfs = h2w_jac
        new_traces = {}

        for rel in self.graph.etp_op_relations:
            yw_to_w = etp_rules_yw_to_w[rel.primitive]
            w_val = weight_vals[rel.weight_path]

            # x value for this op
            if rel.x_var is not None:
                x_val = etrace_xs.get(id(rel.x_var))
            else:
                x_val = None

            for group in rel.hidden_groups:
                key = etrace_param_key(rel.weight_path, rel.y_var, group.index)
                df = etrace_dfs[etrace_df_key(rel.y_var, group.index)]
                h2h = h2h_jacs[group.index]
                old_e = old_traces[key]

                # --- Step 1: ∂h^t/∂θ^t via VJP of the primitive ---
                phg_to_pw = self._compute_dh_dtheta(
                    rel, x_val, df, w_val,
                )

                # --- Step 2: J_hh @ old_trace ---
                yw_kw = {k: v for k, v in rel.eqn_params.items() if k != 'has_bias'}
                new_e_pre = self._apply_h2h(
                    yw_to_w, h2h, old_e, group, **yw_kw,
                )

                # --- Step 3: ε^t = pre + dh/dθ ---
                new_e = jax.tree.map(jnp.add, new_e_pre, phg_to_pw)
                new_traces[key] = new_e

        return new_traces

    def _compute_dh_dtheta(
        self,
        rel: ETPOpRelation,
        x_val,
        df,
        w_val,
    ):
        """
        Compute ∂h/∂θ at the current step using the registered
        ``xy_to_dw`` rule.

        ``df`` has shape ``(hidden_concat, ..., n_state)``.
        """
        xy_to_dw = etp_rules_xy_to_dw[rel.primitive]
        params = {k: v for k, v in rel.eqn_params.items() if k != 'has_bias'}

        @partial(jax.vmap, in_axes=-1, out_axes=-1)
        def _over_states(df_col):
            return xy_to_dw(x_val, u.get_mantissa(df_col), w_val, **params)

        return _over_states(df)

    def _apply_h2h(self, yw_to_w, h2h, old_trace, group, **yw_kw):
        """
        Apply the hidden-to-hidden Jacobian to the old trace::

            ε^t_pre[w, i] = Σ_j J_hh[i, j] · yw_to_w(J_hh[i, j], old_e[w, j])
        """
        # vmap over j (source hidden states)
        fn_over_j = lambda d: _sum_dim(
            jax.vmap(
                lambda d_, t_: yw_to_w(d_, t_, **yw_kw),
                in_axes=-1,
                out_axes=-1,
            )(d, old_trace),
            axis=-1,
        )
        # vmap over i (target hidden states)
        result = jax.vmap(fn_over_j, in_axes=-2, out_axes=-1)(h2h)
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def show_graph(self):
        self.graph_executor.show_graph()

    def get_etrace_of(self, weight):
        """Get eligibility traces for a given weight."""
        if not self.is_compiled:
            raise RuntimeError('Not compiled.')

        weight_id = (
            id(weight)
            if isinstance(weight, brainstate.ParamState)
            else id(self.graph_executor.path_to_states[weight])
        )
        result = {}
        for rel in self.graph.etp_op_relations:
            if id(rel.weight) != weight_id:
                continue
            for group in rel.hidden_groups:
                key = etrace_param_key(rel.weight_path, rel.y_var, group.index)
                result[key] = self.etrace_bwg[key].value
        if not result:
            raise ValueError(f'No traces found for weight: {weight}')
        return result
