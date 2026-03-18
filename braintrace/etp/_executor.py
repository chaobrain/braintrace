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
ETP Graph Executor.

Runs the compiled ETP graph at each time step, producing:
  * model outputs
  * h2w Jacobians (structured, per-relation)
  * h2h Jacobians (per hidden group)
"""

from typing import Any, Dict, Sequence, Tuple

import brainstate
import brainunit as u
import jax
import jax.core
import jax.numpy as jnp

from braintrace._compatible_imports import Var
from braintrace._etrace_compiler.hidden_group import HiddenGroup
from braintrace._etrace_input_data import get_single_step_data
from braintrace._misc import etrace_df_key, etrace_x_key
from braintrace._state_managment import (
    assign_dict_state_values,
    split_dict_states_v2,
)
from braintrace._typing import (
    ETraceX_Key,
    ETraceDF_Key,
    Hid2WeightJacobian,
    HiddenGroupJacobian,
    Path,
)
from ._compiler import ETPOpRelation
from ._graph import ETPGraph, compile_etp_graph

__all__ = [
    'ETPGraphExecutor',
]


class ETPGraphExecutor:
    r"""
    Executor for the compiled ETP graph.

    Mirrors ``ETraceGraphExecutor`` / ``ETraceVjpGraphExecutor`` but
    works with ``ETPGraph`` (primitive-based compilation).

    Parameters
    ----------
    model : brainstate.nn.Module
        The recurrent model (single-step).
    """
    __module__ = 'braintrace.etp'

    def __init__(self, model: brainstate.nn.Module):
        if not isinstance(model, brainstate.nn.Module):
            raise TypeError(
                'model must be a brainstate.nn.Module, '
                f'got {type(model)}'
            )
        self.model = model
        self._compiled_graph: ETPGraph | None = None
        self._state_id_to_path: Dict[int, Path] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def graph(self) -> ETPGraph:
        if self._compiled_graph is None:
            raise RuntimeError('Call compile_graph() first.')
        return self._compiled_graph

    @property
    def path_to_states(self):
        return self.graph.module_info.retrieved_model_states

    @property
    def state_id_to_path(self) -> Dict[int, Path]:
        if self._state_id_to_path is None:
            self._state_id_to_path = {
                id(st): path
                for path, st in self.path_to_states.items()
            }
        return self._state_id_to_path

    @property
    def states(self):
        return self.graph.module_info.retrieved_model_states.split(
            brainstate.ParamState,
            brainstate.HiddenState,
            ...,
        )

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile_graph(self, *args) -> None:
        """Compile the ETP graph from example inputs."""
        args = get_single_step_data(*args)
        self._compiled_graph = compile_etp_graph(self.model, *args)
        self._state_id_to_path = None

    # ------------------------------------------------------------------
    # Jacobian computation
    # ------------------------------------------------------------------

    def solve_h2w_h2h_jacobian(
        self,
        *args,
    ) -> Tuple[
        Any,          # model outputs
        Dict,         # new hidden vals
        Dict,         # new other vals
        Hid2WeightJacobian,
        HiddenGroupJacobian,
    ]:
        """
        Run the model forward and compute h2w / h2h Jacobians.
        """
        graph = self.graph

        # --- Forward pass ---
        (
            out,
            new_hidden_vals,
            new_other_vals,
            temp_data,
        ) = graph.module_info(*args)

        # --- h2w Jacobians ---
        # Collect x values and df values for each (relation, group)
        etrace_xs: Dict[ETraceX_Key, jax.Array] = {}
        etrace_dfs: Dict[ETraceDF_Key, jax.Array] = {}

        for relation in graph.etp_op_relations:
            # x value
            if relation.x_var is not None:
                x_key = id(relation.x_var)
                if x_key not in etrace_xs:
                    etrace_xs[x_key] = temp_data[relation.x_var]

            # df values (transition: y -> hidden groups)
            y_val = temp_data[relation.y_var]
            const_vals = {v: temp_data[v] for v in temp_data}

            for jaxpr_y2h, group in zip(
                relation.y_to_hidden_group_jaxprs,
                relation.hidden_groups,
            ):
                # Evaluate: given y_val=1, what is df = ∂h/∂y?
                # Use JVP with tangent = identity-like
                consts = [const_vals.get(v, temp_data.get(v)) for v in jaxpr_y2h.constvars]

                # df: apply transition jaxpr's Jacobian
                # For simple cases (h = f(y)), df = f'(y)
                # We compute this via JVP: tangent_out = J @ tangent_in
                # with tangent_in = ones_like(y)
                def _eval_transition(y_in):
                    return jax.core.eval_jaxpr(jaxpr_y2h, consts, y_in)

                # Get df as Jacobian-vector product with identity
                df_vals = jax.jacfwd(_eval_transition)(y_val)
                # df_vals is a list of arrays, one per hidden outvar
                df_concat = group.concat_hidden(
                    [jnp.squeeze(d, axis=0) if d.ndim > y_val.ndim else d
                     for d in df_vals]
                )

                df_key = etrace_df_key(relation.y_var, group.index)
                etrace_dfs[df_key] = df_concat

        h2w_jac = (etrace_xs, etrace_dfs)

        # --- h2h Jacobians ---
        h2h_jacs = []
        for group in graph.hidden_groups:
            jac = self._compute_h2h_for_group(group, temp_data)
            h2h_jacs.append(jac)

        return out, new_hidden_vals, new_other_vals, h2w_jac, h2h_jacs

    def _compute_h2h_for_group(
        self,
        group: HiddenGroup,
        temp_data: Dict,
    ) -> jax.Array:
        """
        Compute ∂h^t / ∂h^{t-1} for a hidden group using
        the transition Jaxpr and perturbation.
        """
        # Collect h_prev values and transition consts
        h_prev_vals = [temp_data[v] for v in group.hidden_invars]
        h_prev_concat = group.concat_hidden(h_prev_vals)

        consts = [temp_data[v] for v in group.transition_jaxpr_constvars]

        def _transition(h_in):
            """h^t = g(h^{t-1}, consts)"""
            # Split h_in into individual hidden state arrays
            h_splits = group.split_hidden(h_in)
            out = jax.core.eval_jaxpr(
                group.transition_jaxpr, consts, *h_splits
            )
            return group.concat_hidden(out)

        # Compute full Jacobian ∂h^t/∂h^{t-1}
        jac = jax.jacfwd(_transition)(h_prev_concat)
        return jac

    def show_graph(self):
        """Print a summary of the compiled ETP graph."""
        graph = self.graph
        print(f'ETP Graph for {self.model.__class__.__name__}')
        print(f'  Hidden groups: {len(graph.hidden_groups)}')
        for i, g in enumerate(graph.hidden_groups):
            print(f'    [{i}] paths={g.hidden_paths}, '
                  f'num_state={g.num_state}')
        print(f'  ETP op relations: {len(graph.etp_op_relations)}')
        for rel in graph.etp_op_relations:
            print(f'    {rel.primitive.name}: '
                  f'weight={rel.weight_path}, '
                  f'groups={[g.index for g in rel.hidden_groups]}')
