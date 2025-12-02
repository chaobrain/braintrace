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
#
# Author: Chaoming Wang <chao.brain@qq.com>
# Copyright: 2024, Chaoming Wang
# Date: 2024-04-03
#
# ==============================================================================
#
# Refinement History:
#   [2024-04-03] Created
#   [2024-04-06] Added the traceback information for the error messages.
#   [2024-04-16] Changed the "op" in the "HiddenWeightOpTracer" to "JaxprEqn".
#                Added the support for the "pjit" operator.
#   [2024-05] Add the support for vjp_time == 't_minus_1'
#   [2024-06] Conditionally support control flows, including `scan`, `while`, and `cond`
#   [2024-09] version 0.0.2
#   [2024-11-22] compatible with `brainstate>=0.1.0` (#17)
#   [2024-11-23] Add the support for vjp_time_ahead > 1, it can combine the
#                advantage of etrace learning and backpropagation through time.
#   [2024-11] version 0.0.3, a complete new revision for better model debugging.
#   [2025-02-06]
#       - [x] split into "_etrace_graph_executor.py" and "graph_executor.py"
#
# ==============================================================================

# -*- coding: utf-8 -*-

from typing import Dict, Tuple

import brainstate
import brainunit as u
import jax.core
import jax.numpy as jnp
from jax.extend import linear_util as lu
from jax.interpreters import partial_eval as pe
from jax.tree_util import register_pytree_node_class

from braintrace._compatible_imports import Var
from braintrace._etrace_compiler_graph import compile_etrace_graph
from braintrace._etrace_compiler_hidden_group import HiddenGroup
from braintrace._etrace_graph_executor import ETraceGraphExecutor
from braintrace._etrace_input_data import (
    get_single_step_data,
    split_input_data_types,
    merge_data,
    has_multistep_data,
)
from braintrace._misc import (
    etrace_df_key,
    etrace_x_key,
)
from braintrace._state_managment import (
    assign_dict_state_values,
    split_dict_states_v2
)
from braintrace._typing import (
    Outputs,
    ETraceVals,
    StateVals,
    ETraceX_Key,
    ETraceDF_Key,
    Hid2WeightJacobian,
    HiddenGroupJacobian,
)

# TODO
#
# - [x] The visualization of the etrace graph.
# - [ ] Evaluate whether the `df` is the same for different weights.
#       For example,
#
#          h = f(x1 @ w1 + x2 @ w2)
#
#       The `df` for w1 and w2 are the same, although them have the different weight y.

__all__ = [
    'ETraceVjpGraphExecutor',
]


@register_pytree_node_class
class VjpResiduals:
    """
    The residuals for storing the backward pass data in a VJP function.

    Args:
      jaxpr: The jaxpr for the backward pass.
      in_tree: The input tree structure.
      out_tree: The output tree structure.
      consts: The constants for the backward pass.
    """

    def __init__(
        self,
        jaxpr,
        in_tree,
        out_tree,
        consts
    ):
        self.jaxpr = jaxpr
        self.in_tree = in_tree
        self.out_tree = out_tree
        self.consts = consts

    def __iter__(self):
        return iter((self.jaxpr, self.in_tree, self.out_tree, self.consts))

    def tree_flatten(self):
        return self.consts, (self.jaxpr, self.in_tree, self.out_tree)

    @classmethod
    def tree_unflatten(cls, aux, consts):
        jaxpr, in_tree, out_tree = aux
        return cls(jaxpr, in_tree, out_tree, consts)


class ETraceVjpGraphExecutor(ETraceGraphExecutor):
    r"""
    The eligibility trace graph executor for the VJP-based online learning algorithms.

    This class is used for executing the eligibility trace graph for the VJP-based online learning algorithms,
    including:

    - :class:`IODimVjpAlgorithm` for the algorithm with input-output dimensional complexity.
    - :class:`ParamDimVjpAlgorithm` for the algorithm with parameter dimensional complexity.
    - :class:`HybridDimVjpAlgorithm` for the algorithm with hybrid dimensional complexity.

    Parameters
    ----------
    model: brainstate.nn.Module
        The model to build the eligibility trace graph. The models should only define the one-step behavior.
    vjp_method: str
        The method for computing the VJP. It should be either "single-step" or "multi-step".

        - "single-step": The VJP is computed at the current time step, i.e., $\partial L^t/\partial h^t$.
        - "multi-step": The VJP is computed at multiple time steps, i.e., $\partial L^t/\partial h^{t-k}$,
          where $k$ is determined by the data input.
    """
    __module__ = 'braintrace'

    def __init__(
        self,
        model: brainstate.nn.Module,
        vjp_method: str = 'single-step'
    ):
        super().__init__(model)

        # the VJP method
        assert vjp_method in ('single-step', 'multi-step'), (
            'The VJP method should be either "single-step" or "multi-step". '
            f'While we got {vjp_method}. '
        )
        self.vjp_method = vjp_method

    @property
    def is_single_step_vjp(self) -> bool:
        """
        Whether the VJP method is ``single-step``.

        Returns:
            bool: Whether the VJP method is ``single-step``.
        """
        return self.vjp_method == 'single-step'

    @property
    def is_multi_step_vjp(self) -> bool:
        """
        Whether the VJP method is ``multi-step``.

        Returns:
            bool: Whether the VJP method is ``multi-step``.
        """
        return self.vjp_method == 'multi-step'

    def compile_graph(self, *args) -> None:
        r"""
        Building the eligibility trace graph for the model according to the given inputs.

        This is the most important method for the eligibility trace graph. It builds the
        graph for the model, which is used for computing the weight spatial gradients and
        the hidden state Jacobian.

        Args:
            *args: The positional arguments for the model.
        """

        # process the inputs
        args = get_single_step_data(*args)

        # compile the graph
        self._compiled_graph = compile_etrace_graph(self.model, *args, include_hidden_perturb=self.is_single_step_vjp)

    def _compute_hid2weight_jacobian(
        self,
        intermediate_values: Dict[Var, jax.Array]
    ) -> Tuple[
        Dict[ETraceX_Key, jax.Array],
        Dict[ETraceDF_Key, jax.Array]
    ]:
        """
        Computing the weight x and df values for the spatial gradients.

        Args:
            intermediate_values: The intermediate values of the model.

        Returns:
            The weight x and df values.
        """

        # the weight x
        xs = {}
        for relation in self.graph.hidden_param_op_relations:
            if relation.x is not None:
                x = etrace_x_key(relation.x)
                xs[x] = intermediate_values[relation.x]

        # the weight df
        dfs = {}
        for relation in self.graph.hidden_param_op_relations:
            y = intermediate_values[relation.y]

            #
            # [ KEY ]
            #
            # # ---- Method 1: using ``backward_pass`` ---- #
            # Assuming the function is linear. One cheap way is to use
            # the backward pass for computing the gradients of the hidden states.
            # For most situations, the ``y --> hidden`` relation is linear. Therefore,
            # we use ``backward_pass`` to compute the ``Df`` while avoids the overhead
            # of computing the forward pass. Otherwise, we should use ``jax.vjp`` instead.
            # Please also see ``jax.linear_transpose()`` for the same purpose.
            #
            # [df] = backward_pass(relation.jaxpr_y2hid, [], True, consts, invars, outvars)
            #
            # # ---- Method 2: using ``jax.vjp`` ---- #
            # For general cases, we should use ``jax.vjp`` to compute the gradients.
            #
            # # ---- Method 3: using ``jax.jvp`` ---- #
            # For computational efficiency, we use ``jax.jvp`` to compute the gradients,
            # since this is the one-to-many mapping.
            #
            primals, hidden_group_tangents = jax.jvp(
                lambda y_val: relation.y_to_hidden_groups(
                    y_val,
                    intermediate_values,
                    concat_hidden_vals=True,
                ),
                [y],
                [u.math.ones_like(y)]
            )

            #
            # compute the df we want
            #
            #    df = jvp gradient of "y -> hidden group"
            #
            for tangent, group in zip(hidden_group_tangents, relation.hidden_groups):
                dfs[etrace_df_key(relation.y, group.index)] = tangent

        # all x and df values
        return jax.lax.stop_gradient(xs), jax.lax.stop_gradient(dfs)

    def _compute_hid2hid_jacobian(
        self,
        intermediate_values: Dict[Var, jax.Array]
    ) -> HiddenGroupJacobian:
        """
        Computing the hidden group-to-hidden group Jacobian according to the given intermediate values.

        Args:
            intermediate_values: The intermediate values of the model.

        Returns:
            The hidden group-to-hidden group Jacobian.
        """

        hid2hid_jacobian = []
        for group in self.graph.hidden_groups:
            group: HiddenGroup

            # data for jacobian computation
            hidden_vals = [intermediate_values[v] for v in group.hidden_invars]
            input_vals = [intermediate_values[v] for v in group.transition_jaxpr_constvars]

            # compute the jacobian
            jac = group.diagonal_jacobian(hidden_vals, input_vals)
            hid2hid_jacobian.append(jac)

        return jax.lax.stop_gradient(hid2hid_jacobian)

    def solve_h2w_h2h_jacobian(
        self,
        *args,
    ) -> Tuple[
        Outputs,
        ETraceVals,
        StateVals,
        Hid2WeightJacobian,
        HiddenGroupJacobian,
    ]:
        r"""
        Solving the hidden-to-weight and hidden-to-hidden Jacobian according to the given inputs and parameters.

        This function is typically used for computing the forward propagation of hidden-to-weight Jacobian.

        Now we mathematically define what computations are done in this function.

        For the state transition function $y, h^t = f(h^{t-1}, \theta, x)$, this function aims to solve:

        1. The function output: $y$
        2. The updated hidden states: $h^t$
        3. The Jacobian matrix of hidden-to-weight, i.e., $\partial h^t / \partial \theta^t$.
        2. The Jacobian matrix of hidden-to-hidden, i.e., $\partial h^t / \partial h^{t-1}$.

        Args:
            *args: The positional arguments for the model.

        Returns:
            The outputs, hidden states, other states, and the spatial gradients of the weights.
            Return the single-step results if inputs do not contain multiple-step data,
            otherwise return the multi-step data.
        """

        input_is_multi_step = has_multistep_data(*args)

        # --- split the states and state values --- #
        (
            etrace_params,
            etrace_states,
            non_etrace_params,
            other_states
        ) = split_dict_states_v2(self.states)

        etrace_param_vals = {path: st.value for path, st in etrace_params.items()}
        etrace_state_vals = {path: st.value for path, st in etrace_states.items()}
        non_etrace_param_vals = {path: st.value for path, st in non_etrace_params.items()}
        other_state_vals = {path: st.value for path, st in other_states.items()}

        # --- processing the inputs information --- #
        (
            args_single_step,
            args_multi_steps,
            tree_def,
        ) = split_input_data_types(*args)

        # --- call the model --- #

        def scan_fn(carray, single_step_of_multistep_arg):
            args_ = merge_data(tree_def, single_step_of_multistep_arg, args_single_step)

            _etrace_state_vals, _oth_state_vals = carray
            # use "restore_value" to recover the hidden states
            # this keeps the reading/writing operations as
            # the same as the original model
            for path, val in _etrace_state_vals.items():
                self.path_to_states[path].restore_value(val)
            for path, val in _oth_state_vals.items():
                self.path_to_states[path].restore_value(val)
            for path, val in non_etrace_param_vals.items():
                self.path_to_states[path].restore_value(val)
            for path, val in etrace_param_vals.items():
                self.path_to_states[path].restore_value(val)

            (
                out,
                _etrace_state_vals,
                _oth_state_vals,
                temps
            ) = self.graph.module_info.jaxpr_call(*args_)

            # compute the hidden-to-weight Jacobian
            hid2weight_jac = self._compute_hid2weight_jacobian(temps)

            # compute the hidden-to-hidden Jacobian
            hid2hid_jac = self._compute_hid2hid_jacobian(temps)

            return (_etrace_state_vals, _oth_state_vals), (out, hid2weight_jac, hid2hid_jac)

        # check the batch size
        if len(args_multi_steps):
            args_dim = [jnp.shape(x)[0] for x in jax.tree.leaves(args_multi_steps)]
            if len(set(args_dim)) != 1:
                raise ValueError(f'The sequence size should be the same for all inputs. But we got {args_dim}.')

        if input_is_multi_step:
            (
                (
                    etrace_state_vals,
                    other_state_vals
                ),
                (
                    outs_single_or_multi_steps,
                    hid2weight_jac_single_or_multi_steps,
                    hid2hid_jac_single_or_multi_steps
                )
            ) = jax.lax.scan(scan_fn, (etrace_state_vals, other_state_vals), args_multi_steps)

        else:
            (
                (
                    etrace_state_vals,
                    other_state_vals
                ),
                (
                    outs_single_or_multi_steps,
                    hid2weight_jac_single_or_multi_steps,
                    hid2hid_jac_single_or_multi_steps
                )
            ) = scan_fn((etrace_state_vals, other_state_vals), {})

        # recovering the other non-etrace weights, although the weights are not changed
        assign_dict_state_values(non_etrace_params, non_etrace_param_vals, write=False)
        assign_dict_state_values(etrace_params, etrace_param_vals, write=False)

        # return the results
        return (
            outs_single_or_multi_steps,
            etrace_state_vals,
            other_state_vals,
            hid2weight_jac_single_or_multi_steps,
            hid2hid_jac_single_or_multi_steps
        )

    def solve_h2w_h2h_l2h_jacobian(
        self,
        *args,
    ) -> Tuple[
        Outputs,
        ETraceVals,
        StateVals,
        Hid2WeightJacobian,
        HiddenGroupJacobian,
        VjpResiduals,
    ]:
        r"""
        Solving the hidden-to-weight and hidden-to-hidden Jacobian and the VJP transformed loss-to-hidden
        gradients according to the given inputs.

        This function is typically used for computing both the forward propagation of hidden-to-weight Jacobian
        and the loss-to-hidden gradients at the current time-step.

        Particularly, this function aims to solve:

        1. The Jacobian matrix of hidden-to-weight. That is,
           :math:`\partial h / \partial w`, where :math:`h` is the hidden state and :math:`w` is the weight.
        2. The Jacobian matrix of hidden-to-hidden. That is,
           :math:`\partial h / \partial h`, where :math:`h` is the hidden state.
        3. The partial gradients of the loss with respect to the hidden states.
           That is, :math:`\partial L / \partial h`, where :math:`L` is the loss and :math:`h` is the hidden state.

        Args:
          *args: The positional arguments for the model.

        Returns:
          The outputs, hidden states, other states, the spatial gradients of the weights, and the residuals.
        """
        input_is_multi_step = has_multistep_data(*args)

        if self.is_single_step_vjp and input_is_multi_step:
            raise NotImplementedError(
                'When the VJP method is "single-step", '
                'we only support the input data that is at a single time step, '
                'while we got the data at multiple time steps. \n'
                'This design is to ensure the correctness of the VJP gradient '
                'computation of hidden states.'
            )

        # ---------------------- [Part 1] ----------------------
        # weights, hidden, and states information
        # for VJP computation
        # ------------------------------------------------------

        #  [KEY]
        #  The most important assumption here is
        #  that the weight values (including etrace weights and normal param weights) are not changed

        # split the states, got initial hidden and weight values

        (
            etrace_param_states,
            etrace_hidden_states,
            non_etrace_param_states,
            other_states
        ) = split_dict_states_v2(self.states)

        if self.is_single_step_vjp:
            etrace_param_vals = dict()
            hidden_perturbs = self.graph.hidden_perturb.init_perturb_data()
            etrace_weight_vals_restore = {path: st.value for path, st in etrace_param_states.items()}

        else:
            etrace_param_vals = {path: st.value for path, st in etrace_param_states.items()}
            etrace_weight_vals_restore = {k: v for k, v in etrace_param_vals.items()}
            hidden_perturbs = []

        non_etrace_param_vals = {path: st.value for path, st in non_etrace_param_states.items()}
        etrace_state_vals = {path: st.value for path, st in etrace_hidden_states.items()}
        other_state_vals = {path: st.value for path, st in other_states.items()}

        def fun_for_vjp(
            inputs,  # functional inputs, original inputs
            etrace_hidden_vals_,  # etrace hidden states
            non_etrace_param_vals_,  # non-etrace weights
            etrace_param_vals_,  # etrace weights
            oth_state_vals_,  # other states
            perturb_vals_  # hidden perturbations, useful when computing \partial L / \partial h
        ):
            # assign state values
            if len(etrace_param_vals_) > 0:
                assign_dict_state_values(etrace_param_states, etrace_param_vals_, write=False)
            assign_dict_state_values(etrace_hidden_states, etrace_hidden_vals_, write=False)
            assign_dict_state_values(non_etrace_param_states, non_etrace_param_vals_, write=False)
            assign_dict_state_values(other_states, oth_state_vals_, write=False)

            # get state values by the "stateful_model", to preserve the order of states
            old_state_vals = [st.value for st in self.graph.module_info.compiled_model_states]

            # calling the function
            if self.is_single_step_vjp:
                assert self.graph.hidden_perturb is not None, (
                    'The hidden_perturb should not be None '
                    'when the vjp method is "single-step".'
                )

                (
                    out, _etrace_state_vals, _oth_state_vals, temps
                ) = self.graph.call_hidden_perturb(
                    inputs,
                    perturb_vals_,
                    old_state_vals,
                )

            else:
                assert len(perturb_vals_) == 0, (
                    'The hidden perturbations should be empty '
                    'when the vjp method is "multi-step".'
                )

                (
                    out, _etrace_state_vals, _oth_state_vals, temps
                ) = self.graph.module_info.jaxpr_call(*inputs, old_state_vals=old_state_vals)

            # --- compute the hidden-to-weight Jacobian --- #
            hid2weight_jac = self._compute_hid2weight_jacobian(temps)

            # --- compute the hidden-to-hidden Jacobian --- #
            hid2hid_jac = self._compute_hid2hid_jacobian(temps)

            return out, _etrace_state_vals, _oth_state_vals, hid2weight_jac, hid2hid_jac

        # ---------------------- [Part 2.1] ----------------------
        # Scan VJP function over multiple time steps
        # --------------------------------------------------------

        # In the following variable names, the suffix "_ss" means "single-step",
        # and the suffix "_ms" means "multi-step".

        def scan_over_multiple_steps(
            inputs_single_or_multi: Dict,  # the inputs for single/multiple time steps
            hidden_vals_ss,  # the initial hidden states
            non_etrace_weight_vals_ss,  # the non-etrace weights
            etrace_weight_vals_ss,  # the etrace weights
            other_vals_ss,  # the initial other states
            hidden_perturbs_ss  # the hidden perturbations, only used when is_single_step_vjp is True
        ):

            # processing the inputs information
            args_single_step, args_multi_steps, tree_def = split_input_data_types(*inputs_single_or_multi)
            assert len(args_multi_steps), 'The inputs should contain at least one multi-step data.'

            # check the batch size
            args_dim = [jnp.shape(x)[0] for x in jax.tree.leaves(args_multi_steps)]
            if len(set(args_dim)) != 1:
                raise ValueError(f'The sequence size should be the same for all inputs. But we got {args_dim}.')

            # scan function
            def scan_fn(carray, x_ss: Dict):
                args_ss = merge_data(tree_def, x_ss, args_single_step)

                hidden_vals_iter, other_vals_iter = carray
                (
                    out,
                    hidden_vals_iter,
                    other_vals_iter,
                    hid2weight_jac,
                    hid2hid_jac
                ) = fun_for_vjp(
                    args_ss,
                    hidden_vals_iter,
                    non_etrace_weight_vals_ss,
                    etrace_weight_vals_ss,
                    other_vals_iter,
                    hidden_perturbs_ss,
                )

                return (
                    (hidden_vals_iter, other_vals_iter),
                    (out, hid2weight_jac, hid2hid_jac)
                )

            # scan over multiple time steps
            (
                (
                    hidden_vals_ss,
                    other_vals_ss
                ),
                (
                    _outs_multi_steps,
                    _hid2weight_jac_multi_steps,
                    _hid2hid_jac_multi_steps
                )
            ) = jax.lax.scan(scan_fn, (hidden_vals_ss, other_vals_ss), args_multi_steps)

            return (
                (
                    _outs_multi_steps,
                    hidden_vals_ss,
                    other_vals_ss
                ),
                (
                    _hid2weight_jac_multi_steps,
                    _hid2hid_jac_multi_steps
                )
            )

        # ---------------------- [Part 2.2] ----------------------
        # Scan VJP function over single time step
        # --------------------------------------------------------

        def call_over_single_step(
            inputs_single_or_multi: Dict,  # the inputs for single/multiple time steps
            hidden_vals_ss,  # the initial hidden states
            non_etrace_weight_vals_ss,  # the non-etrace weights
            etrace_weight_vals_ss,  # the etrace weights
            other_vals_ss,  # the initial other states
            hidden_perturbs_ss  # the hidden perturbations, only used when is_single_step_vjp is True
        ):
            (
                out,
                hidden_vals_iter,
                other_vals_iter,
                hid2weight_jac,
                hid2hid_jac
            ) = fun_for_vjp(
                inputs_single_or_multi,
                hidden_vals_ss,
                non_etrace_weight_vals_ss,
                etrace_weight_vals_ss,
                other_vals_ss,
                hidden_perturbs_ss,
            )
            return (
                (out, hidden_vals_iter, other_vals_iter),
                (hid2weight_jac, hid2hid_jac)
            )

        # ---------------------- [Part 3] ------------------------
        # Compile the AutoGrad of the VJP function that over time
        # into the residual jaxpr representation
        # ---------------------------------------------------------

        # format VJP calling, compile the autograd information into the residual jaxpr representation
        # so that it can be computed when they are needed.
        (
            (
                out_single_or_multi_steps,
                etrace_state_vals,
                other_state_vals
            ),
            f_vjp,
            (
                hid2weight_jac_single_or_multi_steps,
                hid2hid_jac_single_or_multi_steps
            )
        ) = jax.vjp(
            (scan_over_multiple_steps if input_is_multi_step else call_over_single_step),  # the function
            args,  # the inputs (multiple/single time)
            etrace_state_vals,  # the inputs (single time)
            non_etrace_param_vals,  # the inputs (single time)
            etrace_param_vals,  # the inputs (single time)
            other_state_vals,  # the inputs (single time)
            hidden_perturbs,  # the inputs (single time)
            has_aux=True
        )
        out_flat, out_tree = jax.tree.flatten(((out_single_or_multi_steps, etrace_state_vals, other_state_vals),))
        rule, in_tree = jax.api_util.flatten_fun_nokwargs(lu.wrap_init(f_vjp), out_tree)
        out_avals = [jax.core.get_aval(x).at_least_vspace() for x in out_flat]
        jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(rule, out_avals)
        residual = VjpResiduals(jaxpr, in_tree(), out_tree, consts)

        # ---------------------- [Part 4] ------------------------
        # Recover the weight states values
        # ---------------------------------------------------------

        # recovering other non-etrace weights, although the weights are not changed
        assign_dict_state_values(non_etrace_param_states, non_etrace_param_vals, write=False)
        assign_dict_state_values(etrace_param_states, etrace_weight_vals_restore, write=False)

        return (
            out_single_or_multi_steps,
            etrace_state_vals,
            other_state_vals,
            hid2weight_jac_single_or_multi_steps,
            hid2hid_jac_single_or_multi_steps,
            residual,
        )
