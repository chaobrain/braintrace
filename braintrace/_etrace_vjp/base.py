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


from typing import Dict, Tuple, Any, List, Optional, Sequence

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

from braintrace._etrace_algorithms import (
    ETraceAlgorithm,
)
from braintrace._etrace_input_data import has_multistep_data
from braintrace._state_managment import assign_state_values_v2
from braintrace._typing import (
    PyTree,
    Outputs,
    WeightID,
    WeightVals,
    HiddenVals,
    StateVals,
    ETraceVals,
    Hid2WeightJacobian,
    dG_Inputs,
    dG_Weight,
    dG_Hidden,
    dG_State,
)
from .graph_executor import ETraceVjpGraphExecutor

__all__ = [
    'ETraceVjpAlgorithm',  # the base class for the eligibility trace algorithm with the VJP gradient computation
]


class ETraceVjpAlgorithm(ETraceAlgorithm):
    r"""
    The base class for the eligibility trace algorithm which supporting the VJP gradient
    computation (reverse-mode differentiation).

    The term ``VJP`` comes from the following two aspects:

    **First**, this module is designed to be compatible with the JAX's VJP mechanism.
    This means that the gradient is computed according to the reverse-mode differentiation
    interface, like the ``jax.grad()`` function, the ``jax.vjp()`` function,
    or the ``jax.jacrev()`` function. The true update function is defined as a custom
    VJP function ``._true_update_fun()``, which receives the inputs, the hidden states,
    other states, and etrace variables at the last time step, and returns the outputs,
    the hidden states, other states, and etrace variables at the current time step.

    For each subclass (or the instance of an etrace algorithm), we should define the
    following methods:

    - ``._update()``: update the eligibility trace states and return the outputs, hidden states, other states, and etrace data.
    - ``._update_fwd()``: the forward pass of the custom VJP rule.
    - ``._update_bwd()``: the backward pass of the custom VJP rule.

    However, this class has provided a default implementation for the ``._update()``,
    ``._update_fwd()``, and ``._update_bwd()`` methods.

    To implement a new etrace algorithm, users just need to override the following methods:

    - ``._solve_weight_gradients()``: solve the gradients of the learnable weights / parameters.
    - ``._update_etrace_data()``: update the eligibility trace data.
    - ``._assign_etrace_data()``: assign the eligibility trace data to the states.
    - ``._get_etrace_data()``: get the eligibility trace data.

    **Second**, the algorithm computes the spatial gradient $\partial L^t / \partial H^t$ using the standard
    back-propagation algorithm. This design can enhance the accuracy and the stability of the algorithm for
    computing gradients.


    Parameters
    ----------
    model: brainstate.nn.Module
        The model function, which receives the input arguments and returns the model output.
    name: str, optional
        The name of the etrace algorithm.
    vjp_method: str
        The method for computing the VJP. It should be either "single-step" or "multi-step".

        - "single-step": The VJP is computed at the current time step, i.e., $\partial L^t/\partial h^t$.
        - "multi-step": The VJP is computed at multiple time steps, i.e., $\partial L^t/\partial h^{t-k}$,
          where $k$ is determined by the data input.

    """

    __module__ = 'braintrace'
    graph_executor: ETraceVjpGraphExecutor

    def __init__(
        self,
        model: brainstate.nn.Module,
        name: Optional[str] = None,
        vjp_method: str = 'single-step'
    ):

        # the VJP method
        assert vjp_method in ('single-step', 'multi-step'), (
            'The VJP method should be either "single-step" or "multi-step". '
            f'While we got {vjp_method}. '
        )
        self.vjp_method = vjp_method

        # graph
        graph_executor = ETraceVjpGraphExecutor(model, vjp_method=vjp_method)

        # super initialization
        super().__init__(model=model, name=name, graph_executor=graph_executor)

        # the update rule
        self._true_update_fun = jax.custom_vjp(self._update_fn)
        self._true_update_fun.defvjp(
            fwd=self._update_fn_fwd,
            bwd=self._update_fn_bwd
        )

    def _assert_compiled(self):
        if not self.is_compiled:
            raise ValueError('The etrace algorithm has not been compiled. Please call `compile_graph()` first. ')

    def update(self, *args) -> Any:
        """
        Update the model states and the eligibility trace.

        The input arguments ``args`` here supports very complex data structures, including
        the combination of :py:class:`SingleStepData` and :py:class:`MultiStepData`.

        - :py:class:`SingleStepData`: indicating the data at the single time step, $x_t$.
        - :py:class:`MultiStepData`: indicating the data at multiple time steps, $[x_{t-k}, ..., x_t]$.

        Suppose all inputs have the shape of ``(10,)``.

        If the input arguments are given by:

        .. code-block:: python

            x = [jnp.ones((10,)), jnp.zeros((10,))]

        Then, two input arguments are considered as the :py:class:`SingleStepData`.

        If the input arguments are given by:

        .. code-block:: python

            x = [braintrace.SingleStepData(jnp.ones((10,))),
                 braintrace.SingleStepData(jnp.zeros((10,)))]

        This is the same as the previous case, they are all considered as the input at the current time step.

        If the input arguments are given by:

        .. code-block:: python

            x = [braintrace.MultiStepData(jnp.ones((5, 10)),
                 jnp.zeros((10,)))]

        or,

        .. code-block:: python

            x = [braintrace.MultiStepData(jnp.ones((5, 10)),
                 braintrace.SingleStepData(jnp.zeros((10,)))]

        Then, the first input argument is considered as the :py:class:`MultiStepData`, and its data will
        be fed into the model within five consecutive steps, and the second input argument will be fed
        into the model at each time of this five consecutive steps.

        Args:
            *args: the input arguments.
        """

        # ----------------------------------------------------------------------------------------------
        #
        # This method is the main function to
        #
        # - update the model
        # - update the eligibility trace states
        # - compute the weight gradients
        #
        # The key here is that we change the object-oriented attributes as the function arguments.
        # Therefore, the function arguments are the states of the current time step, and the function
        # returns the states of the next time step.
        #
        # Particularly, the model calls the "_true_update_fun()" function to update the states.
        #
        # ----------------------------------------------------------------------------------------------

        #
        # This function need to process the following multiple cases:
        #
        # 1. if vjp_method = 'single-step', input = SingleStepData, then output is single step
        #
        # 2. if vjp_method = 'single-step', input = MultiStepData, then output is multiple step data
        #
        # 3. if vjp_method = 'multi-step', input = SingleStepData, then output is single step
        #
        # 4. if vjp_method = 'multi-step', input = MultiStepData, then output is multiple step data
        #

        # check the compilation
        self._assert_compiled()

        # state values
        weight_vals = {
            key: st.value
            for key, st in self.param_states.items()
        }
        hidden_vals = {
            key: st.value
            for key, st in self.hidden_states.items()
        }
        other_vals = {
            key: st.value
            for key, st in self.other_states.items()
        }
        # etrace data
        last_etrace_vals = self._get_etrace_data()

        # update all states
        #
        # [KEY] The key here is that we change the object-oriented attributes as the function arguments.
        #       Therefore, the function arguments are the states of the current time step, and the function
        #       returns the states of the next time step.
        #
        # out: is always multiple step
        (
            out,
            hidden_vals,
            other_vals,
            new_etrace_vals
        ) = self._true_update_fun(
            args,
            weight_vals,
            hidden_vals,
            other_vals,
            last_etrace_vals,
            self.running_index.value
        )

        # assign/restore the weight values back
        #
        # [KEY] assuming the weight values are not changed
        #       This is a key assumption in the RTRL algorithm.
        #       This is very important for the implementation.
        assign_state_values_v2(self.param_states, weight_vals, write=False)

        # assign the new hidden and state values
        assign_state_values_v2(self.hidden_states, hidden_vals)
        assign_state_values_v2(self.other_states, other_vals)

        #
        # assign the new etrace values
        #
        # "self._assign_etrace_data()" is a protocol method that should be implemented in the subclass.
        # It's logic may be different for different etrace algorithms.
        #
        self._assign_etrace_data(new_etrace_vals)  # call the protocol method

        # update the running index
        running_index = self.running_index.value + 1
        self.running_index.value = jax.lax.stop_gradient(jnp.where(running_index >= 0, running_index, 0))

        # return the model output
        return out

    def _update_fn(
        self,
        args,
        weight_vals: WeightVals,
        hidden_vals: HiddenVals,
        oth_state_vals: StateVals,
        etrace_vals: ETraceVals,
        running_index,
    ) -> Tuple[Outputs, HiddenVals, StateVals, ETraceVals]:
        """
        The main function to update the [model] and the [eligibility trace] states.

        Particularly, ``self.graph.solve_h2w_h2h_jacobian()`` is called to:
          - compute the model output, the hidden states, and the other states
          - compute the hidden-to-weight Jacobian and the hidden-to-hidden Jacobian

        Then, ``self._update_etrace_data`` is called to:
          - update the eligibility trace data

        Moreover, this function returns:
          - the model output
          - the updated hidden states
          - the updated other states
          - the updated eligibility trace states

        Note that the weight values are assumed not changed in this function.

        """
        input_is_multi_step = has_multistep_data(*args)

        # state value assignment
        assign_state_values_v2(self.param_states, weight_vals, write=False)
        assign_state_values_v2(self.hidden_states, hidden_vals, write=False)
        assign_state_values_v2(self.other_states, oth_state_vals, write=False)

        # necessary jacobian information of the weights
        (
            out,
            hidden_vals,
            oth_state_vals,
            hid2weight_jac_single_or_multi_steps,
            hid2hid_jac_single_or_multi_steps
        ) = self.graph_executor.solve_h2w_h2h_jacobian(*args)

        # eligibility trace update
        #
        # "self._update_etrace_data()" is a protocol method that should be implemented in the subclass.
        # It's logic may be different for different etrace algorithms.
        #
        etrace_vals = self._update_etrace_data(
            running_index,
            etrace_vals,
            hid2weight_jac_single_or_multi_steps,
            hid2hid_jac_single_or_multi_steps,
            weight_vals,
            input_is_multi_step,
        )

        # returns
        return out, hidden_vals, oth_state_vals, etrace_vals

    def _update_fn_fwd(
        self,
        args,
        weight_vals: WeightVals,
        hidden_vals: HiddenVals,
        othstate_vals: StateVals,
        etrace_vals: ETraceVals,
        running_index: int,
    ) -> Tuple[Tuple[Outputs, HiddenVals, StateVals, ETraceVals], Any]:
        """
        The forward function to update the [model] and the [eligibility trace] states when computing
        the VJP gradients.

        Particularly, ``self.graph.solve_h2w_h2h_jacobian_and_l2h_vjp()`` is called to:

        - compute the model output, the hidden states, and the other states
        - compute the hidden-to-weight Jacobian and the hidden-to-hidden Jacobian
        - compute the loss-to-hidden or loss-to-weight Jacobian

        Then, ``self._update_etrace_data`` is called to:

        - update the eligibility trace data

        The forward function returns two parts of data:

        - The first part is the functional returns (same as "self._update()" function):
              * the model output
              * the updated hidden states
              * the updated other states
              * the updated eligibility trace states

        - The second part is the data used for backward gradient computation:
              * the residuals of the model
              * the eligibility trace data at the current/last time step
              * the weight id to its value mapping
              * the running index
        """
        input_is_multi_step = has_multistep_data(*args)

        # state value assignment
        assign_state_values_v2(self.param_states, weight_vals, write=False)
        assign_state_values_v2(self.hidden_states, hidden_vals, write=False)
        assign_state_values_v2(self.other_states, othstate_vals, write=False)

        # necessary gradients of the weights
        (
            out,
            hiddens,
            oth_states,
            hid2weight_jac_single_or_multi_steps,
            hid2hid_jac_single_or_multi_steps,
            residuals
        ) = self.graph_executor.solve_h2w_h2h_l2h_jacobian(*args)

        # eligibility trace update
        #
        # "self._update_etrace_data()" is a protocol method that should be implemented in the subclass.
        # It's logic may be different for different etrace algorithms.
        #
        new_etrace_vals = self._update_etrace_data(
            running_index,
            etrace_vals,
            hid2weight_jac_single_or_multi_steps,
            hid2hid_jac_single_or_multi_steps,
            weight_vals,
            input_is_multi_step
        )

        # returns
        old_etrace_vals = etrace_vals
        fwd_out = (out, hiddens, oth_states, new_etrace_vals)
        fwd_res = (
            residuals,
            (
                old_etrace_vals
                if self.graph_executor.is_multi_step_vjp else
                new_etrace_vals
            ),
            weight_vals,
            running_index
        )
        return fwd_out, fwd_res

    def _update_fn_bwd(
        self,
        fwd_res,
        grads,
    ) -> Tuple[dG_Inputs, dG_Weight, dG_Hidden, dG_State, None, None]:
        """
        The backward function to compute the VJP gradients when the learning signal is arrived at
        this time step.

        There are three steps:

        1. Interpret the forward results (eligibility trace) and top-down gradients (learning signal)
        2. Compute the gradients of input arguments
           (maybe necessary, but it can be optimized away but the XLA compiler)
        3. Compute the gradients of the weights

        """

        # [1] Interpret the fwd results
        #
        (
            residuals,  # the residuals of the VJP computation, for computing the gradients of input arguments
            etrace_vals_at_t_or_t_minus_1,  # the eligibility trace data at the current or last time step
            weight_vals,  # the weight id to its value mapping
            running_index  # the running index
        ) = fwd_res

        (
            jaxpr,
            in_tree,
            out_tree,
            consts
        ) = residuals

        # [2] Interpret the top-down gradient signals
        #
        # Since
        #
        #     dg_out, dg_hiddens, dg_others, dg_etrace = grads
        #
        # we need to remove the "dg_etrace" iterm from the gradients for matching
        # the jaxpr vjp gradients.
        #
        grad_flat, grad_tree = jax.tree.flatten((grads[:-1],))

        # [3] Compute the gradients of the input arguments
        #     It may be unnecessary, but it can be optimized away by the XLA compiler after it is computed.
        #
        # The input argument gradients are computed through the normal back-propagation algorithm.
        #
        if out_tree != grad_tree:
            raise TypeError(
                f'Gradient tree should be the same as the function output tree. '
                f'While we got: \n'
                f'out_tree  = {out_tree}\n!=\n'
                f'grad_tree = {grad_tree}'
            )
        cts_out = jax.core.eval_jaxpr(jaxpr, consts, *grad_flat)

        #
        # We compute:
        #
        #   - the gradients of input arguments,
        #     maybe necessary to propagate the gradients to the last layer
        #
        #   - the gradients of the hidden states at the last time step,
        #     maybe unnecessary but can be optimized away by the XLA compiler
        #
        #   - the gradients of the non-etrace parameters, defined by "NonTempParam"
        #
        #   - the gradients of the other states
        #
        #   - the gradients of the loss-to-hidden at the current time step
        #

        # the `_jaxpr_compute_model_with_vjp()` in `ETraceGraphExecutor`
        (
            dg_args,
            dg_last_hiddens,
            dg_non_etrace_params,
            dg_etrace_params,
            dg_oth_states,
            dg_hid_perturb_or_dl2h
        ) = jax.tree.unflatten(in_tree, cts_out)

        #
        # get the gradients of the hidden states at the last time step
        #
        if self.graph_executor.is_single_step_vjp:
            # TODO: the correspondence between the hidden states and the gradients
            #       should be checked.
            #
            assert len(dg_etrace_params) == 0  # gradients all etrace weights are updated by the RTRL algorithm
            assert len(self.graph.hidden_perturb.perturb_vars) == len(dg_hid_perturb_or_dl2h)
            dl2h_at_t_or_t_minus_1 = self.graph.hidden_perturb.perturb_data_to_hidden_group_data(
                dg_hid_perturb_or_dl2h,
                self.graph.hidden_groups,
            )

        else:
            assert len(dg_last_hiddens) == len(self.hidden_states)
            assert set(dg_last_hiddens.keys()) == set(self.hidden_states.keys()), (
                f'The hidden states should be the same. Bug got \n'
                f'{set(dg_last_hiddens.keys())}\n'
                f'!=\n'
                f'{set(self.hidden_states.keys())}'
            )
            dl2h_at_t_or_t_minus_1 = [
                group.concat_hidden(
                    [
                        # dimensionless processing
                        u.get_mantissa(dg_last_hiddens[path])
                        for path in group.hidden_paths
                    ]
                )
                for group in self.graph.hidden_groups
            ]

        #
        # [4] Compute the gradients of the weights
        #
        # the gradients of the weights are computed through the RTRL algorithm.
        #
        # "self._solve_weight_gradients()" is a protocol method that should be implemented in the subclass.
        # It's logic may be different for different etrace algorithms.
        #
        dg_weights = self._solve_weight_gradients(
            running_index,
            etrace_vals_at_t_or_t_minus_1,
            dl2h_at_t_or_t_minus_1,
            weight_vals,
            dg_non_etrace_params,
            dg_etrace_params,
        )

        # Note that there are no gradients flowing through the etrace data and the running index.
        dg_etrace = None
        dg_running_index = None

        return (
            dg_args,
            dg_weights,
            dg_last_hiddens,
            dg_oth_states,
            dg_etrace,
            dg_running_index
        )

    def _solve_weight_gradients(
        self,
        running_index: Optional[int],
        etrace_h2w_at_t: Any,
        dl_to_hidden_groups: Sequence[jax.Array],
        weight_vals: Dict[WeightID, PyTree],
        dl_to_nonetws_at_t: List[PyTree],
        dl_to_etws_at_t: Optional[List[PyTree]],
    ):
        r"""
        The method to solve the weight gradients, i.e., :math:`\partial L / \partial W`.

        .. note::

            This is the protocol method that should be implemented in the subclass.


        Particularly, the weight gradients are computed through::

        .. math::

            \frac{\partial L^t}{\partial W} = \frac{\partial L^t}{\partial h^t} \frac{\partial h^t}{\partial W}

        Or,

        .. math::

            \frac{\partial L^t}{\partial W} = \frac{\partial L^{t-1}}{\partial h^{t-1}}
                                              \frac{\partial h^{t-1}}{\partial W}
                                              + \frac{\partial L^t}{\partial W^t}


        Args:
          running_index: Optional[int], the running index.
          etrace_h2w_at_t: Any, the eligibility trace data (which track the hidden-to-weight Jacobian)
              that have accumulated util the time ``t``.
          dl_to_hidden_groups: Dict[HiddenOutVar, jax.Array], the gradients of the loss-to-hidden
              at the time ``t``.
          weight_vals: Dict[WeightID, PyTree], the weight values.
          dl_to_nonetws_at_t: List[PyTree], the gradients of the loss-to-non-etrace parameters
              at the time ``t``, i.e., :math:``\partial L^t / \partial W^t``.
          dl_to_etws_at_t: List[PyTree], the gradients of the loss-to-etrace parameters
              at the time ``t``, i.e., :math:``\partial L^t / \partial W^t``.
        """
        raise NotImplementedError

    def _update_etrace_data(
        self,
        running_index: Optional[int],
        etrace_vals_util_t_1: ETraceVals,
        hid2weight_jac_single_or_multi_times: Hid2WeightJacobian,
        hid2hid_jac_single_or_multi_times: Sequence[jax.Array],
        weight_vals: WeightVals,
        input_is_multi_step: bool,
    ) -> ETraceVals:
        """
        The method to update the eligibility trace data.

        .. note::

            This is the protocol method that should be implemented in the subclass.

        Args:
          running_index: Optional[int], the running index.
          etrace_vals_util_t_1: ETraceVals, the history eligibility trace data that have accumulated util :math:`t-1`.
          hid2weight_jac_single_or_multi_times: ETraceVals, the current eligibility trace data at the time :math:`t`.
          hid2hid_jac_single_or_multi_times: The data for computing the hidden-to-hidden Jacobian at the time :math:`t`.
          weight_vals: Dict[WeightID, PyTree], the weight values.

        Returns:
          ETraceVals, the updated eligibility trace data that have accumulated util :math:`t`.
        """
        raise NotImplementedError

    def _get_etrace_data(self) -> ETraceVals:
        """
        Get the eligibility trace data at the last time-step.

        .. note::

            This is the protocol method that should be implemented in the subclass.

        Returns:
            ETraceVals, the eligibility trace data.
        """
        raise NotImplementedError

    def _assign_etrace_data(self, etrace_vals: ETraceVals) -> None:
        """
        Assign the eligibility trace data to the states at the current time-step.

        .. note::

            This is the protocol method that should be implemented in the subclass.

        Args:
          etrace_vals: ETraceVals, the eligibility trace data.
        """
        raise NotImplementedError
