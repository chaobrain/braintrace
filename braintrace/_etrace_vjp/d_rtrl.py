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

from functools import partial
from typing import Dict, Tuple, Optional, Sequence

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

from braintrace._etrace_algorithms import EligibilityTrace
from braintrace._etrace_compiler_hid_param_op import HiddenParamOpRelation
from braintrace._etrace_compiler_hidden_group import HiddenGroup
from braintrace._etrace_concepts import ElemWiseParam
from braintrace._etrace_operators import ETraceOp
from braintrace._misc import (
    etrace_param_key,
    etrace_df_key,
)
from braintrace._typing import (
    PyTree,
    WeightID,
    Path,
    ETraceX_Key,
    ETraceDF_Key,
    ETraceWG_Key,
    Hid2WeightJacobian,
    HiddenGroupJacobian,
    dG_Weight,
)
from .base import ETraceVjpAlgorithm
from .misc import _reset_state_in_a_dict, _batched_zeros_like, _sum_dim, _update_dict

__all__ = [
    'ParamDimVjpAlgorithm',
    'D_RTRL',
]


def _init_param_dim_state(
    mode: brainstate.mixin.Mode,
    etrace_bwg: Dict[ETraceWG_Key, brainstate.State],
    relation: HiddenParamOpRelation
):
    """
    Initialize the eligibility trace states for parameter dimensions.

    This function sets up the eligibility trace states for the weights and
    differential functions (df) associated with a given relation. It assumes
    that the batch size is the first dimension of the output shape if batching
    is enabled.

    Args:
        mode (brainstate.mixin.Mode): The mode indicating whether batching is enabled.
        etrace_bwg (Dict[ETraceWG_Key, brainstate.State]): A dictionary to store the
            eligibility trace states, keyed by a unique identifier for each
            weight group.
        relation (HiddenParamOpRelation): The relation object containing
            information about the weights and hidden groups involved in the
            computation.

    Raises:
        ValueError: If a relation with the same key has already been added to
            the eligibility trace states.
    """
    # For the relation
    #
    #   h1, h2, ... = f(x, w)
    #
    # we need to initialize the eligibility trace states for the weight x and the df.

    # TODO: assume the batch size is the first dimension
    y_shape = relation.y.aval.shape
    batch_size = y_shape[0] if mode.has(brainstate.mixin.Batching) else None
    for group in relation.hidden_groups:
        group: HiddenGroup
        bwg_key = etrace_param_key(relation.path, relation.y, group.index)
        if bwg_key in etrace_bwg:  # The key should be unique
            raise ValueError(f'The relation {bwg_key} has been added. ')
        etrace_bwg[bwg_key] = EligibilityTrace(
            jax.tree.map(
                partial(_batched_zeros_like, batch_size, group.num_state),
                relation.weight.value
            )
        )


def _update_param_dim_etrace_scan_fn(
    hist_etrace_vals: Dict[ETraceWG_Key, jax.Array],
    jacobians: Tuple[
        Dict[ETraceX_Key, jax.Array],  # the weight x
        Dict[ETraceDF_Key, jax.Array],  # the weight df
        Sequence[jax.Array],  # the hidden group Jacobians
    ],
    weight_path_to_vals: Dict[Path, PyTree],
    hidden_param_op_relations,
    mode: brainstate.mixin.Mode,
    normalize_matrix_spectrum: bool = False,
):
    """
    Update the eligibility trace values for parameter dimensions.

    This function updates the eligibility trace values for the parameter dimensions
    based on the provided Jacobians and the current mode. It computes the new eligibility
    trace values by applying vector-Jacobian products and incorporating the current
    Jacobian values.

    Args:
        hist_etrace_vals (Dict[ETraceWG_Key, jax.Array]): A dictionary containing
            historical eligibility trace values for the weight gradients, keyed by
            ETraceWG_Key.
        jacobians (Tuple[Dict[ETraceX_Key, jax.Array], Dict[ETraceDF_Key, jax.Array], Sequence[jax.Array]]):
            A tuple containing dictionaries of current Jacobian values for the weight x
            and df, and a sequence of hidden group Jacobians.
        weight_path_to_vals (Dict[Path, PyTree]): A dictionary mapping weight paths to
            their corresponding PyTree values.
        hidden_param_op_relations: A sequence of HiddenParamOpRelation objects representing
            the relationships between hidden parameters and operations.
        mode (brainstate.mixin.Mode): The mode indicating whether batching is enabled.

    Returns:
        Tuple[Dict[ETraceWG_Key, jax.Array], None]: A tuple containing a dictionary of
        updated eligibility trace values for the weight gradients, keyed by ETraceWG_Key,
        and None.
    """
    # --- the data --- #

    #
    # + "hist_etrace_vals" has the following structure:
    #    - key: the weight id, the weight-x jax var, the hidden state var
    #    - value: the batched weight gradients
    #

    # + "hid2weight_jac" has the following structure:
    #    - a dict of weight x gradients
    #       * key: the weight x jax var
    #       * value: the weight x gradients
    #    - a dict of weight y gradients
    #       * key: the tuple of the weight y jax var and the hidden state jax var
    #       * value: the weight y gradients
    #
    etrace_xs_at_t: Dict[ETraceX_Key, jax.Array] = jacobians[0]
    etrace_ys_at_t: Dict[ETraceDF_Key, jax.Array] = jacobians[1]

    #
    # the hidden-to-hidden Jacobians
    #
    hid_group_jacobians: Sequence[jax.Array] = jacobians[2]
    if normalize_matrix_spectrum:
        hid_group_jacobians = [_normalize_matrix_spectrum(diag) for diag in hid_group_jacobians]

    # The etrace weight gradients at the current time step.
    # i.e., The "hist_etrace_vals" at the next time step
    #
    new_etrace_bwg = dict()

    for relation in hidden_param_op_relations:
        relation: HiddenParamOpRelation

        #
        # Step 1:
        #
        # Necessary information for the etrace computation
        #
        # 1. the etrace operation for computing etrace updates
        # 2. the weight information
        # 3. the operator information
        #
        weight_path = relation.path
        weight_val = weight_path_to_vals[weight_path]
        etrace_op: ETraceOp = relation.weight.op
        if isinstance(relation.weight, ElemWiseParam):
            x = None
        else:
            x = etrace_xs_at_t[id(relation.x)]

        def comp_dw_with_x(x_, df_):
            """
            Computes the vector-Jacobian product (VJP) of the output with respect to the weight parameter.

            Args:
                x_: The input to the weight operation (can be None for element-wise parameters).
                df_: The cotangent (adjoint) vector for the output, used in the VJP computation.

            Returns:
                The VJP result, representing the gradient of the output with respect to the weight,
                contracted with the provided cotangent vector.
            """

            def to_y(w):
                # Returns the mantissa (unitless value) of the output of the weight operation.
                return u.get_mantissa(etrace_op.xw_to_y(x_, w))

            # Compute the VJP of to_y with respect to weight_val, evaluated at df_.
            return jax.vjp(to_y, weight_val)[1](df_)[0]

        @partial(jax.vmap, in_axes=-1, out_axes=-1)
        def comp_dw_without_x(df_):
            """
            Vectorized version of fn_dw for cases where x is not None.

            If batching is enabled, applies fn_dw over the batch dimension using jax.vmap.
            Otherwise, applies fn_dw directly.

            Args:
                df_: The cotangent (adjoint) vector for the output, used in the VJP computation.

            Returns:
                The VJP result(s) for the provided df_.
            """
            if mode.has(brainstate.mixin.Batching):
                return jax.vmap(comp_dw_with_x)(x, df_)
            else:
                return comp_dw_with_x(x, df_)

        for group in relation.hidden_groups:
            group: HiddenGroup

            #
            # Step 2:
            #
            # compute the current step weight gradients:
            #
            #       \partial h^t / \partial W^t = vjp(f(x, w))(df)
            #
            df = etrace_ys_at_t[etrace_df_key(relation.y, group.index)]
            # jax.debug.print('df = {g}', g=jax.tree.map(lambda x: jnp.abs(x).max(), df))

            #
            # vmap over the different hidden states,
            #
            # x: (n_input, ..., )
            # df: (n_hidden, ..., n_state)
            # phg_to_pw: (n_param, ..., n_state)
            phg_to_pw = comp_dw_without_x(df)
            phg_to_pw = jax.tree.map(_normalize_vector, phg_to_pw)
            # jax.debug.print('phg_to_pw = {g}', g=jax.tree.map(lambda x: jnp.abs(x).max(), phg_to_pw))

            #
            # Step 3:
            #
            # computing the following vector-Jacobian product:
            #  ϵ^t_{pre} = D_h ⊙ ϵ^{t-1}
            #
            # i.e., the hidden-to-hidden Jacobian diagonal matrix * the hidden df at the previous time step
            #
            #  ∂V^t/∂θ1 = ∂V^t/∂V^t-1 * ∂V^t-1/∂θ1 + ∂V^t/∂a^t-1 * ∂a^t-1/∂θ1 + ...
            #
            w_key = etrace_param_key(weight_path, relation.y, group.index)
            diag = hid_group_jacobians[group.index]

            #
            # vmap over j, over the different hidden states \partial h_i^t / \partial h_j^t
            #
            # d: (n_hidden, ..., [n_state])
            # old_bwg: (n_param, ..., [n_state])
            old_bwg = hist_etrace_vals[w_key]
            fn_bwg_pre = lambda d: _sum_dim(
                jax.vmap(etrace_op.yw_to_w, in_axes=-1, out_axes=-1)(d, old_bwg), axis=-1
            )
            if isinstance(relation.weight, ElemWiseParam) and mode.is_a(brainstate.mixin.Batching):
                raise NotImplementedError

            #
            # vmap over i, over the different hidden states \partial h_i^t / \partial h_j^t
            #
            # diag: (n_hidden, ..., [n_state], n_state)
            # old_bwg: (n_param, ..., n_state)
            # new_bwg_pre: (n_param, ..., n_state)
            new_bwg_pre = jax.vmap(fn_bwg_pre, in_axes=-2, out_axes=-1)(diag)

            #
            # Step 4:
            #
            # update: eligibility trace * hidden diagonal Jacobian + new hidden df
            #        ϵ^t = ϵ^t_{pre} + df^t, where D_h is the hidden-to-hidden Jacobian diagonal matrix.
            #
            new_bwg = jax.tree.map(u.math.add, new_bwg_pre, phg_to_pw, is_leaf=u.math.is_quantity)
            if normalize_matrix_spectrum:
                new_bwg = jax.tree.map(_normalize_vector, new_bwg)
            new_etrace_bwg[w_key] = new_bwg

    return new_etrace_bwg, None


def _normalize_matrix_spectrum(diag):
    def base_fn(matrix):
        # Compute the eigenvalues of the matrix
        eigenvalues = jnp.linalg.eigvals(matrix)

        # Get the maximum eigenvalue
        max_eigenvalue = jnp.max(jnp.abs(eigenvalues))

        # Normalize the matrix by dividing it by the maximum eigenvalue
        normalized_matrix = jax.lax.cond(
            max_eigenvalue > 1,
            lambda: matrix / max_eigenvalue,
            lambda: matrix,
        )
        return normalized_matrix

    fn = base_fn
    for i in range(diag.ndim - 2):
        fn = jax.vmap(fn)
    return fn(diag)


def _normalize_vector(v):
    max_elem = jnp.abs(v).max()
    normalized_vector = jax.lax.cond(
        max_elem > 1,
        lambda: v / max_elem,
        lambda: v,
    )

    # # Normalize the vector by dividing it by its norm
    # normalized_vector = v / jnp.linalg.norm(v)
    return normalized_vector


def _solve_param_dim_weight_gradients(
    hist_etrace_data: Dict[ETraceWG_Key, PyTree],  # the history etrace data
    dG_weights: Dict[Path, dG_Weight],  # weight gradients
    dG_hidden_groups: Sequence[jax.Array],  # hidden group gradients
    weight_hidden_relations: Sequence[HiddenParamOpRelation],
    mode: brainstate.mixin.Mode,
):
    """
    Compute and update the weight gradients for parameter dimensions using eligibility trace data.

    This function calculates the weight gradients by utilizing the eligibility trace data and the
    hidden-to-hidden Jacobians. It applies a correction factor to avoid exponential smoothing bias
    at the beginning of the computation.

    Args:
        hist_etrace_data (Dict[ETraceWG_Key, PyTree]): A dictionary containing historical eligibility
            trace data for the weight gradients, keyed by ETraceWG_Key.
        dG_weights (Dict[Path, dG_Weight]): A dictionary to store the computed weight gradients,
            keyed by the path of the weight.
        dG_hidden_groups (Sequence[jax.Array]): A sequence of hidden group gradients, with the same
            length as the total number of hidden groups.
        weight_hidden_relations (Sequence[HiddenParamOpRelation]): A sequence of HiddenParamOpRelation
            objects representing the relationships between hidden parameters and operations.
        mode (brainstate.mixin.Mode): The mode indicating whether batching is enabled.

    Returns:
        None: The function updates the dG_weights dictionary in place with the computed weight gradients.
    """
    # update the etrace weight gradients
    temp_data = dict()
    for relation in weight_hidden_relations:

        #
        # Step 1:
        #
        # Necessary information for the etrace computation
        #
        # 1. the etrace operation for computing etrace updates
        # 2. the weight information
        # 3. the operator information
        #
        weight_path = relation.path
        etrace_op: ETraceOp = relation.weight.op
        yw_to_w = jax.vmap(etrace_op.yw_to_w) if mode.has(brainstate.mixin.Batching) else etrace_op.yw_to_w

        for group in relation.hidden_groups:
            group: HiddenGroup

            #
            # Step 2:
            #
            # compute the weight gradients:
            #
            #   dE/dW = dE/dH * dH/dW, computing the final weight gradients
            #
            w_key = etrace_param_key(weight_path, relation.y, group.index)
            etrace_data = hist_etrace_data[w_key]
            dg_hidden = dG_hidden_groups[group.index]
            # dimensionless processing
            etrace_data, fn_unit_restore = _remove_units(etrace_data)
            dg_hidden, _ = _remove_units(dg_hidden)

            #
            # etrace_data: [n_batch, n_param, ..., n_state]
            #               or,
            #              [n_param, ..., n_state]
            # dg_hidden:   [n_batch, n_hidden, ..., n_state]
            #               or,
            #              [n_hidden, ..., n_state]
            dg_weight = _sum_dim(
                jax.vmap(yw_to_w, in_axes=-1, out_axes=-1)(dg_hidden, etrace_data)
            )
            # unit restoration
            dg_weight = fn_unit_restore(dg_weight)

            # update the weight gradients
            _update_dict(temp_data, weight_path, dg_weight)

    #
    # Step 3:
    #
    # sum up the batched weight gradients
    if mode.has(brainstate.mixin.Batching):
        for key, val in temp_data.items():
            temp_data[key] = jax.tree.map(lambda x: u.math.sum(x, axis=0), val)

    # update the weight gradients
    for key, val in temp_data.items():
        _update_dict(dG_weights, key, val)


def _remove_units(xs_maybe_quantity: brainstate.typing.PyTree):
    """
    Removes units from a PyTree of quantities, returning a unitless PyTree and a function to restore the units.

    This function traverses a PyTree structure, removing units from each quantity and returning a new PyTree
    with the same structure but without units. It also returns a function that can be used to restore the
    original units to the unitless PyTree.

    Args:
        xs_maybe_quantity (brainstate.typing.PyTree): A PyTree structure containing quantities with units.

    Returns:
        Tuple[brainstate.typing.PyTree, Callable]: A tuple containing:
            - A PyTree with the same structure as the input, but with units removed from each quantity.
            - A function that takes a unitless PyTree and restores the original units to it.
    """
    leaves, treedef = jax.tree.flatten(xs_maybe_quantity, is_leaf=u.math.is_quantity)
    new_leaves, units = [], []
    for leaf in leaves:
        leaf, unit = u.split_mantissa_unit(leaf)
        new_leaves.append(leaf)
        units.append(unit)

    def restore_units(xs_unitless: brainstate.typing.PyTree):
        leaves, treedef2 = jax.tree.flatten(xs_unitless)
        assert treedef == treedef2, 'The tree structure should be the same. '
        new_leaves = [
            leaf if unit.dim.is_dimensionless else leaf * unit
            for leaf, unit in zip(leaves, units)
        ]
        return jax.tree.unflatten(treedef, new_leaves)

    return jax.tree.unflatten(treedef, new_leaves), restore_units


class ParamDimVjpAlgorithm(ETraceVjpAlgorithm):
    r"""
    The online gradient computation algorithm with the diagonal approximation and the parameter dimension complexity.

    This algorithm computes the gradients of the weights with the diagonal approximation and the parameter dimension complexity.
    Its algorithm is based on the RTRL algorithm, and has the following learning rule:

    $$
    \begin{aligned}
    &\boldsymbol{\epsilon}^t \approx \mathbf{D}^t \boldsymbol{\epsilon}^{t-1}+\operatorname{diag}\left(\mathbf{D}_f^t\right) \otimes \mathbf{x}^t \\
    & \nabla_{\boldsymbol{\theta}} \mathcal{L}=\sum_{t^{\prime} \in \mathcal{T}} \frac{\partial \mathcal{L}^{t^{\prime}}}{\partial \mathbf{h}^{t^{\prime}}} \circ \boldsymbol{\epsilon}^{t^{\prime}}
    \end{aligned}
    $$

    For more details, please see `the D-RTRL algorithm presented in our manuscript <https://www.biorxiv.org/content/10.1101/2024.09.24.614728v2>`_.

    Note than the :py:class:`ParamDimVjpAlgorithm` is a subclass of :py:class:`brainstate.nn.Module`,
    and it is sensitive to the context/mode of the computation. Particularly,
    the :py:class:`ParamDimVjpAlgorithm` is sensitive to ``brainstate.mixin.Batching`` behavior.

    This algorithm has the :math:`O(B\theta)` memory complexity, where :math:`\theta` is the number of parameters,
    and :math:`B` the batch size.

    For a convolutional layer, the algorithm computes the weight gradients with the :math:`O(B\theta)`
    memory complexity, where :math:`\theta` is the dimension of the convolutional kernel.

    For a Linear transformation layer, the algorithm computes the weight gradients with the :math:`O(BIO)``
    computational complexity, where :math:`I` and :math:`O` are the number of input and output dimensions.

    Parameters
    -----------
    model: brainstate.nn.Module
        The model function, which receives the input arguments and returns the model output.
    vjp_method: str, optional
        The method for computing the VJP. It should be either "single-step" or "multi-step".

        - "single-step": The VJP is computed at the current time step, i.e., $\partial L^t/\partial h^t$.
        - "multi-step": The VJP is computed at multiple time steps, i.e., $\partial L^t/\partial h^{t-k}$,
          where $k$ is determined by the data input.
    name: str, optional
        The name of the etrace algorithm.
    mode: braintrace.mixin.Mode
        The computing mode, indicating the batching behavior.
    """

    # batch of weight gradients
    etrace_bwg: Dict[ETraceWG_Key, brainstate.State]

    def __init__(
        self,
        model: brainstate.nn.Module,
        mode: Optional[brainstate.mixin.Mode] = None,
        name: Optional[str] = None,
        vjp_method: str = 'single-step'
    ):
        super().__init__(model, name=name, vjp_method=vjp_method)

        # computing mode
        if mode is None:
            self.mode = brainstate.environ.get('mode', brainstate.mixin.Mode())
        else:
            self.mode = mode
        assert isinstance(self.mode, brainstate.mixin.Mode), (
            f'The mode should be an instance of brainstate.mixin.Mode. But we got {self.mode}.'
        )

    def init_etrace_state(self, *args, **kwargs):
        """
        Initialize the eligibility trace states of the etrace algorithm.

        This method is needed after compiling the etrace graph. See
        `.compile_graph()` for the details.
        """
        # The states of batched weight gradients
        self.etrace_bwg = dict()
        for relation in self.graph.hidden_param_op_relations:
            _init_param_dim_state(self.mode, self.etrace_bwg, relation)

    def reset_state(self, batch_size: int = None, **kwargs):
        """
        Reset the eligibility trace states.
        """
        self.running_index.value = 0
        _reset_state_in_a_dict(self.etrace_bwg, batch_size)

    def get_etrace_of(self, weight: brainstate.ParamState | Path) -> Dict:
        """
        Get the eligibility trace of the given weight.

        The eligibility trace contains the following structures:

        """

        self._assert_compiled()

        # get the wight id
        weight_id = (
            id(weight)
            if isinstance(weight, brainstate.ParamState) else
            id(self.graph_executor.path_to_states[weight])
        )

        find_this_weight = False
        etraces = dict()
        for relation in self.graph.hidden_param_op_relations:
            relation: HiddenParamOpRelation
            if id(relation.weight) != weight_id:
                continue
            find_this_weight = True

            # retrieve the etrace data
            for group in relation.hidden_groups:
                group: HiddenGroup
                key = etrace_param_key(relation.path, relation.y, group.index)
                etraces[key] = self.etrace_bwg[key].value

        if not find_this_weight:
            raise ValueError(f'Do not the etrace of the given weight: {weight}.')
        return etraces

    def _get_etrace_data(self) -> Dict:
        """Retrieve the current eligibility trace data from all trace states.

        This method collects all eligibility trace values from the internal state dictionary,
        extracting the current values from the brainstate.State objects that store them.
        It returns these values in a dictionary with the same keys as the original state
        dictionary, making the current trace values available for processing.

        This is an internal method used in the parameter dimension eligibility trace algorithm
        to access the current trace state for updates and gradient calculations.

        Returns:
            Dict[ETraceWG_Key, jax.Array]: A dictionary mapping eligibility trace keys to
                their current values. Each key represents a specific trace component
                (typically involving a parameter and hidden state relationship), and
                the corresponding value represents the accumulated eligibility trace.
        """
        return {
            k: v.value
            for k, v in self.etrace_bwg.items()
        }

    def _assign_etrace_data(self, etrace_vals: Dict) -> None:
        """Assign eligibility trace values to their corresponding state objects.

        This method updates the internal eligibility trace state dictionary (etrace_bwg)
        with new values from the provided dictionary. It iterates through each key-value
        pair in the input dictionary and assigns the value to the corresponding state
        object's value attribute.

        This is an implementation of the abstract method from the parent class,
        customized for the parameter dimension eligibility trace algorithm which
        stores traces in a single dictionary rather than separate ones for inputs
        and differential functions.

        Args:
            etrace_vals: Dict[ETraceWG_Key, jax.Array]
                Dictionary mapping eligibility trace keys to their updated values.
                Each key represents a specific parameter-hidden state relationship,
                and the value represents the updated eligibility trace value.

        Returns:
            None
        """
        for x, val in etrace_vals.items():
            self.etrace_bwg[x].value = val

    def _update_etrace_data(
        self,
        running_index: Optional[int],
        hist_etrace_vals: Dict[ETraceWG_Key, PyTree],
        hid2weight_jac_single_or_multi_times: Hid2WeightJacobian,
        hid2hid_jac_single_or_multi_times: HiddenGroupJacobian,
        weight_vals: Dict[Path, PyTree],
        input_is_multi_step: bool,
    ) -> Dict[ETraceWG_Key, PyTree]:
        """Update eligibility trace data for the parameter dimension-based algorithm.

        This method implements the core update equation for the D-RTRL algorithm's eligibility traces:

        ε^t ≈ D^t·ε^{t-1} + diag(D_f^t)⊗x^t

        It uses JAX's scan operation to efficiently process the historical trace values and
        combines them with current Jacobians to compute updated traces according to the
        parameter-dimension approximation approach.

        Args:
            running_index: Optional[int]
                Current timestep counter, used for correcting exponential smoothing bias.
            hist_etrace_vals: Dict[ETraceWG_Key, PyTree]
                Dictionary containing historical eligibility trace values from previous timestep.
                Keys are tuples identifying parameter-hidden state relationships.
            hid2weight_jac_single_or_multi_times: Hid2WeightJacobian
                Jacobians of hidden states with respect to weights at the current timestep.
                Contains input gradients and differential function gradients.
            hid2hid_jac_single_or_multi_times: HiddenGroupJacobian
                Jacobians between hidden states (recurrent connections) at the current timestep.
            weight_vals: Dict[Path, PyTree]
                Dictionary mapping paths to current weight values in the model.

        Returns:
            Dict[ETraceWG_Key, PyTree]: Updated eligibility trace values dictionary with the
                same structure as hist_etrace_vals but containing new values for the current timestep.
        """

        scan_fn = partial(
            _update_param_dim_etrace_scan_fn,
            weight_path_to_vals=weight_vals,
            hidden_param_op_relations=self.graph.hidden_param_op_relations,
            mode=self.mode,
        )

        if input_is_multi_step:
            new_etrace = jax.lax.scan(
                scan_fn,
                hist_etrace_vals,
                (
                    hid2weight_jac_single_or_multi_times[0],
                    hid2weight_jac_single_or_multi_times[1],
                    hid2hid_jac_single_or_multi_times,
                )
            )[0]

        else:
            new_etrace = scan_fn(
                hist_etrace_vals,
                (
                    hid2weight_jac_single_or_multi_times[0],
                    hid2weight_jac_single_or_multi_times[1],
                    hid2hid_jac_single_or_multi_times,
                )
            )[0]

        return new_etrace

    def _solve_weight_gradients(
        self,
        running_index: int,
        etrace_h2w_at_t: Dict[ETraceWG_Key, PyTree],
        dl_to_hidden_groups: Sequence[jax.Array],
        weight_vals: Dict[WeightID, PyTree],
        dl_to_nonetws_at_t: Dict[Path, PyTree],
        dl_to_etws_at_t: Optional[Dict[Path, PyTree]],
    ):
        """Compute weight gradients using parameter dimension eligibility traces.

        This method implements the parameter dimension D-RTRL algorithm's weight gradient
        computation. It combines the eligibility traces with the gradients of the loss
        with respect to hidden states to compute the full parameter gradients according to:

        ∇_θ L = ∑_{t' ∈ T} ∂L^{t'}/∂h^{t'} ∘ ε^{t'}

        Where ε represents the eligibility traces and ∂L/∂h are the gradients of the loss
        with respect to hidden states.

        Args:
            running_index: int
                Current timestep counter used for bias correction.
            etrace_h2w_at_t: Dict[ETraceWG_Key, PyTree]
                Eligibility trace values at the current timestep, mapping parameter-hidden
                state relationship keys to trace values.
            dl_to_hidden_groups: Sequence[jax.Array]
                Gradients of the loss with respect to hidden states at the current timestep.
            weight_vals: Dict[WeightID, PyTree]
                Current values of all weights in the model.
            dl_to_nonetws_at_t: Dict[Path, PyTree]
                Gradients of non-eligibility trace parameters at the current timestep.
            dl_to_etws_at_t: Optional[Dict[Path, PyTree]]
                Optional additional gradients for eligibility trace parameters at the
                current timestep.

        Returns:
            Dict[Path, PyTree]: Dictionary mapping parameter paths to their gradient values.
        """
        dG_weights = {path: None for path in self.param_states}

        # update the etrace weight gradients
        _solve_param_dim_weight_gradients(
            etrace_h2w_at_t,
            dG_weights,
            dl_to_hidden_groups,
            self.graph.hidden_param_op_relations,
            self.mode,
        )

        # update the non-etrace weight gradients
        for path, dg in dl_to_nonetws_at_t.items():
            _update_dict(dG_weights, path, dg)

        # update the etrace parameters when "dl_to_etws_at_t" is not None
        if dl_to_etws_at_t is not None:
            for path, dg in dl_to_etws_at_t.items():
                _update_dict(dG_weights, path, dg, error_when_no_key=True)
        return dG_weights


D_RTRL = ParamDimVjpAlgorithm
