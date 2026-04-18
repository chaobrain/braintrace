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
import saiunit as u
import jax
import jax.numpy as jnp

from braintrace._etrace_algorithms import EligibilityTrace
from braintrace._etrace_compiler import HiddenParamOpRelation, HiddenGroup
from braintrace._etrace_op import (
    etp_elemwise_p,
    ETP_RULES_YW_TO_W,
    ETP_RULES_XY_TO_DW,
    ETP_RULES_INIT_DRTRL,
    is_batched_primitive,
)
from braintrace._misc import etrace_df_key
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
from .misc import (
    _extract_leaf,
    _reset_state_in_a_dict,
    _route_grads_by_path,
    _sum_dim,
    _update_dict,
)

__all__ = [
    'ParamDimVjpAlgorithm',
    'D_RTRL',
]


def _init_param_dim_state(
    etrace_bwg: Dict[ETraceWG_Key, brainstate.State],
    relation: HiddenParamOpRelation
):
    """
    Initialize the eligibility trace states for parameter dimensions.

    Traces are stored as ``Dict[str, Array]`` keyed by the primitive's
    trainable-input names (dict-based rule API).
    """
    for group in relation.hidden_groups:
        group: HiddenGroup
        bwg_key = (id(relation.y_var), group.index)
        if bwg_key in etrace_bwg:
            raise ValueError(f'The relation {bwg_key} has been added. ')
        init_fn = ETP_RULES_INIT_DRTRL[relation.primitive]
        init_val = init_fn(
            relation.x_var,
            relation.y_var,
            relation.trainable_vars,
            group.num_state,
        )
        if not isinstance(init_val, dict):
            raise TypeError(
                f'Primitive {relation.primitive.name} init_drtrl must return a dict; '
                f'got {type(init_val).__name__}.'
            )
        etrace_bwg[bwg_key] = EligibilityTrace(init_val)


def _update_param_dim_etrace_scan_fn(
    hist_etrace_vals: Dict[ETraceWG_Key, jax.Array],
    jacobians: Tuple[
        Dict[ETraceX_Key, jax.Array],  # the weight x
        Dict[ETraceDF_Key, jax.Array],  # the weight df
        Sequence[jax.Array],  # the hidden group Jacobians
    ],
    weight_path_to_vals: Dict[Path, PyTree],
    hidden_param_op_relations,
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

        # Build the weights dict the rules consume.
        weights_dict = {
            key: _extract_leaf(
                weight_path_to_vals[relation.trainable_paths[key]],
                relation.trainable_leaf_indices[key],
            )
            for key in relation.trainable_vars
        }

        xy_to_dw_rule = ETP_RULES_XY_TO_DW[relation.primitive]
        yw_to_w_rule = ETP_RULES_YW_TO_W[relation.primitive]
        eqn_params = relation.eqn_params
        is_elemwise = relation.primitive is etp_elemwise_p
        batched = is_batched_primitive(relation.primitive)

        if is_elemwise:
            x = None
        else:
            x = etrace_xs_at_t[id(relation.x_var)]

        def _call_xy_to_dw_dict(x_, df_, weights_, _rule=xy_to_dw_rule, _params=eqn_params):
            return _rule(x_, df_, weights_, **_params)

        def _call_yw_to_w_dict(d, trace_, _rule=yw_to_w_rule, _params=eqn_params):
            return _rule(d, trace_, **_params)

        def comp_dw_with_x(x_, df_, _wdict=weights_dict):
            return _call_xy_to_dw_dict(x_, df_, _wdict)

        @partial(jax.vmap, in_axes=-1, out_axes=-1)
        def comp_dw_without_x(df_, _x=x, _batched=batched):
            if _batched:
                return jax.vmap(comp_dw_with_x)(_x, df_)
            else:
                return comp_dw_with_x(_x, df_)

        for group in relation.hidden_groups:
            group: HiddenGroup

            df = etrace_ys_at_t[etrace_df_key(relation.y, group.index)]

            # phg_to_pw is a Dict[str, Array].
            phg_to_pw = comp_dw_without_x(df)
            phg_to_pw = jax.tree.map(_normalize_vector, phg_to_pw)

            w_key = (id(relation.y_var), group.index)
            diag = hid_group_jacobians[group.index]

            old_bwg = hist_etrace_vals[w_key]   # Dict[str, Array]

            fn_bwg_pre = lambda d, _old=old_bwg: jax.tree.map(
                lambda arr: _sum_dim(arr, axis=-1),
                jax.vmap(
                    _call_yw_to_w_dict, in_axes=-1, out_axes=-1,
                )(d, _old),
            )

            new_bwg_pre = jax.vmap(fn_bwg_pre, in_axes=-2, out_axes=-1)(diag)

            # new_bwg_pre + phg_to_pw per-leaf.
            new_bwg = jax.tree.map(
                u.math.add, new_bwg_pre, phg_to_pw, is_leaf=u.math.is_quantity,
            )
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
    weight_vals: Dict[Path, PyTree],  # current ParamState pytree values for structure
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
    temp_data: Dict[Path, PyTree] = dict()
    for relation in weight_hidden_relations:
        yw_to_w_rule = ETP_RULES_YW_TO_W[relation.primitive]
        eqn_params = relation.eqn_params
        batched = is_batched_primitive(relation.primitive)

        def _call_yw_to_w_dict(d, trace_, _rule=yw_to_w_rule, _params=eqn_params):
            return _rule(d, trace_, **_params)

        yw_to_w = (
            jax.vmap(_call_yw_to_w_dict)
            if batched
            else _call_yw_to_w_dict
        )

        for group in relation.hidden_groups:
            group: HiddenGroup

            w_key = (id(relation.y_var), group.index)
            etrace_data = hist_etrace_data[w_key]   # Dict[str, Array]
            dg_hidden = dG_hidden_groups[group.index]

            # dimensionless processing (unit strip + restore). Apply per-leaf.
            etrace_data_unitless, fn_unit_restore = _remove_units(etrace_data)
            dg_hidden_unitless, _ = _remove_units(dg_hidden)

            # dg_weight_dict: Dict[str, Array]
            dg_weight_dict = jax.tree.map(
                lambda arr: _sum_dim(arr, axis=-1),
                jax.vmap(yw_to_w, in_axes=-1, out_axes=-1)(
                    dg_hidden_unitless, etrace_data_unitless
                ),
            )
            dg_weight_dict = fn_unit_restore(dg_weight_dict)

            # Route per-key to owning ParamState path.
            _route_grads_by_path(relation, dg_weight_dict, weight_vals, temp_data)

    #
    # Step 3:
    #
    # sum up the batched weight gradients
    # Check if ANY relation uses a batched primitive
    has_batched = any(is_batched_primitive(r.primitive) for r in weight_hidden_relations)
    if has_batched:
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
        name: Optional[str] = None,
        vjp_method: str = 'single-step',
        **kwargs,
    ):
        super().__init__(model, name=name, vjp_method=vjp_method)

    def init_etrace_state(self, *args, **kwargs):
        """
        Initialize the eligibility trace states of the etrace algorithm.

        This method is needed after compiling the etrace graph. See
        `.compile_graph()` for the details.
        """
        # The states of batched weight gradients
        self.etrace_bwg = dict()
        for relation in self.graph.hidden_param_op_relations:
            _init_param_dim_state(self.etrace_bwg, relation)

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
            primary_state = next(iter(relation.trainable_param_states.values()), None)
            if primary_state is None or id(primary_state) != weight_id:
                continue
            find_this_weight = True

            # retrieve the etrace data
            for group in relation.hidden_groups:
                group: HiddenGroup
                key = (id(relation.y_var), group.index)
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
            weight_vals,
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
