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

from collections import defaultdict
from functools import partial
from typing import Dict, Tuple, List, Optional, Sequence

import brainstate
import brainunit as u
import jax

from braintrace._etrace_compiler_hid_param_op import HiddenParamOpRelation
from braintrace._etrace_compiler_hidden_group import HiddenGroup
from braintrace._etrace_concepts import (
    ETraceParam,
    ElemWiseParam,
    ETraceGrad,
)
from braintrace._misc import (
    etrace_x_key,
    etrace_param_key,
    etrace_df_key,
)
from braintrace._typing import (
    PyTree,
    Path,
    ETraceX_Key,
    ETraceDF_Key,
    ETraceWG_Key,
    Hid2WeightJacobian,
    Hid2HidJacobian,
)
from .base import ETraceVjpAlgorithm
from .d_rtrl import (
    _init_param_dim_state,
    _update_param_dim_etrace_scan_fn,
    _solve_param_dim_weight_gradients,
)
from .esd_rtrl import (
    _init_IO_dim_state,
    _update_IO_dim_etrace_scan_fn,
    _solve_IO_dim_weight_gradients,
    _format_decay_and_rank,
)
from .misc import _reset_state_in_a_dict, _update_dict

__all__ = ['ETraceVjpAlgorithm']


def _numel(pytree: PyTree):
    """
    Calculate the total number of elements in a PyTree.

    This function traverses a PyTree structure and sums up the number of elements
    in each array contained within the PyTree.

    Args:
        pytree (PyTree): A PyTree structure, which can be a nested combination of
                         lists, tuples, and dictionaries containing JAX arrays.

    Returns:
        int: The total number of elements across all arrays in the PyTree.
    """
    return sum(u.math.size(x) for x in jax.tree_leaves(pytree))


def _is_weight_need_full_grad(
    relation: HiddenParamOpRelation,
    mode: brainstate.mixin.Mode
):
    """
    Determine whether the weight requires a full gradient computation.

    This function evaluates the type of gradient computation needed for a given weight
    based on its characteristics and the current mode. It decides whether to use an
    O(n^2) algorithm for full gradient computation or an O(n) algorithm for approximate
    gradient computation.

    Args:
        relation (HiddenParamOpRelation): The relation object containing information
            about the weights and hidden groups involved in the computation.
        mode (brainstate.mixin.Mode): The mode indicating whether batching is enabled.

    Returns:
        bool: True if the weight requires a full gradient computation using the O(n^2)
        algorithm, False if an approximate gradient computation using the O(n) algorithm
        is sufficient.
    """
    if isinstance(relation.weight, ETraceParam):
        #
        # When
        #     weight.gradient == ETraceGrad.full
        #
        # the weights will be forced to use O(n^2) algorithm
        # to compute the eligibility trace.
        #
        if relation.weight.gradient == ETraceGrad.full:
            return True

        #
        # When
        #     weight.gradient == ETraceGrad.approx
        #
        # the weights will be forced to use O(n) algorithm
        # to compute the eligibility trace.
        #
        if relation.weight.gradient == ETraceGrad.approx:
            return False

    if isinstance(relation.weight, ElemWiseParam):
        #
        # When
        #     weight is an element-wise parameter
        #
        # the weights will be forced to use O(n^2) algorithm
        # to compute the eligibility trace.
        #
        return True

    batch_size = relation.x.aval.shape[0] if mode.has(brainstate.mixin.Batching) else 1
    if _numel(relation.x) + _numel(relation.y) > batch_size * _numel(relation.weight.value):
        #
        # When the number of elements in the inputs and outputs are bigger than the weight number,
        # we will use the O(n^2) algorithm to compute the eligibility trace, since
        # storing the batched weight gradients will be less expensive.
        #
        return True
    else:
        #
        # For most cases, we will use the O(n) algorithm to compute the eligibility trace.
        # Since the number of elements in input and output (I + O) is greatly less than the number
        # of elements in the weight (W = I * O).
        #
        return False


class HybridDimVjpAlgorithm(ETraceVjpAlgorithm):
    r"""
    The hybrid online gradient computation algorithm with the diagonal approximation and hybrid complexity.

    Similar to :py:class:`ParamDimVjpAlgorithm`, :py:class:`HybridDimVjpAlgorithm` is a subclass of
    :py:class:`brainstate.nn.Module`, and it is sensitive to the context/mode of the computation.
    Particularly, the :py:class:`ParamDimVjpAlgorithm` is sensitive to ``brainstate.mixin.Batching`` behavior.

    For a function :math:`O = f(I, \theta)`, where :math:`I` is the input, :math:`\theta` is the parameters,
    and :math:`O` is the output, the algorithm computes the weight gradients with the ``O(BI + BO)`` memory complexity
    when :math:`I + O < \theta`, or the ``O(B\theta)`` memory complexity when :math:`I + O \geq \theta`.

    This means that the algorithm combine the memory efficiency of the :py:class:`ParamDimVjpAlgorithm` and the
    computational efficiency of the :py:class:`IODimVjpAlgorithm` together.

    Parameters
    -----------
    model: Callable
        The model function, which receives the input arguments and returns the model output.
    vjp_method: str, optional
        The method for computing the VJP. It should be either "single-step" or "multi-step".

        - "single-step": The VJP is computed at the current time step, i.e., $\partial L^t/\partial h^t$.
        - "multi-step": The VJP is computed at multiple time steps, i.e., $\partial L^t/\partial h^{t-k}$,
          where $k$ is determined by the data input.
    name: str, optional
        The name of the etrace algorithm.
    decay_or_rank: float, int
        The exponential smoothing factor for the eligibility trace. If it is a float,
        it is the decay factor, should be in the range of (0, 1). If it is an integer,
        it is the number of approximation rank for the algorithm, should be greater than 0.
    mode: braintrace.mixin.Mode
        The computing mode, indicating the batching behavior.
    """

    # the spatial gradients of the weights
    etrace_xs: Dict[ETraceX_Key, brainstate.State]

    # the spatial gradients of the hidden states
    etrace_dfs: Dict[ETraceDF_Key, brainstate.State]

    # the mapping from the etrace x to the weight operations
    etrace_xs_to_weights = Dict[ETraceX_Key, List[Path]]

    # batch of weight gradients
    etrace_bwg: Dict[ETraceWG_Key, brainstate.State]

    # the exponential smoothing decay factor
    decay: float

    def __init__(
        self,
        model: brainstate.nn.Module,
        decay_or_rank: float | int,
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
        assert isinstance(self.mode, brainstate.mixin.Mode), 'The mode should be an instance of brainstate.mixin.Mode.'

        # the learning parameters
        self.decay, num_rank = _format_decay_and_rank(decay_or_rank)

    def init_etrace_state(self, *args, **kwargs):
        """
        Initialize the eligibility trace states of the etrace algorithm.

        This method is needed after compiling the etrace graph. See
        `.compile_graph()` for the details.
        """
        #
        # The states of weight spatial gradients:
        #   1. x
        #   2. df
        #   3. batched weight gradients
        #
        self.etrace_xs = dict()
        self.etrace_dfs = dict()
        self.etrace_bwg = dict()
        self.etrace_xs_to_weights = defaultdict(list)

        for relation in self.graph.hidden_param_op_relations:
            if _is_weight_need_full_grad(relation, self.mode):
                _init_param_dim_state(
                    self.mode,
                    self.etrace_bwg,
                    relation
                )
            else:
                _init_IO_dim_state(
                    self.etrace_xs,
                    self.etrace_dfs,
                    self.etrace_xs_to_weights,
                    self.graph_executor.state_id_to_path,
                    relation,
                    self.mode,
                )

    def reset_state(self, batch_size: int = None, **kwargs):
        """
        Reset the eligibility trace states.

        This function resets the internal state of the eligibility traces, which are used
        in the computation of gradients in the etrace algorithm. It is typically called
        at the beginning of a new batch or sequence to ensure that the state is clean.

        Parameters
        -----------
        batch_size : int, optional
            The size of the batch for which the state is being reset. If not provided,
            the default behavior is to reset the state without considering batch size.

        **kwargs
            Additional keyword arguments that may be used for resetting the state.
            These are not explicitly used in this function but can be passed for
            compatibility with other functions or methods that require them.

        Returns:
        --------
        None
            This function does not return any value. It performs an in-place reset
            of the internal state variables.
        """
        self.running_index.value = 0
        _reset_state_in_a_dict(self.etrace_xs, batch_size)
        _reset_state_in_a_dict(self.etrace_dfs, batch_size)
        _reset_state_in_a_dict(self.etrace_bwg, batch_size)

    def get_etrace_of(self, weight: brainstate.ParamState | Path) -> Tuple[Dict, Dict, Dict]:
        """
        Retrieve the eligibility trace for a specified weight.

        This function extracts the eligibility trace data associated with a given weight,
        which includes the spatial gradients of the weight inputs, the spatial gradients
        of the hidden states, and the batched weight gradients.

        Parameters
        -----------
        weight : brainstate.ParamState | Path
            The weight for which the eligibility trace is to be retrieved. It can be
            specified either as a `brainstate.ParamState` object or a `Path` object.

        Returns:
        --------
        Tuple[Dict, Dict, Dict]
            A tuple containing three dictionaries:
            - etrace_xs: The spatial gradients of the weight inputs.
            - etrace_dfs: The spatial gradients of the hidden states.
            - etrace_bws: The batched weight gradients.

        Raises:
        -------
        ValueError
            If the eligibility trace for the specified weight cannot be found.
        """

        self._assert_compiled()

        # the weight ID
        weight_id = (
            id(weight)
            if isinstance(weight, brainstate.ParamState) else
            id(self.graph_executor.path_to_states[weight])
        )

        etrace_xs = dict()
        etrace_dfs = dict()
        etrace_bws = dict()
        find_this_weight = False
        for relation in self.graph.hidden_param_op_relations:
            relation: HiddenParamOpRelation
            if id(relation.weight) != weight_id:
                continue
            find_this_weight = True

            wx_var = etrace_x_key(relation.x)
            if wx_var in self.etrace_xs:
                # get the weight_op input
                etrace_xs[wx_var] = self.etrace_xs[wx_var].value

                # get the weight_op df
                for group in relation.hidden_groups:
                    group: HiddenGroup
                    df_key = etrace_df_key(relation.y, group.index)
                    etrace_dfs[df_key] = self.etrace_dfs[df_key].value

            # get the batched weight gradients
            for group in relation.hidden_groups:
                group: HiddenGroup
                bwg_key = etrace_param_key(relation.path, relation.y, group.index)
                if bwg_key in self.etrace_bwg:
                    etrace_bws[bwg_key] = self.etrace_bwg[bwg_key].value

        if not find_this_weight:
            raise ValueError(f'Do not the etrace of the given weight: {weight}.')

        return etrace_xs, etrace_dfs, etrace_bws

    def _get_etrace_data(self) -> Tuple[Dict, ...]:
        """Retrieve all eligibility trace data from internal state dictionaries.

        This method collects the current eligibility trace values from all three internal
        state dictionaries that store different components of the trace information:
        - etrace_xs: Input spatial gradients
        - etrace_dfs: Hidden state differential function values
        - etrace_wgrads: Weight gradients in parameter dimension

        It extracts the current values from the brainstate.State objects and returns them
        as a tuple of dictionaries with the same structure as the state dictionaries.

        This method is used internally during the update process to provide the current
        trace state for computation in the hybrid dimension algorithm.

        Returns:
            Tuple[Dict, ...]: A tuple containing three dictionaries:
                - Input spatial gradients (etrace_xs)
                - Hidden state differential values (etrace_dfs)
                - Weight gradients (etrace_wgrads)
        """
        etrace_xs = {x: val.value for x, val in self.etrace_xs.items()}
        etrace_dfs = {x: val.value for x, val in self.etrace_dfs.items()}
        etrace_wgrads = {x: val.value for x, val in self.etrace_bwg.items()}
        return etrace_xs, etrace_dfs, etrace_wgrads

    def _assign_etrace_data(self, etrace_vals: Sequence[Dict]) -> None:
        """Assign eligibility trace values to their corresponding state objects.

        This method updates the eligibility trace states with new values provided in the input
        dictionary. For the parameter dimension algorithm, it iterates through each key-value
        pair in the input dictionary and assigns the value to the corresponding state's value
        attribute in the etrace_bwg dictionary.

        This is an implementation of the abstract method from the parent ETraceVjpAlgorithm class,
        customized for storing traces specific to this algorithm's approach.

        Args:
            etrace_vals: Sequence[Dict]
                Dictionary mapping eligibility trace keys to their updated values.
                Each key identifies a specific weight-hidden state relationship,
                and the value contains the updated trace information.
        """
        etrace_xs, etrace_dfs, etrace_wgrads = etrace_vals
        for x, val in etrace_xs.items():
            self.etrace_xs[x].value = val
        for x, val in etrace_dfs.items():
            self.etrace_dfs[x].value = val
        for x, val in etrace_wgrads.items():
            self.etrace_bwg[x].value = val

    def _update_etrace_data(
        self,
        running_index: Optional[int],
        hist_etrace_vals: Tuple[Dict, ...],
        hid2weight_jac_single_or_multi_times: Hid2WeightJacobian,
        hid2hid_jac_single_or_multi_times: Hid2HidJacobian,
        weight_vals: Dict[Path, PyTree],
        input_is_multi_step: bool,
    ) -> Tuple[Dict, ...]:
        """
        Update eligibility trace data for the hybrid dimension algorithm.

        This method combines the approaches from both IO dimension and parameter dimension
        algorithms to update eligibility traces. It decides which algorithm to use for each
        weight-hidden relationship based on the complexity characteristics of each operation.

        The hybrid algorithm chooses between:
        - IO dimension approach (O(BI+BO) complexity) when I+O < theta
        - Parameter dimension approach (O(B*theta) complexity) when I+O >= theta

        Where B is batch size, I is input dimensions, O is output dimensions, and theta is
        the number of parameters.

        Args:
            running_index: Optional[int]
                Current timestep counter used for decay corrections.
            hist_etrace_vals: Tuple[Dict, ...]
                Historical eligibility trace values as a tuple containing three dictionaries:
                (etrace_xs, etrace_dfs, etrace_wgrads) for input traces, differential function
                traces, and weight gradient traces respectively.
            hid2weight_jac_single_or_multi_times: Hid2WeightJacobian
                Jacobians of hidden states with respect to weights at current timestep.
            hid2hid_jac_single_or_multi_times: Hid2HidJacobian
                Jacobians of hidden states with respect to previous hidden states.
            weight_vals: Dict[Path, PyTree]
                Current values of all weights in the model.

        Returns:
            Tuple[Dict, ...]: Updated eligibility trace values as a tuple of three dictionaries
                containing the updated traces for inputs, differential functions, and weight
                gradients, maintaining the same structure as the input hist_etrace_vals.
        """

        # the history etrace values
        hist_xs, hist_dfs, hist_bwg = hist_etrace_vals

        # ---- separate the etrace gradients into two parts --- #
        #
        #  1. O(n^2) etrace gradients
        #  2. O(n) etrace gradients
        #

        on_weight_hidden_relations = []
        on2_weight_hidden_relations = []
        for relation in self.graph.hidden_param_op_relations:
            if _is_weight_need_full_grad(relation, self.mode):
                on2_weight_hidden_relations.append(relation)
            else:
                on_weight_hidden_relations.append(relation)

        scan_fn_on2 = partial(
            _update_param_dim_etrace_scan_fn,
            weight_path_to_vals=weight_vals,
            hidden_param_op_relations=on2_weight_hidden_relations,
            mode=self.mode,
        )
        scan_fn_on = partial(
            _update_IO_dim_etrace_scan_fn,
            hid_weight_op_relations=self.graph.hidden_param_op_relations,
            decay=self.decay,
        )

        if input_is_multi_step:
            # ---- O(n^2) etrace gradients update ---- #
            new_bwg = jax.lax.scan(
                scan_fn_on2,
                hist_bwg,
                (
                    hid2weight_jac_single_or_multi_times[0],
                    hid2weight_jac_single_or_multi_times[1],
                    hid2hid_jac_single_or_multi_times,
                )
            )[0]

            # ---- O(n) etrace gradients update ---- #
            new_xs, new_dfs = jax.lax.scan(
                scan_fn_on,
                (hist_xs, hist_dfs),
                (
                    hid2weight_jac_single_or_multi_times[0],
                    hid2weight_jac_single_or_multi_times[1],
                    hid2hid_jac_single_or_multi_times,
                ),
            )[0]

        else:
            # ---- O(n^2) etrace gradients update ---- #
            new_bwg = scan_fn_on2(
                hist_bwg,
                (
                    hid2weight_jac_single_or_multi_times[0],
                    hid2weight_jac_single_or_multi_times[1],
                    hid2hid_jac_single_or_multi_times,
                ),
            )[0]

            # ---- O(n) etrace gradients update ---- #
            new_xs, new_dfs = scan_fn_on(
                (hist_xs, hist_dfs),
                (
                    hid2weight_jac_single_or_multi_times[0],
                    hid2weight_jac_single_or_multi_times[1],
                    hid2hid_jac_single_or_multi_times,
                ),
            )[0]

        return new_xs, new_dfs, new_bwg

    def _solve_weight_gradients(
        self,
        running_index: int,
        etrace_h2w_at_t: Tuple,
        dl_to_hidden_groups: Sequence[jax.Array],
        weight_vals: Dict[Path, PyTree],
        dl_to_nonetws_at_t: Dict[Path, PyTree],
        dl_to_etws_at_t: Optional[Dict[Path, PyTree]],
    ):
        """
        Solve the weight gradients according to the eligibility trace data.

        Particularly, for each weight, we compute its gradients according to the batched weight gradients.
        """

        #
        # dl_to_hidden_groups:
        #         The gradients of the loss-to-hidden-group at the time "t".
        #         It has the shape of [n_hidden, ..., n_state].
        #         - `l` is the loss,
        #         - `h` is the hidden group,
        #
        # dl_to_nonetws_at_t:
        #         The gradients of the loss-to-non-etrace parameters
        #         at the time "t", i.e., ∂L^t / ∂W^t.
        #         It has the shape of [n_param, ...].
        #
        # dl_to_etws_at_t:
        #         The gradients of the loss-to-etrace parameters
        #         at the time "t", i.e., ∂L^t / ∂W^t.
        #         It has the shape of [n_param, ...].
        #

        xs, dfs, wgrads = etrace_h2w_at_t
        dG_weights = {path: None for path in self.param_states.keys()}

        # ---- separate the etrace gradients into two parts --- #
        #
        #  1. O(n^2) etrace gradients
        #  2. O(n) etrace gradients
        #

        on_weight_hidden_relations = []
        on2_weight_hidden_relations = []
        for relation in self.graph.hidden_param_op_relations:
            if _is_weight_need_full_grad(relation, self.mode):
                on2_weight_hidden_relations.append(relation)
            else:
                on_weight_hidden_relations.append(relation)

        # --- update the etrace weight gradients by the O(n) algorithm --- #

        _solve_IO_dim_weight_gradients(
            (xs, dfs),
            dG_weights,
            dl_to_hidden_groups,
            on_weight_hidden_relations,
            weight_vals,
            running_index,
            self.decay,
            self.mode,
        )

        # --- update the etrace weight gradients by the O(n^2) algorithm --- #

        _solve_param_dim_weight_gradients(
            wgrads,
            dG_weights,
            dl_to_hidden_groups,
            on2_weight_hidden_relations,
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
