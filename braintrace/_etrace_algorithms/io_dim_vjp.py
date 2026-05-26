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
# Date: 2024-04-03
# Copyright: 2024, Chaoming Wang
#
# Refinement History:
#    [2025-02-06]
#       - [x] split into "_etrace_algorithms.py" and "_etrace_vjp_algorithms.py"
#
# ==============================================================================

# -*- coding: utf-8 -*-

from functools import partial
from typing import Dict, Tuple, Optional, Sequence, Any

import brainstate
import jax
import jax.numpy as jnp
import saiunit as u

from braintrace._etrace_compiler import HiddenGroup, HiddenParamOpRelation
from braintrace._etrace_op import (
    etp_elemwise_p,
    ETP_RULES_XY_TO_DW,
    ETP_RULES_INIT_PP,
    is_batched_primitive,
)
from braintrace._misc import (
    check_dict_keys,
    etrace_x_key,
    etrace_df_key,
)
from braintrace._typing import (
    PyTree,
    WeightVals,
    Path,
    ETraceX_Key,
    ETraceDF_Key,
    Hid2WeightJacobian,
    HiddenGroupJacobian,
    dG_Weight,
)
from ._common import (
    _extract_leaf,
    _reset_state_in_a_dict,
    _route_grads_by_path,
    _sum_dim,
    _update_dict,
)
from .base import EligibilityTrace
from .vjp_base import ETraceVjpAlgorithm

__all__ = [
    'IODimVjpAlgorithm',
]


def _format_decay_and_rank(decay_or_rank) -> Tuple[float, int]:
    """
    Determines the decay factor and the number of approximation ranks based on the input.

    This function takes either a decay factor or a number of approximation ranks as input
    and returns both the decay factor and the number of approximation ranks. If the input
    is a float, it is treated as a decay factor, and the number of ranks is calculated.
    If the input is an integer, it is treated as the number of ranks, and the decay factor
    is calculated.

    Args:
        decay_or_rank (float or int): The decay factor (a float between 0 and 1) or the
                                      number of approximation ranks (a positive integer).

    Returns:
        Tuple[float, int]: A tuple containing the decay factor and the number of approximation ranks.

    Raises:
        ValueError: If the input is neither a float nor an integer, or if the float is not in the range (0, 1),
                    or if the integer is not greater than 0.
    """
    # number of approximation rank and the decay factor
    if isinstance(decay_or_rank, float):
        assert 0 < decay_or_rank < 1, f'The decay should be in (0, 1). While we got {decay_or_rank}. '
        decay = decay_or_rank  # (num_rank - 1) / (num_rank + 1)
        num_rank = round(2. / (1 - decay) - 1)
    elif isinstance(decay_or_rank, int):
        assert decay_or_rank > 0, f'The num_rank should be greater than 0. While we got {decay_or_rank}. '
        num_rank = decay_or_rank
        decay = (num_rank - 1) / (num_rank + 1)  # (num_rank - 1) / (num_rank + 1)
    else:
        raise ValueError('Please provide "num_rank" (int) or "decay" (float, 0 < decay < 1). ')
    return decay, num_rank


def _expon_smooth(old, new, decay):
    """
    Apply exponential smoothing to update a value.

    This function performs exponential smoothing, which is a technique used to
    smooth out data by applying a decay factor to the old value and combining it
    with the new value. If the new value is None, the function returns the old
    value scaled by the decay factor.

    Args:
        old: The old value to be smoothed.
        new: The new value to be incorporated into the smoothing. If None, only
             the old value scaled by the decay factor is returned.
        decay: The decay factor, a float between 0 and 1, that determines the
               weight of the old value in the smoothing process.

    Returns:
        The smoothed value, which is a combination of the old and new values
        weighted by the decay factor.
    """
    if new is None:
        return decay * old
    return decay * old + (1 - decay) * new


def _low_pass_filter(old, new, alpha):
    """
    Apply a low-pass filter to smooth the transition between old and new values.

    This function implements a simple low-pass filter, which is used to smooth
    out fluctuations in data by blending the old value with the new value based
    on a specified filter factor.

    Parameters
    ----------
    old : Any
        The previous value that needs to be smoothed.
    new : Any
        The current value to be incorporated into the smoothing process. If None,
        the function will return the old value scaled by the filter factor.
    alpha : float
        The filter factor, a value between 0 and 1, that determines the weight
        of the old value in the smoothing process. A higher alpha gives more
        weight to the old value, resulting in slower changes.

    Returns
    -------
    Any
        The filtered value, which is a combination of the old and new values
        weighted by the filter factor.
    """
    if new is None:
        return alpha * old
    return alpha * old + new


def _init_IO_dim_state(
    etrace_xs: Dict[ETraceX_Key, brainstate.State],
    etrace_dfs: Dict[ETraceDF_Key, brainstate.State],
    relation: HiddenParamOpRelation,
):
    """
    Initialize the eligibility trace states for input-output dimensions.

    This function sets up the eligibility trace states for the weights and
    differential functions (df) associated with a given relation. It ensures
    that the eligibility trace states are initialized for the weight x and
    the df, and records the target paths of the weight x if it is used
    repeatedly in the graph.

    Args:
        etrace_xs (Dict[ETraceX_Key, brainstate.State]): A dictionary to store the
            eligibility trace states for the weight x, keyed by ETraceX_Key.
        etrace_dfs (Dict[ETraceDF_Key, brainstate.State]): A dictionary to store the
            eligibility trace states for the differential functions, keyed by
            ETraceDF_Key.
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

    # "relation.x_var" may be repeatedly used in the graph
    if not (relation.primitive is etp_elemwise_p):
        assert relation.x_var is not None  # non-elemwise primitives always have an x_var
        x_key = id(relation.x_var)
        if x_key not in etrace_xs:
            shape = relation.x_var.aval.shape
            dtype = relation.x_var.aval.dtype
            etrace_xs[x_key] = EligibilityTrace(u.math.zeros(shape, dtype))

    y_shape = relation.y_var.aval.shape
    y_dtype = relation.y_var.aval.dtype
    group: HiddenGroup
    for group in relation.hidden_groups:
        # Exact match required, or (elemwise only) allow trailing-dim match
        # where a batched hidden group wraps an unbatched elemwise weight.
        shape_ok = (
            y_shape == group.varshape
            or (
                relation.primitive is etp_elemwise_p
                and y_shape == group.varshape[1:]
            )
        )
        if not shape_ok:
            raise ValueError(
                f'The shape of the hidden states should be the '
                f'same as the shape of the hidden group. '
                f'While we got {y_shape} != {group.varshape}. '
            )
        key = etrace_df_key(relation.y_var, group.index)
        if key in etrace_dfs:  # relation.y_var is a unique output of the weight operation
            raise ValueError(f'The relation {key} has been added. ')

        #
        # Group 1:
        #
        #   [∂a^t-1/∂θ1, ∂b^t-1/∂θ1, ...]
        #
        # Group 2:
        #
        #   [∂A^t-1/∂θ1, ∂B^t-1/∂θ1, ...]
        #
        init_fn = ETP_RULES_INIT_PP[relation.primitive]
        etrace_dfs[key] = EligibilityTrace(
            init_fn(
                relation.x_var,
                relation.y_var,
                relation.trainable_vars,
                group.num_state
            )
        )


def _update_IO_dim_etrace_scan_fn(
    hist_etrace_vals: Tuple[
        Dict[ETraceX_Key, jax.Array],
        Dict[ETraceDF_Key, jax.Array]
    ],
    jacobians: Tuple[
        Dict[ETraceX_Key, jax.Array],  # the weight x
        Dict[ETraceDF_Key, jax.Array],  # the weight df
        Sequence[jax.Array],  # the hidden group Jacobians
    ],
    hid_weight_op_relations: Sequence[HiddenParamOpRelation],
    decay: float,
):
    """
    Update the eligibility trace values for input-output dimensions.

    This function updates the eligibility trace values for the weight x and
    differential functions (df) based on the provided Jacobians and decay
    factor. It computes the new eligibility trace values by applying a
    low-pass filter to the historical values and incorporating the current
    Jacobian values.

    Args:
        hist_etrace_vals (Tuple[Dict[ETraceX_Key, jax.Array], Dict[ETraceDF_Key, jax.Array]]):
            A tuple containing dictionaries of historical eligibility trace
            values for the weight x and df, keyed by ETraceX_Key and
            ETraceDF_Key, respectively.
        jacobians (Tuple[Dict[ETraceX_Key, jax.Array], Dict[ETraceDF_Key, jax.Array], Sequence[jax.Array]]):
            A tuple containing dictionaries of current Jacobian values for the
            weight x and df, and a sequence of hidden group Jacobians.
        hid_weight_op_relations (Sequence[HiddenParamOpRelation]):
            A sequence of HiddenParamOpRelation objects representing the
            relationships between hidden parameters and operations.
        decay (float): The decay factor used in the low-pass filter, a value
            between 0 and 1.

    Returns:
        Tuple[Dict[ETraceX_Key, jax.Array], Dict[ETraceDF_Key, jax.Array]]:
            A tuple containing dictionaries of updated eligibility trace values
            for the weight x and df, keyed by ETraceX_Key and ETraceDF_Key,
            respectively.
    """
    # --- the data --- #

    #
    # the etrace data at the current time step (t) of the O(n) algorithm
    # is a tuple, including the weight x and df values.
    #
    # For the weight x, it is a dictionary,
    #    {ETraceX_Key: jax.Array}
    #
    # For the weight df, it is a dictionary,
    #    {ETraceDF_Key: jax.Array}
    #
    xs: Dict[ETraceX_Key, jax.Array] = jacobians[0]
    dfs: Dict[ETraceDF_Key, jax.Array] = jacobians[1]

    #
    # the hidden-to-hidden Jacobians
    #
    hid_group_jacobians: Sequence[jax.Array] = jacobians[2]

    #
    # the history etrace values
    #
    # - hist_xs is a dictionary,
    #       {ETraceX_Key: brainstate.State}
    #
    # - hist_dfs is a dictionary,
    #       {ETraceDF_Key: brainstate.State}
    #
    hist_xs, hist_dfs = hist_etrace_vals

    #
    # the new etrace values
    #
    new_etrace_xs, new_etrace_dfs = dict(), dict()

    # --- the update --- #

    #
    # Step 1:
    #
    #   update the weight x using the equation:
    #           x^t = α * x^t-1 + x^t, where α is the decay factor.
    #
    check_dict_keys(hist_xs, xs)
    for xkey in hist_xs.keys():
        new_etrace_xs[xkey] = _low_pass_filter(hist_xs[xkey], xs[xkey], decay)

    relation: HiddenParamOpRelation
    for relation in hid_weight_op_relations:

        group: HiddenGroup
        for group in relation.hidden_groups:

            #
            # Step 2:
            #
            # update the eligibility trace * hidden diagonal Jacobian
            #         dϵ^t_{pre} = D_h ⊙ dϵ^t-1, where D_h is the hidden-to-hidden Jacobian diagonal matrix.
            #
            #
            # JVP equation for the following Jacobian computation:
            #
            # [∂V^t/∂V^t-1, ∂V^t/∂a^t-1,  [∂V^t-1/∂θ1,
            #  ∂a^t/∂V^t-1, ∂a^t/∂a^t-1]   ∂a^t-1/∂θ1,]
            #
            # [∂V^t/∂V^t-1, ∂V^t/∂a^t-1,  [∂V^t-1/∂θ2,
            #  ∂a^t/∂V^t-1, ∂a^t/∂a^t-1]   ∂a^t-1/∂θ2]
            #
            df_key = etrace_df_key(relation.y_var, group.index)
            hid_jac = hid_group_jacobians[group.index]
            pre_trace_df = jnp.einsum(
                '...ij,...j->...i',
                hid_jac,
                hist_dfs[df_key]
            )

            #
            # Step 3:
            #
            # update: eligibility trace * hidden diagonal Jacobian + new hidden df
            #        dϵ^t = dϵ^t_{pre} + df^t, where D_h is the hidden-to-hidden Jacobian diagonal matrix.
            #
            new_etrace_dfs[df_key] = _expon_smooth(pre_trace_df, dfs[df_key], decay)

    return (new_etrace_xs, new_etrace_dfs), None


def _solve_IO_dim_weight_gradients(
    hist_etrace_data: Tuple[
        Dict[ETraceX_Key, jax.Array],
        Dict[ETraceDF_Key, jax.Array]
    ],
    dG_weights: Dict[Path, dG_Weight],
    dG_hidden_groups: Sequence[jax.Array],  # same length as total hidden groups
    weight_hidden_relations: Sequence[HiddenParamOpRelation],
    weight_vals: Dict[Path, WeightVals],
    running_index: int,
    decay: float,
    fast_solve: bool = True,
):
    """
    Compute and update the weight gradients for input-output dimensions using eligibility trace data.

    This function calculates the weight gradients by utilizing the eligibility trace data and the
    hidden-to-hidden Jacobians. It applies a correction factor to avoid exponential smoothing bias
    at the beginning of the computation.

    Args:
        hist_etrace_data (Tuple[Dict[ETraceX_Key, jax.Array], Dict[ETraceDF_Key, jax.Array]]):
            A tuple containing dictionaries of historical eligibility trace values for the weight x
            and differential functions (df), keyed by ETraceX_Key and ETraceDF_Key, respectively.
        dG_weights (Dict[Path, dG_Weight]):
            A dictionary to store the computed weight gradients, keyed by the path of the weight.
        dG_hidden_groups (Sequence[jax.Array]):
            A sequence of hidden group Jacobians, with the same length as the total number of hidden groups.
        weight_hidden_relations (Sequence[HiddenParamOpRelation]):
            A sequence of HiddenParamOpRelation objects representing the relationships between hidden
            parameters and operations.
        weight_vals (Dict[Path, WeightVals]):
            A dictionary containing the current values of the weights, keyed by their paths.
        running_index (int):
            The current index in the running sequence, used to compute the correction factor.
        decay (float):
            The decay factor used in the exponential smoothing process, a value between 0 and 1.

    Returns:
        None: The function updates the dG_weights dictionary in place with the computed weight gradients.
    """
    # Bias correction for exponential smoothing
    #   ε_f^t = α ε_f^{t-1} + (1-α) x_t  =>  E[ε_f^t] = x · (1 - α^{t+1})
    # so unbiased estimator divides by (1 - α^{t+1}) = (1 - decay^{t+1}).
    correction_factor = 1. - u.math.power(decay, running_index + 1)
    correction_factor = u.math.where(running_index < 1000, correction_factor, 1.)
    # Clamp guards degenerate decay=0 (rank=1): correction is exactly 1 then,
    # but keep clamp for numerical safety in the early-step power computation.
    correction_factor = u.math.maximum(correction_factor, 1e-8)
    correction_factor = jax.lax.stop_gradient(correction_factor)

    xs, dfs = hist_etrace_data

    relation: HiddenParamOpRelation
    for relation in weight_hidden_relations:

        if not (relation.primitive is etp_elemwise_p):
            x = xs[id(relation.x_var)]
        else:
            x = None

        # Build the weights dict consumed by xy_to_dw.
        weights_dict = {
            key: _extract_leaf(
                weight_vals[relation.trainable_paths[key]],
                relation.trainable_leaf_indices[key],
            )
            for key in relation.trainable_vars
        }

        xy_to_dw_rule = ETP_RULES_XY_TO_DW[relation.primitive]
        eqn_params = relation.eqn_params
        batched = is_batched_primitive(relation.primitive)

        def _call(df_, w_, _rule=xy_to_dw_rule, _params=eqn_params, _x=x):
            return _rule(_x, df_, w_, **_params)

        group: HiddenGroup
        for group in relation.hidden_groups:
            df_key = etrace_df_key(relation.y_var, group.index)
            df = dfs[df_key] / correction_factor
            df_hid = df * dG_hidden_groups[group.index]

            if fast_solve:
                # Fast path: sum over n_state first, then ONE xy_to_dw call.
                # Valid because every xy_to_dw rule is a VJP of a linear map
                # in its cotangent argument, so sum-then-apply == apply-then-sum.
                df_summed = u.math.sum(df_hid, axis=-1)
                if (relation.primitive is etp_elemwise_p) and batched:
                    # Elemwise-in-batched-hidden: strip batch dim via a single
                    # vmap over batch, then sum batch after.
                    dg_dict = jax.tree.map(
                        lambda a: _sum_dim(a, axis=0),
                        jax.vmap(lambda d_: _call(d_, weights_dict))(df_summed),
                    )
                else:
                    dg_dict = _call(df_summed, weights_dict)
            else:
                # Legacy path: vmap xy_to_dw across n_state slices, then sum.
                fn_vmap = jax.vmap(lambda df_: _call(df_, weights_dict), in_axes=-1, out_axes=-1)
                if (relation.primitive is etp_elemwise_p) and batched:
                    fn_vmap2 = jax.vmap(fn_vmap)
                    dg_dict = jax.tree.map(
                        lambda a: _sum_dim(_sum_dim(a, axis=-1), axis=0),
                        fn_vmap2(df_hid),
                    )
                else:
                    dg_dict = jax.tree.map(_sum_dim, fn_vmap(df_hid))

            # Route per-key to owning ParamState path and assemble per-path pytrees.
            _route_grads_by_path(relation, dg_dict, weight_vals, dG_weights)


class IODimVjpAlgorithm(ETraceVjpAlgorithm):
    r"""Online gradient algorithm with diagonal approximation and input-output-dimension complexity.

    This algorithm computes the gradients of the weights with the diagonal
    approximation and the input-output dimensional complexity. It is based on the
    RTRL algorithm (Real-Time Recurrent Learning).

    Parameters
    ----------
    model : brainstate.nn.Module
        The model function, which receives the input arguments and returns the
        model output.
    decay_or_rank : float or int
        The exponential smoothing factor for the eligibility trace. If a float,
        it is the decay factor and should be in the range :math:`(0, 1)`. If an
        integer, it is the number of approximation ranks for the algorithm and
        should be greater than 0.
    vjp_method : str, optional
        The method for computing the VJP. It should be either ``"single-step"``
        or ``"multi-step"``.

        - ``"single-step"``: the VJP is computed at the current time step, i.e.,
          :math:`\partial L^t/\partial h^t`.
        - ``"multi-step"``: the VJP is computed at multiple time steps, i.e.,
          :math:`\partial L^t/\partial h^{t-k}`, where :math:`k` is determined by
          the data input.
    name : str, optional
        The name of the etrace algorithm.
    mode : braintrace.mixin.Mode, optional
        The computing mode, indicating the batching information.

    Notes
    -----
    The learning rule is

    .. math::

        \begin{aligned}
        & \boldsymbol{\epsilon}^t \approx \boldsymbol{\epsilon}_{\mathbf{f}}^t \otimes \boldsymbol{\epsilon}_{\mathbf{x}}^t \\
        & \boldsymbol{\epsilon}_{\mathbf{x}}^t=\alpha \boldsymbol{\epsilon}_{\mathbf{x}}^{t-1}+\mathbf{x}^t \\
        & \boldsymbol{\epsilon}_{\mathbf{f}}^t=\alpha \operatorname{diag}\left(\mathbf{D}^t\right) \circ \boldsymbol{\epsilon}_{\mathbf{f}}^{t-1}+(1-\alpha) \operatorname{diag}\left(\mathbf{D}_f^t\right) \\
        & \nabla_{\boldsymbol{\theta}} \mathcal{L}=\sum_{t^{\prime} \in \mathcal{T}} \frac{\partial \mathcal{L}^{t^{\prime}}}{\partial \mathbf{h}^{t^{\prime}}} \circ \boldsymbol{\epsilon}^{t^{\prime}}
        \end{aligned}

    where :math:`\boldsymbol{\epsilon}_{\mathbf{x}}^t` is the input-side trace,
    :math:`\boldsymbol{\epsilon}_{\mathbf{f}}^t` the output-side trace,
    :math:`\alpha` the exponential-smoothing factor, :math:`\mathbf{D}^t` the
    hidden-to-hidden Jacobian, :math:`\mathbf{D}_f^t` the state-to-output
    Jacobian, and :math:`\mathbf{x}^t` the presynaptic input.

    The full per-parameter D-RTRL trace
    :math:`\boldsymbol{\epsilon}^t \in \mathbb{R}^{I\times O}` is approximated by
    the outer product of two exponentially-smoothed *vectors* — one over the
    input dimension and one over the output dimension. Storing the two factors
    instead of the full matrix drops the memory from :math:`O(I\cdot O)` to
    :math:`O(I+O)` per layer. The decay :math:`\alpha` (equivalently an
    approximation rank) controls how much temporal history the factored trace
    retains; the bias of the exponential estimator is corrected at solve time.

    This algorithm has :math:`O(BI+BO)` memory complexity and :math:`O(BIO)`
    computational complexity, where :math:`I` and :math:`O` are the number of
    input and output dimensions, and :math:`B` the batch size. In particular, for
    a linear transformation layer, the weight gradients are computed with
    :math:`O(Bn)` memory complexity and :math:`O(Bn^2)` computational complexity,
    where :math:`n` is the number of hidden dimensions.

    For more details, please see `the ES-D-RTRL algorithm presented in our manuscript <https://www.biorxiv.org/content/10.1101/2024.09.24.614728v2>`_.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>>
        >>> class RNN(brainstate.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.cell = braintrace.nn.ValinaRNNCell(1, 20, activation='tanh')
        ...         self.out = braintrace.nn.Linear(20, 1)
        ...     def update(self, x):
        ...         return x >> self.cell >> self.out
        >>>
        >>> model = RNN()
        >>> _ = brainstate.nn.init_all_states(model)
        >>> learner = braintrace.pp_prop(model, decay_or_rank=0.9)  # or rank: decay_or_rank=19
        >>> x0 = brainstate.random.randn(1)
        >>> learner.compile_graph(x0)   # trace the graph once
        >>> y = learner(x0)             # forward pass + eligibility-trace update

    References
    ----------
    .. [1] Wang, C., Dong, X., Ji, Z., Xiao, M., Jiang, J., Liu, X., Huan, Y., &
       Wu, S. (2026). "Model-agnostic linear-memory online learning in spiking
       neural networks." *Nature Communications*.
       https://doi.org/10.1038/s41467-026-68453-w
       (preprint: bioRxiv 2024.09.24.614728)
    .. [2] Williams, R. J., & Zipser, D. (1989). "A Learning Algorithm for
       Continually Running Fully Recurrent Neural Networks" (RTRL). *Neural
       Computation*, 1(2), 270-280. https://doi.org/10.1162/neco.1989.1.2.270
    """

    # the spatial gradients of the weights
    etrace_xs: Dict[ETraceX_Key, brainstate.State]

    # the spatial gradients of the hidden states
    etrace_dfs: Dict[ETraceDF_Key, brainstate.State]

    # the exponential smoothing decay factor
    decay: float

    def __init__(
        self,
        model: brainstate.nn.Module,
        decay_or_rank: float | int,
        name: Optional[str] = None,
        vjp_method: str = 'single-step',
        fast_solve: bool = True,
        **kwargs,
    ):
        super().__init__(model, name=name, vjp_method=vjp_method)
        self.decay, num_rank = _format_decay_and_rank(decay_or_rank)
        self.fast_solve = fast_solve

    def init_etrace_state(self, *args, **kwargs):
        """Initialize the eligibility trace states of the etrace algorithm.

        This method is needed after compiling the etrace graph. See
        :meth:`compile_graph` for the details.
        """
        # The states of weight spatial gradients:
        #   1. x
        #   2. df
        self.etrace_xs = dict()
        self.etrace_dfs = dict()
        for relation in self.graph.hidden_param_op_relations:
            relation: HiddenParamOpRelation
            _init_IO_dim_state(self.etrace_xs, self.etrace_dfs, relation)

    def reset_state(self, batch_size: int = None, **kwargs):
        """Reset the eligibility trace states.

        Parameters
        ----------
        batch_size : int, optional
            The batch size used to reshape the reset trace states. Default ``None``.
        """
        self.running_index.value = 0
        _reset_state_in_a_dict(self.etrace_xs, batch_size)
        _reset_state_in_a_dict(self.etrace_dfs, batch_size)

    def get_etrace_of(self, weight: brainstate.ParamState | Path) -> Tuple[Dict, Dict]:
        """Get the eligibility trace of the given weight.

        Parameters
        ----------
        weight : brainstate.ParamState or Path
            The weight whose eligibility trace is requested, given either as a
            :class:`brainstate.ParamState` instance or as its path in the model.

        Returns
        -------
        etrace_xs : dict
            The input-side eligibility traces keyed by the weight-input variable.
        etrace_dfs : dict
            The output-side eligibility traces keyed by
            ``(y_var, hidden-group index)``.

        Raises
        ------
        ValueError
            If no eligibility trace is found for the given weight.
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
        find_this_weight = False
        relation: HiddenParamOpRelation
        for relation in self.graph.hidden_param_op_relations:
            primary_state = next(iter(relation.trainable_param_states.values()), None)
            if primary_state is None or id(primary_state) != weight_id:
                continue
            find_this_weight = True

            # get the weight_op input
            wx_var = etrace_x_key(relation.x_var)
            if wx_var is not None:
                etrace_xs[wx_var] = self.etrace_xs[wx_var].value

            # get the weight_op df
            wy_var = relation.y_var
            group: HiddenGroup
            for group in relation.hidden_groups:
                df_key = etrace_df_key(wy_var, group.index)
                etrace_dfs[df_key] = self.etrace_dfs[df_key].value
        if not find_this_weight:
            raise ValueError(f'Do not the etrace of the given weight: {weight}.')
        return etrace_xs, etrace_dfs

    def _get_etrace_data(self) -> Tuple[
        Dict[ETraceX_Key, jax.Array],
        Dict[ETraceDF_Key, jax.Array]
    ]:
        """
        Get the eligibility trace data at the last time-step.

        .. note::

            This is the protocol method that should be implemented in the subclass.

        Returns:
            ETraceVals, the eligibility trace data.
        """
        etrace_xs = {k: v.value for k, v in self.etrace_xs.items()}
        etrace_dfs = {k: v.value for k, v in self.etrace_dfs.items()}
        return etrace_xs, etrace_dfs

    def _assign_etrace_data(
        self,
        hist_etrace_vals: Tuple[
            Dict[ETraceX_Key, jax.Array],
            Dict[ETraceDF_Key, jax.Array]
        ]
    ):
        """Assign the eligibility trace data to the states at the current time-step.

        .. note::

            This is the protocol method that should be implemented in the subclass.

        Args:
            hist_etrace_vals: ETraceVals, the eligibility trace data.
        """
        #
        # For any operation:
        #
        #           h^t = f(x^t \theta)
        #
        # etrace_xs:
        #           x^t
        #
        # etrace_dfs:
        #           df^t = ∂h^t / ∂y^t, where y^t = x^t \theta
        #
        (etrace_xs, etrace_dfs) = hist_etrace_vals

        # the weight x and df
        for x, val in etrace_xs.items():
            self.etrace_xs[x].value = val
        for df, val in etrace_dfs.items():
            self.etrace_dfs[df].value = val

    def _update_etrace_data(
        self,
        running_index: Optional[int],
        hist_etrace_vals: Tuple[
            Dict[ETraceX_Key, jax.Array],
            Dict[ETraceDF_Key, jax.Array]
        ],
        hid2weight_jac_single_or_multi_times: Hid2WeightJacobian,
        hid2hid_jac_single_or_multi_times: HiddenGroupJacobian,
        weight_vals: WeightVals,
        input_is_multi_step: bool,
    ) -> Tuple[
        Dict[ETraceX_Key, jax.Array],
        Dict[ETraceDF_Key, jax.Array]
    ]:
        """Update the eligibility trace data for a given timestep.

        This method implements the core update equations for the eligibility trace
        algorithm with input-output dimensional complexity. It processes historical
        trace values along with current Jacobians to compute the updated eligibility
        traces according to the algorithm's update rules.

        Args:
            running_index: Optional[int]
                The current timestep index. Used for decay correction factors.
            hist_etrace_vals: Tuple[Dict[ETraceX_Key, jax.Array], Dict[ETraceDF_Key, jax.Array]]
                The eligibility trace values from the previous timestep, containing:
                - Dictionary mapping weight inputs to their trace values
                - Dictionary mapping differential functions to their trace values
            hid2weight_jac_single_or_multi_times: Hid2WeightJacobian
                The current hidden-to-weight Jacobians at time t (or t-1 depending on vjp_method).
            hid2hid_jac_single_or_multi_times: HiddenGroupJacobian
                The current hidden-to-hidden Jacobians for propagating gradients.
            weight_vals: WeightVals
                The current values of the model weights.

        Returns:
            Tuple[Dict[ETraceX_Key, jax.Array], Dict[ETraceDF_Key, jax.Array]]:
                Updated eligibility trace values for both input traces and differential
                function traces, computed according to the exponential smoothing rules
                of the algorithm.
        """
        #
        # "running_index":
        #            the running index
        #
        # "hist_etrace_vals":
        #            the history etrace values,
        #            including the x and df values, see "etrace_xs" and "etrace_dfs".
        #
        # "hid2weight_jac_single_or_multi_times":
        #           the current etrace values at the time "t", \epsilon^t, if vjp_time == "t".
        #           Otherwise, the etrace values at the time "t-1", \epsilon^{t-1}.
        #
        # "hid2hid_jac_single_or_multi_times":
        #           the data for computing the hidden-to-hidden Jacobian at the time "t".
        #
        # "weight_path_to_vals":
        #           the weight values.
        #

        scan_fn = partial(
            _update_IO_dim_etrace_scan_fn,
            hid_weight_op_relations=self.graph.hidden_param_op_relations,
            decay=self.decay,
        )

        if input_is_multi_step:
            hist_etrace_vals = jax.lax.scan(
                scan_fn,
                hist_etrace_vals,
                (
                    hid2weight_jac_single_or_multi_times[0],
                    hid2weight_jac_single_or_multi_times[1],
                    hid2hid_jac_single_or_multi_times,
                ),
            )[0]

        else:
            hist_etrace_vals = scan_fn(
                hist_etrace_vals,
                (
                    hid2weight_jac_single_or_multi_times[0],
                    hid2weight_jac_single_or_multi_times[1],
                    hid2hid_jac_single_or_multi_times,
                ),
            )[0]

        return hist_etrace_vals

    def _solve_weight_gradients(
        self,
        running_index: int,
        etrace_h2w_at_t: Tuple[
            Dict[ETraceX_Key, jax.Array],
            Dict[ETraceDF_Key, jax.Array]
        ],
        dl_to_hidden_groups: Sequence[jax.Array],
        weight_vals: Dict[Path, PyTree],
        dl_to_nonetws_at_t: Dict[Path, PyTree],
        dl_to_etws_at_t: Optional[Dict[Path, PyTree]],
    ):
        """Compute weight gradients using eligibility trace data and loss gradients.

        This method implements the final stage of the eligibility trace algorithm, where
        the eligibility traces are combined with the loss gradients to compute the weight
        parameter gradients. It follows the mathematical equation:

        ∇_θ L = ∑ (∂L/∂h) ⊙ ϵ

        where ϵ represents the eligibility traces and ∂L/∂h are the gradients of
        the loss with respect to hidden states.

        Args:
            running_index: int
                The current timestep index, used for correction factor calculation.
            etrace_h2w_at_t: Tuple[Dict[ETraceX_Key, jax.Array], Dict[ETraceDF_Key, jax.Array]]
                The eligibility trace data at the current timestep, containing:
                - Dictionary mapping weight inputs to their trace values
                - Dictionary mapping differential functions to their trace values
            dl_to_hidden_groups: Sequence[jax.Array]
                Gradients of the loss with respect to each hidden group/state.
            weight_vals: Dict[Path, PyTree]
                Current values of the model weights.
            dl_to_nonetws_at_t: Dict[Path, PyTree]
                Gradients for non-eligibility trace weights computed through standard backprop.
            dl_to_etws_at_t: Optional[Dict[Path, PyTree]]
                Optional additional gradients for eligibility trace weights.

        Returns:
            Dict[Path, jax.Array]: Computed gradients for all weights in the model.
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
        dG_weights: Dict[Path, Any] = {path: None for path in self.param_states.keys()}

        # update the etrace parameters
        _solve_IO_dim_weight_gradients(
            etrace_h2w_at_t,
            dG_weights,
            dl_to_hidden_groups,
            self.graph.hidden_param_op_relations,
            weight_vals,
            running_index,
            self.decay,
            fast_solve=self.fast_solve,
        )

        # update the non-etrace parameters
        for path, dg in dl_to_nonetws_at_t.items():
            _update_dict(dG_weights, path, dg)

        # update the etrace parameters when "dl_to_etws_at_t" is not None
        if dl_to_etws_at_t is not None:
            for path, dg in dl_to_etws_at_t.items():
                _update_dict(dG_weights, path, dg, error_when_no_key=True)
        return dG_weights
