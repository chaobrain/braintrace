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

from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Tuple, Optional, Sequence, Any

import brainstate
import jax
import jax.numpy as jnp
import brainunit as u

from braintrace._compiler import (
    ControlFlowPolicy,
    DEFAULT_MAX_JACOBIAN_ELEMENTS,
    HiddenGroup,
    HiddenParamOpRelation,
    gather_learning_signal,
    widen_instant_term,
)
from braintrace._op import (
    etp_conv_p,
    etp_elemwise_p,
    ETP_RULES_DT_TO_T,
    ETP_RULES_XY_TO_DW,
    ETP_RULES_INIT_DRTRL,
    is_batched_primitive,
    get_fast_path_rules,
)
from braintrace._op._registries import (
    get_instant_drtrl_rule,
    get_solve_drtrl_rule,
)
from braintrace._misc import etrace_df_key, etrace_x_key, suffix_products
from braintrace._typing import (
    PyTree,
    WeightID,
    Path,
    DTypeLike,
    ETraceX_Key,
    ETraceDF_Key,
    ETraceWG_Key,
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
from .axes import ETraceConfig
from .vjp_base import ETraceVjpAlgorithm

__all__ = [
    'ParamDimVjpAlgorithm',
]

def _cast_to_dtype(tree: Any, dtype: Any) -> Any:
    """Cast every array leaf of ``tree`` to ``dtype`` (unit-safe; ``None`` -> no-op).

    Used to store the eligibility trace — and the inputs to its update — at a
    reduced precision (e.g. ``bfloat16``). The fast path operates on unitless
    arrays, but the ``is_leaf`` guard keeps the helper correct if a leaf ever
    carries a unit.
    """
    if dtype is None:
        return tree
    return jax.tree.map(lambda a: a.astype(dtype), tree, is_leaf=u.math.is_quantity)


def _init_param_dim_state(
    etrace_bwg: Dict[ETraceWG_Key, brainstate.State],
    relation: HiddenParamOpRelation,
    trace_dtype: Optional[DTypeLike] = None,
    fast_solve: bool = True,
) -> None:
    """
    Initialize the eligibility trace states for parameter dimensions.

    Traces are stored as ``Dict[str, Array]`` keyed by the primitive's
    trainable-input names (dict-based rule API). When ``trace_dtype`` is set,
    the trace is only allocated at that reduced precision when the runtime
    update will actually take the fast path for this relation — i.e. the same
    predicate used by :func:`_update_param_dim_etrace_scan_fn` /
    :func:`_solve_param_dim_weight_gradients`: ``fast_solve and fp is not None
    and fp.applicable(relation.eqn_params)``. Otherwise the trace stays at
    native precision, because the legacy (non-fast) update always emits
    native-precision arrays and a mismatched scan-carry dtype would raise at
    trace time.
    """
    group: HiddenGroup
    for group in relation.hidden_groups:
        bwg_key = (id(relation.y_var), group.index)
        if bwg_key in etrace_bwg:
            raise ValueError(f'The relation {bwg_key} has been added. ')
        init_fn = ETP_RULES_INIT_DRTRL[relation.primitive]
        # ``etp_elemwise`` has no x/y batch carrier (its output is the weight),
        # so it needs the hidden group to size the trace's leading (position /
        # batch) axes. ``etp_conv``'s per-position trace shape depends on the
        # equation's layout (``dimension_numbers`` / ``strides``) and grouped
        # convolutions must be rejected at init, so it receives the eqn
        # params. Other primitives are unchanged.
        init_kw: dict = {}
        if relation.primitive is etp_elemwise_p:
            init_kw['group'] = group
        elif relation.primitive is etp_conv_p:
            init_kw['eqn_params'] = relation.eqn_params
        # ``trace_state_width`` is ``num_state`` at every recurrence_scope but
        # ``sparse_n``, where the trailing state axis carries the SnAp-n
        # neighbourhood as well: ``M = K * num_state``. Every per-primitive rule
        # is generic in the size of that axis, so widening is a sizing change
        # here and nowhere else.
        init_val = init_fn(
            relation.x_var,
            relation.y_var,
            relation.trainable_vars,
            group.trace_state_width,
            **init_kw,
        )
        if not isinstance(init_val, dict):
            raise TypeError(
                f'Primitive {relation.primitive.name} init_drtrl must return a dict; '
                f'got {type(init_val).__name__}.'
            )
        fp = get_fast_path_rules(relation.primitive)
        use_fast = fast_solve and fp is not None and fp.applicable(relation.eqn_params)
        if use_fast:
            init_val = _cast_to_dtype(init_val, trace_dtype)
        etrace_bwg[bwg_key] = EligibilityTrace(init_val)


def _update_param_dim_etrace_scan_fn(
    hist_etrace_vals: Dict[ETraceWG_Key, jax.Array],
    jacobians: Tuple[
        Dict[ETraceX_Key, jax.Array],  # the weight x
        Dict[ETraceDF_Key, jax.Array],  # the weight df
        Sequence[jax.Array],  # the hidden group Jacobians
    ],
    weight_path_to_vals: Dict[Path, PyTree],
    hidden_param_op_relations: Any,
    fast_solve: bool = True,
    trace_dtype: Optional[DTypeLike] = None,
) -> Any:
    """
    Update the eligibility trace values for parameter dimensions.

    This function updates the eligibility trace values for the parameter dimensions
    based on the provided Jacobians and the current mode. It computes the new eligibility
    trace values by applying vector-Jacobian products and incorporating the current
    Jacobian values.

    Parameters
    ----------
    hist_etrace_vals : Dict[ETraceWG_Key, jax.Array]
        A dictionary containing historical eligibility trace values for the
        weight gradients, keyed by ETraceWG_Key.
    jacobians : Tuple[Dict[ETraceX_Key, jax.Array], Dict[ETraceDF_Key, jax.Array], Sequence[jax.Array]]
        A tuple containing dictionaries of current Jacobian values for the weight x
        and df, and a sequence of hidden group Jacobians.
    weight_path_to_vals : Dict[Path, PyTree]
        A dictionary mapping weight paths to their corresponding PyTree values.
    hidden_param_op_relations : Any
        A sequence of HiddenParamOpRelation objects representing the
        relationships between hidden parameters and operations.
    fast_solve : bool, optional
        Whether to use the per-primitive fast contraction instead of the legacy
        vmap path.
    trace_dtype : DTypeLike, optional
        Optional dtype override for the updated trace values.

    Returns
    -------
    Tuple[Dict[ETraceWG_Key, jax.Array], None]
        A tuple containing a dictionary of updated eligibility trace values for
        the weight gradients, keyed by ETraceWG_Key, and None.
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

    # --- partition: flat relations vs descended relations (by owning scan) --- #
    #
    # Descended relations (structured scan descent, Phase 4) carry stacked
    # per-substep Jacobians with a leading axis ``L``; their trace update is
    # folded over that axis with an inner ``jax.lax.scan``. Flat relations
    # take the historical single-application path unchanged.
    regular = [r for r in hidden_param_op_relations
               if r.control_flow_context is None]
    by_scan: Dict[int, list] = {}
    for r in hidden_param_op_relations:
        if r.control_flow_context is not None:
            by_scan.setdefault(id(r.control_flow_context.scan), []).append(r)

    # The etrace weight gradients at the current time step.
    # i.e., The "hist_etrace_vals" at the next time step
    #
    new_etrace_bwg = dict(hist_etrace_vals)

    if regular:
        new_etrace_bwg.update(_apply_relation_step(
            hist_etrace_vals, etrace_xs_at_t, etrace_ys_at_t,
            hid_group_jacobians, regular, weight_path_to_vals,
            fast_solve, trace_dtype,
        ))

    for rels in by_scan.values():
        trace_keys = [(id(r.y_var), g.index)
                      for r in rels for g in r.hidden_groups]
        x_keys = [etrace_x_key(r.x_var) for r in rels if r.x_var is not None]
        df_keys = [etrace_df_key(r.y_var, g.index)
                   for r in rels for g in r.hidden_groups]
        g_idx = sorted({g.index for r in rels for g in r.hidden_groups})

        sub_carry = {k: hist_etrace_vals[k] for k in trace_keys}
        sub_xs = (
            {k: etrace_xs_at_t[k] for k in x_keys},      # (L, ...)
            {k: etrace_ys_at_t[k] for k in df_keys},     # (L, ...)
            {i: hid_group_jacobians[i] for i in g_idx},  # (L, ...)
        )

        def _substep(
            carry: Dict[ETraceWG_Key, jax.Array],
            sliced: Any,
            _rels: Any = rels,
        ) -> Tuple[Dict[ETraceWG_Key, jax.Array], None]:
            xs_t, dfs_t, diags_t = sliced
            carry = {**carry, **_apply_relation_step(
                carry, xs_t, dfs_t, diags_t, _rels,
                weight_path_to_vals, fast_solve, trace_dtype)}
            return carry, None

        sub_carry, _ = jax.lax.scan(_substep, sub_carry, sub_xs)
        new_etrace_bwg.update(sub_carry)

    return new_etrace_bwg, None


def relation_weights_dict(
    relation: HiddenParamOpRelation,
    weight_path_to_vals: Dict[Path, PyTree],
) -> Dict[str, Any]:
    """The ``{trainable name: leaf}`` dict the per-primitive rules consume.

    Parameters
    ----------
    relation : HiddenParamOpRelation
        The relation whose trainable inputs are wanted.
    weight_path_to_vals : dict
        Mapping from ``ParamState`` path to its PyTree value.

    Returns
    -------
    dict
        One entry per trainable input name of the relation's primitive.
    """
    return {
        key: _extract_leaf(
            weight_path_to_vals[relation.trainable_paths[key]],
            relation.trainable_leaf_indices[key],
        )
        for key in relation.trainable_vars
    }


def relation_instant_term(
    relation: HiddenParamOpRelation,
    group: HiddenGroup,
    x: Any,
    df: Any,
    weights_dict: Dict[str, Any],
    fast_solve: bool = True,
    trace_dtype: Optional[DTypeLike] = None,
) -> Dict[str, Any]:
    r"""The instantaneous term :math:`\mathrm{diag}(D_f^t) \otimes x^t`.

    The instantaneous half of one trace-update application, extracted so that a
    second engine can build it without also rolling a recurrent term.
    :class:`~braintrace._algorithm.random_projection_vjp.RandomProjectionVjpAlgorithm`
    needs exactly this array (contracted with a random projection instead of
    accumulated), and building it twice in two spellings is how the two engines
    would drift apart on a primitive's documented regime.

    Parameters
    ----------
    relation : HiddenParamOpRelation
        The relation being applied.
    group : HiddenGroup
        The hidden group this application targets. Only ``group.snap`` is read,
        to widen the term onto neighbour slot 0 under ``sparse_n``.
    x : array or None
        The relation's x-side carrier, or ``None`` for ``etp_elemwise`` (whose
        output *is* the weight, so it has no x carrier).
    df : array
        The ``y -> hidden`` tangent for this ``(relation, group)`` pair, with a
        trailing state axis.
    weights_dict : dict
        As returned by :func:`relation_weights_dict`.
    fast_solve : bool, default True
        Whether a registered closed-form kernel may be used.
    trace_dtype : dtype-like, optional
        Reduced precision for the returned term (fast path only).

    Returns
    -------
    dict
        ``{trainable name: array}`` in trace coordinates -- parameter-shaped
        plus whatever leading (batch / position) and trailing (state) axes the
        primitive's ``init_drtrl`` rule declares.
    """
    xy_to_dw_rule = (
        get_instant_drtrl_rule(relation.primitive)
        or ETP_RULES_XY_TO_DW[relation.primitive]
    )
    eqn_params = relation.eqn_params
    batched = is_batched_primitive(relation.primitive)
    has_bias = eqn_params.get('has_bias', False)
    fp = get_fast_path_rules(relation.primitive)
    use_fast = fast_solve and fp is not None and fp.applicable(eqn_params)

    if group.snap is not None:
        # `sparse_n`: the instantaneous term lands on the position the relation
        # anchors at, i.e. neighbour slot 0. Padding it here -- rather than
        # inside each primitive's `instant` rule -- keeps every rule generic in
        # the width of the trailing axis.
        df = widen_instant_term(df, group.snap.num_neighbour)

    if use_fast:
        assert fp is not None  # use_fast implies a registered fast path
        return fp.instant(
            _cast_to_dtype(x, trace_dtype),
            _cast_to_dtype(df, trace_dtype),
            has_bias,
        )

    def comp_dw_with_x(x_: Any, df_: Any) -> Any:
        return xy_to_dw_rule(x_, df_, weights_dict, **eqn_params)

    # Legacy nested-vmap path: vmap xy_to_dw over num_state (and batch).
    @partial(jax.vmap, in_axes=-1, out_axes=-1)
    def _inner(df_slice: Any) -> Any:
        if batched:
            df_b = df_slice
            # Under ``brainstate.nn.Vmap(vmap_states='new')`` the hidden-state
            # trace (df) is per-lane and has lost its leading batch axis, while a
            # conv input still carries the singleton batch its forward API
            # requires (x = [1, *spatial, C]). Re-insert the matching singleton so
            # the per-sample vmap maps consistent leading axes (collapsed again by
            # the solve-time batch sum).
            if x is not None and x.ndim == df_b.ndim + 1:
                df_b = df_b[None]
            return jax.vmap(comp_dw_with_x)(x, df_b)
        return comp_dw_with_x(x, df_slice)

    return _inner(df)


def _apply_relation_step(
    hist_etrace_vals: Dict[ETraceWG_Key, jax.Array],
    etrace_xs_at_t: Dict[ETraceX_Key, jax.Array],
    etrace_ys_at_t: Dict[ETraceDF_Key, jax.Array],
    hid_group_jacobians: Any,
    relations: Any,
    weight_path_to_vals: Dict[Path, PyTree],
    fast_solve: bool = True,
    trace_dtype: Optional[DTypeLike] = None,
) -> Dict[ETraceWG_Key, jax.Array]:
    """One trace-update application over ``relations`` (a subset is fine).

    The body of the historical per-relation loop of
    :func:`_update_param_dim_etrace_scan_fn`, extracted verbatim so the
    descended-scan substep fold can reuse it per substep.

    Parameters
    ----------
    hist_etrace_vals : dict
        Previous trace values; only the keys touched by ``relations`` are
        read.
    etrace_xs_at_t : dict
        Weight-x values keyed by :func:`~braintrace._misc.etrace_x_key`.
    etrace_ys_at_t : dict
        Weight-df values keyed by :func:`~braintrace._misc.etrace_df_key`.
    hid_group_jacobians : sequence or dict
        The hidden-group Jacobians; may be the executor's list (indexed by
        ``group.index``) or an int-keyed dict holding only the indices these
        relations touch — both support ``[group.index]``.
    relations : sequence of HiddenParamOpRelation
        The relations to apply (a subset of the graph's relations is fine).
    weight_path_to_vals : dict
        Mapping from weight paths to their PyTree values.
    fast_solve : bool, default True
        Whether closed-form fast-path kernels may be used.
    trace_dtype : dtype-like, optional
        Reduced trace precision (fast path only).

    Returns
    -------
    dict
        ONLY the updated trace keys (``(id(y_var), group.index)``).
    """
    new_etrace_bwg: Dict[ETraceWG_Key, jax.Array] = dict()

    relation: HiddenParamOpRelation
    for relation in relations:

        # Build the weights dict the rules consume.
        weights_dict = relation_weights_dict(relation, weight_path_to_vals)

        dt_to_t_rule = ETP_RULES_DT_TO_T[relation.primitive]
        eqn_params = relation.eqn_params
        is_elemwise = relation.primitive is etp_elemwise_p
        batched = is_batched_primitive(relation.primitive)
        # Fast path only applies to primitives with elementwise dt_to_t, and
        # only when no parameter-transform hook is present (the closed-form
        # kernels drop the f'(W) factor — gated by ``fp.applicable``).
        fp = get_fast_path_rules(relation.primitive)
        use_fast = fast_solve and fp is not None and fp.applicable(eqn_params)

        if is_elemwise:
            x = None
        else:
            x = etrace_xs_at_t[id(relation.x_var)]

        def _call_dt_to_t_dict(d: Any, trace_: Any, _rule: Any = dt_to_t_rule, _params: Any = eqn_params) -> Any:
            return _rule(d, trace_, **_params)

        def _comp_recurrent_legacy(diag_: Any, old_bwg_: Any, num_state_: Any) -> Any:
            """Legacy nested-vmap dt_to_t + sum path."""

            # Under ``brainstate.nn.Vmap(vmap_states='new')`` the hidden-state
            # Jacobian (diag) is per-lane and has lost its leading batch axis,
            # while the weight trace (old_bwg) still carries the singleton batch
            # from ``init_drtrl`` (``batch = x_var.shape[0]`` == 1 for a conv whose
            # forward forces a batch axis). Re-insert the matching singleton on
            # diag so the ``dt_to_t`` rule sees a consistent batch prefix on both
            # the cotangent and the trace (collapsed again by the solve-time sum).
            if batched and x is not None and diag_.ndim == x.ndim + 1:
                diag_ = diag_[None]

            def fn_bwg_pre(d: Any, _old: Any = old_bwg_) -> Any:
                return jax.tree.map(
                    lambda arr: _sum_dim(arr, axis=-1),
                    jax.vmap(_call_dt_to_t_dict, in_axes=-1, out_axes=-1)(d, _old),
                )

            # num_state == 1 shortcut: squeeze the size-1 alpha axis to skip
            # outer vmap overhead; re-expand at the end.
            if num_state_ == 1:
                d_squeezed = u.math.squeeze(diag_, axis=-2)
                res = fn_bwg_pre(d_squeezed)
                return jax.tree.map(lambda a: u.math.expand_dims(a, axis=-1), res)
            return jax.vmap(fn_bwg_pre, in_axes=-2, out_axes=-1)(diag_)

        group: HiddenGroup
        for group in relation.hidden_groups:

            df = etrace_ys_at_t[etrace_df_key(relation.y, group.index)]

            # Instantaneous term: diag(D_f^t) ⊗ x^t  (Dict[str, Array]).
            # Cast the update inputs to ``trace_dtype`` (no-op when None) so the
            # multiply-add runs in the trace precision and the new trace stays
            # there; Jacobians/learning-signal remain full precision elsewhere.
            phg_to_pw = relation_instant_term(
                relation, group, x, df, weights_dict,
                fast_solve=fast_solve, trace_dtype=trace_dtype,
            )

            w_key = (id(relation.y_var), group.index)
            diag = hid_group_jacobians[group.index]

            old_bwg = hist_etrace_vals[w_key]  # Dict[str, Array]

            # Recurrent term: D^t · ε^{t-1}.
            if use_fast:
                assert fp is not None  # use_fast implies a registered fast path
                new_bwg_pre = fp.recurrent(
                    _cast_to_dtype(diag, trace_dtype),
                    old_bwg,
                    group.trace_state_width,
                )
            else:
                new_bwg_pre = _comp_recurrent_legacy(
                    diag, old_bwg, group.trace_state_width)

            # new_bwg_pre + phg_to_pw per-leaf.
            new_bwg = jax.tree.map(
                u.math.add, new_bwg_pre, phg_to_pw, is_leaf=u.math.is_quantity,
            )
            new_etrace_bwg[w_key] = new_bwg

    return new_etrace_bwg


def _chunk_supported(relation: HiddenParamOpRelation, fast_solve: bool) -> bool:
    """Whether *relation* can take the chunk-factorized multi-step update.

    Requires a flat (non-descended) relation whose primitive has a registered
    chunk kernel with an applicable fast path, and no dedicated
    ``instant_drtrl`` / ``solve_drtrl`` rule (trace structure == parameter
    structure). Everything else falls back to the per-step scan.
    """
    if relation.control_flow_context is not None:
        return False
    if not fast_solve:
        return False
    if get_instant_drtrl_rule(relation.primitive) is not None:
        return False
    if get_solve_drtrl_rule(relation.primitive) is not None:
        return False
    fp = get_fast_path_rules(relation.primitive)
    return (
        fp is not None
        and fp.chunk is not None
        and fp.applicable(relation.eqn_params)
    )


def _update_param_dim_etrace_chunked(
    hist_etrace_vals: Dict[ETraceWG_Key, PyTree],
    stacked_jacobians: Tuple[
        Dict[ETraceX_Key, jax.Array],   # stacked weight x, leading axis T
        Dict[ETraceDF_Key, jax.Array],  # stacked weight df, leading axis T
        Sequence[jax.Array],            # stacked hidden-group Jacobians
    ],
    hidden_param_op_relations: Any,
    weight_path_to_vals: Dict[Path, PyTree],
    fast_solve: bool = True,
    trace_dtype: Optional[DTypeLike] = None,
) -> Dict[ETraceWG_Key, PyTree]:
    """Chunk-factorized multi-step trace update.

    Rolls the eligibility trace over a whole multi-step window in closed form
    (see ``FastPathRules.chunk`` and :func:`braintrace._misc.suffix_products`)
    for chunk-eligible relations, and falls back to the historical per-step
    ``jax.lax.scan`` (:func:`_update_param_dim_etrace_scan_fn`) for the rest —
    including descended-scan relations. Equal to the per-step roll up to
    floating-point reassociation.
    """
    xs_seq, dfs_seq, diags_seq = stacked_jacobians
    chunkable = [r for r in hidden_param_op_relations
                 if _chunk_supported(r, fast_solve)]
    legacy = [r for r in hidden_param_op_relations
              if not _chunk_supported(r, fast_solve)]

    new_etrace_bwg: Dict[ETraceWG_Key, PyTree] = dict(hist_etrace_vals)

    # suffix products are per hidden group; share them across relations
    p_cache: Dict[int, Tuple[jax.Array, jax.Array]] = {}
    relation: HiddenParamOpRelation
    for relation in chunkable:
        fp = get_fast_path_rules(relation.primitive)
        assert fp is not None and fp.chunk is not None  # _chunk_supported
        x_seq = (
            None if relation.x_var is None
            else _cast_to_dtype(xs_seq[etrace_x_key(relation.x_var)], trace_dtype)
        )
        group: HiddenGroup
        for group in relation.hidden_groups:
            gi = group.index
            if gi not in p_cache:
                p_seq, m_full = suffix_products(
                    diags_seq[gi], group.trace_state_width)
                p_cache[gi] = (
                    _cast_to_dtype(p_seq, trace_dtype),
                    _cast_to_dtype(m_full, trace_dtype),
                )
            p_seq, m_full = p_cache[gi]
            df_seq = dfs_seq[etrace_df_key(relation.y_var, gi)]
            if group.snap is not None:
                # same widening as the per-step body; ``df_seq`` is stacked over
                # time but the padded axis is still the trailing one.
                df_seq = widen_instant_term(df_seq, group.snap.num_neighbour)
            df_seq = _cast_to_dtype(df_seq, trace_dtype)
            w_key = (id(relation.y_var), gi)
            new_etrace_bwg[w_key] = fp.chunk(
                x_seq, df_seq, p_seq, m_full,
                hist_etrace_vals[w_key], group.trace_state_width,
            )

    if legacy:
        scan_fn = partial(
            _update_param_dim_etrace_scan_fn,
            weight_path_to_vals=weight_path_to_vals,
            hidden_param_op_relations=legacy,
            fast_solve=fast_solve,
            trace_dtype=trace_dtype,
        )
        legacy_keys = {
            (id(r.y_var), g.index) for r in legacy for g in r.hidden_groups
        }
        sub_hist = {k: hist_etrace_vals[k] for k in legacy_keys}
        sub_new = jax.lax.scan(scan_fn, sub_hist, stacked_jacobians)[0]
        new_etrace_bwg.update(sub_new)

    return new_etrace_bwg


def relation_solve_to_param(
    relation: HiddenParamOpRelation,
    group: HiddenGroup,
    dg_hidden: Any,
    etrace_data: Dict[str, Any],
    weight_vals: Dict[Path, PyTree],
    fast_solve: bool = True,
) -> Tuple[Dict[str, Any], bool]:
    r"""Contract one ``(relation, group)`` trace with a hidden-side signal.

    Computes :math:`\mathrm{signal} \cdot \varepsilon` for a single relation and
    group, reducing the hidden and trailing state axes and leaving the result in
    *trace-key* coordinates (``{trainable name: array}``), units restored.

    This is the composition the framework uses to turn a trace into a parameter
    gradient, and — with the trace replaced by a bare instantaneous term and the
    signal replaced by a random projection — it is also exactly
    :math:`\nu^\top J_f`. Extracted so that
    :class:`~braintrace._algorithm.random_projection_vjp.RandomProjectionVjpAlgorithm`
    inherits each primitive's documented regime instead of re-deriving it.

    Parameters
    ----------
    relation : HiddenParamOpRelation
        The relation being solved.
    group : HiddenGroup
        The hidden group whose signal is being contracted. ``group.snap`` selects
        the ``sparse_n`` gather; ``group.trace_state_width`` sizes the trailing
        axis.
    dg_hidden : array
        The hidden-side signal, shaped ``(*varshape, trace_state_width)``.
    etrace_data : dict
        The trace (or instantaneous term) in trace coordinates.
    weight_vals : dict
        Current ``ParamState`` values; read only by primitives with a dedicated
        ``solve_drtrl`` rule (LoRA chains the signal through the weights).
    fast_solve : bool, default True
        Whether a registered closed-form kernel may be used.

    Returns
    -------
    dg_weight_dict : dict
        ``{trainable name: array}``, ready for
        :func:`~braintrace._misc._route_grads_by_path`.
    batch_folded : bool
        Whether the batch reduction was folded into the closed-form einsum. When
        true the caller must **not** batch-sum the routed result again; see
        :func:`reduce_param_batch_axes`.
    """
    dt_to_t_rule = ETP_RULES_DT_TO_T[relation.primitive]
    solve_drtrl_rule = get_solve_drtrl_rule(relation.primitive)
    eqn_params = relation.eqn_params
    batched = is_batched_primitive(relation.primitive)
    # Fast path only for elementwise-yw primitives with no transform hook
    # (the closed-form solve drops f'(W) — gated by ``fp.applicable``).
    fp = get_fast_path_rules(relation.primitive)
    use_fast = fast_solve and fp is not None and fp.applicable(eqn_params)

    _call_rule_dict: Callable[..., Any]
    if solve_drtrl_rule is not None:
        # Dedicated solve rule (trace structure != parameter structure,
        # e.g. LoRA's effective-weight trace): chain the learning signal
        # through the current weights. The rule sees batch-free,
        # num_state-free slices — the vmap scaffolding below is shared
        # with the legacy ``dt_to_t`` path.
        weights_dict = relation_weights_dict(relation, weight_vals)

        def _call_solve_drtrl_dict(
            d: Any, trace_: Any,
            _rule: Any = solve_drtrl_rule, _params: Any = eqn_params,
            _weights: Any = weights_dict,
        ) -> Any:
            return _rule(d, trace_, _weights, **_params)

        _call_rule_dict = _call_solve_drtrl_dict
    else:
        def _call_dt_to_t_dict(d: Any, trace_: Any, _rule: Any = dt_to_t_rule, _params: Any = eqn_params) -> Any:
            return _rule(d, trace_, **_params)

        _call_rule_dict = _call_dt_to_t_dict

    dt_to_t = jax.vmap(_call_rule_dict) if batched else _call_rule_dict

    if group.snap is not None:
        # `sparse_n`: the trace's trailing axis indexes (neighbour, state)
        # pairs, so the learning signal has to be expressed in the same
        # coordinates before the contraction.
        dg_hidden = gather_learning_signal(dg_hidden, group.snap)

    # dimensionless processing (unit strip + restore). Apply per-leaf.
    etrace_data_unitless, fn_unit_restore = _remove_units(etrace_data)
    dg_hidden_unitless, _ = _remove_units(dg_hidden)

    # Under ``brainstate.nn.Vmap(vmap_states='new')`` a batched primitive
    # (necessarily conv here — dense/lora/sparse dispatch to their
    # unbatched variants when per-lane) has a per-lane hidden cotangent
    # that lost its leading batch axis, while the weight trace keeps the
    # singleton batch from ``init_drtrl``. Both the batched ``dt_to_t`` and
    # the closed-form solve map a shared leading batch axis, so re-insert
    # the matching singleton on the cotangent; the trailing solve-time sum
    # (``has_batched`` branch in :func:`reduce_param_batch_axes`) collapses it
    # again.
    if batched:
        _trace_lead = jax.tree.leaves(etrace_data_unitless)[0].shape[0]
        dg_hidden_unitless = jax.tree.map(
            lambda a: a[None] if (a.ndim >= 1 and a.shape[0] != _trace_lead) else a,
            dg_hidden_unitless,
        )

    if use_fast:
        assert fp is not None  # use_fast implies a registered fast path
        # Upcast a reduced-precision trace to (at least) the learning-
        # signal dtype so the gradient reduction accumulates in full
        # precision. ``promote_types`` never downcasts, so this is a
        # no-op for the default fp32 trace.
        sig_dtype = jax.tree.leaves(dg_hidden_unitless)[0].dtype
        etrace_for_solve = jax.tree.map(
            lambda a: a.astype(jnp.promote_types(a.dtype, sig_dtype)),
            etrace_data_unitless,
        )
        # Closed-form einsum path for mm/mv/elemwise primitives. For a
        # batched primitive, fold the batch reduction into the einsum so
        # no (B, I, O) intermediate is materialized; record the routed
        # paths so the trailing batch-sum skips them (already reduced).
        dg_weight_dict = fp.solve(
            dg_hidden_unitless, etrace_for_solve, fold_batch=batched,
        )
    elif group.trace_state_width == 1:
        # width==1 shortcut: skip outer vmap of size 1. Reads the widened
        # width, not num_state: under `sparse_n` a single-state group
        # with K > 1 has a size-K trailing axis and must NOT take it.
        dg_hid_squeezed = jax.tree.map(
            lambda a: u.math.squeeze(a, axis=-1), dg_hidden_unitless
        )
        etr_squeezed = jax.tree.map(
            lambda a: u.math.squeeze(a, axis=-1), etrace_data_unitless
        )
        dg_weight_dict = dt_to_t(dg_hid_squeezed, etr_squeezed)
    else:
        dg_weight_dict = jax.tree.map(
            lambda arr: _sum_dim(arr, axis=-1),
            jax.vmap(dt_to_t, in_axes=-1, out_axes=-1)(
                dg_hidden_unitless, etrace_data_unitless
            ),
        )
    return fn_unit_restore(dg_weight_dict), bool(use_fast and batched)


def reduce_param_batch_axes(
    temp_data: Dict[Path, PyTree],
    weight_vals: Dict[Path, PyTree],
    folded_paths: set,
    batched_paths: set,
) -> None:
    """Collapse leading batch axes on routed parameter gradients, in place.

    Paths routed through the fast-path einsum (*folded_paths*) were already
    reduced via ``fold_batch``; unbatched-primitive paths (not in
    *batched_paths*) never grew a batch axis and must be left intact.

    Parameters
    ----------
    temp_data : dict
        ``{path: gradient}``, mutated in place.
    weight_vals : dict
        ``{path: ParamState value}``, used as the rank reference.
    folded_paths : set
        Paths whose batch axis was folded into the einsum.
    batched_paths : set
        Paths owned by a batched primitive.
    """
    for key, val in temp_data.items():
        if key in folded_paths:
            continue
        if key in batched_paths:
            temp_data[key] = jax.tree.map(lambda x: u.math.sum(x, axis=0), val)
        else:
            # Unbatched-primitive paths usually carry no batch axis. But under
            # ``brainstate.mixin.Batching()`` a diagonal op with no ``x`` carrier
            # (``etp_elemwise``: its output is the weight itself, so neither its
            # input nor output rank reveals the batch) still acquires a leading
            # batch axis from the batched hidden state it feeds. ``is_batched_
            # primitive`` does not flag it, so reduce any leading axes the
            # parameter itself does not have. This is a no-op for genuinely
            # unbatched paths (e.g. ``etp_mv``) whose gradient already matches
            # the parameter rank, and for the per-lane vmap path.
            ref = weight_vals[key]
            temp_data[key] = jax.tree.map(
                lambda g, p: (
                    u.math.sum(g, axis=tuple(range(u.math.ndim(g) - u.math.ndim(p))))
                    if u.math.ndim(g) > u.math.ndim(p) else g
                ),
                val, ref,
            )


def _solve_param_dim_weight_gradients(
    hist_etrace_data: Dict[ETraceWG_Key, PyTree],  # the history etrace data
    dG_weights: Dict[Path, dG_Weight],  # weight gradients
    dG_hidden_groups: Sequence[jax.Array],  # hidden group gradients
    weight_hidden_relations: Sequence[HiddenParamOpRelation],
    weight_vals: Dict[Path, PyTree],  # current ParamState pytree values for structure
    fast_solve: bool = True,
) -> None:
    """
    Compute and update the weight gradients for parameter dimensions using eligibility trace data.

    This function calculates the weight gradients by utilizing the eligibility trace data and the
    hidden-to-hidden Jacobians. It applies a correction factor to avoid exponential smoothing bias
    at the beginning of the computation, and updates the ``dG_weights`` dictionary in place.

    Parameters
    ----------
    hist_etrace_data : Dict[ETraceWG_Key, PyTree]
        A dictionary containing historical eligibility trace data for the weight
        gradients, keyed by ETraceWG_Key.
    dG_weights : Dict[Path, dG_Weight]
        A dictionary to store the computed weight gradients, keyed by the path
        of the weight.
    dG_hidden_groups : Sequence[jax.Array]
        A sequence of hidden group gradients, with the same length as the total
        number of hidden groups.
    weight_hidden_relations : Sequence[HiddenParamOpRelation]
        A sequence of HiddenParamOpRelation objects representing the
        relationships between hidden parameters and operations.
    weight_vals : Dict[Path, PyTree]
        Current ParamState pytree values, used as the structure template.
    fast_solve : bool, optional
        Whether to use the per-primitive fast contraction instead of the legacy
        vmap path.
    """
    # update the etrace weight gradients
    temp_data: Dict[Path, PyTree] = dict()
    # Paths whose gradient was already batch-reduced inside the fast-path einsum
    # (fold_batch). The trailing batch-sum must skip these.
    folded_paths: set = set()
    # Paths owned by a *batched* primitive: only these carry a leading batch axis
    # in ``temp_data`` and so only these may be batch-summed. A model can mix
    # batched and unbatched primitives in one solve — under
    # ``brainstate.nn.Vmap(vmap_states='new')`` a conv stays batched while a
    # ``Linear`` dispatches to the unbatched ``etp_mv`` (1-D per-lane input) — and
    # summing the unbatched gradient would collapse its leading (in-feature) axis.
    batched_paths: set = set()
    for relation in weight_hidden_relations:
        if is_batched_primitive(relation.primitive):
            batched_paths.update(relation.trainable_paths.values())

        group: HiddenGroup
        for group in relation.hidden_groups:

            w_key = (id(relation.y_var), group.index)
            dg_weight_dict, batch_folded = relation_solve_to_param(
                relation,
                group,
                dG_hidden_groups[group.index],
                hist_etrace_data[w_key],
                weight_vals,
                fast_solve=fast_solve,
            )
            if batch_folded:
                folded_paths.update(relation.trainable_paths.values())

            # Route per-key to owning ParamState path.
            _route_grads_by_path(relation, dg_weight_dict, weight_vals, temp_data)

    #
    # Step 3:
    #
    # sum up the batched weight gradients
    reduce_param_batch_axes(temp_data, weight_vals, folded_paths, batched_paths)

    # update the weight gradients
    for key, val in temp_data.items():
        _update_dict(dG_weights, key, val)


def _remove_units(xs_maybe_quantity: PyTree) -> Any:
    """
    Removes units from a PyTree of quantities, returning a unitless PyTree and a function to restore the units.

    This function traverses a PyTree structure, removing units from each quantity and returning a new PyTree
    with the same structure but without units. It also returns a function that can be used to restore the
    original units to the unitless PyTree.

    Parameters
    ----------
    xs_maybe_quantity : PyTree
        A PyTree structure containing quantities with units.

    Returns
    -------
    Tuple[PyTree, Callable]
        A tuple containing:

        - A PyTree with the same structure as the input, but with units removed
          from each quantity.
        - A function that takes a unitless PyTree and restores the original
          units to it.
    """
    leaves, treedef = jax.tree.flatten(xs_maybe_quantity, is_leaf=u.math.is_quantity)
    new_leaves, units = [], []
    for leaf in leaves:
        leaf, unit = u.split_mantissa_unit(leaf)
        new_leaves.append(leaf)
        units.append(unit)

    def restore_units(xs_unitless: PyTree) -> Any:
        leaves, treedef2 = jax.tree.flatten(xs_unitless)
        # jax's PyTreeDef stubs omit __eq__; the comparison is valid at runtime.
        assert treedef == treedef2, 'The tree structure should be the same. '  # type: ignore[operator]
        new_leaves = [
            leaf if unit.dim.is_dimensionless else leaf * unit
            for leaf, unit in zip(leaves, units)
        ]
        return jax.tree.unflatten(treedef, new_leaves)

    return jax.tree.unflatten(treedef, new_leaves), restore_units


class ParamDimVjpAlgorithm(ETraceVjpAlgorithm):
    r"""Online gradient algorithm with diagonal approximation and parameter-dimension complexity.

    This algorithm computes the gradients of the weights with the diagonal
    approximation and the parameter-dimension complexity. It implements the
    linear-memory estimator of Wang et al. [1]_ on the RTRL foundation of
    Williams and Zipser [2]_.

    Parameters
    ----------
    model : brainstate.nn.Module
        The model function, which receives the input arguments and returns the
        model output.
    name : str, optional
        Name of the eligibility-trace algorithm.
    vjp_method : str, optional
        The method for computing the VJP. It should be either ``"single-step"``
        or ``"multi-step"``.

        - ``"single-step"``: the VJP is computed at the current time step, i.e.,
          :math:`\partial L^t/\partial h^t`.
        - ``"multi-step"``: the VJP is computed at multiple time steps, i.e.,
          :math:`\partial L^t/\partial h^{t-k}`, where :math:`k` is determined by
          the data input.
    fast_solve : bool, optional
        Whether to use the closed-form per-primitive contractions when
        available. The default is ``True``.
    trace_dtype : dtype, optional
        Storage dtype for supported fast-path eligibility traces. ``None``
        preserves the native dtype.
    chunked_trace : bool, optional
        When ``True`` (default) and the input spans multiple time steps, the
        eligibility-trace roll over the window is computed in closed form —
        suffix products of the hidden-to-hidden Jacobians plus a single
        time-contracting einsum — instead of a per-step scan. Mathematically
        identical to the per-step roll (up to floating-point reassociation),
        but converts the dominant per-step elementwise passes over the
        parameter-sized trace into matmul-class kernels (~an order of
        magnitude faster on long windows). Relations without a chunk kernel
        (conv / sparse / LoRA / grouped), descended-scan relations, and
        relations with an active ``weight_fn`` / ``bias_fn`` transform fall
        back to the per-step scan automatically. Single-step input is
        unaffected. Note: chunking stacks the per-step Jacobians
        (``O(T · B · (I + H))`` memory for a window of length ``T``) instead
        of fusing the roll into the forward scan; for very long windows either
        feed the sequence in smaller windows (the trace carries across calls)
        or set ``chunked_trace=False``.
    control_flow : ControlFlowPolicy, optional
        Policy governing control-flow canonicalization (cond if-conversion,
        scan unrolling, structured scan descent, ...) during graph
        compilation. ``None`` (default) uses
        ``ControlFlowPolicy()``.
    config : ETraceConfig, optional
        Learning-rule coordinates. ``None`` uses the parameter-dimensional
        preset.
    random_feedback_key : jax.Array, optional
        Key used to initialize fixed random-feedback projections when the
        selected config requests them.
    snap_max_jacobian_elements : int, optional
        Maximum number of elements permitted in each SnAp widened block
        Jacobian. The default is ``16777216``.

    Notes
    -----
    The learning rule is

    .. math::

        \begin{aligned}
        &\boldsymbol{\epsilon}^t \approx \mathbf{D}^t \boldsymbol{\epsilon}^{t-1}+\operatorname{diag}\left(\mathbf{D}_f^t\right) \otimes \mathbf{x}^t \\
        & \nabla_{\boldsymbol{\theta}} \mathcal{L}=\sum_{t^{\prime} \in \mathcal{T}} \frac{\partial \mathcal{L}^{t^{\prime}}}{\partial \mathbf{h}^{t^{\prime}}} \circ \boldsymbol{\epsilon}^{t^{\prime}}
        \end{aligned}

    where :math:`\boldsymbol{\epsilon}^t` is the per-parameter eligibility
    trace, :math:`\mathbf{D}^t` the hidden-to-hidden Jacobian, :math:`\mathbf{D}_f^t`
    the state-to-output Jacobian, :math:`\mathbf{x}^t` the presynaptic input, and
    :math:`\partial \mathcal{L}^{t'}/\partial \mathbf{h}^{t'}` the learning
    signal back-propagated from the loss at each step.

    :math:`\mathbf{D}_f^t` is read off by
    ``ETraceVjpGraphExecutor._compute_hid2weight_jacobian``
    from a single all-ones-tangent ``jax.jvp`` of the ``y -> hidden`` map; see
    that method's docstring for when this is exact (elementwise maps) versus
    an approximation (non-elementwise maps, e.g. a normalization layer
    between the weight op and the neuron) — the same approximation is shared
    with :class:`~braintrace._algorithm.io_dim_vjp.IODimVjpAlgorithm`.

    Real-Time Recurrent Learning (RTRL) propagates the full sensitivity
    :math:`\partial \mathbf{h}^t/\partial \boldsymbol{\theta}` forward in time,
    which costs :math:`O(|\theta| \cdot H)` memory. D-RTRL keeps only the
    *diagonal* of the hidden-to-hidden Jacobian, collapsing the trace to one
    value per parameter. The trace is then contracted with the instantaneous
    learning signal at each step to accumulate the gradient — no backward pass
    through time and memory linear in the parameter count.

    :class:`ParamDimVjpAlgorithm` is a subclass of :class:`brainstate.nn.Module`
    and is sensitive to the context/mode of the computation. In particular, it is
    sensitive to ``brainstate.mixin.Batching`` behavior.

    For dense (linear) transformation layers this algorithm has
    :math:`O(B\theta)` memory complexity, where :math:`\theta` is the number
    of parameters and :math:`B` the batch size — the weight gradients are
    computed with :math:`O(BIO)` complexity, where :math:`I` and :math:`O`
    are the number of input and output dimensions.

    For a convolutional layer the exact eligibility trace must keep one
    kernel-shaped slot **per spatial output position** — the kernel is
    spatially shared while the diagonal discount acts per output element,
    so a spatially pre-summed (kernel-shaped) trace cannot follow the
    recurrence. The conv trace therefore costs :math:`O(B S \theta)` memory,
    where :math:`S` is the number of spatial output positions and
    :math:`\theta` the kernel parameter count. For large convolutions prefer
    the IO-dim algorithm (``pp_prop`` / :class:`IODimVjpAlgorithm`), whose
    conv trace stays output-shaped.

    For more details, please see `the D-RTRL algorithm presented in our manuscript <https://www.biorxiv.org/content/10.1101/2024.09.24.614728v2>`_.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>> import jax.numpy as jnp
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
        >>> x0 = brainstate.random.randn(1)
        >>> # ``D_RTRL`` is the concrete parameter-dimensional preset; one call
        >>> # initialises states, builds the trace graph, and returns a learner.
        >>> learner = braintrace.compile(model, braintrace.D_RTRL, x0)
        >>> y = learner(x0)             # forward pass + eligibility-trace update
        >>>
        >>> # etrace_grad drives the sequence and accumulates the online gradients
        >>> xs = brainstate.random.randn(10, 1)   # (T, ...)
        >>> ys = brainstate.random.randn(10, 1)
        >>> def step_loss(x, y):
        ...     return jnp.mean((learner(x) - y) ** 2)
        >>> grads, losses = learner.etrace_grad(xs, ys, step_fn=step_loss, return_value=True)

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

    # batch of weight gradients
    etrace_bwg: Dict[ETraceWG_Key, brainstate.State]

    #: ``trace_filter='kappa'`` state, keyed exactly like :attr:`etrace_bwg`
    #: (``ETraceWG_Key == (id(y_var), group.index)``): one filter per trainable
    #: weight / HiddenGroup relation, holding a PyTree that mirrors the raw
    #: trace's own structure and shape. Empty under ``trace_filter='none'``.
    _trace_filters: Dict[ETraceWG_Key, brainstate.State]

    _supports_scan_descent = True
    """Structured scan descent (Phase 4): the param-dim trace update folds
    per-substep injections with an inner scan; see
    ``braintrace._compiler.scan_descent``."""

    def __init__(
        self,
        model: brainstate.nn.Module,
        name: Optional[str] = None,
        vjp_method: str = 'single-step',
        fast_solve: bool = True,
        trace_dtype: Optional[DTypeLike] = None,
        chunked_trace: bool = True,
        control_flow: Optional[ControlFlowPolicy] = None,
        config: Optional[ETraceConfig] = None,
        random_feedback_key: Optional[jax.Array] = None,
        snap_max_jacobian_elements: int = DEFAULT_MAX_JACOBIAN_ELEMENTS,
    ) -> None:
        super().__init__(model, name=name, vjp_method=vjp_method,
                         control_flow=control_flow, config=config,
                         random_feedback_key=random_feedback_key,
                         snap_max_jacobian_elements=snap_max_jacobian_elements)
        if self.config.trace_factorization != 'per_param':
            raise ValueError(
                f'{type(self).__name__} is the per-parameter trace engine, but '
                f'the config asks for '
                f'trace_factorization={self.config.trace_factorization!r}. Use '
                f'the engine matching the factorization, or '
                f"braintrace.compile(model, config, ...) which picks it for you."
            )
        # ``fast_solve=True`` enables closed-form einsum kernels for
        # mm/mv/elemwise primitives, replacing the nested-vmap legacy path.
        # Conv / sparse / LoRA primitives always use the legacy path.
        self.fast_solve = fast_solve
        # Optional reduced-precision storage for the eligibility trace (e.g.
        # ``jnp.bfloat16`` / ``jnp.float16``); ``None`` keeps native fp32. Only
        # applies on the fast path: a relation's trace is cast to
        # ``trace_dtype`` iff ``fast_solve`` is ``True`` *and* the relation's
        # primitive has a registered fast-path bundle whose ``applicable``
        # gate accepts its ``eqn_params`` (false, e.g., when a ``weight_fn`` /
        # ``bias_fn`` transform is active). When that predicate is false the
        # trace stays at native precision, since the legacy update always
        # produces native-precision arrays and a mismatched scan-carry dtype
        # would raise at trace time.
        self.trace_dtype = trace_dtype
        # ``chunked_trace=True`` computes the multi-step trace roll in closed
        # form (chunk factorization) instead of a per-step scan. Identical to
        # the per-step roll up to floating-point reassociation; relations
        # without a chunk kernel fall back automatically. Single-step input is
        # unaffected.
        self.chunked_trace = chunked_trace

    def init_etrace_state(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the eligibility trace states of the etrace algorithm.

        This method is needed after compiling the etrace graph. See
        :meth:`~braintrace.ETraceAlgorithm.compile_graph` for the details.
        """
        # The states of batched weight gradients
        self.etrace_bwg = dict()
        for relation in self.graph.hidden_param_op_relations:
            _init_param_dim_state(
                self.etrace_bwg, relation, self.trace_dtype, self.fast_solve,
            )

        # `trace_filter`: one filter per raw-trace key, initialised to zeros
        # with the exact PyTree structure/shape of that trace (batch axis,
        # weight shape, and the trailing num_state axis all included) -- no
        # reduction, so per-state channels never mix. It mirrors the trace
        # pytree and so cannot be sized before the traces exist.
        self._trace_filters = {}
        if self.config.trace_filter == 'kappa':
            for trace_key, trace_state in self.etrace_bwg.items():
                self._trace_filters[trace_key] = brainstate.ShortTermState(
                    jax.tree.map(jnp.zeros_like, trace_state.value)
                )

        # Last: the base allocates the axis-side state that is not ours.
        super().init_etrace_state(*args, **kwargs)

    def reset_state(self, batch_size: int | None = None, **kwargs: Any) -> None:
        """Reset the eligibility trace states.

        Parameters
        ----------
        batch_size : int, optional
            The batch size used to reshape the reset trace states. Default ``None``.
        """
        self.running_index.value = 0
        _reset_state_in_a_dict(self.etrace_bwg, batch_size)
        _reset_state_in_a_dict(self._trace_filters, batch_size)

    def get_etrace_of(self, weight: brainstate.ParamState | Path) -> Dict:
        """Get the eligibility trace of the given weight.

        Parameters
        ----------
        weight : brainstate.ParamState or Path
            The weight whose eligibility trace is requested, given either as a
            :class:`brainstate.ParamState` instance or as its path in the model.

        Returns
        -------
        dict
            A dictionary mapping ``(y_var id, hidden-group index)`` keys to the
            eligibility-trace values associated with the given weight.

        Raises
        ------
        ValueError
            If no eligibility trace is found for the given weight.
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
        relation: HiddenParamOpRelation
        for relation in self.graph.hidden_param_op_relations:
            primary_state = next(iter(relation.trainable_param_states.values()), None)
            if primary_state is None or id(primary_state) != weight_id:
                continue
            find_this_weight = True

            # retrieve the etrace data
            group: HiddenGroup
            for group in relation.hidden_groups:
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

        Returns
        -------
        Dict[ETraceWG_Key, jax.Array]
            A dictionary mapping eligibility trace keys to their current values.
            Each key represents a specific trace component (typically involving a
            parameter and hidden state relationship), and the corresponding value
            represents the accumulated eligibility trace.
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

        Parameters
        ----------
        etrace_vals : Dict[ETraceWG_Key, jax.Array]
            Dictionary mapping eligibility trace keys to their updated values.
            Each key represents a specific parameter-hidden state relationship,
            and the value represents the updated eligibility trace value.
        """
        for x, val in etrace_vals.items():
            self.etrace_bwg[x].value = val

    def _make_scan_fn(self, weight_vals: Dict[Path, PyTree]) -> Callable:
        """Build the per-step D-RTRL eligibility-trace stepper (scan body).

        Returns the ``partial`` of :func:`_update_param_dim_etrace_scan_fn`
        used both as the fused in-scan stepper (see
        :meth:`_make_etrace_stepper`) and as the body of
        :meth:`_update_etrace_data`'s own trace scan.
        """
        return partial(
            _update_param_dim_etrace_scan_fn,
            weight_path_to_vals=weight_vals,
            hidden_param_op_relations=self.graph.hidden_param_op_relations,
            fast_solve=self.fast_solve,
            trace_dtype=self.trace_dtype,
        )

    def _make_etrace_stepper(self, weight_vals: Dict[Path, PyTree]) -> Optional[Callable]:
        """Per-step stepper for in-scan fusion, or ``None`` under chunking.

        When ``chunked_trace`` is disabled, returns the per-step scan body so
        the graph executor fuses the roll into its over-time scan (see the
        base-class :meth:`_make_etrace_stepper`). When ``chunked_trace`` is
        enabled, returns ``None`` so the executor stacks the per-step
        Jacobians, which :meth:`_update_etrace_data` then consumes in closed
        form (:func:`_update_param_dim_etrace_chunked`).
        """
        if self.chunked_trace:
            return None
        return self._make_scan_fn(weight_vals)

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

        Parameters
        ----------
        running_index : int, optional
            Current timestep counter, used for correcting exponential smoothing bias.
        hist_etrace_vals : Dict[ETraceWG_Key, PyTree]
            Dictionary containing historical eligibility trace values from previous timestep.
            Keys are tuples identifying parameter-hidden state relationships.
        hid2weight_jac_single_or_multi_times : Hid2WeightJacobian
            Jacobians of hidden states with respect to weights at the current timestep.
            Contains input gradients and differential function gradients.
        hid2hid_jac_single_or_multi_times : HiddenGroupJacobian
            Jacobians between hidden states (recurrent connections) at the current timestep.
        weight_vals : Dict[Path, PyTree]
            Dictionary mapping paths to current weight values in the model.
        input_is_multi_step : bool
            Whether the Jacobian inputs span multiple time steps, selecting the
            scan (or chunked) path over the single-step path.

        Returns
        -------
        Dict[ETraceWG_Key, PyTree]
            Updated eligibility trace values dictionary with the same structure as
            ``hist_etrace_vals`` but containing new values for the current timestep.
        """

        jacobians = (
            hid2weight_jac_single_or_multi_times[0],
            hid2weight_jac_single_or_multi_times[1],
            hid2hid_jac_single_or_multi_times,
        )

        if input_is_multi_step:
            if self.chunked_trace:
                new_etrace = _update_param_dim_etrace_chunked(
                    hist_etrace_vals,
                    jacobians,
                    self.graph.hidden_param_op_relations,
                    weight_vals,
                    fast_solve=self.fast_solve,
                    trace_dtype=self.trace_dtype,
                )
            else:
                new_etrace = jax.lax.scan(
                    self._make_scan_fn(weight_vals),
                    hist_etrace_vals,
                    jacobians,
                )[0]

        else:
            new_etrace = self._make_scan_fn(weight_vals)(
                hist_etrace_vals,
                jacobians,
            )[0]

        return new_etrace

    def _solve_weight_gradients(
        self,
        running_index: int,
        etrace_h2w_at_t: Dict[ETraceWG_Key, PyTree],
        dl_to_hidden_groups: Sequence[jax.Array],
        weight_vals: Dict[Path, PyTree],
        dl_to_nonetws_at_t: Dict[Path, PyTree],
        dl_to_etws_at_t: Optional[Dict[Path, PyTree]],
    ) -> Any:
        """Compute weight gradients using parameter dimension eligibility traces.

        This method implements the parameter dimension D-RTRL algorithm's weight gradient
        computation. It combines the eligibility traces with the gradients of the loss
        with respect to hidden states to compute the full parameter gradients according to:

        ∇_θ L = ∑_{t' ∈ T} ∂L^{t'}/∂h^{t'} ∘ ε^{t'}

        Where ε represents the eligibility traces and ∂L/∂h are the gradients of the loss
        with respect to hidden states.

        Parameters
        ----------
        running_index : int
            Current timestep counter used for bias correction.
        etrace_h2w_at_t : Dict[ETraceWG_Key, PyTree]
            Eligibility trace values at the current timestep, mapping parameter-hidden
            state relationship keys to trace values.
        dl_to_hidden_groups : Sequence[jax.Array]
            Gradients of the loss with respect to hidden states at the current timestep.
        weight_vals : Dict[Path, PyTree]
            Current values of all weights in the model.
        dl_to_nonetws_at_t : Dict[Path, PyTree]
            Gradients of non-eligibility trace parameters at the current timestep.
        dl_to_etws_at_t : Dict[Path, PyTree], optional
            Optional additional gradients for eligibility trace parameters at the
            current timestep.

        Returns
        -------
        Dict[Path, PyTree]
            Dictionary mapping parameter paths to their gradient values.
        """
        # `trace_filter`: low-pass the raw trace before it is contracted with the
        # learning signal. Applied elementwise (`jax.tree.map` over the trace's
        # own PyTree, including its trailing num_state axis) so multi-state
        # HiddenGroups are filtered independently per state -- never summed
        # across states and broadcast back.
        #
        #     e_bar^t_{ji} = kappa * e_bar^{t-1}_{ji} + e^t_{ji}
        #
        if self._trace_filters:
            kappa = self.config.kappa
            filtered: Dict[ETraceWG_Key, PyTree] = {}
            for key, trace in etrace_h2w_at_t.items():
                flt = self._trace_filters.get(key)
                if flt is None:
                    filtered[key] = trace
                    continue
                new_val = jax.tree.map(
                    lambda prev, e: kappa * prev + e, flt.value, trace
                )
                flt.value = new_val
                filtered[key] = new_val
            etrace_h2w_at_t = filtered

        dG_weights: Dict[Path, Any] = {path: None for path in self.param_states}

        # update the etrace weight gradients
        _solve_param_dim_weight_gradients(
            etrace_h2w_at_t,
            dG_weights,
            dl_to_hidden_groups,
            self.graph.hidden_param_op_relations,
            weight_vals,
            fast_solve=self.fast_solve,
        )

        # update the non-etrace weight gradients
        for path, dg in dl_to_nonetws_at_t.items():
            _update_dict(dG_weights, path, dg)

        # update the etrace parameters when "dl_to_etws_at_t" is not None
        if dl_to_etws_at_t is not None:
            for path, dg in dl_to_etws_at_t.items():
                _update_dict(dG_weights, path, dg, error_when_no_key=True)
        return dG_weights
