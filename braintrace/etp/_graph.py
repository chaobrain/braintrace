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
ETP compiled graph and the ``compile_etp_graph`` entry-point.

Reuses existing infrastructure (``ModuleInfo``, ``HiddenGroup``,
``HiddenPerturbation``) but replaces the operator identification step
with primitive matching via ``find_etp_relations_from_minfo``.
"""

from typing import Dict, NamedTuple, Optional, Sequence, Tuple

import brainstate
import jax

from braintrace._etrace_compiler.graph import compiler_context, order_hidden_group_index
from braintrace._etrace_compiler.hidden_group import (
    HiddenGroup,
    find_hidden_groups_from_minfo,
)
from braintrace._etrace_compiler.hidden_pertubation import (
    HiddenPerturbation,
    add_hidden_perturbation_from_minfo,
)
from braintrace._etrace_compiler.module_info import (
    ModuleInfo,
    extract_module_info,
)
from braintrace._typing import Inputs, Path
from ._compiler import ETPOpRelation, find_etp_relations_from_minfo

__all__ = [
    'ETPGraph',
    'compile_etp_graph',
]


class ETPGraph(NamedTuple):
    """
    Compiled ETP graph.

    This is the ETP equivalent of ``ETraceGraph``.  It contains all
    structural information needed for eligibility-trace computation.
    """

    module_info: ModuleInfo
    hidden_groups: Sequence[HiddenGroup]
    hid_path_to_group: Dict[Path, HiddenGroup]
    etp_op_relations: Sequence[ETPOpRelation]
    hidden_perturb: Optional[HiddenPerturbation]

    def call_hidden_perturb(
        self,
        args: Inputs,
        perturb_data: Sequence[jax.Array],
        old_state_vals: Optional[Sequence[jax.Array]] = None,
    ):
        if old_state_vals is None:
            old_state_vals = [st.value for st in self.module_info.compiled_model_states]
        jaxpr_outs = self.hidden_perturb.eval_jaxpr(
            jax.tree.leaves((args, old_state_vals)),
            perturb_data,
        )
        return self.module_info._process(*args, jaxpr_outs=jaxpr_outs)

    def dict(self) -> Dict:
        return self._asdict()

    def __repr__(self) -> str:
        return repr(
            brainstate.util.PrettyMapping(
                self._asdict(), type_name=self.__class__.__name__
            )
        )


ETPGraph.__module__ = 'braintrace.etp'


def compile_etp_graph(
    model: brainstate.nn.Module,
    *model_args: Tuple,
    include_hidden_perturb: bool = True,
) -> ETPGraph:
    """
    Compile an ETP graph for *model*.

    This mirrors ``compile_etrace_graph`` but uses primitive matching
    (``find_etp_relations_from_minfo``) instead of JIT-name matching.

    Args:
        model: A ``brainstate.nn.Module`` whose ``update`` (or ``__call__``)
            defines a single recurrent step.
        model_args: Example inputs (used for Jaxpr tracing).
        include_hidden_perturb: Whether to add hidden-state perturbation
            variables for Jacobian computation.

    Returns:
        An ``ETPGraph`` containing all structural information.
    """
    with compiler_context('compile_etp_graph'):
        assert isinstance(model_args, tuple)
        minfo = extract_module_info(model, *model_args)

        # --- Hidden groups ---
        hidden_groups, hid_path_to_group = find_hidden_groups_from_minfo(minfo)
        order_hidden_group_index(hidden_groups)

        # --- ETP op relations (replaces hid_param_op) ---
        etp_relations = find_etp_relations_from_minfo(
            minfo=minfo,
            hid_path_to_group=hid_path_to_group,
        )

        # --- Rewrite Jaxpr to expose intermediate variables ---
        original_outvars = set(minfo.jaxpr.outvars)
        extra_vars = []

        # Weight x vars
        for rel in etp_relations:
            if rel.x_var is not None and rel.x_var not in original_outvars:
                extra_vars.append(rel.x_var)

        # y-to-hidden transition vars
        for rel in etp_relations:
            for tj in rel.y_to_hidden_group_jaxprs:
                for v in tj.invars + tj.constvars:
                    if v not in original_outvars:
                        extra_vars.append(v)

        # Hidden-to-hidden transition vars
        for group in hidden_groups:
            for v in group.hidden_invars:
                if v not in original_outvars:
                    extra_vars.append(v)
            for v in group.transition_jaxpr_constvars:
                if v not in original_outvars:
                    extra_vars.append(v)

        # State vars
        for v in minfo.jaxpr.outvars[minfo.num_var_out:]:
            if v not in original_outvars:
                extra_vars.append(v)

        # Deduplicate while preserving order
        seen = set(original_outvars)
        deduped = []
        for v in extra_vars:
            if v not in seen:
                seen.add(v)
                deduped.append(v)

        if deduped:
            minfo = minfo.add_jaxpr_outs(deduped)

        # --- Hidden perturbation ---
        hidden_perturb = (
            add_hidden_perturbation_from_minfo(minfo)
            if include_hidden_perturb
            else None
        )

        return ETPGraph(
            module_info=minfo,
            hidden_groups=hidden_groups,
            hid_path_to_group=hid_path_to_group,
            etp_op_relations=etp_relations,
            hidden_perturb=hidden_perturb,
        )
