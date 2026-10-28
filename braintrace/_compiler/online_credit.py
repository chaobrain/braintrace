# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Which weights an online algorithm can credit at all (#199).

An online algorithm credits a weight in two ways. The eligibility trace of each
hidden group the weight feeds carries the weight's influence across steps
*within that group*. The learning signal then reads, in the same step, how the
loss depends on the group's hidden values: on the new values (every algorithm),
and also on the old ones under a multi-step VJP that is given one step at a
time. Nothing else crosses a step. A value written into a state the trace does
not follow -- a delay line held in a ``ShortTermState``, or, for a single-step
algorithm, a hidden state of another group that only a later step reads -- is
a given input to the next step.

Reverse mode through time (BPTT) follows every state. So a weight whose
influence reaches the loss only through such a state has a non-zero BPTT
gradient and an online gradient of exactly zero. This module finds those
weights from the step's jaxpr:

- ``B_now``: the variables that reach a used output within the step (one
  backward sweep);
- ``B_all``: the same, closed over steps: a state whose old value is in it
  makes its new value a target too (one backward sweep with a work list);
- the online credit of a weight is non-zero iff one of its ETP outputs, its
  weight inputs or its groups' new hidden values (and, under a multi-step VJP
  of one step, old hidden values) is in ``B_now``; its BPTT credit is non-zero
  iff one of its ETP outputs or weight inputs is in ``B_all``.

Both sweeps are linear in the number of equations. The non-hidden states each
weight writes in a step come from one more backward pass that propagates
bitsets, so no sweep runs per weight.

Equations are opaque: every input reaches every output, including through
nested bodies. A reference (``jax.new_ref``) taken by an equation counts as an
output of it, so a value stored in a reference reaches its later reads. Both
are over-approximations, which can only hide a report, never invent one.

Known false negative (#216): a buffer written and then read back in the same
step (a ring updated before it is read, as ``brainstate.nn.Delay.update``
followed by ``retrieve_at_step`` does) is tracked as one array, so its old
slots look like part of the new value and the weight is not reported.
"""

from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

import brainstate
import jax

from braintrace._compatible_imports import Var
from braintrace._typing import Path

__all__ = [
    'CreditFinding',
    'OnlineCreditGraph',
    'credit_message',
]


def _is_ref(v: Var) -> bool:
    return type(getattr(v, 'aval', None)).__name__ == 'AbstractRef'


class CreditFinding(NamedTuple):
    """A weight whose online credit misses part or all of its BPTT credit."""

    weight_path: Path
    """The weight's state path."""

    hidden_paths: Tuple[Path, ...]
    """The hidden states of the weight's hidden groups."""

    through: Tuple[Path, ...]
    """The states whose old values carry the missing credit into later steps."""

    zero: bool
    """``True``: the online gradient is exactly zero. ``False``: it is non-zero
    but misses the credit through ``through``."""


class OnlineCreditGraph:
    """Reachability tables of one compiled step, built once per compile.

    Parameters
    ----------
    graph : ETraceGraph
        The compiled graph. Weights with a relation inside a descended scan
        are left out: their values live in the scan body.
    """

    def __init__(self, graph) -> None:
        self.graph = graph
        minfo = graph.module_info
        jaxpr = minfo.jaxpr

        # one entry per equation: the variables it reads and the ones it may
        # change (its outputs, plus every reference it takes)
        self._ins: List[List[Var]] = []
        self._writers: Dict[Var, List[int]] = {}
        self._readers: Dict[Var, List[int]] = {}
        for i, eqn in enumerate(jaxpr.eqns):
            ins = [v for v in eqn.invars if isinstance(v, Var)]
            outs = [v for v in eqn.outvars if isinstance(v, Var)]
            outs += [v for v in ins if _is_ref(v)]
            self._ins.append(ins)
            for v in outs:
                self._writers.setdefault(v, []).append(i)
            for v in ins:
                self._readers.setdefault(v, []).append(i)
        self._outs: List[List[Var]] = [[] for _ in self._ins]
        for v, idx in self._writers.items():
            for i in idx:
                self._outs[i].append(v)

        # one entry per output leaf, literals included, in the output order
        self.model_outputs: Tuple = tuple(jaxpr.outvars[:minfo.num_var_out])
        self.model_outvars: Tuple[Var, ...] = tuple(
            v for v in self.model_outputs if isinstance(v, Var))

        # non-parameter states: their old values (invars) and new values (outvars)
        self.state_invars: Dict[Path, Tuple[Var, ...]] = {}
        self.state_outvars: Dict[Path, Tuple[Var, ...]] = {}
        self.hidden: Set[Path] = set()
        self.kind: Dict[Path, str] = {}
        self._invar_states: Dict[Var, List[Path]] = {}
        for ins, outs, state in zip(minfo.state_tree_invars, minfo.state_tree_outvars,
                                    minfo.compiled_model_states):
            if isinstance(state, brainstate.ParamState):
                continue
            path = minfo.state_id_to_path[id(state)]
            self.kind[path] = type(state).__name__
            self.state_invars[path] = tuple(v for v in jax.tree.leaves(ins) if isinstance(v, Var))
            self.state_outvars[path] = tuple(v for v in jax.tree.leaves(outs) if isinstance(v, Var))
            if isinstance(state, brainstate.HiddenState):
                self.hidden.add(path)
            for v in self.state_invars[path]:
                self._invar_states.setdefault(v, []).append(path)

        # trainable weights with ETP relations, outside descended scans
        relations: Dict[Path, list] = {}
        for r in graph.hidden_param_op_relations:
            for p in dict.fromkeys(r.trainable_paths.values()):
                relations.setdefault(p, []).append(r)
        self.weights: Dict[Path, Tuple[Tuple[Var, ...], tuple]] = {}
        for p, rels in relations.items():
            if any(r.control_flow_context is not None for r in rels):
                continue
            seeds = [r.y_var for r in rels]
            seeds += [v for v in minfo.weight_path_to_invars.get(p, ()) if isinstance(v, Var)]
            groups = tuple({id(g): g for r in rels for g in r.hidden_groups}.values())
            self.weights[p] = (tuple(dict.fromkeys(seeds)), groups)

    # ------------------------------------------------------------------ sweeps

    def _co_reach(self, targets: Iterable[Var], across_steps: bool) -> Set[Var]:
        """Variables that reach *targets* (within a step, or over steps)."""
        need: Set[Var] = set()
        done: Set[int] = set()
        relevant: Set[Path] = set()
        stack = list(targets)
        while stack:
            v = stack.pop()
            if v in need:
                continue
            need.add(v)
            for i in self._writers.get(v, ()):
                if i not in done:
                    done.add(i)
                    stack.extend(self._ins[i])
            if across_steps:
                for p in self._invar_states.get(v, ()):
                    if p not in relevant:
                        relevant.add(p)
                        stack.extend(self.state_outvars[p])
        return need

    def _written_labels(self, labelled: Dict[Var, int]) -> Dict[Var, int]:
        """Bitsets of the labelled variables each variable reaches in a step."""
        label = dict(labelled)
        for i in range(len(self._ins) - 1, -1, -1):
            bits = 0
            for v in self._outs[i]:
                bits |= label.get(v, 0)
            if bits:
                for v in self._ins[i]:
                    label[v] = label.get(v, 0) | bits
        return label

    def _forward(self, seeds: Iterable[Var]) -> Set[Var]:
        reached = set(seeds)
        stack = list(reached)
        done: Set[int] = set()
        while stack:
            v = stack.pop()
            for i in self._readers.get(v, ()):
                if i not in done:
                    done.add(i)
                    for o in self._outs[i]:
                        if o not in reached:
                            reached.add(o)
                            stack.append(o)
        return reached

    # ---------------------------------------------------------------- analysis

    def find(
        self,
        old_hidden_credit: bool,
        used: Optional[Iterable[Var]] = None,
        partial: bool = True,
    ) -> List[CreditFinding]:
        """Weights whose online credit misses BPTT credit.

        Parameters
        ----------
        old_hidden_credit : bool
            Whether the learning signal also reads the loss's dependence on the
            old values of a weight's hidden groups (a multi-step VJP given one
            step at a time).
        used : iterable of Var, optional
            The step outputs the loss depends on: model outputs and new state
            values. Default: every model output.
        partial : bool
            Also return weights whose online credit is non-zero but misses the
            credit through a non-hidden state.

        Returns
        -------
        list of CreditFinding
        """
        used = set(self.model_outvars if used is None else used)
        if not used or not self.weights:
            return []
        b_all = self._co_reach(used, across_steps=True)

        # the non-hidden states whose old value matters later
        buffers = [p for p in self.state_invars
                   if p not in self.hidden and any(v in b_all for v in self.state_invars[p])]
        labelled: Dict[Var, int] = {v: 1 for v in used}  # bit 0: a used output
        for k, p in enumerate(buffers, start=1):
            for v in self.state_outvars[p]:
                labelled[v] = labelled.get(v, 0) | (1 << k)
        label = self._written_labels(labelled)

        findings = []
        for path, (seeds, groups) in self.weights.items():
            if not any(v in b_all for v in seeds):
                continue  # no BPTT credit either: nothing is missing
            hidden_paths = tuple(dict.fromkeys(p for g in groups for p in g.hidden_paths))
            group_outs = [v for g in groups for v in g.hidden_outvars]
            bits = 0
            for v in list(seeds) + group_outs:
                bits |= label.get(v, 0)
            online_seeds = list(seeds) + group_outs
            if old_hidden_credit:
                online_seeds += [v for g in groups for v in g.hidden_invars]
            online = any(label.get(v, 0) & 1 for v in online_seeds)
            through = tuple(p for k, p in enumerate(buffers, start=1) if bits >> k & 1)
            if online:
                if partial and through:
                    findings.append(CreditFinding(path, hidden_paths, through, zero=False))
                continue
            if not through:
                through = self._hidden_through(hidden_paths, used, b_all)
            findings.append(CreditFinding(path, hidden_paths, through, zero=True))
        return findings

    def _hidden_through(
        self,
        own: Sequence[Path],
        used: Set[Var],
        b_all: Set[Var],
    ) -> Tuple[Path, ...]:
        """Hidden states of the weight's groups whose old values leave the groups.

        Run only for a reported weight: its credit crosses no non-hidden state,
        so it leaves its groups through the old value of one of their states.
        """
        own_set = set(own)
        targets = set(used)
        for p, outs in self.state_outvars.items():
            if p not in own_set and any(v in b_all for v in self.state_invars[p]):
                targets.update(outs)
        return tuple(p for p in own
                     if self._forward(self.state_invars.get(p, ())) & targets)


def credit_message(
    finding: CreditFinding,
    graph: OnlineCreditGraph,
    algorithm: str,
    old_hidden_credit: bool,
    coupled: bool,
    stage: str,
) -> str:
    """The text of an ``ONLINE_GRADIENT_ZERO`` or ``ONLINE_GRADIENT_PARTIAL`` record.

    Parameters
    ----------
    finding : CreditFinding
        The weight and the states its missing credit crosses.
    graph : OnlineCreditGraph
        The tables the finding came from (state kinds).
    algorithm : str
        The algorithm, as the user would name it.
    old_hidden_credit : bool
        Whether the algorithm is a multi-step VJP (it credits old hidden values).
    coupled : bool
        Whether the algorithm couples the states of a hidden group
        (``recurrence_scope`` ``'coupled'`` or ``'sparse_n'``).
    stage : str
        ``'compile'`` (the loss is not known yet) or ``'gradient'``.

    Returns
    -------
    str
    """
    through = list(finding.through)
    states = ', '.join(f'{p} ({graph.kind.get(p, "state")})' for p in through) or 'a state'
    hidden_line = bool(through) and all(p in graph.hidden for p in through)
    if not finding.zero:
        return (
            f'The online gradient of weight {finding.weight_path} under {algorithm} '
            f'omits the credit that reaches the outputs in later steps through the '
            f'old value of {states}: only the credit within the step and within its '
            f'hidden group(s) is learned. The result can be strongly biased (on a '
            f'two-layer net with a one-step ShortTermState line and a same-step '
            f'readout of the first layer, the online gradient pointed away from '
            f'BPTT\'s, cosine -0.92).'
        )
    if hidden_line:
        why = (f'{algorithm} reads the learning signal at the new hidden values of '
               f'a step only, so the credit that leaves the weight\'s hidden group '
               f'through an old value is lost')
        remedy = ("use an algorithm that credits old hidden values or couples the "
                  "groups (D_RTRL(vjp_method='multi-step') given one step per call, "
                  "OSTLRecurrent or SnAp), or train with BPTT")
    else:
        why = ('online algorithms carry credit across steps only in the eligibility '
               'traces of hidden groups, and a non-hidden state is a given input to '
               'the next step')
        if old_hidden_credit:
            remedy = ('hold the state in a brainstate.HiddenState (this VJP then '
                      'credits it through its old value), or train with BPTT')
        elif coupled:
            remedy = ('hold the state in a brainstate.HiddenState (this algorithm then '
                      'couples it into the weight\'s hidden group), or train with BPTT')
        else:
            remedy = ("hold the state in a brainstate.HiddenState and use an algorithm "
                      "that credits it (D_RTRL(vjp_method='multi-step') given one step "
                      "per call, OSTLRecurrent or SnAp), or train with BPTT")
    head = ('If the loss reads only the model outputs, the online gradient'
            if stage == 'compile' else 'The online gradient')
    when = ' given one step per call' if old_hidden_credit else ''
    tail = (' Given MultiStepData of two or more steps per call, the multi-step VJP '
            'credits the path within each call and drops it across calls.'
            if old_hidden_credit else '')
    return (
        f'{head} of weight {finding.weight_path} under {algorithm}{when} is zero: the loss '
        f'depends on it only in later steps, through the old value of {states}, and '
        f'{why}. To learn it online, {remedy}.{tail}'
    )
