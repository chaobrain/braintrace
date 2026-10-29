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
loss depends on the group's hidden values:

- on the new values, under every algorithm;
- on the old values along the output bypass, under a single-step VJP: an
  output computed from an old hidden value without crossing a new one (a
  readout taken before the update) gets the exact same-step signal
  ``dL^t/dh^{t-1}``, paired with the previous trace (#185);
- on the old values along every path, under a multi-step VJP given one step
  per call.

Nothing else crosses a step. A value written into a state the trace does not
follow -- a delay line held in a ``ShortTermState``, or, for a single-step
algorithm, a hidden state of another group that only a later step reads -- is
a given input to the next step.

Reverse mode through time (BPTT) follows every state. So a weight whose
influence reaches the loss only through such a state has a non-zero BPTT
gradient and an online gradient of exactly zero. This module finds those
weights from the step's jaxpr:

- ``B_now``: the variables that reach a used output within the step;
- ``B_all``: the same, closed over steps: a state whose old value is in it
  makes its new value a target too;
- the online credit of a weight is non-zero iff one of its ETP outputs, its
  weight inputs, its groups' new hidden values or a credited old hidden value
  reaches a used output within the step; its BPTT credit is non-zero iff one
  of its ETP outputs or weight inputs is in ``B_all``.

Each is one backward sweep, linear in the number of equations. The states
each weight writes come from one more backward pass that propagates bitsets,
so no sweep runs per weight.

Values carry no derivative through ``stop_gradient``, through primitives
whose derivative is zero (``round``, ``floor``, ``ceil``, ``sign``) and
through integer or boolean values; every sweep stops there. Calls to a jaxpr
(``jit``, ``closed_call``, ``remat``) are followed into their bodies, so a
cut inside them is seen. Every other equation is opaque: each input reaches
each output, including through ``scan``, ``while`` and ``cond`` bodies and
custom derivative rules. A reference (``jax.new_ref``) taken by an equation
counts as an output of it, so a value stored in a reference reaches its later
reads.

The check is structural: a path counts if a derivative can flow along it. A
derivative that vanishes for the values taken (``jnp.heaviside`` keeps its
input on a branch taken only for NaN; a ReLU that is never active) is not
told apart, so such a weight can be reported although its BPTT gradient is
zero too.

Known false negative (#216): a buffer written and then read back in the same
step (a ring updated before it is read, as ``brainstate.nn.Delay.update``
followed by ``retrieve_at_step`` does) is tracked as one array, so its old
slots look like part of the new value and the weight is not reported.
"""

from typing import Any, Dict, FrozenSet, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

import brainstate
import jax
import jax.numpy as jnp

from braintrace._compatible_imports import Var
from braintrace._typing import Path

try:  # the registry of forward-mode rules, to spot zero derivatives
    from jax._src.interpreters import ad as _jax_ad
    _ad: Any = _jax_ad
except ImportError:  # pragma: no cover - a jax without the private module
    _ad = None

__all__ = [
    'CreditFinding',
    'OnlineCreditGraph',
    'credit_message',
]

# primitives whose derivative is zero everywhere
_ZERO_DERIVATIVE = frozenset({'stop_gradient', 'round', 'floor', 'ceil', 'sign'})
# primitives that call a jaxpr whose inputs and outputs are their own
_CALLS = frozenset({'jit', 'pjit', 'closed_call', 'core_call', 'remat', 'remat2', 'checkpoint'})

# a block: the variables it reads and the ones it may change, each of which
# may depend on each of the reads
_Block = Tuple[List[Var], List[Var]]


def _is_ref(v: Var) -> bool:
    return type(getattr(v, 'aval', None)).__name__ == 'AbstractRef'


_INEXACT: Dict[Any, bool] = {}


def _carries_derivative(v) -> bool:
    """A variable of an inexact dtype (a reference: of its contents)."""
    if not isinstance(v, Var):
        return False
    aval = getattr(v, 'aval', None)
    aval = getattr(aval, 'inner_aval', aval)
    dtype = getattr(aval, 'dtype', None)
    if dtype is None:  # pragma: no cover - a token
        return False
    known = _INEXACT.get(dtype)
    if known is None:
        try:
            known = bool(jnp.issubdtype(dtype, jnp.inexact))
        except TypeError:  # pragma: no cover - an extended dtype
            known = False
        _INEXACT[dtype] = known
    return known


def _has_zero_derivative(eqn) -> bool:
    if eqn.primitive.name in _ZERO_DERIVATIVE:
        return True
    rule = None if _ad is None else _ad.primitive_jvps.get(eqn.primitive)
    zero = getattr(_ad, 'zero_jvp', None)
    return zero is not None and getattr(rule, 'func', None) is zero


def _call_body(eqn):
    """The jaxpr *eqn* calls with its own inputs and outputs, else ``None``."""
    if eqn.primitive.name not in _CALLS:
        return None
    body = eqn.params.get('jaxpr', eqn.params.get('call_jaxpr'))
    body = getattr(body, 'jaxpr', body)  # a closed jaxpr: its jaxpr
    if (getattr(body, 'eqns', None) is None
            or len(body.invars) != len(eqn.invars)
            or len(body.outvars) != len(eqn.outvars)):  # pragma: no cover - not a plain call
        return None
    return body


def _blocks(eqns, memo: Dict[int, tuple]) -> List[_Block]:
    """The dependence blocks of *eqns*, in order; calls are followed."""
    blocks: List[_Block] = []
    for eqn in eqns:
        if _has_zero_derivative(eqn):
            continue
        ins = [v for v in eqn.invars if _carries_derivative(v)]
        refs = [v for v in ins if _is_ref(v)]
        body = None if refs else _call_body(eqn)
        if body is not None:
            deps = _body_deps(body, memo)
            by_reads: Dict[FrozenSet[int], List[Var]] = {}
            for k, o in enumerate(eqn.outvars):
                if _carries_derivative(o) and deps[k]:
                    by_reads.setdefault(deps[k], []).append(o)
            for reads, outs in by_reads.items():
                block_ins = [eqn.invars[j] for j in sorted(reads)
                             if _carries_derivative(eqn.invars[j])]
                if block_ins:
                    blocks.append((block_ins, outs))
            continue
        outs = [v for v in eqn.outvars if _carries_derivative(v)] + refs
        if ins and outs:
            blocks.append((ins, outs))
    return blocks


def _body_deps(body, memo: Dict[int, tuple]) -> List[FrozenSet[int]]:
    """For each output of *body*, the inputs it depends on differentiably."""
    key = id(body)
    if key not in memo:
        label: Dict[Var, int] = {}
        for k, o in enumerate(body.outvars):
            if _carries_derivative(o):
                label[o] = label.get(o, 0) | (1 << k)
        for ins, outs in reversed(_blocks(body.eqns, memo)):
            bits = 0
            for v in outs:
                bits |= label.get(v, 0)
            if bits:
                for v in ins:
                    label[v] = label.get(v, 0) | bits
        deps: List[Set[int]] = [set() for _ in body.outvars]
        for j, v in enumerate(body.invars):
            for k in _bits(label.get(v, 0)):
                deps[k].add(j)
        memo[key] = (body, [frozenset(d) for d in deps])  # keep the body alive
    return memo[key][1]


def _bits(x: int):
    while x:
        low = x & -x
        yield low.bit_length() - 1
        x ^= low


class CreditFinding(NamedTuple):
    """A weight whose online credit misses part or all of its BPTT credit."""

    weight_path: Path
    """The weight's state path."""

    hidden_paths: Tuple[Path, ...]
    """The hidden states of the weight's hidden groups."""

    through: Tuple[Path, ...]
    """The states whose old values carry the missing credit into later steps."""

    zero: bool
    """``True``: the online gradient is exactly zero. ``False``: the weight
    also has a credited path to the loss, so the online gradient is not known
    to be zero; it misses the credit through ``through``."""


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

        # one entry per block: the variables it reads and the ones it may
        # change (its outputs, plus every reference it takes)
        blocks = _blocks(jaxpr.eqns, {})
        self._ins: List[List[Var]] = [ins for ins, _ in blocks]
        self._outs: List[List[Var]] = [outs for _, outs in blocks]
        self._writers: Dict[Var, List[int]] = {}
        self._readers: Dict[Var, List[int]] = {}
        for i, (ins, outs) in enumerate(blocks):
            for v in outs:
                self._writers.setdefault(v, []).append(i)
            for v in ins:
                self._readers.setdefault(v, []).append(i)

        # one entry per output leaf, literals included, in the output order
        self.model_outputs: Tuple = tuple(jaxpr.outvars[:minfo.num_var_out])
        self.model_outvars: Tuple[Var, ...] = tuple(
            v for v in self.model_outputs if isinstance(v, Var))

        # non-parameter states: the leaves of their old values (invars) and new
        # values (outvars) that carry a derivative
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
            self.state_invars[path] = tuple(v for v in jax.tree.leaves(ins) if _carries_derivative(v))
            self.state_outvars[path] = tuple(v for v in jax.tree.leaves(outs) if _carries_derivative(v))
            if isinstance(state, brainstate.HiddenState):
                self.hidden.add(path)
            for v in self.state_invars[path]:
                self._invar_states.setdefault(v, []).append(path)
        self._hidden_outvars: FrozenSet[Var] = frozenset(
            v for p in self.hidden for v in self.state_outvars[p])

        # the hidden states whose old value reaches an output along the output
        # bypass (a single-step VJP credits it, #185)
        perturb = getattr(graph, 'hidden_perturb', None)
        self.bypass_hidden: FrozenSet[Path] = frozenset(
            getattr(perturb, 'bypass_hidden_paths', ()) or ())

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

    def _co_reach(
        self,
        targets: Iterable[Var],
        across_steps: bool,
        stop: FrozenSet[Var] = frozenset(),
    ) -> Set[Var]:
        """Variables that reach *targets* (within a step, or over steps),
        without passing through a variable of *stop*."""
        need: Set[Var] = set()
        done: Set[int] = set()
        relevant: Set[Path] = set()
        stack = [v for v in targets if _carries_derivative(v)]
        while stack:
            v = stack.pop()
            if v in need:
                continue
            need.add(v)
            if v in stop:
                continue
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
            Whether the learning signal reads the loss's dependence on the old
            values of a weight's hidden groups along every path (a multi-step
            VJP given one step at a time). Otherwise it reads it along the
            output bypass only.
        used : iterable of Var, optional
            The step outputs the loss depends on: model outputs and new state
            values. Default: every model output.
        partial : bool
            Also return weights whose online credit is not known to be zero but
            misses the credit through a non-hidden state.

        Returns
        -------
        list of CreditFinding
        """
        used = {v for v in (self.model_outvars if used is None else used)
                if _carries_derivative(v)}
        if not used or not self.weights:
            return []
        b_all = self._co_reach(used, across_steps=True)
        b_bypass: Optional[Set[Var]] = None

        # bit 0: a used output; one bit per state, on its new value
        order = list(self.state_outvars)
        bit = {p: k for k, p in enumerate(order, start=1)}
        labelled: Dict[Var, int] = {v: 1 for v in used}
        for p in order:
            for v in self.state_outvars[p]:
                labelled[v] = labelled.get(v, 0) | (1 << bit[p])
        label = self._written_labels(labelled)
        # the non-hidden states whose old value matters later
        buffer_mask = 0
        for p in order:
            if p not in self.hidden and any(v in b_all for v in self.state_invars[p]):
                buffer_mask |= 1 << bit[p]

        findings = []
        for path, (seeds, groups) in self.weights.items():
            if not any(v in b_all for v in seeds):
                continue  # no BPTT credit either: nothing is missing
            hidden_paths = tuple(dict.fromkeys(p for g in groups for p in g.hidden_paths))
            group_outs = [v for g in groups for v in g.hidden_outvars]
            online = any(label.get(v, 0) & 1 for v in list(seeds) + group_outs)
            if not online and old_hidden_credit:
                online = any(label.get(v, 0) & 1 for g in groups for v in g.hidden_invars)
            elif not online:
                old = [v for p in hidden_paths if p in self.bypass_hidden
                       for v in self.state_invars.get(p, ())]
                if old:
                    if b_bypass is None:
                        b_bypass = self._co_reach(used, across_steps=False,
                                                  stop=self._hidden_outvars)
                    online = any(v in b_bypass for v in old)
            through = self._through(seeds, hidden_paths, label, bit, buffer_mask, order)
            if online:
                if partial and through:
                    findings.append(CreditFinding(path, hidden_paths, through, zero=False))
                continue
            if not through:
                through = self._hidden_through(hidden_paths, used, b_all)
            findings.append(CreditFinding(path, hidden_paths, through, zero=True))
        return findings

    def _through(
        self,
        seeds: Sequence[Var],
        own: Sequence[Path],
        label: Dict[Var, int],
        bit: Dict[Path, int],
        buffer_mask: int,
        order: Sequence[Path],
    ) -> Tuple[Path, ...]:
        """The buffers the weight itself writes.

        Its influence: what its seeds reach in a step, then, over steps, what
        the old values of the hidden states of its groups that it reaches
        reach (the part its eligibility trace carries).
        """
        own_bits = {bit[p]: p for p in own if p in bit}
        own_mask = 0
        for k in own_bits:
            own_mask |= 1 << k
        reach = 0
        for v in seeds:
            reach |= label.get(v, 0)
        seen = 0
        while True:
            new = reach & own_mask & ~seen
            if not new:
                break
            seen |= new
            for k in _bits(new):
                for v in self.state_invars[own_bits[k]]:
                    reach |= label.get(v, 0)
        return tuple(order[k - 1] for k in _bits(reach & buffer_mask))

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
        ``'compile'`` (the loss is not known yet) or ``'gradient'`` (one call,
        with the loss's reads known).

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
            f'old value of {states}: it keeps only the credit that reaches the loss '
            f'within a step or through the eligibility traces of its hidden '
            f'group(s). The omitted part can bias the gradient, or even reverse its '
            f'direction; train with BPTT if that credit matters.'
        )
    if hidden_line:
        why = (f'{algorithm} reads the learning signal at the new hidden values of '
               f'a step (and at old ones only along a same-step path to the outputs), '
               f'so the credit that leaves the weight\'s hidden group through an old '
               f'value is lost')
        remedy = ("use an algorithm that credits old hidden values or couples the "
                  "groups (D_RTRL(vjp_method='multi-step') given one step per call, "
                  "OSTLRecurrent or SnAp), or train with BPTT")
    else:
        why = ('online algorithms carry credit across steps only in the eligibility '
               'traces of hidden groups; a non-hidden state is a given input to the '
               'next step')
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
    weight = f'weight {finding.weight_path} under {algorithm}'
    if stage == 'compile':
        when = ' given one step per call' if old_hidden_credit else ''
        lead = f'If the loss reads only the model outputs, the online gradient of {weight}{when} is zero'
    elif old_hidden_credit:
        lead = f'Within this call of one step, the online gradient of {weight} is zero'
    else:
        lead = f'The online gradient of {weight} is zero'
    tail = (' Given MultiStepData of two or more steps per call, the multi-step VJP '
            'credits the path within each call and drops it across calls.'
            if old_hidden_credit else '')
    return (
        f'{lead}: the loss depends on it only in later steps, through the old value '
        f'of {states}, and {why}. To learn it online, {remedy}.{tail}'
    )
