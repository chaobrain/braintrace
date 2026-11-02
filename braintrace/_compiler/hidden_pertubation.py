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
from typing import Any, Collection, Dict, FrozenSet, List, Mapping, NamedTuple, Optional, Sequence, Set, Tuple, Union

import brainstate
import jax.core
import jax.numpy as jnp
import brainunit as u

from braintrace._compatible_imports import (
    Var,
    Literal,
    JaxprEqn,
    Jaxpr,
    ClosedJaxpr,
    new_var,
    new_jaxpr_eqn,
    stop_gradient_p,
    is_while_primitive,
    shaped_aval,
)
from braintrace._misc import (
    git_issue_addr,
)
from braintrace._op import (
    is_etp_enable_gradient_primitive,
    is_etp_primitive,
)
from braintrace._typing import (
    HiddenInVar,
    HiddenOutVar,
    Path,
)
from .base import (
    JaxprEvaluation,
    check_unsupported_op,
)
from .canonicalize import ControlFlowPolicy, DEFAULT_CONTROL_FLOW_POLICY
from .diagnostics import DiagnosticKind, DiagnosticLevel, emit
from .hidden_group import (
    HiddenGroup,
)
from .position_graph import eqn_preserves_positions
from .module_info import (
    extract_module_info,
    ModuleInfo,
)
from .output_bypass import OutputBypass, add_output_bypass

__all__ = [
    'HiddenPerturbation',
    'add_hidden_perturbation_from_minfo',
    'add_hidden_perturbation_in_module',
]


class HiddenPerturbation(NamedTuple):
    r"""The hidden-perturbation information.

    Hidden perturbation adds a perturbation variable to each hidden state in the
    jaxpr and replaces the hidden states with the perturbed states:

    .. math::

        h^t = f(x) \;\Rightarrow\; h^t = f(x) + \mathrm{perturb\_var},

    where :math:`h` is the hidden state, :math:`f` is the function, :math:`x` is
    the input, and :math:`\mathrm{perturb\_var}` is the perturbation variable.

    Attributes
    ----------
    perturb_vars : sequence of Var
        The perturbation variables.
    perturb_hidden_paths : sequence of Path
        The hidden-state paths that are perturbed.
    perturb_hidden_states : sequence of brainstate.HiddenState
        The hidden states that are perturbed.
    perturb_jaxpr : ClosedJaxpr
        The perturbed jaxpr.
    new_read_cuts : sequence of (Path, Path)
        The in-step reads ``(k, j)``, by the update of ``h_j``, of the new
        value of ``h_k`` that the learning signal holds fixed, because every
        eligibility trace paired with ``h_k``'s signal already carries them,
        within the step and across steps (see the Notes). Empty when nothing
        is cut.
    new_read_conflicts : sequence of str
        Why some in-step reads could not be cut: the relations disagree, the
        traces are not proven to carry a read across steps (it crosses hidden
        groups, or goes through an equation the group transition leaves out
        or one not proven position-preserving, i.e. not elementwise), or the
        reads cannot be emitted apart. When
        non-empty, no read is cut: the signal is the total derivative, which
        counts some in-step paths twice, and the graph carries a
        ``HIDDEN_NEW_READ_DOUBLE_COUNT`` record (also warned).
    new_read_exact : bool
        ``True`` iff the reads were planned and no conflict was found: then no
        in-step read of a new hidden value is double counted in the
        single-step learning signal. It does not say that the single-step
        gradient is exact: the approximations of the learner itself (a
        diagonal trace of a coupled recurrence, a path the traces do not
        cover) are separate. ``False`` when the reads were not planned (no
        relations, as in :func:`add_hidden_perturbation_in_module`) or a
        conflict was found.

    See Also
    --------
    add_hidden_perturbation_in_module : Build perturbations directly from a model.

    Notes
    -----
    Internally a new variable :math:`\hat{h}^t = f(x)` is defined and an extra
    equation :math:`h^t = \hat{h}^t + \mathrm{perturb\_var}` is added. The
    perturbation lets the hidden-state gradient be read off the perturbation
    variable

    .. math::

        \frac{\partial L^t}{\partial h^t}
        = \frac{\partial L^t}{\partial \mathrm{perturb\_var}}.

    Every reader of :math:`h^t_k` in the step sees the perturbed value, so this
    is the total derivative: it includes the update of another hidden state
    :math:`h^t_j` that reads the new :math:`h^t_k`,
    :math:`s_k = \partial_k L + J_{jk}^\top s_j` with
    :math:`J_{jk} = \partial \hat{h}^t_j / \partial h^t_k`. When the graph is
    compiled with its relations, a read that every trace paired with
    :math:`s_k` already carries is *cut* (``new_read_cuts``): it sees
    :math:`\hat{h}^t_k` instead, and :math:`s_k` is the partial derivative with
    that read held fixed. Otherwise the path through the read would be
    counted twice, once in :math:`s_j e_j` and once in :math:`s_k e_k`
    (issue #239). A trace carries a read when the ``y -> hidden`` jaxpr of
    :math:`h_j`'s group walks through it (within the step) and the transition
    of that group holds it too (across steps): :math:`h_k` and :math:`h_j`
    share the group, and every equation on the read is one the transition
    computes (:meth:`HiddenGroup.carried_vars`) and is proven
    position-preserving (elementwise; see
    :meth:`NewReadPolicy.why_not_carried`). A read the relations would cut
    but the transition does not provably carry is a conflict.
    An equation that serves both a cut read and the outputs is emitted
    twice, once for each.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>> gru = braintrace.nn.GRUCell(3, 4)
        >>> _ = brainstate.nn.init_all_states(gru)
        >>> inputs = brainstate.random.randn(3)
        >>> hidden_perturb = braintrace.add_hidden_perturbation_in_module(gru, inputs)
        >>> isinstance(hidden_perturb, braintrace.HiddenPerturbation)
        True
    """
    perturb_vars: Sequence[Var]  # the perturbation variables
    perturb_hidden_paths: Sequence[Path]  # the hidden state paths that are perturbed
    perturb_hidden_states: Sequence[brainstate.HiddenState]  # the hidden states that are perturbed
    perturb_jaxpr: ClosedJaxpr  # the perturbed jaxpr
    # parameters whose path to an output bypasses every new hidden value
    # (see ``_compiler/output_bypass.py``), and the copies of their leaves
    bypass_param_paths: Sequence[Path] = ()
    bypass_vars: Sequence[Var] = ()
    # hidden states whose previous value reaches an output that way, and the
    # copies of their previous values
    bypass_hidden_paths: Sequence[Path] = ()
    bypass_hidden_vars: Sequence[Var] = ()
    # in-step reads of new hidden values (#239); see the class Notes
    new_read_cuts: Sequence[Tuple[Path, Path]] = ()
    new_read_conflicts: Sequence[str] = ()
    new_read_exact: bool = False

    def eval_jaxpr(
        self,
        inputs: Sequence[jax.Array],
        perturb_data: Sequence[jax.Array],
        bypass_data: Sequence[jax.Array] = (),
        bypass_hidden_data: Sequence[jax.Array] = (),
    ) -> Sequence[jax.Array]:
        """Evaluate the perturbed jaxpr.

        Parameters
        ----------
        inputs : sequence of jax.Array
            The flat input values of the original jaxpr.
        perturb_data : sequence of jax.Array
            The perturbation values, one per entry of ``perturb_vars``.
        bypass_data : sequence of jax.Array, optional
            The values of the bypass parameter copies, one per entry of
            ``bypass_vars`` (the leaves of each ``bypass_param_paths`` value).
            Differentiating the outputs with respect to them gives exactly the
            parameters' same-step bypass gradient. Defaults to none.
        bypass_hidden_data : sequence of jax.Array, optional
            The previous values of the ``bypass_hidden_paths`` hidden states,
            one per entry of ``bypass_hidden_vars``. Differentiating the
            outputs with respect to them gives exactly the bypass learning
            signal ``dL^t/dh^{t-1}``. Defaults to none.

        Returns
        -------
        sequence of jax.Array
            The outputs of the perturbed jaxpr.
        """
        if len(bypass_data) != len(self.bypass_vars):
            raise ValueError(
                f'Expected {len(self.bypass_vars)} bypass value(s) for the '
                f'parameters {list(self.bypass_param_paths)}, got {len(bypass_data)}.'
            )
        if len(bypass_hidden_data) != len(self.bypass_hidden_vars):
            raise ValueError(
                f'Expected {len(self.bypass_hidden_vars)} bypass value(s) for the '
                f'hidden states {list(self.bypass_hidden_paths)}, '
                f'got {len(bypass_hidden_data)}.'
            )
        return jax.core.eval_jaxpr(
            self.perturb_jaxpr.jaxpr,
            self.perturb_jaxpr.consts,
            *(tuple(inputs) + tuple(perturb_data) + tuple(bypass_data)
              + tuple(bypass_hidden_data))
        )

    def init_perturb_data(self) -> Sequence[jax.Array]:
        """Initialize the perturbation data to zeros.

        Returns
        -------
        sequence of jax.Array
            One zero array per perturbation variable, matching its shape and
            dtype.
        """
        return [jax.numpy.zeros(shaped_aval(v).shape, dtype=shaped_aval(v).dtype) for v in self.perturb_vars]

    def bypass_hidden_data_to_hidden_group_data(
        self,
        bypass_hidden_data: Sequence[jax.Array],
        hidden_groups: Sequence[HiddenGroup],
    ) -> Sequence[jax.Array]:
        """Assemble per-group arrays from per-hidden bypass data.

        Parameters
        ----------
        bypass_hidden_data : sequence of jax.Array
            One array per entry of ``bypass_hidden_vars`` (typically the
            cotangents ``dL^t/dh^{t-1}`` along the bypass paths).
        hidden_groups : sequence of HiddenGroup
            The hidden groups to assemble, as in
            :meth:`perturb_data_to_hidden_group_data`.

        Returns
        -------
        sequence of jax.Array
            One concatenated array per hidden group; a perturbed hidden state
            without a bypass contributes zeros.

        Raises
        ------
        ValueError
            If ``bypass_hidden_data`` does not have the same length as
            ``bypass_hidden_vars``.
        """
        bypass_hidden_data = list(bypass_hidden_data)
        if len(bypass_hidden_data) != len(self.bypass_hidden_vars):
            raise ValueError(
                f'Expected {len(self.bypass_hidden_vars)} bypass value(s) for the '
                f'hidden states {list(self.bypass_hidden_paths)}, '
                f'got {len(bypass_hidden_data)}.'
            )
        by_path = dict(zip(self.bypass_hidden_paths, bypass_hidden_data))
        per_perturb = []
        for path, v in zip(self.perturb_hidden_paths, self.perturb_vars):
            aval = shaped_aval(v)
            per_perturb.append(by_path[path] if path in by_path else jnp.zeros(aval.shape, aval.dtype))
        return self.perturb_data_to_hidden_group_data(per_perturb, hidden_groups)

    def perturb_data_to_hidden_group_data(
        self,
        perturb_data: Sequence[jax.Array],
        hidden_groups: Sequence[HiddenGroup],
    ) -> Sequence[jax.Array]:
        """Convert the perturbation data to per-hidden-group data.

        Parameters
        ----------
        perturb_data : sequence of jax.Array
            The perturbation values, one per entry of ``perturb_vars``.
        hidden_groups : sequence of HiddenGroup
            The hidden groups to map the perturbation data onto.

        Returns
        -------
        sequence of jax.Array
            One concatenated perturbation array per hidden group.

        Raises
        ------
        ValueError
            If ``perturb_data`` does not have the same length as
            ``perturb_vars``, or if a hidden group needs a path that was not
            perturbed.

        Notes
        -----
        Both guards are ``if ... raise`` rather than ``assert`` so that
        ``python -O`` cannot strip them. The mapping from paths to perturbation
        data is positional (``zip(perturb_hidden_paths, perturb_data)``), so a
        length disagreement silently re-attributes every cotangent past the
        mismatch; and a group asking for an unperturbed path used to surface as
        a bare ``KeyError`` naming a path tuple and nothing else. Both checks
        read only Python-level lengths and dict keys, never array data.
        """
        perturb_data = list(perturb_data)
        if len(perturb_data) != len(self.perturb_vars):
            raise ValueError(
                f'The length of the perturb data is not correct. '
                f'Expected: {len(self.perturb_vars)} '
                f'(one per perturbed hidden state '
                f'{list(self.perturb_hidden_paths)}), '
                f'Got: {len(perturb_data)}'
            )
        path_to_perturb_data = {
            path: data
            for path, data in zip(self.perturb_hidden_paths, perturb_data)
        }

        group_data = []
        for group in hidden_groups:
            vals = []
            for path in group.hidden_paths:
                if path not in path_to_perturb_data:
                    raise ValueError(
                        f'Hidden group {group.index} needs the hidden state '
                        f'{path}, but that path was not perturbed. The perturbed '
                        f'paths are {list(self.perturb_hidden_paths)}. The graph '
                        f'and its hidden perturbation disagree; recompile the '
                        f'graph.'
                    )
                # dimensionless processing
                vals.append(u.get_mantissa(path_to_perturb_data[path]))
            group_data.append(group.concat_hidden(vals))
        return group_data

    def dict(self) -> Dict[str, Any]:
        """Return this perturbation's named fields as a plain dictionary.

        Returns
        -------
        dict
            An ordered mapping from field name to value, as produced by the
            underlying :class:`typing.NamedTuple`.
        """
        return self._asdict()

    def __repr__(self) -> str:
        return repr(brainstate.util.PrettyMapping(self._asdict(), type_name=self.__class__.__name__))


HiddenPerturbation.__module__ = 'braintrace'


class NewReadPolicy:
    """Which in-step reads of a new hidden value the traces carry.

    Used by :func:`plan_new_read_cuts` (issue #239). The update of ``h_j``
    reads the new value of ``h_k`` in the same step. The relations whose
    groups include ``h_k``'s group are the ones whose traces pair with
    ``h_k``'s learning signal. :meth:`edge` decides the read from them:

    ``'cut'``
        when every one of them also includes ``h_j``'s group. Each of their
        traces then carries the in-step path through the read, because the
        ``y -> hidden`` jaxpr of ``h_j``'s group walks back through the
        producer of ``h_k`` (``hid_param_op._build_transition_jaxpr``), so
        the signal of ``h_k`` would count it a second time. The read is cut
        only if the traces also carry it across steps
        (:meth:`why_not_carried`); otherwise it is a conflict.
    ``'keep'``
        when none of them includes ``h_j``'s group, when ``h_k`` or ``h_j``
        is in no group, or when no relation traces ``h_k``'s group. No trace
        paired with ``h_k`` carries the read, so the signal keeps it.
    ``'conflict'``
        when some do and some do not: one signal cannot serve them all.

    Parameters
    ----------
    hid_path_to_group : mapping of Path to HiddenGroup
        The group of each grouped hidden state; :meth:`why_not_carried` reads
        their :meth:`HiddenGroup.carried_vars` and ``varshape``.
    relations : sequence of HiddenParamOpRelation
        The graph's relations; only their ``hidden_groups`` are read.
    """

    def __init__(self, hid_path_to_group: Mapping[Path, HiddenGroup], relations: Sequence[Any]):
        self._group = {path: group.index for path, group in hid_path_to_group.items()}
        self._groups = {group.index: group for group in hid_path_to_group.values()}
        self._carried: Dict[int, FrozenSet[Var]] = {}
        self._relation_groups = tuple(
            frozenset(group.index for group in relation.hidden_groups) for relation in relations
        )

    def group(self, path: Path) -> Optional[int]:
        """The index of ``path``'s group, or ``None`` if it is in no group."""
        return self._group.get(path)

    def traced(self, path: Path) -> FrozenSet[int]:
        """The groups of the relations whose groups include ``path``'s group."""
        index = self.group(path)
        out: Set[int] = set()
        if index is not None:
            for groups in self._relation_groups:
                if index in groups:
                    out |= groups
        return frozenset(out)

    def edge(self, path_k: Path, path_j: Path) -> str:
        """``'cut'``, ``'keep'`` or ``'conflict'`` for the read of the new
        ``path_k`` by the update of ``path_j``."""
        gk, gj = self.group(path_k), self.group(path_j)
        if gk is None or gj is None:
            return 'keep'
        covers = [gj in groups for groups in self._relation_groups if gk in groups]
        if covers and all(covers):
            return 'cut'
        if not any(covers):
            return 'keep'
        return 'conflict'

    def why_not_carried(
        self,
        path_k: Path,
        path_j: Path,
        read_eqns: Sequence[JaxprEqn],
        read_vars: Collection[Var],
    ) -> Optional[str]:
        """Why the traces do not carry the read across steps, or ``None``.

        A trace carries the history of ``h_j`` through the transition of its
        group: ``e_j^t = D e^{t-1} + (in-step term)``, where ``D`` and the
        in-step term are taken per position. When ``h_j``'s update reads the
        new ``h_k``, the true history of ``h_j`` includes
        ``h_k^{t-1} -> h_k^t -> h_j^t``. A cut drops that path from the signal
        of ``h_k``, so it is exact only if ``D`` holds it instead. That is
        proven when ``h_k`` and ``h_j`` share a group and every equation on
        the read (``read_eqns``, from the new ``h_k`` to ``h_j``):

        - computes a value of :meth:`HiddenGroup.carried_vars` (the
          transition contains it; a value bound from the forward pass is held
          fixed), and
        - is proven to keep positions, in value and in derivative
          (:func:`~braintrace._compiler.position_graph.eqn_preserves_positions`:
          an allow-listed elementwise primitive on the full position shape, or
          a call whose body and derivative are made of them).

        Anything else is not proven, so not carried: a read across groups;
        through an equation the transition leaves out (a ``dot_general``/
        ``conv``, an ETP mixing op, control flow that mixes); or through any
        equation that is not proven elementwise (a ``roll``, a reduction, a
        ``gather``, a ``reshape``, a ``custom_vjp``, ...), which the
        per-position traces carry only in part, if at all.

        Parameters
        ----------
        path_k, path_j : Path
            The source and the reader.
        read_eqns : sequence of JaxprEqn
            The equations between the new ``h_k`` and ``h_j``.
        read_vars : collection of Var
            The values on the read: the new ``h_k`` and the outvars of
            ``read_eqns``. An equation's operands among them are its path
            operands.

        Returns
        -------
        str or None
            The reason, for the conflict record; ``None`` when carried.
        """
        gk, gj = self.group(path_k), self.group(path_j)
        if gk is None or gk != gj:
            return f'the read {path_k} -> {path_j} crosses hidden groups, whose traces do not carry it across steps'
        group = self._groups[gj]
        if gj not in self._carried:
            self._carried[gj] = group.carried_vars()
        carried = self._carried[gj]
        varshape = tuple(group.varshape)
        for eqn in read_eqns:
            if not all(ov in carried for ov in eqn.outvars):
                return (f'the read {path_k} -> {path_j} goes through {eqn.primitive}, which the '
                        f'transition of their group leaves out or holds fixed')
            hit = [i for i, v in enumerate(eqn.invars) if isinstance(v, Var) and v in read_vars]
            if not eqn_preserves_positions(eqn, varshape, hit):
                return (f'the read {path_k} -> {path_j} goes through {eqn.primitive}, which is not proven '
                        f'to keep positions (only elementwise equations on the full position shape are)')
        return None


class NewReadPlan(NamedTuple):
    """The plan :func:`plan_new_read_cuts` returns (issue #239).

    Attributes
    ----------
    sources : dict of Var to None
        The hidden outvars, in order, with at least one cut read. A cut read
        sees the unperturbed new value; every other reader sees it perturbed.
    state_eqns : frozenset of int
        Indices of the equations on a cut read, between a source and the
        hidden state that reads it without an ETP op. They read the
        unperturbed sources.
    output_eqns : frozenset of int
        Indices of the equations between a source and a kept read, a model
        output or a non-hidden state. They read the perturbed sources. An
        equation in both sets is emitted twice.
    cuts : tuple of (Path, Path)
        The cut reads ``(path_k, path_j)``.
    conflicts : tuple of str
        Why the reads cannot be separated. When non-empty the plan must not
        be applied.
    """
    sources: Dict[Var, None]
    state_eqns: FrozenSet[int]
    output_eqns: FrozenSet[int]
    cuts: Tuple[Tuple[Path, Path], ...]
    conflicts: Tuple[str, ...]


def _is_trace_boundary(eqn: JaxprEqn) -> bool:
    """An ETP op without gradient: the ``y -> hidden`` jaxprs stop there."""
    return is_etp_primitive(eqn.primitive) and not is_etp_enable_gradient_primitive(eqn.primitive)


def _has_no_derivative(eqn: JaxprEqn) -> bool:
    """A ``stop_gradient``: no path through it carries a derivative, neither
    in a trace (its ``jvp`` is zero, and the group transition leaves it out)
    nor in a learning signal, so it is no part of a read."""
    return eqn.primitive.name == 'stop_gradient'


def plan_new_read_cuts(
    jaxpr: Jaxpr,
    hidden_outvars: Sequence[Var],
    outvar_to_hidden_path: Mapping[Var, Path],
    merge_positions: Sequence[int],
    policy: NewReadPolicy,
) -> NewReadPlan:
    """Find the in-step reads of new hidden values that the traces already
    carry, and the equations that serve them (issue #239).

    A *read* of the new ``h_k`` by ``h_j``: ``h_j`` is the first hidden outvar
    on a path of top-level equations from ``h_k`` that crosses no
    ``stop_gradient`` (no derivative flows there). A read on a path without
    an ETP op that has no gradient (a *direct* read) is decided by
    ``policy.edge``, and a read it would cut is cut only if
    ``policy.why_not_carried`` finds the group transition carries it; a read
    only through such an op is kept, because the traces stop there. A read
    inside one equation (a ``jit`` or ``scan`` that produces both states) is
    not a read here: the perturbation is added after the equation.

    Parameters
    ----------
    jaxpr : Jaxpr
        The module jaxpr.
    hidden_outvars : sequence of Var
        The hidden outvars, in order.
    outvar_to_hidden_path : mapping of Var to Path
        The path of each hidden outvar.
    merge_positions : sequence of int
        The outvar positions of the model outputs and the non-hidden states.
    policy : NewReadPolicy
        Decides each direct read.

    Returns
    -------
    NewReadPlan
        The cut reads, the equations to emit in each world and the
        conflicts. A conflict is recorded, not raised, when:

        - the relations disagree on a read (``policy.edge``);
        - the relations would cut a read that the transition is not proven to
          carry across steps (``policy.why_not_carried``);
        - a cut read is also made through an ETP op;
        - a kept direct read reaches, without an ETP op, a group that the
          relations tracing the source trace (one signal cannot keep and cut
          the same path);
        - a hidden state has a cut read and a kept direct read of a source
          with cuts;
        - an equation serving both worlds is a ``while``, has effects or
          produces a hidden state that a cut source reads directly and keeps.
    """
    eqns = jaxpr.eqns
    producer: Dict[Var, int] = {}
    consumers: Dict[Var, List[int]] = defaultdict(list)
    for i, eqn in enumerate(eqns):
        for v in eqn.invars:
            if isinstance(v, Var):
                consumers[v].append(i)
        for v in eqn.outvars:
            producer[v] = i
    hidden = set(hidden_outvars)
    path_of = outvar_to_hidden_path
    # a read-only hidden state has no producer: its readers see h^{t-1}
    sources = [v for v in dict.fromkeys(hidden_outvars) if v in producer]

    def forward(k: Var) -> Tuple[Set[int], Set[int], Dict[Var, Set[str]]]:
        """The equations reached from ``k`` (all, and those on a direct
        path), and the hidden outvars reached first, with how."""
        seen: Set[Tuple[Var, bool]] = set()
        stack = [(k, False)]
        direct: Set[int] = set()
        every: Set[int] = set()
        sinks: Dict[Var, Set[str]] = {}
        while stack:
            v, through = stack.pop()
            if (v, through) in seen:
                continue
            seen.add((v, through))
            if v is not k and v in hidden:
                sinks.setdefault(v, set()).add('etp' if through else 'direct')
                continue
            for i in consumers.get(v, ()):
                if _has_no_derivative(eqns[i]):
                    continue
                t = through or _is_trace_boundary(eqns[i])
                every.add(i)
                if not t:
                    direct.add(i)
                stack.extend((ov, t) for ov in eqns[i].outvars)
        return direct, every, sinks

    def direct_closure(m: Var) -> Set[Var]:
        """The hidden outvars reached from ``m`` without an ETP op, through
        other hidden outvars too (what a ``y -> hidden`` jaxpr walks)."""
        seen: Set[Var] = set()
        stack = [m]
        out: Set[Var] = set()
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            if v is not m and v in hidden:
                out.add(v)
            for i in consumers.get(v, ()):
                if not _is_trace_boundary(eqns[i]) and not _has_no_derivative(eqns[i]):
                    stack.extend(eqns[i].outvars)
        return out

    def backward(starts: Sequence[Var], direct_only: bool) -> Set[int]:
        """The equations a var of ``starts`` depends on, stopping at other
        hidden outvars (and at ETP ops when ``direct_only``)."""
        first = set(starts)
        seen: Set[Var] = set()
        stack = list(starts)
        out: Set[int] = set()
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            if v in hidden and v not in first:
                continue
            i = producer.get(v)
            if i is None or _has_no_derivative(eqns[i]) or (direct_only and _is_trace_boundary(eqns[i])):
                continue
            out.add(i)
            stack.extend(iv for iv in eqns[i].invars if isinstance(iv, Var))
        return out

    outvars = jaxpr.outvars
    merge_nonhidden: List[Var] = [
        v for v in (outvars[i] for i in merge_positions if i < len(outvars))
        if isinstance(v, Var) and v not in hidden
    ]

    conflicts: List[str] = []
    found = {}
    kept_direct: List[Tuple[Var, Var]] = []
    for k in sources:
        direct, every, sinks = forward(k)
        cut, kept = [], []
        for j in sorted(sinks, key=producer.__getitem__):
            how = sinks[j]
            if 'direct' not in how:
                kept.append(j)  # only through an ETP op: no trace carries it
                continue
            decision = policy.edge(path_of[k], path_of[j])
            if decision == 'conflict':
                conflicts.append(f'the relations disagree on the read {path_of[k]} -> {path_of[j]}')
            elif decision == 'cut' and 'etp' in how:
                conflicts.append(f'{path_of[j]} reads {path_of[k]} both directly and through an ETP op')
            elif decision == 'cut':
                read_eqns = [eqns[i] for i in sorted(direct & backward([j], direct_only=True))]
                read_vars = {k, *(ov for e in read_eqns for ov in e.outvars)}
                why = policy.why_not_carried(path_of[k], path_of[j], read_eqns, read_vars)
                if why is None:
                    cut.append(j)
                else:
                    conflicts.append(why)
            else:
                kept.append(j)
                kept_direct.append((k, j))
        found[k] = (direct, every, cut, kept)

    cut_sources = {k: None for k in sources if found[k][2]}
    cut_sinks = {j for k in cut_sources for j in found[k][2]}
    for k, m in kept_direct:
        traced = policy.traced(path_of[k])
        if any(policy.group(path_of[j]) in traced for j in direct_closure(m)):
            conflicts.append(
                f'the kept read {path_of[k]} -> {path_of[m]} reaches a group that '
                f'the relations of {path_of[k]} trace'
            )
        if m in cut_sinks and k in cut_sources:
            conflicts.append(f'{path_of[m]} has a cut read and a kept read of {path_of[k]}')

    state_eqns: Set[int] = set()
    output_eqns: Set[int] = set()
    cuts: List[Tuple[Path, Path]] = []
    for k in cut_sources:
        direct, every, cut, kept = found[k]
        cuts.extend((path_of[k], path_of[j]) for j in cut)
        state_eqns |= direct & backward(cut, direct_only=True)
        output_eqns |= every & backward(kept + merge_nonhidden, direct_only=False)
    for i in sorted(state_eqns & output_eqns):
        eqn = eqns[i]
        if is_while_primitive(eqn) or eqn.effects:
            conflicts.append(f'an equation ({eqn.primitive}) serves a cut read and the outputs, '
                             f'and cannot be emitted twice')
        for ov in eqn.outvars:
            if ov in hidden and ov not in cut_sinks:
                conflicts.append(f'one equation ({eqn.primitive}) computes a cut read and the kept '
                                 f'read of {path_of[ov]}')
    return NewReadPlan(
        sources=cut_sources,
        state_eqns=frozenset(state_eqns),
        output_eqns=frozenset(output_eqns),
        cuts=tuple(cuts),
        conflicts=tuple(dict.fromkeys(conflicts)),
    )


class JaxprEvalForHiddenPerturbation(JaxprEvaluation):
    """
    Adding perturbations to the hidden states in the jaxpr, and replacing the hidden states with the perturbed states.

    Parameters
    ----------
    closed_jaxpr : ClosedJaxpr
        The closed jaxpr for the model.
    outvar_to_hidden_path : dict
        The mapping from the outvar to the state id.
    hidden_outvar_to_invar : dict
        The mapping from the hidden output variable to the hidden input variable.
    weight_invars : set
        The weight input variables.
    invar_to_hidden_path : dict
        The mapping from the weight input variable to the hidden state path.
    control_flow : ControlFlowPolicy
        The :class:`~braintrace.ControlFlowPolicy` governing opaque
        control-flow handling.
    descended_scan_eqn_ids : frozenset of int
        ``id()`` values of scan equations rewritten by structured scan descent
        (Phase 4); exempt from the unsupported-op checks.
    new_read_policy : NewReadPolicy, optional
        Decides which in-step reads of new hidden values the traces carry
        (issue #239). ``None`` (default): no read is cut, and every reader
        sees the perturbed value.

    Returns
    -------
    HiddenPerturbation
        The revised closed jaxpr with the perturbations.

    Notes
    -----
    A hidden-producing ``while`` equation gets special treatment: every
    ``Var`` input of the loop is detached with ``stop_gradient`` in the
    perturbed jaxpr (JAX has no transpose rule for ``while``, so an
    undetached loop would make the VJP of the perturbed step
    untraceable). The perturbation add ``h = fresh + p`` stays *outside*
    the detach, so the loop's *own* hidden group keeps an exact per-step
    learning signal ``dL/dp``. Consequence: any *reverse-mode* path
    through the loop body within the same step contributes zero — the
    loop's temporal credit is carried instead by the forward-computed
    hidden-to-hidden Jacobian ``D^t`` (see
    ``hidden_group.jacfwd_last_dim``), but a parameter or *other* hidden
    group whose only same-step path to the loss crosses the loop (e.g.
    an upstream layer feeding the loop) receives a ZERO learning signal.
    A WARNING-level ``CONTROL_FLOW_OPAQUE_FWD`` diagnostic
    (``site='perturbation'``) records every detach.
    """
    __module__ = 'braintrace'

    def __init__(
        self,
        closed_jaxpr: ClosedJaxpr,
        hidden_outvar_to_invar: Dict[HiddenOutVar, HiddenInVar],
        weight_invars: Set[Var],
        invar_to_hidden_path: Dict[HiddenInVar, Path],
        outvar_to_hidden_path: Dict[Var, Path],
        path_to_state: Dict[Path, brainstate.HiddenState],
        control_flow: ControlFlowPolicy = DEFAULT_CONTROL_FLOW_POLICY,
        descended_scan_eqn_ids: FrozenSet[int] = frozenset(),
        weight_path_to_invars: Optional[Mapping[Path, Sequence[Var]]] = None,
        bypass_merge_positions: Sequence[int] = (),
        new_read_policy: Optional[NewReadPolicy] = None,
    ):
        # necessary data structures
        self.closed_jaxpr = closed_jaxpr
        self.new_read_policy = new_read_policy

        # Same-step bypass gradients (issue #185): the parameter leaves, and
        # the outvar positions (model outputs, non-hidden state outputs) where
        # a bypass is attached. ``None`` disables the rewrite.
        self.weight_path_to_invars = weight_path_to_invars
        self.bypass_merge_positions = tuple(bypass_merge_positions)

        # Structured scan descent (Phase 4): scan equations rewritten by
        # ``scan_descent.apply_scan_descent`` (keyed by ``id(eqn)``) bypass
        # ``check_unsupported_op`` — their weight usage is legal by
        # construction, and ``_eval_eqn`` perturbs their hidden carry output
        # exactly like any other producing equation (one perturbation per
        # outer step; scans are reverse-differentiable, so no detach).
        self.descended_scan_eqn_ids = descended_scan_eqn_ids

        # initialize the super class
        # Use dict.fromkeys to deduplicate while preserving insertion order
        # (avoids non-deterministic set iteration for jaxpr variable ordering)
        hidden_outvars_ordered = list(dict.fromkeys(hidden_outvar_to_invar.keys()))
        hidden_invars_ordered = list(dict.fromkeys(hidden_outvar_to_invar.values()))
        super().__init__(
            weight_invars=weight_invars,
            hidden_invars=set(hidden_invars_ordered),
            hidden_outvars=set(hidden_outvars_ordered),
            invar_to_hidden_path=invar_to_hidden_path,
            outvar_to_hidden_path=outvar_to_hidden_path,
            control_flow=control_flow,
        )
        # Keep an ordered version for deterministic iteration in compile()
        self._hidden_outvars_ordered = hidden_outvars_ordered
        self._hidden_outvar_to_invar = hidden_outvar_to_invar

        self.path_to_state = path_to_state

    def compile(self) -> HiddenPerturbation:
        # new invars, the var order is the same as the hidden_outvars (use ordered list for determinism)
        self.perturb_invars = {
            v: self._new_var_like(v)
            for v in self._hidden_outvars_ordered
        }

        # the hidden states that are not found in the code
        self.hidden_jaxvars_to_remove = set(self.hidden_outvars)

        # final revised equations
        self.revised_eqns: list = []

        # ``while`` equations to detach: every loop whose result reaches a
        # hidden outvar, directly or through further equations
        self._while_ids_to_detach = _while_eqns_reaching(
            self.closed_jaxpr.jaxpr, set(self.hidden_outvars))

        # [in-step reads of new hidden values, #239]
        # ``_perturbed`` maps a cut source and every output of an equation
        # emitted twice to its value in the perturbed world
        self._new_read_plan = self._plan_new_reads()
        self._perturbed: Dict[Var, Var] = {}
        self._eqn_index = 0

        # revising equations
        self._eval_jaxpr(self.closed_jaxpr.jaxpr)
        plan = self._new_read_plan
        if plan is not None and self._eqn_index != len(self.closed_jaxpr.jaxpr.eqns):
            raise RuntimeError(
                f'Internal error: the perturbation visited {self._eqn_index} of '
                f'{len(self.closed_jaxpr.jaxpr.eqns)} equations. Please report '
                f'an issue at {git_issue_addr}.'
            )

        # [read-only hidden states]
        # A hidden state that is read but never written has no producing
        # equation: its jaxpr outvar IS its invar (or a constvar). Synthesize
        # the perturbed passthrough  ``h^t = h^{t-1} + p``  and substitute the
        # fresh var into the jaxpr outvar slots that referenced the old one.
        # The equation invars are NOT substituted: reads inside the step see
        # ``h^{t-1}``, which is unperturbed by definition.
        outvar_subst: Dict[Var, Var] = {}
        source_vars = set(self.closed_jaxpr.jaxpr.invars) | set(self.closed_jaxpr.jaxpr.constvars)
        for hidden_var in tuple(self._hidden_outvars_ordered):
            if hidden_var not in self.hidden_jaxvars_to_remove:
                continue
            if hidden_var not in source_vars:
                continue  # truly unexplained; reported below
            self.hidden_jaxvars_to_remove.remove(hidden_var)
            perturb_var = self.perturb_invars[hidden_var]
            fresh = self._new_var_like(hidden_var)
            self.revised_eqns.append(
                new_jaxpr_eqn(
                    [hidden_var, perturb_var],
                    [fresh],
                    jax.lax.add_p,
                    {},
                    set(),
                )
            )
            outvar_subst[hidden_var] = fresh

        # [final checking]
        # If there are hidden states that are not found in the code, we raise an error.
        if len(self.hidden_jaxvars_to_remove) > 0:
            hid_paths = [self.outvar_to_hidden_path[v] for v in self.hidden_jaxvars_to_remove]
            hid_info = '\n'.join([f'{v} -> {path}' for v, path in zip(self.hidden_jaxvars_to_remove, hid_paths)])
            raise ValueError(
                f'Error: we did not found your defined hidden state '
                f'(see the following information) in the code. \n'
                f'Please report an issue to the developers at {git_issue_addr}. \n'
                f'The missed hidden states are: \n'
                f'{hid_info}'
            )

        # new jaxpr
        new_outvars = [
            outvar_subst.get(v, v) if isinstance(v, Var) else v
            for v in self.closed_jaxpr.jaxpr.outvars
        ]
        if plan is not None:
            # the outputs, the non-hidden states and the new values of the cut
            # sources come from the perturbed world
            merge = set(self.bypass_merge_positions)
            for pos, v in enumerate(new_outvars):
                if isinstance(v, Var) and (pos in merge or v in plan.sources):
                    new_outvars[pos] = self._perturbed.get(v, v)
            for v in plan.sources:
                outvar_subst[v] = self._perturbed[v]

        # [same-step bypass gradients]
        bypass = self._add_output_bypass(
            new_outvars,
            cut_vars={outvar_subst.get(v, v) for v in self._hidden_outvars_ordered},
        )
        param_bypass_keys = [k for k in bypass.bypass_keys if k[0] == 'param']
        hidden_bypass_keys = [k for k in bypass.bypass_keys if k[0] == 'hidden']
        # one copy per hidden bypass (a hidden invar is a single leaf)
        n_param_leaves = len(bypass.bypass_vars) - len(hidden_bypass_keys)
        jaxpr = Jaxpr(
            constvars=list(self.closed_jaxpr.jaxpr.constvars),
            invars=(
                list(self.closed_jaxpr.jaxpr.invars)
                + list(self.perturb_invars.values())
                + list(bypass.bypass_vars)
            ),
            outvars=bypass.outvars,
            eqns=bypass.eqns,
            effects=self.closed_jaxpr.jaxpr.effects,
            debug_info=self.closed_jaxpr.jaxpr.debug_info,
        )
        revised_closed_jaxpr = ClosedJaxpr(jaxpr, self.closed_jaxpr.literals)

        # finalizing
        perturb_hidden_paths = [self.outvar_to_hidden_path[v] for v in self._hidden_outvars_ordered]
        perturb_hidden_states = [self.path_to_state[self.outvar_to_hidden_path[v]] for v in
                                 self._hidden_outvars_ordered]
        info = HiddenPerturbation(
            perturb_vars=brainstate.util.PrettyList(self.perturb_invars.values()),
            perturb_hidden_paths=brainstate.util.PrettyList(perturb_hidden_paths),
            perturb_hidden_states=brainstate.util.PrettyList(perturb_hidden_states),
            perturb_jaxpr=revised_closed_jaxpr,
            bypass_param_paths=brainstate.util.PrettyList([k[1] for k in param_bypass_keys]),
            bypass_vars=brainstate.util.PrettyList(bypass.bypass_vars[:n_param_leaves]),
            bypass_hidden_paths=brainstate.util.PrettyList([k[1] for k in hidden_bypass_keys]),
            bypass_hidden_vars=brainstate.util.PrettyList(bypass.bypass_vars[n_param_leaves:]),
            new_read_cuts=() if plan is None else plan.cuts,
            new_read_conflicts=self._new_read_conflicts,
            new_read_exact=self.new_read_policy is not None and not self._new_read_conflicts,
        )

        # remove the temporal data
        self.perturb_invars = dict()
        self.revised_eqns = []
        self.hidden_jaxvars_to_remove = set()
        self._perturbed = {}
        self._new_read_plan = None
        return info

    def _plan_new_reads(self) -> Optional[NewReadPlan]:
        """Plan the cut reads (#239) and record the outcome; return the plan
        to apply, or ``None`` (no policy, nothing to cut, or a conflict)."""
        self._new_read_conflicts: Tuple[str, ...] = ()
        if self.new_read_policy is None:
            return None
        plan = plan_new_read_cuts(
            self.closed_jaxpr.jaxpr,
            self._hidden_outvars_ordered,
            self.outvar_to_hidden_path,
            self.bypass_merge_positions,
            self.new_read_policy,
        )
        paths = tuple(dict.fromkeys(p for cut in plan.cuts for p in cut))
        if plan.conflicts:
            self._new_read_conflicts = plan.conflicts
            emit(
                kind=DiagnosticKind.HIDDEN_NEW_READ_DOUBLE_COUNT,
                level=DiagnosticLevel.ERROR,
                message=(
                    f'Hidden updates read the new value of another hidden state '
                    f'in the same step, and these reads cannot be separated: '
                    f'{list(plan.conflicts)}. The single-step learning signal '
                    f'is the total derivative, which counts the path through '
                    f'such a read twice when the eligibility trace carries it '
                    f'too, so the gradient can be wrong. '
                    f"vjp_method='multi-step' builds no perturbation and "
                    f'avoids it.'
                ),
                hidden_paths=paths,
                context={'cuts': plan.cuts, 'conflicts': plan.conflicts},
            )
            return None
        if not plan.cuts:
            return None
        emit(
            kind=DiagnosticKind.HIDDEN_NEW_READ_CUT,
            level=DiagnosticLevel.INFO,
            message=(
                f'The single-step learning signal holds the in-step reads '
                f'{list(plan.cuts)} fixed: the eligibility traces already carry '
                f'them.'
            ),
            hidden_paths=paths,
            context={'cuts': plan.cuts},
        )
        return plan

    def _add_output_bypass(self, outvars: List, cut_vars: Set[Var]) -> OutputBypass:
        """Rewrite the perturbed step to also yield the same-step bypass
        gradients of the parameters and of the previous hidden values (see
        ``_compiler/output_bypass.py``).

        The sources are keyed ``('param', path)`` (parameters first, so their
        copies come first in ``bypass_vars``) and ``('hidden', path)``.
        """
        if self.weight_path_to_invars is None:
            return OutputBypass(self.revised_eqns, outvars, [], [], [])
        sources: Dict = {('param', p): invars for p, invars in self.weight_path_to_invars.items()}
        for ov in self._hidden_outvars_ordered:
            sources[('hidden', self.outvar_to_hidden_path[ov])] = [self._hidden_outvar_to_invar[ov]]
        bypass = add_output_bypass(
            self.revised_eqns,
            outvars,
            sources=sources,
            cut_vars=cut_vars,
            merge_positions=self.bypass_merge_positions,
        )
        bypass_params = [k[1] for k in bypass.bypass_keys if k[0] == 'param']
        bypass_hidden = [k[1] for k in bypass.bypass_keys if k[0] == 'hidden']
        dropped = [k[1] for k in bypass.dropped_keys]
        if bypass_params or bypass_hidden:
            emit(
                kind=DiagnosticKind.OUTPUT_BYPASSES_HIDDEN,
                level=DiagnosticLevel.INFO,
                message=(
                    f'Parameters {bypass_params} and previous hidden values '
                    f'{bypass_hidden} reach the step output through a value that '
                    f'is not a new hidden state; their exact same-step gradient '
                    f'along that path is added to the eligibility-trace gradient.'
                ),
                context={'params': tuple(bypass_params), 'hidden': tuple(bypass_hidden)},
            )
        if dropped:
            emit(
                kind=DiagnosticKind.OUTPUT_BYPASS_DROPPED,
                level=DiagnosticLevel.WARNING,
                message=(
                    f'Parameters or previous hidden values {dropped} reach the '
                    f'step output, bypassing every new hidden state, through a '
                    f'while loop or an equation with effects. That part of their '
                    f'same-step gradient cannot be computed and is zero. Read the '
                    f'output from a new hidden state, or move the loop out of the '
                    f'output path.'
                ),
                context={'sources': tuple(dropped)},
            )
        return bypass

    def _eval_pjit(self, eqn: JaxprEqn) -> None:
        """
        Evaluating the pjit primitive.

        Note: Unlike the base class, the perturbation compiler must process ALL
        equations (including etrace ops without gradient) to maintain a valid jaxpr.
        Skipping equations would leave variable references unresolved.
        """
        self._eval_eqn(eqn)

    def _eval_while(self, eqn: JaxprEqn) -> None:
        """Evaluate a ``while`` equation, detaching its inputs when its
        result reaches a hidden state.

        JAX cannot transpose ``while``, so a loop on the path to a hidden
        state left as-is would make ``jax.vjp`` of the perturbed step
        untraceable. That holds whether the loop writes the hidden state
        itself or its result passes through further equations first
        (``h <- while(...)(h) + I``). Every ``Var`` input of such a loop is
        therefore routed through a ``stop_gradient`` equation first; the
        perturbation add stays outside the detach (see the class Notes).
        A loop that reaches no hidden state (e.g. on the readout only) is
        left unchanged.
        """
        check_unsupported_op(self, eqn, 'while')
        if id(eqn) in self._while_ids_to_detach:
            eqn = self._detach_invars(eqn)
        self._eval_eqn(eqn)

    def _detach_invars(self, eqn: JaxprEqn) -> JaxprEqn:
        """Route every ``Var`` input of *eqn* through ``stop_gradient``.

        Emits the ``stop_gradient`` equations into ``revised_eqns`` and
        returns a copy of *eqn* consuming the detached variables. Literal
        inputs are left untouched.
        """
        new_invars: List[Union[Var, Literal]] = []
        n_detached = 0
        for iv in eqn.invars:
            if not isinstance(iv, Var):
                new_invars.append(iv)
                continue
            fresh = new_var('', iv.aval)
            self.revised_eqns.append(
                new_jaxpr_eqn(
                    [iv],
                    [fresh],
                    stop_gradient_p,
                    {},
                    set(),
                    eqn.source_info.replace(),
                )
            )
            new_invars.append(fresh)
            n_detached += 1
        emit(
            kind=DiagnosticKind.CONTROL_FLOW_OPAQUE_FWD,
            level=DiagnosticLevel.WARNING,
            message=(
                'Detached the inputs of a hidden-producing while loop in the '
                'perturbed jaxpr: same-step reverse-mode signals THROUGH the '
                'loop are zero there. The loop\'s own hidden-state learning '
                'signal stays exact and its temporal credit flows via the '
                'forward-computed hidden-to-hidden Jacobian, but any '
                'parameter or other hidden group whose only same-step path '
                'to the loss crosses this loop receives a ZERO learning '
                'signal (e.g. the weights of an upstream layer feeding the '
                'loop). Move such parameters\' influence out of the loop '
                'inputs, or avoid stacking trainable layers behind a '
                'while-hidden layer.'
            ),
            context={'site': 'perturbation', 'n_detached': n_detached},
        )
        return eqn.replace(invars=new_invars)

    def _eval_eqn(self, eqn: JaxprEqn) -> None:
        # ------------------------------------------------
        #
        # For every hidden outvar the equation produces (any number, at any
        # position), we add a perturbation:
        #    y = f(x)  =>  y = f(x) + perturb_var
        #
        # Particularly, each hidden outvar slot is first redirected to a
        # fresh variable
        #    new_outvar = f(x)
        #
        # then a perturbation equation re-defines the hidden var
        #    y = new_outvar + perturb_var
        #
        # With cut reads (#239) an equation is emitted in the world it
        # serves: on a cut read it reads the unperturbed sources; towards a
        # kept read, an output or a non-hidden state it reads the perturbed
        # ones; serving both, it is emitted once for each.
        # ------------------------------------------------
        plan = self._new_read_plan
        if plan is None:
            self._perturb_outvars(eqn)
            return
        index = self._eqn_index
        self._eqn_index += 1
        if index not in plan.output_eqns:
            self._perturb_outvars(eqn)
            return
        perturbed_in = [self._perturbed.get(v, v) if isinstance(v, Var) else v for v in eqn.invars]
        if index not in plan.state_eqns:
            self._perturb_outvars(eqn.replace(invars=perturbed_in))
            return
        self._perturb_outvars(eqn)
        fresh = [self._new_var_like(ov) for ov in eqn.outvars]
        self.revised_eqns.append(eqn.replace(invars=perturbed_in, outvars=fresh))
        for ov, nv in zip(eqn.outvars, fresh):
            if ov not in self.hidden_outvars:  # its hidden outputs are cut reads: unused
                self._perturbed[ov] = nv

    def _perturb_outvars(self, eqn: JaxprEqn) -> None:
        """Emit ``eqn`` and redefine its hidden outvars ``h = f(x) + p``. A
        cut source keeps its outvar for the cut reads, and its perturbed
        value goes to a fresh var for every other reader."""
        hidden_positions = [
            i for i, ov in enumerate(eqn.outvars)
            if ov in self.hidden_jaxvars_to_remove
        ]
        if not hidden_positions:
            self.revised_eqns.append(eqn.replace())
            return
        plan = self._new_read_plan
        sources = {} if plan is None else plan.sources

        new_outvars = list(eqn.outvars)
        for i in hidden_positions:
            hidden_var = eqn.outvars[i]
            self.hidden_jaxvars_to_remove.remove(hidden_var)
            if hidden_var not in sources:
                new_outvars[i] = self._new_var_like(hidden_var)
        self.revised_eqns.append(eqn.replace(outvars=new_outvars))

        for i in hidden_positions:
            hidden_var = eqn.outvars[i]
            if hidden_var in sources:
                perturbed = self._new_var_like(hidden_var)
                self._perturbed[hidden_var] = perturbed
                ins, outs = [hidden_var, self.perturb_invars[hidden_var]], [perturbed]
            else:
                ins, outs = [new_outvars[i], self.perturb_invars[hidden_var]], [hidden_var]
            self.revised_eqns.append(
                new_jaxpr_eqn(ins, outs, jax.lax.add_p, {}, set(), eqn.source_info.replace())
            )

    def _new_var_like(self, v: Var) -> Var:
        return new_var('', jax.core.ShapedArray(shaped_aval(v).shape, shaped_aval(v).dtype))


def _while_eqns_reaching(jaxpr: Jaxpr, targets: Set[Var]) -> Set[int]:
    """Return the ``id()`` of every top-level ``while`` equation whose
    outputs reach one of ``targets``.

    One backward pass: a variable is *needed* if a target depends on it,
    and an equation is on a path to a target if any of its outputs is
    needed.
    """
    needed = set(targets)
    found: Set[int] = set()
    for eqn in reversed(jaxpr.eqns):
        if not any(ov in needed for ov in eqn.outvars):
            continue
        if is_while_primitive(eqn):
            found.add(id(eqn))
        needed.update(iv for iv in eqn.invars if isinstance(iv, Var))
    return found


def add_hidden_perturbation_in_jaxpr(
    closed_jaxpr: ClosedJaxpr,
    hidden_outvar_to_invar: Dict[HiddenOutVar, HiddenInVar],
    weight_invars: Set[Var],
    invar_to_hidden_path: Dict[HiddenInVar, Path],
    outvar_to_hidden_path: Dict[Var, Path],
    path_to_state: Dict[Path, brainstate.HiddenState],
    control_flow: ControlFlowPolicy = DEFAULT_CONTROL_FLOW_POLICY,
    descended_scan_eqn_ids: FrozenSet[int] = frozenset(),
    weight_path_to_invars: Optional[Mapping[Path, Sequence[Var]]] = None,
    bypass_merge_positions: Sequence[int] = (),
    new_read_policy: Optional[NewReadPolicy] = None,
) -> HiddenPerturbation:
    """
    Adding perturbations to the hidden states in the jaxpr, and replacing the hidden states with the perturbed states.

    Parameters
    ----------
    closed_jaxpr : ClosedJaxpr
        The closed jaxpr for the model.
    hidden_outvar_to_invar : dict
        The mapping from the hidden output variable to the hidden input variable.
    weight_invars : set
        The weight input variables.
    invar_to_hidden_path : dict
        The mapping from the weight input variable to the hidden state path.
    outvar_to_hidden_path : dict
        The mapping from the outvar to the state id.
    path_to_state : dict
        The mapping from the hidden state path to the state.
    control_flow : ControlFlowPolicy, optional
        The :class:`~braintrace.ControlFlowPolicy` governing opaque
        control-flow handling. Under the default policy a hidden-producing
        ``while`` is kept and its inputs are detached in the perturbed jaxpr
        (see :class:`JaxprEvalForHiddenPerturbation`).
    descended_scan_eqn_ids : frozenset of int, optional
        ``id()`` values of scan equations rewritten by structured scan descent
        (Phase 4); exempt from the unsupported-op checks.
    weight_path_to_invars : dict, optional
        Parameter path to its leaf invars. When given, parameters whose path
        to an output at ``bypass_merge_positions`` bypasses every new hidden
        value get differentiable copies in the perturbed jaxpr (see
        :attr:`HiddenPerturbation.bypass_vars`). ``None`` (default) disables
        this.
    bypass_merge_positions : sequence of int, optional
        Positions of the jaxpr outvars where a bypass is attached.
    new_read_policy : NewReadPolicy, optional
        Decides which in-step reads of new hidden values to cut (issue
        #239; see :class:`HiddenPerturbation`). ``None`` (default) cuts none.

    Returns
    -------
    HiddenPerturbation
        The revised closed jaxpr with the perturbations.
    """
    return JaxprEvalForHiddenPerturbation(
        closed_jaxpr=closed_jaxpr,
        hidden_outvar_to_invar=hidden_outvar_to_invar,
        weight_invars=weight_invars,
        invar_to_hidden_path=invar_to_hidden_path,
        outvar_to_hidden_path=outvar_to_hidden_path,
        path_to_state=path_to_state,
        control_flow=control_flow,
        descended_scan_eqn_ids=descended_scan_eqn_ids,
        weight_path_to_invars=weight_path_to_invars,
        bypass_merge_positions=bypass_merge_positions,
        new_read_policy=new_read_policy,
    ).compile()


def add_hidden_perturbation_from_minfo(
    minfo: ModuleInfo,
    descended_scan_eqn_ids: FrozenSet[int] = frozenset(),
    include_output_bypass: bool = True,
    new_read_policy: Optional[NewReadPolicy] = None,
) -> HiddenPerturbation:
    """Add hidden-state perturbations from a ``ModuleInfo``.

    Adds perturbations to the hidden states in the module jaxpr and replaces
    the hidden states with the perturbed states.

    Parameters
    ----------
    minfo : ModuleInfo
        The model information.
    descended_scan_eqn_ids : frozenset of int, default ``frozenset()``
        ``id()`` values of scan equations rewritten by structured scan
        descent (Phase 4); exempt from the unsupported-op checks.
    include_output_bypass : bool, default ``True``
        Whether to add the same-step bypass gradients of the parameters and
        of the previous hidden values (see
        :attr:`HiddenPerturbation.bypass_param_paths`). Disable it for a
        learning signal that replaces ``dL/dh`` (``'modulatory'``).
    new_read_policy : NewReadPolicy, optional
        Decides which in-step reads of new hidden values to cut (issue
        #239; see :class:`HiddenPerturbation`). ``None`` (default) cuts none:
        the signal is the total derivative.

    Returns
    -------
    HiddenPerturbation
        The hidden-perturbation information.

    See Also
    --------
    add_hidden_perturbation_in_module : Equivalent helper starting from a model.
    """
    return add_hidden_perturbation_in_jaxpr(
        closed_jaxpr=minfo.closed_jaxpr,
        hidden_outvar_to_invar=minfo.hidden_outvar_to_invar,
        weight_invars=set(minfo.weight_invars),
        invar_to_hidden_path=minfo.invar_to_hidden_path,
        outvar_to_hidden_path=minfo.outvar_to_hidden_path,
        path_to_state=minfo.retrieved_model_states,
        control_flow=minfo.control_flow,
        descended_scan_eqn_ids=descended_scan_eqn_ids,
        weight_path_to_invars=minfo.weight_path_to_invars if include_output_bypass else None,
        bypass_merge_positions=_bypass_merge_positions(minfo),
        new_read_policy=new_read_policy,
    )


def _bypass_merge_positions(minfo: ModuleInfo) -> List[int]:
    """Outvar positions where a parameter's bypass is attached: the model
    outputs and the leaves of non-hidden, non-param states (whose incoming
    cotangent the backward pass honours). Hidden-state slots and the extra
    Jacobian temporaries are excluded."""
    positions = list(range(minfo.num_var_out))
    k = minfo.num_var_out
    for st, outvar in zip(minfo.compiled_model_states, minfo.state_tree_outvars):
        n = len(jax.tree.leaves(outvar))
        if not isinstance(st, (brainstate.HiddenState, brainstate.ParamState)):
            positions.extend(range(k, k + n))
        k += n
    return positions


def add_hidden_perturbation_in_module(
    model: brainstate.nn.Module,
    *model_args: Any,
    **model_kwargs: Any,
) -> HiddenPerturbation:
    """Add hidden-state perturbations from a model.

    Adds perturbations to the hidden states of the given module and replaces the
    hidden states with the perturbed states.

    Parameters
    ----------
    model : brainstate.nn.Module
        The neural-network module to which hidden-state perturbations are added.
    *model_args
        Additional positional arguments passed to the model.
    **model_kwargs
        Additional keyword arguments passed to the model.

    Returns
    -------
    HiddenPerturbation
        Information about the perturbations added to the hidden states,
        including the perturbed variables, paths, states, and the revised
        jaxpr.

    See Also
    --------
    add_hidden_perturbation_from_minfo : Equivalent helper starting from ``ModuleInfo``.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintrace
        >>> gru = braintrace.nn.GRUCell(3, 4)
        >>> _ = brainstate.nn.init_all_states(gru)
        >>> inputs = brainstate.random.randn(3)
        >>> hidden_perturb = braintrace.add_hidden_perturbation_in_module(gru, inputs)
        >>> len(hidden_perturb.perturb_vars)
        1
    """
    minfo = extract_module_info(model, *model_args, **model_kwargs)
    return add_hidden_perturbation_from_minfo(minfo)
