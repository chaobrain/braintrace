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

"""Graph-hash gate: fingerprint what the compiler builds, per test (test support).

A refactor of the compiler should leave every compiled graph, every eligibility
trace shape and every diagnostic unchanged unless it says otherwise. This
pytest plugin records, for every test, three channels:

``graphs``
    One SHA-1 per :class:`~braintrace.ETraceGraph` built, in build order, over
    :func:`graph_summary`: the module jaxpr; per hidden group its paths,
    descent flag, transition jaxpr and constvars, ``is_diagonal_recurrence``
    and SnAp pattern; per relation its weight paths, leaf indices, groups,
    connected paths, path classification, control-flow context and
    ``y -> hidden`` jaxprs; the hidden perturbation (jaxpr, perturbed paths,
    bypass paths); and the compile-time diagnostics as a multiset of
    ``(kind, level)``.
``traces``
    The sorted set of eligibility-trace shapes and dtypes after
    ``init_etrace_state`` and at the output of each ``_update_etrace_data``,
    labelled by relation (primitive, weight paths, group index), never by
    object id.
``records``
    The diagnostics an algorithm adds after compiling (the online-credit
    records), as a sorted multiset of ``(kind, level)``.

Usage::

    BRAINTRACE_GRAPH_HASH_OUT=/tmp/gh/base pytest -p braintrace._testing.graph_hash braintrace
    BRAINTRACE_GRAPH_HASH_OUT=/tmp/gh/head pytest -p braintrace._testing.graph_hash braintrace
    python -m braintrace._testing.graph_hash compare /tmp/gh/base /tmp/gh/head

Each process (each xdist worker) writes ``<prefix>.<worker>.json``;
:func:`load` merges them. ``compare`` exits with status 1 when any test
present on both sides changed in any channel.

The plugin only observes: every hook calls the original and returns its
result unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import json
import os
import sys
from collections import Counter
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    'CHANNELS',
    'Recorder',
    'graph_summary',
    'graph_digest',
    'trace_labels',
    'install',
    'recording',
    'load',
    'compare',
    'main',
]

#: The recorded channels, in report order.
CHANNELS: Tuple[str, ...] = ('graphs', 'traces', 'records')

#: Environment variable naming the output prefix of the pytest plugin.
OUT_ENV = 'BRAINTRACE_GRAPH_HASH_OUT'


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def _var_index(jaxpr: Any) -> Dict[int, int]:
    """Number the variables of ``jaxpr`` in definition order (by ``id``).

    ``str(Var)`` carries the object id, which differs between runs; a
    position in the module jaxpr does not.
    """
    index: Dict[int, int] = {}
    inner = getattr(jaxpr, 'jaxpr', jaxpr)
    for v in list(inner.constvars) + list(inner.invars):
        index.setdefault(id(v), len(index))
    for eqn in inner.eqns:
        for v in eqn.outvars:
            index.setdefault(id(v), len(index))
    return index


def _vars(vs: Sequence[Any], index: Dict[int, int]) -> str:
    parts = []
    for v in vs:
        aval = getattr(v, 'aval', None)
        where = index.get(id(v))
        parts.append(f'{"?" if where is None else where}:{aval}')
    return '[' + ', '.join(parts) + ']'


def _array(a: Any) -> str:
    if a is None:
        return 'None'
    arr = np.asarray(a)
    return f'{arr.dtype}{arr.shape}:{hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]}'


def _snap(snap: Any) -> str:
    if snap is None:
        return 'None'
    return (f'snap(n={snap.n}, varshape={tuple(snap.varshape)}, '
            f'axes={[_array(a) for a in snap.axes]}, neighbours={_array(snap.neighbours)}, '
            f'valid={_array(snap.valid)}, conservative={snap.conservative}, reason={snap.reason!r})')


def _diagnostics(records: Sequence[Any]) -> str:
    counts = Counter((r.kind.value, r.level.name) for r in records)
    return str(sorted(counts.items()))


def graph_summary(graph: Any) -> str:
    """A deterministic text summary of a compiled :class:`~braintrace.ETraceGraph`.

    Parameters
    ----------
    graph : ETraceGraph
        The compiled graph.

    Returns
    -------
    str
        The summary; equal graphs give equal text across processes.
    """
    minfo = graph.module_info
    index = _var_index(minfo.jaxpr)
    parts = [str(minfo.jaxpr)]
    for grp in graph.hidden_groups:
        parts.append(
            f'G{grp.index} paths={list(grp.hidden_paths)} descent={grp.descent is not None} '
            f'diagonal={grp.is_diagonal_recurrence} '
            f'constvars={_vars(grp.transition_jaxpr_constvars, index)} snap={_snap(grp.snap)}')
        parts.append(str(grp.transition_jaxpr))
    for r in graph.hidden_param_op_relations:
        ctx = r.control_flow_context
        parts.append(
            f'R{r.primitive.name} paths={sorted(r.trainable_paths.items())} '
            f'leaves={sorted(r.trainable_leaf_indices.items())} '
            f'groups={[g.index for g in r.hidden_groups]} connected={list(r.connected_hidden_paths)} '
            f'classes={sorted(r.path_classification.items())} '
            f'ctx={None if ctx is None else (ctx.scan.length, bool(getattr(ctx, "lifted", False)))}')
        parts.extend(str(j) for j in r.y_to_hidden_group_jaxprs)
    perturb = graph.hidden_perturb
    if perturb is None:
        parts.append('perturb=None')
    else:
        parts.append(
            f'perturb paths={list(perturb.perturb_hidden_paths)} '
            f'bypass_params={list(perturb.bypass_param_paths)} '
            f'bypass_hidden={list(perturb.bypass_hidden_paths)}')
        parts.append(str(perturb.perturb_jaxpr))
    parts.append(f'diagnostics={_diagnostics(graph.diagnostics)}')
    return '\n'.join(parts)


def graph_digest(graph: Any) -> str:
    """SHA-1 of :func:`graph_summary`, or ``'ERR <type>'`` if it cannot be built."""
    try:
        text = graph_summary(graph)
    except Exception as e:  # noqa: BLE001 -- an observer never raises
        return f'ERR {type(e).__name__}'
    return hashlib.sha1(text.encode()).hexdigest()


def _relation_label(relation: Any) -> str:
    paths = sorted(set(map(str, relation.trainable_paths.values())))
    return f'{relation.primitive.name}{paths}'


def _id_labels(graph: Any) -> Dict[int, str]:
    """Map the ``id`` of every relation's x/y var to a label.

    The trace dictionaries are keyed by ``id(var)``; the label replaces it.
    """
    labels: Dict[int, str] = {}
    for r in getattr(graph, 'hidden_param_op_relations', ()):
        name = _relation_label(r)
        if r.y_var is not None:
            labels.setdefault(id(r.y_var), f'y:{name}')
        if r.x_var is not None:
            labels.setdefault(id(r.x_var), f'x:{name}')
    return labels


#: Integers above this in a trace key are object ids, not indices.
_MAX_INDEX = 1 << 24


def _key(k: Any, labels: Dict[int, str]) -> str:
    if isinstance(k, tuple):
        return '(' + ','.join(_key(e, labels) for e in k) + ')'
    if isinstance(k, int) and not isinstance(k, bool):
        if k in labels:
            return labels[k]
        if k > _MAX_INDEX:  # an unlabelled object id: never print it
            return '<id>'
    return str(k)


def _path(path: Sequence[Any], labels: Dict[int, str]) -> str:
    out = []
    for entry in path:
        for attr in ('key', 'idx', 'name'):
            if hasattr(entry, attr):
                out.append(_key(getattr(entry, attr), labels))
                break
        else:
            out.append(str(entry))
    return '/'.join(out)


def trace_labels(algorithm: Any, values: Any, stage: str) -> List[str]:
    """Label every leaf of an eligibility-trace pytree with its shape and dtype.

    Parameters
    ----------
    algorithm : ETraceAlgorithm
        The algorithm the traces belong to (its graph names the keys).
    values : PyTree
        Trace values: arrays or tracers.
    stage : str
        ``'init'`` or ``'update'``.

    Returns
    -------
    list of str
        ``'<Algorithm>|<stage>|<key path>|<shape>:<dtype>'`` per leaf.
    """
    import jax
    labels = _id_labels(getattr(algorithm, 'graph', None))
    out = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(values)[0]:
        shape = tuple(getattr(leaf, 'shape', ()))
        dtype = getattr(leaf, 'dtype', type(leaf).__name__)
        out.append(f'{type(algorithm).__name__}|{stage}|{_path(path, labels)}|{shape}:{dtype}')
    return out


# ---------------------------------------------------------------------------
# recording
# ---------------------------------------------------------------------------

class Recorder:
    """Per-test accumulator of the three channels."""

    def __init__(self) -> None:
        self.current: Optional[str] = None
        self.graphs: Dict[str, List[str]] = {}
        self.traces: Dict[str, set] = {}
        self.records: Dict[str, Counter] = {}

    @property
    def test(self) -> str:
        return self.current or '<no test>'

    def add_graph(self, graph: Any) -> None:
        self.graphs.setdefault(self.test, []).append(graph_digest(graph))

    def add_traces(self, items: Sequence[str]) -> None:
        self.traces.setdefault(self.test, set()).update(items)

    def add_records(self, records: Sequence[Any]) -> None:
        bucket = self.records.setdefault(self.test, Counter())
        for r in records:
            bucket[f'{r.kind.value}:{r.level.name}'] += 1

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        """The channels in their JSON form."""
        return {
            'graphs': {t: list(v) for t, v in self.graphs.items()},
            'traces': {t: sorted(v) for t, v in self.traces.items()},
            'records': {t: sorted(f'{k}x{n}' for k, n in v.items()) for t, v in self.records.items()},
        }


def _algorithm_classes() -> List[type]:
    from braintrace._algorithm.base import ETraceAlgorithm
    seen, stack = [], [ETraceAlgorithm]
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.append(cls)
        stack.extend(cls.__subclasses__())
    return seen


def install(recorder: Recorder) -> Callable[[], None]:
    """Patch the observation points; return a function that restores them.

    Patched: ``ETraceGraph.__new__``, every ``init_etrace_state`` and
    ``_update_etrace_data`` an algorithm class defines, and
    ``ETraceVjpAlgorithm._add_records``. Each wrapper calls the original and
    returns its result.
    """
    import braintrace  # noqa: F401 -- registers every algorithm class
    from braintrace._algorithm.vjp_base import ETraceVjpAlgorithm
    from braintrace._compiler import graph as graph_module

    restore: List[Tuple[Any, str, Any]] = []

    def patch(owner: Any, name: str, new: Any) -> None:
        restore.append((owner, name, owner.__dict__[name]))
        setattr(owner, name, new)

    graph_cls = graph_module.ETraceGraph
    orig_new = graph_cls.__new__

    def new(cls: Any, *args: Any, **kwargs: Any) -> Any:
        obj = orig_new(cls, *args, **kwargs)
        recorder.add_graph(obj)
        return obj

    patch(graph_cls, '__new__', new)

    # Nesting depth of init_etrace_state per algorithm object (a subclass calls
    # super()); kept here, never on the module, so the module graph is untouched.
    depths: Dict[int, int] = {}

    for cls in _algorithm_classes():
        if 'init_etrace_state' in cls.__dict__:
            def init(self: Any, *args: Any, _orig: Any = cls.__dict__['init_etrace_state'], **kwargs: Any) -> Any:
                depth = depths.get(id(self), 0)
                depths[id(self)] = depth + 1
                try:
                    out = _orig(self, *args, **kwargs)
                finally:
                    if depth:
                        depths[id(self)] = depth
                    else:
                        depths.pop(id(self), None)
                if depth == 0:
                    try:
                        recorder.add_traces(trace_labels(self, self._get_etrace_data(), 'init'))
                    except Exception as e:  # noqa: BLE001 -- an observer never raises
                        recorder.add_traces([f'{type(self).__name__}|init|ERR {type(e).__name__}'])
                return out

            patch(cls, 'init_etrace_state', init)
        if '_update_etrace_data' in cls.__dict__:
            def update(self: Any, *args: Any, _orig: Any = cls.__dict__['_update_etrace_data'], **kwargs: Any) -> Any:
                out = _orig(self, *args, **kwargs)
                try:
                    recorder.add_traces(trace_labels(self, out, 'update'))
                except Exception as e:  # noqa: BLE001
                    recorder.add_traces([f'{type(self).__name__}|update|ERR {type(e).__name__}'])
                return out

            patch(cls, '_update_etrace_data', update)

    orig_add = ETraceVjpAlgorithm.__dict__['_add_records']

    def add_records(self: Any, records: Sequence[Any]) -> None:
        recorder.add_records(records)
        return orig_add(self, records)

    patch(ETraceVjpAlgorithm, '_add_records', add_records)

    def uninstall() -> None:
        for owner, name, value in reversed(restore):
            setattr(owner, name, value)
        restore.clear()

    return uninstall


@contextlib.contextmanager
def recording(test: str = 'test') -> Iterator[Recorder]:
    """Record within a ``with`` block (for tests and ad-hoc probes)."""
    recorder = Recorder()
    recorder.current = test
    uninstall = install(recorder)
    try:
        yield recorder
    finally:
        uninstall()


# ---------------------------------------------------------------------------
# pytest plugin (active only when loaded with ``-p braintrace._testing.graph_hash``)
# ---------------------------------------------------------------------------

_RECORDER = Recorder()
_UNINSTALL: List[Callable[[], None]] = []


def pytest_configure(config: Any) -> None:
    _UNINSTALL.append(install(_RECORDER))


def pytest_runtest_setup(item: Any) -> None:
    _RECORDER.current = item.nodeid


def pytest_sessionfinish(session: Any) -> None:
    out = os.environ.get(OUT_ENV)
    if not out:
        return
    worker = os.environ.get('PYTEST_XDIST_WORKER', 'main')
    with open(f'{out}.{worker}.json', 'w') as f:
        json.dump(_RECORDER.as_dict(), f)


def pytest_unconfigure(config: Any) -> None:
    while _UNINSTALL:
        _UNINSTALL.pop()()


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def load(prefix: str) -> Dict[str, Dict[str, Any]]:
    """Merge the per-process files ``<prefix>.*.json``."""
    merged: Dict[str, Dict[str, Any]] = {c: {} for c in CHANNELS}
    files = sorted(glob.glob(f'{glob.escape(prefix)}.*.json'))
    if not files:
        raise FileNotFoundError(f'no {prefix}.*.json files')
    for path in files:
        with open(path) as f:
            data = json.load(f)
        for channel in CHANNELS:
            merged[channel].update(data.get(channel, {}))
    return merged


def compare(base: Dict[str, Dict[str, Any]], head: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    """Per channel: the tests that changed, and the tests found on one side only.

    A test missing from a channel on one side but present on the other counts
    as added or removed for that channel.
    """
    report: Dict[str, Dict[str, List[str]]] = {}
    for channel in CHANNELS:
        b, h = base.get(channel, {}), head.get(channel, {})
        report[channel] = {
            'changed': sorted(t for t in b.keys() & h.keys() if b[t] != h[t]),
            'only_base': sorted(b.keys() - h.keys()),
            'only_head': sorted(h.keys() - b.keys()),
        }
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m braintrace._testing.graph_hash compare BASE HEAD``."""
    parser = argparse.ArgumentParser(prog='python -m braintrace._testing.graph_hash')
    sub = parser.add_subparsers(dest='command', required=True)
    cmp = sub.add_parser('compare', help='compare two recorded runs')
    cmp.add_argument('base')
    cmp.add_argument('head')
    cmp.add_argument('--show', type=int, default=20, help='tests listed per category')
    args = parser.parse_args(argv)

    base, head = load(args.base), load(args.head)
    report = compare(base, head)
    changed = 0
    for channel in CHANNELS:
        r = report[channel]
        n_tests = len(base[channel].keys() | head[channel].keys())
        n_items = sum(len(v) for v in head[channel].values())
        print(f'{channel}: {n_tests} tests ({n_items} entries on head); changed {len(r["changed"])}, '
              f'only on base {len(r["only_base"])}, only on head {len(r["only_head"])}')
        for category in ('changed', 'only_base', 'only_head'):
            for t in r[category][:args.show]:
                print(f'  {category}: {t}')
            if len(r[category]) > args.show:
                print(f'  {category}: ... {len(r[category]) - args.show} more')
        changed += len(r['changed'])
    return 1 if changed else 0


if __name__ == '__main__':
    sys.exit(main())
