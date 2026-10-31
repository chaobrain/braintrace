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

"""Precondition census: how often a run violates what the compiler assumes (test support).

The online algorithms rely on in-step facts the compiler does not prove
(issues #233-#239). This module measures them on real runs without changing
any value:

``DIAG`` (runtime, #233)
    For a default-mode hidden group (``is_diagonal_recurrence``, no SnAp
    pattern), the Jacobian the algorithm uses
    (:meth:`HiddenGroup.diagonal_jacobian`) against the true per-position
    block (:func:`block_diagonal_last_dim`), when the group has at most
    :data:`MAX_FULL` entries.
``TAIL`` (runtime, #234)
    For a flat relation, the ``df`` read off with an all-ones tangent
    against the true ``dh/dy``: for a fixed tangent ``v``,
    ``jvp(y -> h)(v)`` must equal ``df * v``.
``CHAIN`` (structural, #235)
    A trainable input of an ETP op that is not a raw ``ParamState`` leaf.
``MULTISOURCE`` (structural, #236)
    A trainable input of an ETP op that depends on more than one leaf.
``NEWREAD`` (structural, #239)
    A relation's output reaches the new value of a hidden state ``g``, and the
    new ``g`` reaches the new value of another hidden state ``V`` of the
    relation's groups in the same step, with no ETP op between them.

Runtime channels record ``max|used - true| / max|true|`` per site;
structural channels record ``1.0`` per site. Structural counts are
exposure: a site is recorded for every compiled graph, including graphs that
run ``vjp_method='multi-step'``, which the defect does not affect. Usage::

    BRAINTRACE_CENSUS_OUT=/tmp/census/run pytest -p braintrace._testing.census braintrace
    python -m braintrace._testing.census summarize /tmp/census/run

Each process writes ``<prefix>.main.json`` or ``<prefix>.gw<N>.json``; the
session removes earlier files of the same prefix when it starts.

Attribution: a runtime value is booked to the test that was current when
the measured function was traced. When a later test reuses a cached
executable (same shapes, same function), its values go to the first test;
the maxima per site are right, the test names may not be.

For a single model use :func:`measuring`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from braintrace._testing.graph_hash import _remove_stale_files, _worker_files

__all__ = [
    'CHANNELS',
    'MAX_FULL',
    'Census',
    'install',
    'measuring',
    'structural_sites',
    'load',
    'summarize',
    'main',
]

#: Census channels, in report order.
CHANNELS: Tuple[str, ...] = ('DIAG', 'TAIL', 'CHAIN', 'MULTISOURCE', 'NEWREAD')

#: Largest group (positions x states) the DIAG check builds a dense Jacobian for.
MAX_FULL = 4096

#: Environment variable naming the output prefix of the pytest plugin.
OUT_ENV = 'BRAINTRACE_CENSUS_OUT'


class Census:
    """Accumulator: ``values[test][channel|site] = max deviation``; ``skipped`` likewise."""

    def __init__(self) -> None:
        self.current: Optional[str] = None
        self.values: Dict[str, Dict[str, float]] = {}
        self.skipped: Dict[str, Dict[str, str]] = {}
        self._lock = threading.Lock()

    @property
    def test(self) -> str:
        return self.current or '<no test>'

    def record(self, channel: str, site: str, value: float, test: Optional[str] = None) -> None:
        key = f'{channel}|{site}'
        with self._lock:
            bucket = self.values.setdefault(test or self.test, {})
            bucket[key] = max(bucket.get(key, 0.0), float(value))

    def skip(self, channel: str, site: str, why: str) -> None:
        with self._lock:
            self.skipped.setdefault(self.test, {})[f'{channel}|{site}'] = why[:120]

    def max(self, channel: str) -> float:
        """Largest value recorded on ``channel`` over every test (0.0 if none)."""
        vals = [v for t in self.values.values() for k, v in t.items() if k.split('|', 1)[0] == channel]
        return max(vals, default=0.0)

    def as_dict(self) -> Dict[str, Any]:
        return {'values': self.values, 'skipped': self.skipped}


def _relative(num: Any, den: Any) -> float:
    num = float(np.max(np.abs(np.asarray(num)))) if np.size(num) else 0.0
    den = float(np.max(np.abs(np.asarray(den)))) if np.size(den) else 0.0
    return num / den if den > 0 else num


def _callback(census: Census, channel: str, site: str) -> Callable[[Any, Any], None]:
    test = census.test  # bound at trace time: the callback may run later

    def cb(num: Any, den: Any) -> None:
        census.record(channel, site, _relative(num, den), test=test)

    return cb


def _tangent(shape: Sequence[int], dtype: Any) -> Any:
    """A fixed, dense tangent with distinct entries (no random state)."""
    import jax.numpy as jnp
    n = int(np.prod(shape)) if len(shape) else 1
    return jnp.sin(jnp.arange(n, dtype=dtype) * 0.7 + 0.3).reshape(tuple(shape))


def _group_site(group: Any) -> str:
    return f'G{[str(p) for p in group.hidden_paths]}' + ('/descended' if group.descent is not None else '')


# ---------------------------------------------------------------------------
# structural channels
# ---------------------------------------------------------------------------

def _cone(v: Any, producers: Dict[Any, Any], leaf_of: Dict[Any, Any]) -> Tuple[List[str], List[Any]]:
    """Primitives and weight leaves upstream of ``v`` (stopping at leaves)."""
    from braintrace._compatible_imports import Var
    prims, leaves, seen, stack = set(), set(), set(), [v]
    while stack:
        w = stack.pop()
        if not isinstance(w, Var) or w in seen:
            continue
        seen.add(w)
        if w in leaf_of:
            leaves.add(leaf_of[w])
            continue
        eqn = producers.get(w)
        if eqn is None:
            continue
        prims.add(eqn.primitive.name)
        stack.extend(eqn.invars)
    return sorted(prims), sorted(leaves, key=str)


def _is_float(v: Any) -> bool:
    dtype = getattr(getattr(v, 'aval', None), 'dtype', None)
    return dtype is not None and np.issubdtype(dtype, np.inexact)


def _reach_new_hidden(start: Any, consumers: Dict[Any, Any], outvar_path: Dict[Any, Any]) -> set:
    """Paths of the hidden outvars reachable from ``start`` in-step.

    The walk follows float values only and stops at ETP ops: a path through a
    second ETP op is cut from the relation (it is not in its ``y -> hidden``
    map), so it cannot be counted twice.
    """
    from braintrace._compatible_imports import Var
    from braintrace._op._registries import is_etp_primitive
    found, seen, stack = set(), {start}, [start]
    while stack:
        w = stack.pop()
        for eqn in consumers.get(w, ()):
            if is_etp_primitive(eqn.primitive):
                continue
            for o in eqn.outvars:
                if not isinstance(o, Var) or o in seen or not _is_float(o):
                    continue
                seen.add(o)
                if o in outvar_path:
                    found.add(outvar_path[o])
                stack.append(o)
    return found


def _new_reads(graph: Any) -> List[str]:
    """``relation: g -> V``: the relation's output reaches the new ``g``, the
    new ``g`` reaches the new ``V`` in the same step without an ETP op between
    them, and ``V`` is one of the relation's hidden paths (its trace carries
    ``V``). Under ``vjp_method='single-step'`` the term ``W -> g -> V`` then
    counts twice (#239). A site is exposure: it is recorded for every graph,
    multi-step ones included, where the double count does not occur."""
    from braintrace._compatible_imports import Var
    from braintrace._compiler.jaxpr_graph import build_consumer_map
    minfo = graph.module_info
    consumers = build_consumer_map(minfo.jaxpr)
    outvar_path = {v: p for p, v in minfo.hidden_path_to_outvar.items() if isinstance(v, Var)}
    path_outvar = {p: v for v, p in outvar_path.items()}
    sites = []
    for r in graph.hidden_param_op_relations:
        if r.control_flow_context is not None:
            continue  # body-scoped vars: not in the module jaxpr
        reached = _reach_new_hidden(r.y_var, consumers, outvar_path)
        traced = {p for grp in r.hidden_groups for p in grp.hidden_paths}
        name = f'{r.primitive.name}{sorted(map(str, r.trainable_paths.values()))}'
        for g in reached:
            if not _is_float(path_outvar[g]):
                continue
            for v in _reach_new_hidden(path_outvar[g], consumers, outvar_path):
                if v != g and v in reached and v in traced:
                    sites.append(f'{name}: {g} -> {v}')
    return sorted(set(sites))


def structural_sites(graph: Any) -> Dict[str, List[str]]:
    """The CHAIN, MULTISOURCE and NEWREAD sites of a compiled graph.

    Parameters
    ----------
    graph : ETraceGraph
        A compiled graph.

    Returns
    -------
    dict
        ``{channel: [site, ...]}``.
    """
    from braintrace._compiler.jaxpr_graph import build_producer_map
    minfo = graph.module_info
    jaxpr = minfo.jaxpr
    producers = build_producer_map(jaxpr)
    leaf_of = {v: (str(p), i) for p, vs in minfo.weight_path_to_invars.items() for i, v in enumerate(vs)}
    eqn_by_id = {id(e): e for e in jaxpr.eqns}
    out: Dict[str, List[str]] = {'CHAIN': [], 'MULTISOURCE': [], 'NEWREAD': []}
    for r in graph.hidden_param_op_relations:
        ctx = r.control_flow_context
        if ctx is not None and not ctx.lifted:
            body = ctx.scan.body_jaxpr
            prod = build_producer_map(body)
            lo = {}
            scan_eqn = eqn_by_id.get(ctx.scan.scan_eqn_id)
            if scan_eqn is not None:
                for bv, ov in zip(body.invars, scan_eqn.invars):
                    if ov in leaf_of:
                        lo[bv] = leaf_of[ov]
        else:
            prod, lo = producers, leaf_of
        for key, v in r.trainable_vars.items():
            prims, leaves = _cone(v, prod, lo)
            site = f'{r.primitive.name}[{key}]{sorted(map(str, r.trainable_paths.values()))}'
            if v not in lo:
                out['CHAIN'].append(f'{site} via {prims}')
            if len(leaves) > 1:
                out['MULTISOURCE'].append(f'{site} from {len(leaves)} leaves')
    out['NEWREAD'] = _new_reads(graph)
    return out


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

def install(census: Census) -> Callable[[], None]:
    """Patch the measurement points; return a function that restores them.

    Patched: ``ETraceGraph.__new__`` (structural channels),
    ``HiddenGroup.diagonal_jacobian`` (DIAG) and
    ``ETraceVjpGraphExecutor._compute_hid2weight_jacobian`` (TAIL). Each
    wrapper returns the original result unchanged.
    """
    import jax
    import brainunit as u
    from braintrace._algorithm import vjp_graph_executor as executor_module
    from braintrace._compiler import graph as graph_module
    from braintrace._compiler import hidden_group as hidden_group_module
    from braintrace._misc import etrace_df_key

    restore: List[Tuple[Any, str, Any]] = []

    def patch(owner: Any, name: str, new: Any) -> None:
        restore.append((owner, name, owner.__dict__[name]))
        setattr(owner, name, new)

    def uninstall() -> None:
        for owner, name, value in reversed(restore):
            setattr(owner, name, value)
        restore.clear()

    try:
        graph_cls = graph_module.ETraceGraph
        orig_new = graph_cls.__new__

        def new(cls: Any, *args: Any, **kwargs: Any) -> Any:
            obj = orig_new(cls, *args, **kwargs)
            try:
                for channel, sites in structural_sites(obj).items():
                    for site in sites:
                        census.record(channel, site, 1.0)
            except Exception as e:  # noqa: BLE001 -- an observer never raises
                census.skip('STRUCT', 'graph', f'{type(e).__name__}: {e}')
            return obj

        patch(graph_cls, '__new__', new)

        group_cls = hidden_group_module.HiddenGroup
        orig_diag = group_cls.__dict__['diagonal_jacobian']

        def diagonal_jacobian(self: Any, hidden_vals: Any, input_vals: Any, *args: Any, **kwargs: Any) -> Any:
            out = orig_diag(self, hidden_vals, input_vals, *args, **kwargs)
            if self.snap is None and self.is_diagonal_recurrence:
                site = _group_site(self)
                try:
                    concat = self.concat_hidden(hidden_vals)
                    size = int(np.prod(concat.shape))
                    if size > MAX_FULL:
                        census.skip('DIAG', site, f'size {size} > {MAX_FULL}')
                    else:
                        def fn(hid: Any) -> Any:
                            return self.concat_hidden(self.transition(self.split_hidden(hid), input_vals))

                        true = hidden_group_module.block_diagonal_last_dim(
                            fn, concat,
                            use_forward_mode=hidden_group_module._transition_contains_while(self.transition_jaxpr))
                        jax.debug.callback(_callback(census, 'DIAG', site),
                                           u.get_mantissa(out) - u.get_mantissa(true), u.get_mantissa(true))
                except Exception as e:  # noqa: BLE001
                    census.skip('DIAG', site, f'{type(e).__name__}: {e}')
            return out

        patch(group_cls, 'diagonal_jacobian', diagonal_jacobian)

        executor_cls = executor_module.ETraceVjpGraphExecutor
        orig_h2w = executor_cls.__dict__['_compute_hid2weight_jacobian']

        def compute_hid2weight_jacobian(self: Any, intermediate_values: Any) -> Any:
            xs, dfs = orig_h2w(self, intermediate_values)
            for relation in self.graph.hidden_param_op_relations:
                if relation.control_flow_context is not None:
                    continue
                y = intermediate_values[relation.y_var]
                for gi, group in enumerate(relation.hidden_groups):
                    site = f'{relation.primitive.name}{sorted(map(str, relation.trainable_paths.values()))}->{_group_site(group)}'
                    try:
                        df = u.get_mantissa(dfs[etrace_df_key(relation.y_var, group.index)])
                        ym = u.get_mantissa(y)
                        v = _tangent(ym.shape, ym.dtype)
                        v_in = u.Quantity(v, unit=u.get_unit(y)) if u.get_unit(y) != u.get_unit(1.0) else v

                        def y2h(y_val: Any, _r: Any = relation, _gi: int = gi) -> Any:
                            return _r.y_to_hidden_groups(y_val, intermediate_values, concat_hidden_vals=True)[_gi]

                        _, actual = jax.jvp(y2h, (y,), (v_in,))
                        actual = u.get_mantissa(actual)
                        varshape = tuple(group.varshape)
                        vm = v
                        while vm.ndim > len(varshape) and vm.shape[0] == 1:
                            vm = vm[0]
                        predicted = df * jax.numpy.broadcast_to(vm, varshape)[..., None]
                        jax.debug.callback(_callback(census, 'TAIL', site), actual - predicted, actual)
                    except Exception as e:  # noqa: BLE001
                        census.skip('TAIL', site, f'{type(e).__name__}: {e}')
            return xs, dfs

        patch(executor_cls, '_compute_hid2weight_jacobian', compute_hid2weight_jacobian)
    except BaseException:
        uninstall()  # a failed install leaves nothing patched
        raise
    return uninstall


@contextlib.contextmanager
def measuring(test: str = 'test') -> Iterator[Census]:
    """Measure every channel within a ``with`` block.

    Examples
    --------
    .. code-block:: python

        >>> from braintrace._testing import census
        >>> with census.measuring() as c:
        ...     pass  # compile and run a learner here
        >>> c.max('DIAG')
        0.0
    """
    import jax
    c = Census()
    c.current = test
    uninstall = install(c)
    try:
        yield c
        jax.effects_barrier()  # flush pending debug callbacks
    finally:
        uninstall()


# ---------------------------------------------------------------------------
# pytest plugin (active only when loaded with ``-p braintrace._testing.census``)
# ---------------------------------------------------------------------------

_CENSUS = Census()

#: The plugin's install, keyed by ``id()`` of the pytest config that made it
#: (see :data:`braintrace._testing.graph_hash._INSTALLS`).
_INSTALLS: Dict[int, Callable[[], None]] = {}


def pytest_configure(config: Any) -> None:
    if _INSTALLS:
        return  # already measuring: patching twice would measure twice
    if not os.environ.get('PYTEST_XDIST_WORKER'):
        _remove_stale_files(os.environ.get(OUT_ENV))
    _INSTALLS[id(config)] = install(_CENSUS)


def pytest_runtest_setup(item: Any) -> None:
    _CENSUS.current = item.nodeid


def pytest_sessionfinish(session: Any) -> None:
    out = os.environ.get(OUT_ENV)
    if not out:
        return
    try:
        import jax
        jax.effects_barrier()
    except Exception:  # noqa: BLE001
        pass
    worker = os.environ.get('PYTEST_XDIST_WORKER', 'main')
    with open(f'{out}.{worker}.json', 'w') as f:
        json.dump(_CENSUS.as_dict(), f)


def pytest_unconfigure(config: Any) -> None:
    uninstall = _INSTALLS.pop(id(config), None)
    if uninstall is not None:
        uninstall()


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def load(prefix: str) -> Dict[str, Any]:
    """Merge the per-process files ``<prefix>.main.json`` and ``<prefix>.gw<N>.json``."""
    files = _worker_files(prefix)
    if not files:
        raise FileNotFoundError(f'no {prefix}.main.json or {prefix}.gw<N>.json files')
    merged: Dict[str, Any] = {'values': {}, 'skipped': {}}
    for path in files:
        with open(path) as f:
            data = json.load(f)
        merged['values'].update(data.get('values', {}))
        merged['skipped'].update(data.get('skipped', {}))
    return merged


def summarize(data: Dict[str, Any], threshold: float = 1e-4) -> Dict[str, Dict[str, Any]]:
    """Per channel: the tests with a site above ``threshold``, and the skipped sites.

    Returns
    -------
    dict
        ``{channel: {'tests': {test: max value}, 'sites': {site: max value}, 'skipped': n}}``.
    """
    out: Dict[str, Dict[str, Any]] = {c: {'tests': {}, 'sites': {}, 'skipped': 0} for c in CHANNELS}
    for test, vals in data['values'].items():
        for key, value in vals.items():
            channel, site = key.split('|', 1)
            if channel not in out or value <= threshold:
                continue
            entry = out[channel]
            entry['tests'][test] = max(entry['tests'].get(test, 0.0), value)
            entry['sites'][site] = max(entry['sites'].get(site, 0.0), value)
    for skipped in data['skipped'].values():
        for key in skipped:
            channel = key.split('|', 1)[0]
            if channel in out:
                out[channel]['skipped'] += 1
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m braintrace._testing.census summarize PREFIX``."""
    parser = argparse.ArgumentParser(prog='python -m braintrace._testing.census')
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('summarize', help='summarize a recorded run')
    s.add_argument('prefix')
    s.add_argument('--threshold', type=float, default=1e-4)
    s.add_argument('--show', type=int, default=20, help='tests listed per channel')
    args = parser.parse_args(argv)

    data = load(args.prefix)
    report = summarize(data, args.threshold)
    print(f'{len(data["values"])} tests measured; threshold {args.threshold:g}')
    for channel in CHANNELS:
        r = report[channel]
        print(f'{channel}: {len(r["tests"])} tests, {len(r["sites"])} sites above threshold; '
              f'{r["skipped"]} sites skipped')
        for test, value in sorted(r['tests'].items(), key=lambda kv: -kv[1])[:args.show]:
            print(f'  {value:.3g}  {test}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
