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

"""Tests for the graph-hash gate (``braintrace._testing.graph_hash``)."""

import json
import os
import subprocess
import sys
import textwrap
import warnings

import brainstate
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._algorithm.param_dim_vjp import ParamDimVjpAlgorithm
from braintrace._algorithm.vjp_base import ETraceVjpAlgorithm
from braintrace._compiler.graph import ETraceGraph
from braintrace._testing import graph_hash as gh
from braintrace._testing import precondition_models as pm


def _case(name):
    return next(c for c in pm.CASES if c.name == name)


def _compile(name, algorithm='d_rtrl', **options):
    case = _case(name)
    model = case.model()
    xs = case.inputs()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        learner = braintrace.compile(model, algorithm, xs[0], **options)
    return learner, model, xs


def _run(name, steps=2, **kw):
    learner, model, xs = _compile(name, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pm.online_gradients(learner, model, xs[:steps])
    return learner


class TestGraphSummary:

    def test_same_model_same_digest_across_compiles(self):
        a, _, _ = _compile('tail-roll')
        b, _, _ = _compile('tail-roll')
        assert a.graph is not b.graph
        assert gh.graph_digest(a.graph) == gh.graph_digest(b.graph)

    def test_different_model_different_digest(self):
        a, _, _ = _compile('tail-roll')
        b, _, _ = _compile('tail-reverse')
        assert gh.graph_digest(a.graph) != gh.graph_digest(b.graph)

    def test_summary_names_no_object_id(self):
        learner, _, _ = _compile('newread-new')
        text = gh.graph_summary(learner.graph)
        assert 'id=' not in text
        assert str(id(learner.graph.hidden_param_op_relations[0].y_var)) not in text

    def test_summary_covers_groups_relations_perturbation_and_diagnostics(self):
        learner, _, _ = _compile('drop-partial')
        text = gh.graph_summary(learner.graph)
        assert 'diagonal=True' in text
        assert 'Retp_mm' in text
        assert 'perturb paths=' in text
        assert "('relation_partial_path', 'WARNING')" in text

    def test_summary_with_snap_pattern(self):
        learner, _, _ = _compile('transition-roll', algorithm=braintrace.SnAp, n=2)
        assert any(g.snap is not None for g in learner.graph.hidden_groups)
        assert 'snap(n=2' in gh.graph_summary(learner.graph)

    def test_digest_of_a_broken_graph_is_an_error_marker(self):
        assert gh.graph_digest(object()).startswith('ERR ')

    def test_digest_sees_the_value_of_a_constant_array(self):
        def make(mask):
            class Net(brainstate.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.p = brainstate.ParamState(jnp.ones((3, 4)) * 0.1)
                    self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

                def update(self, x):
                    self.h.value = 0.9 * self.h.value * jnp.asarray(mask) + braintrace.matmul(x, self.p.value)
                    return self.h.value

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return braintrace.compile(Net(), 'd_rtrl', jnp.ones((1, 3))).graph

        a = make(np.array([[1., 0., 1., 1.]]))
        b = make(np.array([[1., 1., 0., 1.]]))
        assert gh._consts(a.module_info.closed_jaxpr)  # the mask is a constant of the module jaxpr
        assert gh.graph_digest(a) != gh.graph_digest(b)
        assert gh.graph_digest(a) == gh.graph_digest(make(np.array([[1., 0., 1., 1.]])))

    def test_consts_of_nested_jaxprs_are_hashed(self):
        import jax
        c1, c2 = np.arange(3.) + 0.5, np.arange(3.) + 1.5
        nested = [jax.make_jaxpr(lambda x, c=c: jax.jit(lambda y: y * c)(x))(jnp.ones(3)) for c in (c1, c2)]
        assert all(any(d.startswith('float32(3,)') for d in gh._consts(j)) for j in nested)
        assert gh._consts(nested[0]) != gh._consts(nested[1])

    def test_consts_of_an_old_closed_jaxpr_and_non_array_constants(self):
        """JAX < 0.11: a ClosedJaxpr has no ``eqns``; its equations sit on
        ``.jaxpr``. A constant that is not an array is named by its type, never
        hashed by its address."""
        import jax
        new = jax.make_jaxpr(lambda x: jax.jit(lambda y: y * (np.arange(3.) + 0.5))(x))(jnp.ones(3))

        class Unconvertible:
            def __array__(self, *args, **kwargs):
                raise TypeError('no array')

        class Old:
            consts = (object(), Unconvertible())
            jaxpr = new

        assert gh._consts(new)
        assert gh._consts(Old()) == ['object', 'Unconvertible'] + gh._consts(new)


class TestTraceLabels:

    def test_keys_are_labelled_by_relation_not_by_id(self):
        learner, _, _ = _compile('drop-partial')  # two ETP ops, weights w and w2
        labels = gh.trace_labels(learner, learner._get_etrace_data(), 'init')
        assert all('<id>' not in l for l in labels)
        assert any('y:etp_mm["(\'w\',)"]' in l for l in labels), labels
        assert any('y:etp_mm["(\'w2\',)"]' in l for l in labels), labels

    def test_unlabelled_ids_are_masked(self):
        big = 1 << 40
        assert gh._key(big, {}) == '<id>'
        assert gh._key((big, 3), {big: 'y:op'}) == '(y:op,3)'
        assert gh._key(7, {}) == '7'
        assert gh._key(True, {}) == 'True'

    def test_algorithm_without_graph(self):
        class NoGraph:
            pass

        labels = gh.trace_labels(NoGraph(), {'a': jnp.zeros((2, 3))}, 'init')
        assert labels == ["NoGraph|init|a|(2, 3):float32"]


class TestRecording:

    def test_records_graphs_traces_and_runtime_records(self):
        with gh.recording('t') as rec:
            _run('credit-4-substeps')
        data = rec.as_dict()
        assert len(data['graphs']['t']) == 1
        stages = {t.split('|')[1] for t in data['traces']['t']}
        assert stages == {'init', 'update'}
        assert any(r.startswith('online_gradient_zero:') for r in data['records']['t'])

    def test_init_traces_are_recorded_once_per_outer_call(self, monkeypatch):
        with gh.recording('t') as rec:
            learner, _, _ = _compile('weight-raw')
            n = len(rec.traces['t'])
            learner.init_etrace_state()
        assert len(rec.traces['t']) == n  # the same labels again: a set, no growth
        assert all('ERR' not in t for t in rec.traces['t'])

    def test_io_dim_algorithm_traces(self):
        with gh.recording('t') as rec:
            _run('weight-raw', algorithm='es_d_rtrl', decay_or_rank=0.9)
        assert any(t.startswith('pp_prop|update|') for t in rec.traces['t']), rec.traces['t']

    def test_failing_trace_read_is_recorded_not_raised(self, monkeypatch):
        def broken(self):
            raise RuntimeError('boom')

        with gh.recording('t') as rec:
            monkeypatch.setattr(ParamDimVjpAlgorithm, '_get_etrace_data', broken)
            _compile('weight-raw')
        assert any('init|ERR RuntimeError' in t for t in rec.traces['t'])

    def test_uninstall_restores_every_patch(self):
        before = (ETraceGraph.__dict__['__new__'], ETraceVjpAlgorithm.__dict__['_add_records'],
                  ParamDimVjpAlgorithm.__dict__['init_etrace_state'],
                  ParamDimVjpAlgorithm.__dict__['_update_etrace_data'])
        with gh.recording():
            assert ETraceVjpAlgorithm.__dict__['_add_records'] is not before[1]
        after = (ETraceGraph.__dict__['__new__'], ETraceVjpAlgorithm.__dict__['_add_records'],
                 ParamDimVjpAlgorithm.__dict__['init_etrace_state'],
                 ParamDimVjpAlgorithm.__dict__['_update_etrace_data'])
        assert after == before

    def test_records_the_environment_of_every_graph(self):
        with gh.recording('t') as rec:
            _compile('weight-raw')
            with brainstate.environ.context(precision=64):
                _compile('weight-raw')
        assert rec.as_dict()['env']['t'] == ['x64=False precision=32', 'x64=True precision=64']

    def test_a_failed_install_leaves_nothing_patched(self, monkeypatch):
        before = ETraceGraph.__dict__['__new__']

        def broken():
            raise RuntimeError('boom')

        monkeypatch.setattr(gh, '_algorithm_classes', broken)
        with pytest.raises(RuntimeError, match='boom'):
            gh.install(gh.Recorder())
        assert ETraceGraph.__dict__['__new__'] is before

    def test_recording_does_not_change_the_gradient(self):
        case = _case('newread-new')
        plain = pm.evaluate(case)
        with gh.recording():
            observed = pm.evaluate(case)
        assert observed.errors == plain.errors


class TestCompare:

    def _write(self, tmp_path, name, worker, data):
        with open(tmp_path / f'{name}.{worker}.json', 'w') as f:
            json.dump(data, f)

    def test_load_merges_workers_and_compare_reports(self, tmp_path):
        self._write(tmp_path, 'base', 'gw0', {'graphs': {'a': ['1'], 'b': ['2']}, 'traces': {'a': ['x']}})
        self._write(tmp_path, 'base', 'gw1', {'graphs': {'c': ['3']}, 'records': {}})
        self._write(tmp_path, 'head', 'main', {'graphs': {'a': ['1'], 'b': ['9'], 'd': ['4']},
                                               'traces': {'a': ['x']}})
        base, head = gh.load(str(tmp_path / 'base')), gh.load(str(tmp_path / 'head'))
        assert set(base['graphs']) == {'a', 'b', 'c'}
        report = gh.compare(base, head)
        assert report['graphs'] == {'changed': ['b'], 'only_base': ['c'], 'only_head': ['d']}
        assert report['traces']['changed'] == []

    def test_main_exit_status(self, tmp_path, capsys):
        self._write(tmp_path, 'base', 'main', {'graphs': {'a': ['1']}})
        self._write(tmp_path, 'same', 'main', {'graphs': {'a': ['1']}})
        self._write(tmp_path, 'diff', 'main', {'graphs': {'a': ['2']}})
        assert gh.main(['compare', str(tmp_path / 'base'), str(tmp_path / 'same')]) == 0
        assert gh.main(['compare', str(tmp_path / 'base'), str(tmp_path / 'diff'), '--show', '0']) == 1
        out = capsys.readouterr().out
        assert 'graphs: 1 tests' in out and 'changed 1' in out and '... 1 more' in out

    def test_main_fails_when_tests_vanish(self, tmp_path, capsys):
        self._write(tmp_path, 'base', 'main', {'graphs': {'a': ['1'], 'b': ['2']}})
        self._write(tmp_path, 'head', 'main', {'graphs': {'a': ['1']}})
        assert gh.main(['compare', str(tmp_path / 'base'), str(tmp_path / 'head')]) == 1
        assert 'missing on the head side' in capsys.readouterr().out
        assert gh.main(['compare', str(tmp_path / 'base'), str(tmp_path / 'head'), '--allow-removed']) == 0
        # a test added on the head side is not a failure
        assert gh.main(['compare', str(tmp_path / 'head'), str(tmp_path / 'base')]) == 0

    def test_compare_lists_environment_changes(self, tmp_path, capsys):
        self._write(tmp_path, 'base', 'main', {'graphs': {'a': ['1']}, 'env': {'a': ['x64=False precision=32']}})
        self._write(tmp_path, 'head', 'main', {'graphs': {'a': ['2']}, 'env': {'a': ['x64=True precision=64']}})
        report = gh.compare(gh.load(str(tmp_path / 'base')), gh.load(str(tmp_path / 'head')))
        assert report['env'] == {'changed': ['a']}
        assert gh.main(['compare', str(tmp_path / 'base'), str(tmp_path / 'head')]) == 1
        out = capsys.readouterr().out
        assert 'env: 1 tests' in out and "precision=32'] -> ['x64=True" in out

    def test_load_reads_only_this_run_s_worker_files(self, tmp_path):
        self._write(tmp_path, 'run', 'gw0', {'graphs': {'a': ['1']}})
        self._write(tmp_path, 'run', 'main', {'graphs': {'b': ['2']}})
        self._write(tmp_path, 'run.rerun', 'gw0', {'graphs': {'a': ['stale']}})  # another prefix
        self._write(tmp_path, 'run', 'gwx', {'graphs': {'c': ['3']}})            # not a worker name
        assert gh.load(str(tmp_path / 'run'))['graphs'] == {'a': ['1'], 'b': ['2']}

    def test_load_without_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            gh.load(str(tmp_path / 'missing'))


class TestPlugin:

    def test_session_writes_one_file_per_worker(self, tmp_path, monkeypatch):
        monkeypatch.setenv(gh.OUT_ENV, str(tmp_path / 'run'))
        monkeypatch.setenv('PYTEST_XDIST_WORKER', 'gw3')
        rec = gh.Recorder()
        rec.current = 'x'
        rec.add_traces(['a'])
        monkeypatch.setattr(gh, '_RECORDER', rec)
        gh.pytest_sessionfinish(None)
        assert gh.load(str(tmp_path / 'run'))['traces'] == {'x': ['a']}

    def test_session_without_output_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.delenv(gh.OUT_ENV, raising=False)
        monkeypatch.chdir(tmp_path)
        gh.pytest_sessionfinish(None)
        assert list(tmp_path.iterdir()) == []

    def test_configure_and_unconfigure_pair(self, monkeypatch):
        before = ETraceGraph.__dict__['__new__']

        class Item:
            nodeid = 'n'

        config = object()
        monkeypatch.setattr(gh, '_RECORDER', gh.Recorder())
        monkeypatch.setattr(gh, '_INSTALLS', {})  # leave a session install (``-p``) alone
        gh.pytest_configure(config)
        try:
            assert ETraceGraph.__dict__['__new__'] is not before
            gh.pytest_runtest_setup(Item())
            assert gh._RECORDER.current == 'n'
        finally:
            gh.pytest_unconfigure(config)
        assert ETraceGraph.__dict__['__new__'] is before

    def test_unconfigure_undoes_only_its_own_install(self, monkeypatch):
        before = ETraceGraph.__dict__['__new__']
        monkeypatch.setattr(gh, '_INSTALLS', {})
        session, nested = object(), object()
        gh.pytest_configure(session)
        try:
            patched = ETraceGraph.__dict__['__new__']
            gh.pytest_configure(nested)  # a second configure installs nothing ...
            assert ETraceGraph.__dict__['__new__'] is patched
            gh.pytest_unconfigure(nested)  # ... and its unconfigure removes nothing
            assert ETraceGraph.__dict__['__new__'] is patched
        finally:
            gh.pytest_unconfigure(session)
        assert ETraceGraph.__dict__['__new__'] is before

    def test_configure_removes_stale_files_once(self, tmp_path, monkeypatch):
        prefix = str(tmp_path / 'run')
        for name in ('run.gw0.json', 'run.main.json', 'other.gw0.json'):
            (tmp_path / name).write_text('{}')
        monkeypatch.setattr(gh, '_INSTALLS', {})
        monkeypatch.setenv(gh.OUT_ENV, prefix)
        monkeypatch.setenv('PYTEST_XDIST_WORKER', 'gw1')  # a worker never cleans
        config = object()
        gh.pytest_configure(config)
        gh.pytest_unconfigure(config)
        assert sorted(p.name for p in tmp_path.iterdir()) == ['other.gw0.json', 'run.gw0.json', 'run.main.json']
        monkeypatch.delenv('PYTEST_XDIST_WORKER')  # the controller (or a plain run) does
        gh.pytest_configure(config)
        gh.pytest_unconfigure(config)
        assert sorted(p.name for p in tmp_path.iterdir()) == ['other.gw0.json']


_PROBE = """
import warnings
import braintrace
from braintrace._testing import precondition_models as pm


def test_probe_after_the_self_tests():
    case = next(c for c in pm.CASES if c.name == 'newread-new')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        braintrace.compile(case.model(), 'd_rtrl', case.inputs()[0])
"""


def run_after_self_tests(tmp_path, plugin, self_test, env_name):
    """Run ``self_test`` and then a probe test in a fresh pytest session that
    loads ``plugin`` with ``-p``; return the output prefix."""
    probe = tmp_path / 'probe_test.py'
    probe.write_text(textwrap.dedent(_PROBE))
    root = os.path.dirname(os.path.dirname(os.path.abspath(braintrace.__file__)))
    prefix = str(tmp_path / 'run')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([root, os.environ.get('PYTHONPATH', '')]))
    env[env_name] = prefix
    env.pop('PYTEST_XDIST_WORKER', None)
    cmd = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-p', 'no:xdist',
           '-p', plugin, '--rootdir', root, self_test, str(probe)]
    done = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    return prefix


def test_recording_continues_after_the_plugin_self_tests(tmp_path):
    """H1 (PR #242 review): the self-tests call the plugin hooks; the session's
    ``-p`` install must survive them, so the probe that runs after is recorded."""
    self_test = os.path.abspath(__file__) + '::TestPlugin'
    prefix = run_after_self_tests(tmp_path, 'braintrace._testing.graph_hash', self_test, gh.OUT_ENV)
    graphs = gh.load(prefix)['graphs']
    assert any(t.endswith('::test_probe_after_the_self_tests') for t in graphs), sorted(graphs)
