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
import warnings

import brainstate
import jax.numpy as jnp
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

        monkeypatch.setattr(gh, '_RECORDER', gh.Recorder())
        gh.pytest_configure(None)
        try:
            assert ETraceGraph.__dict__['__new__'] is not before
            gh.pytest_runtest_setup(Item())
            assert gh._RECORDER.current == 'n'
        finally:
            gh.pytest_unconfigure(None)
        assert ETraceGraph.__dict__['__new__'] is before
