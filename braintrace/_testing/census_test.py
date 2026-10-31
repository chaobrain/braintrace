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

"""Tests for the precondition census (``braintrace._testing.census``)."""

import json
import warnings

import brainstate
import pytest

import braintrace
from braintrace._algorithm.vjp_graph_executor import ETraceVjpGraphExecutor
from braintrace._compiler.graph import ETraceGraph
from braintrace._compiler.hidden_group import HiddenGroup
from braintrace._testing import census
from braintrace._testing import precondition_models as pm


def _case(name):
    return next(c for c in pm.CASES if c.name == name)


def _measure(name, steps=2):
    """Compile ``name`` and run ``steps`` steps in float64 while measuring."""
    case = _case(name)
    with brainstate.environ.context(precision=64), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with census.measuring(name) as c:
            model = case.model()
            xs = case.inputs()
            learner = braintrace.compile(model, 'd_rtrl', xs[0])
            pm.online_gradients(learner, model, xs[:steps])
    return c


class TestRuntimeChannels:

    @pytest.mark.parametrize('name', ['transition-roll', 'transition-mean', 'transition-reverse'])
    def test_diag_flags_a_coupled_transition(self, name):
        assert _measure(name).max('DIAG') > 0.1

    def test_diag_is_zero_on_a_diagonal_transition(self):
        c = _measure('transition-none')
        assert any(k.startswith('DIAG|') for k in c.values['transition-none'])
        assert c.max('DIAG') < 1e-12

    @pytest.mark.parametrize('name', ['tail-roll', 'tail-reverse', 'tail-transpose', 'tail-softmax'])
    def test_tail_flags_a_non_elementwise_tail(self, name):
        assert _measure(name).max('TAIL') > 0.1

    def test_tail_is_zero_on_an_elementwise_tail(self):
        c = _measure('tail-elementwise')
        assert any(k.startswith('TAIL|') for k in c.values['tail-elementwise'])
        assert c.max('TAIL') < 1e-12

    def test_large_groups_are_skipped(self, monkeypatch):
        monkeypatch.setattr(census, 'MAX_FULL', 0)
        c = _measure('transition-roll')
        assert c.max('DIAG') == 0.0
        assert any(k.startswith('DIAG|') and 'size' in v for k, v in c.skipped['transition-roll'].items())

    def test_failures_inside_a_check_are_skipped_not_raised(self, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError('boom')

        monkeypatch.setattr(census, '_tangent', broken)
        monkeypatch.setattr(braintrace._compiler.hidden_group, 'block_diagonal_last_dim', broken)
        c = _measure('transition-roll')
        reasons = list(c.skipped['transition-roll'].values())
        assert sum('RuntimeError: boom' in r for r in reasons) == 2

    def test_measuring_does_not_change_the_gradient(self):
        case = _case('tail-roll')
        with brainstate.environ.context(precision=64), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            xs = case.inputs()
            grads = []
            for observe in (False, True):
                model = case.model()
                learner = braintrace.compile(model, 'd_rtrl', xs[0])
                if observe:
                    with census.measuring():
                        grads.append(pm.online_gradients(learner, model, xs))
                else:
                    grads.append(pm.online_gradients(learner, model, xs))
        assert pm.relative_errors(grads[1], grads[0]) == {"[('w',)]": 0.0}


class TestStructuralChannels:

    @pytest.mark.parametrize('name,channel', [
        ('weight-2p', 'CHAIN'),
        ('weight-mask', 'CHAIN'),
        ('source-two-params', 'MULTISOURCE'),
        ('source-two-leaves', 'MULTISOURCE'),
        ('newread-new', 'NEWREAD'),
    ])
    def test_flags_the_violation(self, name, channel):
        c = _measure(name, steps=1)
        assert any(k.startswith(channel + '|') for k in c.values[name])

    @pytest.mark.parametrize('name', ['weight-raw', 'weight-mask-as-weight-fn', 'source-one', 'newread-old'])
    def test_controls_are_clean(self, name):
        c = _measure(name, steps=1)
        flagged = [k for k in c.values.get(name, {}) if k.split('|')[0] in ('CHAIN', 'MULTISOURCE', 'NEWREAD')]
        assert flagged == []

    def test_newread_names_both_states(self):
        c = _measure('newread-new', steps=1)
        assert "NEWREAD|('g',) -> ('V',)" in c.values['newread-new']

    def test_descended_relation_is_read_through_the_scan_body(self):
        c = _measure('credit-20-substeps', steps=1)
        flagged = [k for k in c.values.get('credit-20-substeps', {}) if k.startswith(('CHAIN|', 'MULTISOURCE|'))]
        assert flagged == []

    def test_structural_failure_is_skipped_not_raised(self, monkeypatch):
        def broken(graph):
            raise RuntimeError('boom')

        monkeypatch.setattr(census, 'structural_sites', broken)
        c = _measure('weight-raw', steps=1)
        assert 'RuntimeError: boom' in c.skipped['weight-raw']['STRUCT|graph']


class TestInstall:

    def test_uninstall_restores_every_patch(self):
        before = (ETraceGraph.__dict__['__new__'], HiddenGroup.__dict__['diagonal_jacobian'],
                  ETraceVjpGraphExecutor.__dict__['_compute_hid2weight_jacobian'])
        with census.measuring():
            assert HiddenGroup.__dict__['diagonal_jacobian'] is not before[1]
        after = (ETraceGraph.__dict__['__new__'], HiddenGroup.__dict__['diagonal_jacobian'],
                 ETraceVjpGraphExecutor.__dict__['_compute_hid2weight_jacobian'])
        assert after == before

    def test_relative_and_record(self):
        assert census._relative([0.5], [2.0]) == 0.25
        assert census._relative([0.5], [0.0]) == 0.5
        assert census._relative([], []) == 0.0
        c = census.Census()
        c.record('DIAG', 's', 0.1)
        c.record('DIAG', 's', 0.05)
        assert c.values == {'<no test>': {'DIAG|s': 0.1}}
        assert c.max('TAIL') == 0.0


class TestSummary:

    def _write(self, tmp_path, worker, data):
        with open(tmp_path / f'run.{worker}.json', 'w') as f:
            json.dump(data, f)

    def test_summarize_counts_tests_above_threshold(self, tmp_path):
        self._write(tmp_path, 'gw0', {'values': {'t1': {'DIAG|a': 0.5, 'TAIL|b': 1e-9}},
                                      'skipped': {'t1': {'DIAG|c': 'size'}}})
        self._write(tmp_path, 'gw1', {'values': {'t2': {'DIAG|a': 0.2, 'OTHER|x': 1.0}}, 'skipped': {}})
        data = census.load(str(tmp_path / 'run'))
        report = census.summarize(data, threshold=1e-4)
        assert report['DIAG']['tests'] == {'t1': 0.5, 't2': 0.2}
        assert report['DIAG']['sites'] == {'a': 0.5}
        assert report['DIAG']['skipped'] == 1
        assert report['TAIL']['tests'] == {}

    def test_main_prints_a_summary(self, tmp_path, capsys):
        self._write(tmp_path, 'main', {'values': {'t1': {'NEWREAD|g -> V': 1.0}}, 'skipped': {}})
        assert census.main(['summarize', str(tmp_path / 'run'), '--show', '1']) == 0
        out = capsys.readouterr().out
        assert '1 tests measured' in out and 'NEWREAD: 1 tests, 1 sites' in out and 't1' in out

    def test_load_without_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            census.load(str(tmp_path / 'missing'))


class TestPlugin:

    def test_session_writes_one_file_per_worker(self, tmp_path, monkeypatch):
        monkeypatch.setenv(census.OUT_ENV, str(tmp_path / 'run'))
        monkeypatch.setenv('PYTEST_XDIST_WORKER', 'gw2')
        c = census.Census()
        c.record('DIAG', 's', 0.3, test='x')
        monkeypatch.setattr(census, '_CENSUS', c)
        census.pytest_sessionfinish(None)
        assert census.load(str(tmp_path / 'run'))['values'] == {'x': {'DIAG|s': 0.3}}

    def test_session_without_output_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.delenv(census.OUT_ENV, raising=False)
        monkeypatch.chdir(tmp_path)
        census.pytest_sessionfinish(None)
        assert list(tmp_path.iterdir()) == []

    def test_configure_and_unconfigure_pair(self, monkeypatch):
        before = HiddenGroup.__dict__['diagonal_jacobian']

        class Item:
            nodeid = 'n'

        monkeypatch.setattr(census, '_CENSUS', census.Census())
        census.pytest_configure(None)
        try:
            assert HiddenGroup.__dict__['diagonal_jacobian'] is not before
            census.pytest_runtest_setup(Item())
            assert census._CENSUS.current == 'n'
        finally:
            census.pytest_unconfigure(None)
        assert HiddenGroup.__dict__['diagonal_jacobian'] is before
