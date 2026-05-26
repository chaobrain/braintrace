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

"""L2 compiler regression baseline (spec section 5.4): the relation enumeration
for every public RNN cell is pinned to a known profile — the number of
hidden->param->op relations, the set of included trainable weights, and the set
of weights excluded by the no-W->W->h invariant. A meta-test ties this matrix to
the cell registry so a newly added cell must add a guardrail row.

Ground truth verified by compiling each cell (cls(3, 4), unbatched input) on
2026-05-27. Sections 5.1-5.3 are covered elsewhere (compiler_oracle_test.py,
compiler_property_test.py, graph_test.py) and are not duplicated here.
"""

import warnings

import brainstate
import pytest

import braintrace
from braintrace._etrace_compiler.diagnostics import DiagnosticKind

# cell name -> (relation count, included weight-path set, excluded W->W->h set)
_CELL_GUARDRAILS = {
    'ValinaRNNCell': (1, {('W', 'weight')}, set()),
    'GRUCell': (2, {('Wz', 'weight'), ('Wh', 'weight')}, {('Wr', 'weight')}),
    'MGUCell': (2, {('Wf', 'weight'), ('Wh', 'weight')}, set()),
    'LSTMCell': (4, {('Wf', 'weight'), ('Wg', 'weight'), ('Wi', 'weight'), ('Wo', 'weight')}, set()),
    'URLSTMCell': (6, {('Wf', 'weight'), ('Wo', 'weight'), ('Wr', 'weight'), ('Wu', 'weight'), ('bias',)}, set()),
    'MinimalRNNCell': (2, {('W_u', 'weight'), ('phi', 'weight')}, set()),
    'MiniGRU': (2, {('W_x', 'weight'), ('W_z', 'weight')}, set()),
    'MiniLSTM': (3, {('W_f', 'weight'), ('W_i', 'weight'), ('W_x', 'weight')}, set()),
    'LRUCell': (5, {('B_im', 'weight'), ('B_re', 'weight'), ('gamma_log',), ('nu_log',), ('theta_log',)}, set()),
}


def _compile_cell(name, n_in=3, n_out=4):
    cls = getattr(braintrace.nn, name)
    cell = cls(n_in, n_out)
    brainstate.nn.init_all_states(cell)
    inp = brainstate.random.rand(n_in)
    with warnings.catch_warnings():
        # GRUCell legitimately warns when it excludes Wr (W->W->h); the guardrail
        # checks the diagnostic records, not the warning.
        warnings.simplefilter('ignore')
        return braintrace.compile_etrace_graph(cell, inp, include_hidden_perturb=False)


@pytest.mark.parametrize('cell_name', list(_CELL_GUARDRAILS))
def test_cell_relation_profile(cell_name):
    """Each public RNN cell compiles to its pinned relation profile. A change in
    count / included / excluded sets is a deliberate compiler-behavior change and
    must update this matrix knowingly."""
    expected_count, expected_incl, expected_excl = _CELL_GUARDRAILS[cell_name]
    graph = _compile_cell(cell_name)

    assert len(graph.hidden_param_op_relations) == expected_count

    included = {r.weight_path for r in graph.explain(kind=DiagnosticKind.RELATION_INCLUDED)}
    excluded = {r.weight_path for r in graph.explain(
        kind=DiagnosticKind.RELATION_EXCLUDED_WEIGHT_TO_WEIGHT)}
    assert included == expected_incl
    assert excluded == expected_excl


# --- Task 2: the guardrail matrix must cover every public RNN cell -----------

def test_every_public_rnn_cell_has_a_guardrail():
    """The guardrail matrix is tied to the cell registry: every cell exported by
    braintrace.nn._rnn must have a pinned profile. Adding a new cell to the
    registry fails this test until a guardrail row is added — the regression hook
    the test strategy requires (spec section 11)."""
    from braintrace.nn import _rnn

    registry = set(_rnn.__all__)
    pinned = set(_CELL_GUARDRAILS)
    missing = registry - pinned
    extra = pinned - registry
    assert not missing, f"public RNN cells without a relation guardrail: {sorted(missing)}"
    assert not extra, f"guardrail rows for non-existent/removed cells: {sorted(extra)}"
