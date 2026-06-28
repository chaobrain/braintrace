# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# ==============================================================================

import brainstate
import jax.numpy as jnp
import pytest

import braintrace
from braintrace._compiler import CompilationReport
from braintrace._algorithm.oracle_models import tanh_rnn


def _gru_graph():
    model = tanh_rnn(n_in=3, n_rec=4, seed=0).factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    return braintrace.compile_etrace_graph(model, jnp.ones((3,), 'float32'))


def test_report_basic_view():
    report = CompilationReport(_gru_graph())
    # tanh_rnn => 1 hidden group, 1 etrace weight relation, 0 excluded
    assert report.counts['hidden_groups'] == 1
    assert report.counts['etrace_weights'] >= 1
    assert isinstance(report.hidden_groups, list)
    g_index, g_paths = report.hidden_groups[0]
    assert isinstance(g_index, int)
    assert isinstance(g_paths, list) and len(g_paths) >= 1
    w_path, w_groups = report.etrace_weights[0]
    assert isinstance(w_groups, list) and len(w_groups) >= 1


def test_report_excluded_weights_for_tanh_rnn():
    report = CompilationReport(_gru_graph())
    # tanh_rnn has win as a plain-op input projection (excluded from ETP)
    # and w as the recurrent ETP weight (included).
    assert report.counts['excluded_weights'] == 1
    excluded_paths = [path for path, _reason in report.excluded_weights]
    assert ('win',) in excluded_paths


def test_report_to_str_levels_and_repr():
    report = CompilationReport(_gru_graph())
    s1 = report.to_str(1)
    s2 = report.to_str(2)
    assert 'The hidden groups are:' in s1
    assert len(s2) >= len(s1)            # level 2 is a superset
    assert 'CompilationReport(' in repr(report)


def test_report_diagnostics_passthrough_is_tuple():
    report = CompilationReport(_gru_graph())
    assert isinstance(report.diagnostics, tuple)
