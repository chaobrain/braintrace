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

"""Tests for the ``braintrace`` package root (``braintrace/__init__.py``).

Two concerns, both owned by the package root:

* **Public-API contract** — every ``braintrace.__all__`` symbol resolves to a
  usable object; public algorithms run a minimal end-to-end step; documented
  error paths fire on their real triggers; legacy/nn deprecations warn; and the
  implementation facts that drifted from CLAUDE.md (F-17) stay pinned.
* **Legacy deprecation forwarding** — the v0.1.x shims served lazily by
  ``__getattr__`` warn on access, stay out of ``__all__``, and appear in
  ``__dir__``.
"""

import brainstate
import jax
import jax.numpy as jnp
import pytest

import braintrace
import braintrace._legacy as legacy
from braintrace._algorithm.oracle_models import tanh_rnn


# ===========================================================================
# Public-API contract
# ===========================================================================

# --- Task 1: every __all__ symbol resolves to a non-None object --------------

def test_all_symbols_are_resolvable():
    missing = [name for name in braintrace.__all__ if not hasattr(braintrace, name)]
    assert not missing, f"__all__ names with no attribute: {missing}"


def test_all_symbols_are_non_none():
    none_valued = [name for name in braintrace.__all__ if getattr(braintrace, name) is None]
    assert not none_valued, f"__all__ names resolving to None: {none_valued}"


def test_marker_sentinel_fails_first():
    # Guard against an empty/trivial __all__ silently passing the above.
    assert len(braintrace.__all__) > 30


# --- Task 2: rate-model algorithms instantiate and run a minimal step --------

def _algo_constructors():
    """Map name -> callable(model)->algo for each public algorithm that works on
    a plain rate model (SNN-specific algos needing leak are covered separately)."""
    return {
        'D_RTRL': lambda m: braintrace.D_RTRL(m),
        'ParamDimVjpAlgorithm': lambda m: braintrace.ParamDimVjpAlgorithm(m),
        'pp_prop': lambda m: braintrace.pp_prop(m, decay_or_rank=0.9),
        'ES_D_RTRL': lambda m: braintrace.ES_D_RTRL(m, decay_or_rank=0.9),
        'IODimVjpAlgorithm': lambda m: braintrace.IODimVjpAlgorithm(m, decay_or_rank=0.9),
        'EProp': lambda m: braintrace.EProp(m),
        'OSTLRecurrent': lambda m: braintrace.OSTLRecurrent(m),
        'OSTLFeedforward': lambda m: braintrace.OSTLFeedforward(m, decay_or_rank=0.9),
    }


@pytest.mark.parametrize('name', list(_algo_constructors().keys()))
def test_algorithm_is_usable_end_to_end(name):
    model = tanh_rnn(n_in=3, n_rec=4, seed=0).factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    algo = _algo_constructors()[name](model)
    x0 = jnp.ones((3,), dtype='float32')
    algo.compile_graph(x0)
    algo.init_etrace_state()
    y = algo(x0)
    assert y.shape == (1, 4)
    assert bool(jnp.all(jnp.isfinite(y)))


# --- Task 3: SNN algos requiring leak + OSTTP B_list contract ----------------

def test_ottt_otpe_require_explicit_leak():
    model = tanh_rnn(seed=0).factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    with pytest.raises(TypeError):
        braintrace.OTTT(model)          # leak is keyword-only & required
    with pytest.raises(TypeError):
        braintrace.OTPE(model)


def test_ottt_otpe_construct_with_leak():
    for ctor in (lambda m: braintrace.OTTT(m, leak=0.9),
                 lambda m: braintrace.OTPE(m, leak=0.9)):
        model = tanh_rnn(seed=0).factory()
        brainstate.nn.init_all_states(model, batch_size=1)
        algo = ctor(model)
        assert algo is not None


def test_osttp_constructs_with_B_list():
    model = tanh_rnn(n_in=3, n_rec=4, seed=0).factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    # OSTTP needs one random-feedback matrix per HiddenGroup; B.shape[1] == n_l (=4).
    B_list = [jnp.eye(4, dtype='float32')]
    algo = braintrace.OSTTP(model, B_list)
    assert algo is not None


# --- Task 4: dynamic weight assignment raises NotSupportedError --------------

def test_dynamic_weight_assignment_raises_not_supported():
    """Writing a ParamState during the forward pass is unsupported and must
    raise braintrace.NotSupportedError at graph compile time."""

    class BadModel(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (4, 4)))
            self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

        def update(self, x):
            self.w.value = self.w.value * 1.001   # illegal: rewriting a ParamState
            self.h.value = jax.nn.tanh(braintrace.matmul(self.h.value, self.w.value))
            return self.h.value

    model = BadModel()
    brainstate.nn.init_all_states(model, batch_size=1)
    algo = braintrace.D_RTRL(model)
    with pytest.raises(braintrace.NotSupportedError):
        algo.compile_graph(jnp.ones((1, 4), dtype='float32'))


# --- Task 5: weight inside control flow raises NotImplementedError -----------

@pytest.mark.xfail(
    strict=True,
    reason="F-SCAN-WEIGHT: the weight-in-control-flow guard intends to raise "
           "NotImplementedError, but its error-message construction at "
           "base.py:133 indexes invar_to_hidden_path with the *weight* var "
           "(absent from that hidden-path map), so a KeyError escapes instead. "
           "The guard fires, but the wrong exception type surfaces.",
)
def test_weight_used_inside_scan_raises_not_implemented():
    """check_unsupported_op should raise NotImplementedError when a weight var is
    used within a control-flow op. Today it raises KeyError while building the
    message (F-SCAN-WEIGHT) — this test pins the intended contract via xfail."""

    class ScanModel(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.ParamState(
                0.1 * jax.random.normal(jax.random.PRNGKey(0), (4, 4)))
            self.h = brainstate.HiddenState(jnp.zeros((1, 4)))

        def update(self, x):
            def body(carry, _):
                return braintrace.matmul(carry, self.w.value), None  # weight inside scan
            out, _ = jax.lax.scan(body, self.h.value, xs=None, length=2)
            self.h.value = jax.nn.tanh(out)
            return self.h.value

    model = ScanModel()
    brainstate.nn.init_all_states(model, batch_size=1)
    algo = braintrace.D_RTRL(model)
    with pytest.raises(NotImplementedError):
        algo.compile_graph(jnp.ones((1, 4), dtype='float32'))


# --- Task 6: legacy + nn deprecations emit DeprecationWarning ----------------

def test_legacy_op_access_warns():
    # Legacy shims are served lazily via the package-root __getattr__, which warns
    # at attribute-access time (not at construction).
    with pytest.warns(DeprecationWarning):
        _ = braintrace.MatMulOp


def test_legacy_param_access_warns():
    with pytest.warns(DeprecationWarning):
        _ = braintrace.ETraceParam


def test_nn_forwarded_name_warns():
    import braintrace.nn as nn
    with pytest.warns(DeprecationWarning):
        _ = nn.LayerNorm


def test_nn_unknown_name_raises_attribute_error():
    import braintrace.nn as nn
    with pytest.raises(AttributeError):
        _ = nn.ThisNameDoesNotExist


# --- Task 7: lock the drifted implementation facts (F-17) --------------------

def test_ostl_is_two_classes_not_a_factory():
    from braintrace._algorithm.param_dim_vjp import ParamDimVjpAlgorithm
    from braintrace._algorithm.io_dim_vjp import IODimVjpAlgorithm
    assert isinstance(braintrace.OSTLRecurrent, type)
    assert isinstance(braintrace.OSTLFeedforward, type)
    assert issubclass(braintrace.OSTLRecurrent, ParamDimVjpAlgorithm)
    assert issubclass(braintrace.OSTLFeedforward, IODimVjpAlgorithm)


def test_iodim_lives_in_io_dim_vjp_module():
    from braintrace._algorithm.io_dim_vjp import IODimVjpAlgorithm
    assert braintrace.IODimVjpAlgorithm is IODimVjpAlgorithm
    assert braintrace.pp_prop is braintrace.ES_D_RTRL  # aliases


def test_expected_rnn_cells_exist():
    import braintrace.nn as nn
    for cell in ('ValinaRNNCell', 'GRUCell', 'MGUCell', 'LSTMCell', 'URLSTMCell',
                 'MinimalRNNCell', 'MiniGRU', 'MiniLSTM', 'LRUCell'):
        assert hasattr(nn, cell), f"missing cell: {cell}"


# ===========================================================================
# Legacy v0.1.x deprecation forwarding
# ===========================================================================

_LEGACY_NAMES = [
    'ETraceOp', 'MatMulOp', 'ElemWiseOp', 'ConvOp', 'SpMatMulOp', 'LoraOp',
    'ETraceParam', 'ElemWiseParam', 'NonTempParam',
    'FakeETraceParam', 'FakeElemWiseParam',
]


@pytest.mark.parametrize('name', _LEGACY_NAMES)
def test_legacy_access_warns_and_returns_class(name):
    with pytest.warns(DeprecationWarning):
        obj = getattr(braintrace, name)
    assert obj is getattr(legacy, name)


@pytest.mark.parametrize('name', _LEGACY_NAMES)
def test_legacy_names_not_in_all(name):
    assert name not in braintrace.__all__


def test_from_import_warns():
    with pytest.warns(DeprecationWarning):
        from braintrace import MatMulOp  # noqa: F401


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = braintrace.ThisNameDoesNotExist


def test_legacy_names_in_dir():
    d = dir(braintrace)
    for name in _LEGACY_NAMES:
        assert name in d
