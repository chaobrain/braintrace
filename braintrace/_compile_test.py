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

import brainstate
import jax.numpy as jnp
import pytest

import braintrace
from braintrace._compile import _resolve_algorithm
from braintrace._etrace_algorithms.oracle_models import tanh_rnn


# --- Task 1: algorithm resolution -------------------------------------------

def test_resolve_class_passthrough():
    assert _resolve_algorithm(braintrace.D_RTRL) is braintrace.D_RTRL


@pytest.mark.parametrize('name,cls_name', [
    ('d_rtrl', 'D_RTRL'),
    ('D_RTRL', 'D_RTRL'),          # case-insensitive
    ('pp_prop', 'pp_prop'),
    ('es_d_rtrl', 'pp_prop'),      # alias
    ('esd_rtrl', 'pp_prop'),       # alias
    ('eprop', 'EProp'),
    ('e_prop', 'EProp'),
    ('ostl_recurrent', 'OSTLRecurrent'),
    ('ostl_feedforward', 'OSTLFeedforward'),
    ('otpe', 'OTPE'),
    ('ottt', 'OTTT'),
    ('osttp', 'OSTTP'),
])
def test_resolve_string_names(name, cls_name):
    assert _resolve_algorithm(name) is getattr(braintrace, cls_name)


def test_resolve_unknown_string_raises_value_error():
    with pytest.raises(ValueError) as exc:
        _resolve_algorithm('not_an_algo')
    assert 'd_rtrl' in str(exc.value)


def test_resolve_bare_ostl_is_rejected():
    with pytest.raises(ValueError):
        _resolve_algorithm('ostl')   # ambiguous; removed in 0.2.0


def test_resolve_instance_raises_type_error():
    class Foo:
        pass
    with pytest.raises(TypeError):
        _resolve_algorithm(Foo())


def test_resolve_unrelated_class_raises_type_error():
    with pytest.raises(TypeError):
        _resolve_algorithm(dict)


# --- Task 2: compile() end-to-end + errors ----------------------------------

def _fresh_model():
    # compile() now owns initialization, so the model is returned uninitialized.
    return tanh_rnn(n_in=3, n_rec=4, seed=0).factory()


def test_compile_returns_ready_learner_and_updates():
    model = _fresh_model()
    x0 = jnp.ones((3,), dtype='float32')
    learner = braintrace.compile(model, 'D_RTRL', x0, batch_size=1)
    assert isinstance(learner, braintrace.D_RTRL)
    assert learner.is_compiled
    y = learner.update(x0)          # must NOT raise "not compiled"
    assert y.shape == (1, 4)
    assert bool(jnp.all(jnp.isfinite(y)))


def test_compile_accepts_class_and_forwards_options():
    model = _fresh_model()
    x0 = jnp.ones((3,), dtype='float32')
    learner = braintrace.compile(model, braintrace.D_RTRL, x0, batch_size=1, trace_dtype=jnp.bfloat16)
    assert learner.trace_dtype == jnp.bfloat16


def test_compile_requires_example_inputs():
    model = _fresh_model()
    with pytest.raises(ValueError) as exc:
        braintrace.compile(model, 'D_RTRL')   # no example inputs
    assert 'example input' in str(exc.value).lower()


def test_compile_propagates_missing_required_option():
    model = _fresh_model()
    x0 = jnp.ones((3,), dtype='float32')
    # OTTT requires keyword-only ``leak``; constructor raises TypeError.
    with pytest.raises(TypeError):
        braintrace.compile(model, 'OTTT', x0, batch_size=1)


def test_compile_matches_manual_construction():
    x0 = jnp.ones((3,), dtype='float32')

    manual_model = tanh_rnn(n_in=3, n_rec=4, seed=0).factory()
    brainstate.nn.init_all_states(manual_model, batch_size=1)
    manual = braintrace.D_RTRL(manual_model)
    manual.compile_graph(x0)
    y_manual = manual.update(x0)

    compiled_model = tanh_rnn(n_in=3, n_rec=4, seed=0).factory()
    compiled = braintrace.compile(compiled_model, 'D_RTRL', x0, batch_size=1)
    y_compiled = compiled.update(x0)

    assert jnp.allclose(y_manual, y_compiled)


import inspect

# --- Task 3: always-init / seed / guardrail / verbose -----------------------

def test_compile_initializes_uninitialized_model():
    model = tanh_rnn(n_in=3, n_rec=4, seed=0).factory()   # NOT pre-initialized
    x0 = jnp.ones((3,), dtype='float32')
    learner = braintrace.compile(model, 'D_RTRL', x0, batch_size=1)
    y = learner.update(x0)
    assert y.shape == (1, 4)


def test_compile_seed_restores_global_rng():
    model = tanh_rnn(n_in=3, n_rec=4, seed=0).factory()
    x0 = jnp.ones((3,), dtype='float32')
    before = brainstate.random.DEFAULT.value
    braintrace.compile(model, 'D_RTRL', x0, batch_size=1, seed=7)
    after = brainstate.random.DEFAULT.value
    assert bool(jnp.array_equal(before, after))


def test_compile_seed_reproducible_state_init():
    x0 = jnp.ones((3,), dtype='float32')

    def states_after(seed):
        m = tanh_rnn(n_in=3, n_rec=4, seed=0).factory()
        learner = braintrace.compile(m, 'D_RTRL', x0, batch_size=1, seed=seed)
        return [s.value for s in learner.hidden_states.values()]

    a = states_after(11)
    b = states_after(11)
    assert all(bool(jnp.array_equal(x, y)) for x, y in zip(a, b))


class _NoEtraceModel(brainstate.nn.Module):
    """A model with a ParamState used through a *plain* JAX op (no ETP routing)."""
    def __init__(self):
        super().__init__()
        self.h = brainstate.HiddenState(jnp.zeros((1, 4)))
        self.w = brainstate.ParamState(jnp.ones((3, 4)))

    def update(self, x):
        # plain matmul -> w is NOT an eligibility-trace weight
        self.h.value = jnp.tanh(jnp.matmul(x, self.w.value))
        return self.h.value


def test_compile_guardrail_no_etrace_weights_raises():
    model = _NoEtraceModel()
    x0 = jnp.ones((1, 3), dtype='float32')
    with pytest.raises(braintrace.CompilationError) as exc:
        braintrace.compile(model, 'D_RTRL', x0)
    assert 'ETP' in str(exc.value)


@pytest.mark.parametrize('bad', [-1, 3, 'x'])
def test_compile_verbose_invalid_raises(bad):
    model = _fresh_model()
    x0 = jnp.ones((3,), dtype='float32')
    with pytest.raises(ValueError):
        braintrace.compile(model, 'D_RTRL', x0, batch_size=1, verbose=bad)


def test_compile_verbose_levels_print(capsys):
    x0 = jnp.ones((3,), dtype='float32')

    braintrace.compile(_fresh_model(), 'D_RTRL', x0, batch_size=1, verbose=0)
    assert capsys.readouterr().out == ''

    braintrace.compile(_fresh_model(), 'D_RTRL', x0, batch_size=1, verbose=1)
    out1 = capsys.readouterr().out
    assert 'The hidden groups are:' in out1


# --- Task 3: docstring drift guard ------------------------------------------

# Option names the compile docstring documents per algorithm. Each MUST be an
# explicit constructor parameter (not absorbed by **kwargs).
_DOC_OPTIONS = {
    'd_rtrl': ['name', 'vjp_method', 'fast_solve', 'trace_dtype'],
    'es_d_rtrl': ['decay_or_rank', 'name', 'vjp_method', 'fast_solve'],
    'eprop': ['feedback', 'kappa_filter_decay', 'random_feedback_key', 'name', 'vjp_method', 'fast_solve'],
    'otpe': ['mode', 'leak', 'name', 'vjp_method', 'trace_clip_abs'],
    'ottt': ['mode', 'leak', 'name', 'vjp_method'],
    'osttp': ['B_list', 'target_timing', 'name', 'vjp_method', 'fast_solve'],
    'ostl_recurrent': ['name', 'vjp_method', 'fast_solve', 'trace_dtype'],
    'ostl_feedforward': ['decay_or_rank', 'name'],
}


@pytest.mark.parametrize('algo_name,options', list(_DOC_OPTIONS.items()))
def test_documented_options_are_real_constructor_params(algo_name, options):
    cls = _resolve_algorithm(algo_name)
    params = inspect.signature(cls.__init__).parameters
    for opt in options:
        assert opt in params, f'{algo_name}: documented option {opt!r} is not a constructor parameter'
