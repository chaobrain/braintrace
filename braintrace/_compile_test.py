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
    model = tanh_rnn(n_in=3, n_rec=4, seed=0).factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    return model


def test_compile_returns_ready_learner_and_updates():
    model = _fresh_model()
    x0 = jnp.ones((3,), dtype='float32')
    learner = braintrace.compile(model, 'D_RTRL', x0)
    assert isinstance(learner, braintrace.D_RTRL)
    assert learner.is_compiled
    y = learner.update(x0)          # must NOT raise "not compiled"
    assert y.shape == (1, 4)
    assert bool(jnp.all(jnp.isfinite(y)))


def test_compile_accepts_class_and_forwards_options():
    model = _fresh_model()
    x0 = jnp.ones((3,), dtype='float32')
    learner = braintrace.compile(model, braintrace.D_RTRL, x0, trace_dtype=jnp.bfloat16)
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
        braintrace.compile(model, 'OTTT', x0)


def test_compile_matches_manual_construction():
    x0 = jnp.ones((3,), dtype='float32')

    manual_model = _fresh_model()
    manual = braintrace.D_RTRL(manual_model)
    manual.compile_graph(x0)
    y_manual = manual.update(x0)

    compiled_model = _fresh_model()
    compiled = braintrace.compile(compiled_model, 'D_RTRL', x0)
    y_compiled = compiled.update(x0)

    assert jnp.allclose(y_manual, y_compiled)
