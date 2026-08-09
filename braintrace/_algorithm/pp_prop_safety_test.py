import brainstate
import jax.numpy as jnp
import pytest

import braintrace
from braintrace._testing import oracle_models


def _compiled_pp_prop(spec, *, limit=1 << 24):
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    config = braintrace.ETraceConfig(
        trace_factorization='io_factorized',
        recurrence_scope='coupled',
        decay=0.9,
    )
    learner = braintrace.pp_prop(
        model,
        0.9,
        config=config,
        snap_max_jacobian_elements=limit,
    )
    return learner, spec.make_inputs(2, 3, seed=1)[0]


def test_coupled_pp_prop_rejects_an_oversized_full_jacobian():
    learner, sample = _compiled_pp_prop(
        oracle_models.tanh_rnn(n_in=3, n_rec=5, seed=0),
        limit=24,
    )

    with pytest.raises(braintrace.NotSupportedError, match='25'):
        learner.compile_graph(sample)

    assert not learner.is_compiled


def test_coupled_pp_prop_accepts_the_jacobian_limit_boundary():
    learner, sample = _compiled_pp_prop(
        oracle_models.tanh_rnn(n_in=3, n_rec=5, seed=0),
        limit=25,
    )

    learner.compile_graph(sample)

    assert learner.is_compiled


def test_pp_prop_rejects_a_non_position_preserving_tail():
    spec = oracle_models.rolled_tail_rnn(
        n_in=3,
        n_rec=5,
        roll=1,
        seed=0,
    )
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    learner = braintrace.pp_prop(model, 0.9)

    with pytest.raises(braintrace.NotSupportedError, match='position-preserving'):
        learner.compile_graph(spec.make_inputs(2, 3, seed=1)[0])

    assert not learner.is_compiled


def test_pp_prop_accepts_a_position_preserving_tail():
    spec = oracle_models.rolled_tail_rnn(
        n_in=3,
        n_rec=5,
        roll=0,
        seed=0,
    )
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    learner = braintrace.pp_prop(model, 0.9)

    learner.compile_graph(jnp.asarray(spec.make_inputs(2, 3, seed=1)[0]))

    assert learner.is_compiled
