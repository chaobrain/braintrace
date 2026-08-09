import brainstate
import pytest

import braintrace
from braintrace._testing import oracle_models


def test_sparse_n_checks_the_transient_full_jacobian_before_graph_creation():
    spec = oracle_models.two_state_rnn(n_in=3, n_rec=5, seed=0)
    model = spec.factory()
    brainstate.nn.init_all_states(model, batch_size=1)
    learner = braintrace.SnAp(
        model,
        n=2,
        snap_max_jacobian_elements=50,
    )

    with pytest.raises(braintrace.NotSupportedError, match='100'):
        learner.compile_graph(spec.make_inputs(2, 3, seed=1)[0])

    assert learner.graph_executor._compiled_graph is None
