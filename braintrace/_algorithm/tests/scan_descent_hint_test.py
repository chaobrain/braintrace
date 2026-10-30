# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""The remedies an algorithm names when it refuses a descended scan (issue #226).

``scan_descent='off'`` does not help with a scan that holds an ETP primitive:
the opaque scan then rejects the weight inside it. Unrolling the scan does,
and so does an algorithm that supports descent.
"""
import warnings

import brainstate
import jax
import jax.numpy as jnp
import pytest

import braintrace
from braintrace import ControlFlowPolicy
from braintrace._algorithm.base import DESCENT_ALGORITHMS, descended_scan_length


class _SubstepScan(brainstate.nn.Module):
    """Two hidden states integrated in ``k`` substeps of one ``lax.scan``, with
    the input projection and a trainable time constant inside the body: the
    scan is descended when ``k`` exceeds ``scan_unroll_limit`` (16)."""

    def __init__(self, k=17, n_in=3, n=4):
        super().__init__()
        self.k = k
        with brainstate.random.seed_context(0):
            self.W = brainstate.ParamState(0.5 * brainstate.random.randn(n_in, n))
        self.tau = brainstate.ParamState(jnp.full(n, 2.0))
        self.V = brainstate.HiddenState(jnp.zeros(n))
        self.w = brainstate.HiddenState(jnp.zeros(n))

    def update(self, x):
        def body(c, _):
            V, w = c
            I = braintrace.matmul(x, self.W.value)
            h = 1.0 / self.k
            tau = braintrace.element_wise(self.tau.value)
            return (V + h / tau * (-V + jnp.tanh(2 * V) - w + I), w + h * (0.5 * V - w)), None

        (self.V.value, self.w.value), _ = jax.lax.scan(
            body, (self.V.value, self.w.value), None, length=self.k)
        return jax.nn.sigmoid(self.V.value)


class _TwoScans(_SubstepScan):
    """A second descended scan of 23 substeps over two more hidden states."""

    def __init__(self, n=4):
        super().__init__()
        self.U = brainstate.HiddenState(jnp.zeros(n))
        self.u = brainstate.HiddenState(jnp.zeros(n))

    def update(self, x):
        out = super().update(x)

        def body(c, _):
            U, u = c
            g = braintrace.element_wise(self.tau.value)
            return (U + (-U + u + x.sum()) / (23 * g), u + (U - u) / 23), None

        (self.U.value, self.u.value), _ = jax.lax.scan(
            body, (self.U.value, self.u.value), None, length=23)
        return out + jax.nn.sigmoid(self.U.value)


class _PreScaled(_SubstepScan):
    """``V`` is scaled before the scan, so its carry starts from a
    transformed value and descent is skipped (``SCAN_DESCENT_SKIPPED``)."""

    def update(self, x):
        self.V.value = self.V.value * 0.5
        return super().update(x)


_REFUSING = {
    'pp_prop': lambda m, **kw: braintrace.pp_prop(m, decay_or_rank=0.9, **kw),
    'ES_D_RTRL': lambda m, **kw: braintrace.ES_D_RTRL(m, decay_or_rank=0.9, **kw),
    'UORO': lambda m, **kw: braintrace.UORO(m, **kw),
    'OSTLFeedforward': lambda m, **kw: braintrace.OSTLFeedforward(m, **kw),
}

_SCOPE_REFUSING = {
    'OSTLRecurrent': lambda m, **kw: braintrace.OSTLRecurrent(m, **kw),
    'SnAp': lambda m, **kw: braintrace.SnAp(m, **kw),
}


def _run(factory, model, **kw):
    brainstate.nn.init_all_states(model)
    algo = factory(model, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        algo.compile_graph(jnp.zeros(3))
        return algo(jnp.ones(3))


def _check_hint(message, length):
    assert f'length-{length} scan' in message
    assert f'ControlFlowPolicy(scan_unroll_limit={length})' in message
    assert "scan_descent='off'" not in message


class TestRefusal:

    @pytest.mark.parametrize('name', sorted(_REFUSING))
    def test_the_refusal_names_unrolling_and_the_descent_algorithms(self, name):
        with pytest.raises(NotImplementedError, match='does not support structured scan descent') as err:
            _run(_REFUSING[name], _SubstepScan())
        message = str(err.value)
        _check_hint(message, 17)
        for algo in DESCENT_ALGORITHMS:
            assert algo in message

    @pytest.mark.parametrize('name', sorted(_SCOPE_REFUSING))
    def test_a_scope_refusal_names_unrolling(self, name):
        with pytest.raises(NotImplementedError, match='is not honoured inside a descended scan') as err:
            _run(_SCOPE_REFUSING[name], _SubstepScan())
        _check_hint(str(err.value), 17)

    def test_the_hint_names_the_longest_descended_scan(self):
        with pytest.raises(NotImplementedError) as err:
            _run(_REFUSING['pp_prop'], _TwoScans())
        _check_hint(str(err.value), 23)

    def test_scan_descent_off_still_fails(self):
        # the hint main gave (#226): the opaque scan rejects its weights
        off = ControlFlowPolicy(scan_descent='off')
        with pytest.raises(NotImplementedError):
            _run(_REFUSING['pp_prop'], _SubstepScan(), control_flow=off)


class TestDescentSkipped:
    """The warning when descent is skipped for a carry initialised from a
    transformed value. The scan then stays opaque and is refused, as it is
    under ``scan_descent='off'``, the hint main gave."""

    def test_the_warning_names_unrolling(self):
        model = _PreScaled()
        brainstate.nn.init_all_states(model)
        algo = braintrace.D_RTRL(model)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(NotImplementedError, match='weight states are used within a scan'):
                algo.compile_graph(jnp.zeros(3))
        (message,) = [str(w.message) for w in caught if 'descent was skipped' in str(w.message)]
        assert 'ControlFlowPolicy(scan_unroll_limit=17)' in message
        assert "scan_descent='off'" not in message

    def test_scan_descent_off_is_refused_too(self):
        with pytest.raises(NotImplementedError, match='weight states are used within a scan'):
            _run(braintrace.D_RTRL, _PreScaled(), control_flow=ControlFlowPolicy(scan_descent='off'))

    def test_unrolling_as_the_warning_says_runs(self):
        out = _run(braintrace.D_RTRL, _PreScaled(), control_flow=ControlFlowPolicy(scan_unroll_limit=17))
        assert out.shape == (4,)


class TestTheRemediesWork:

    @pytest.mark.parametrize('name', sorted(_REFUSING) + sorted(_SCOPE_REFUSING))
    def test_unrolling_as_the_hint_says_runs(self, name):
        factory = {**_REFUSING, **_SCOPE_REFUSING}[name]
        out = _run(factory, _SubstepScan(), control_flow=ControlFlowPolicy(scan_unroll_limit=17))
        assert out.shape == (4,)

    @pytest.mark.parametrize('name', DESCENT_ALGORITHMS)
    def test_each_named_algorithm_runs_the_descended_scan(self, name):
        model = _SubstepScan()
        brainstate.nn.init_all_states(model)
        algo = getattr(braintrace, name)(model)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            algo.compile_graph(jnp.zeros(3))
            assert descended_scan_length(algo.graph) == 17
            assert algo(jnp.ones(3)).shape == (4,)


class TestDescendedScanLength:

    def test_no_descended_scan(self):
        model = _SubstepScan(k=8)  # within scan_unroll_limit: unrolled
        brainstate.nn.init_all_states(model)
        algo = braintrace.D_RTRL(model)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            algo.compile_graph(jnp.zeros(3))
        assert descended_scan_length(algo.graph) is None

    def test_two_descended_scans(self):
        model = _TwoScans()
        brainstate.nn.init_all_states(model)
        algo = braintrace.D_RTRL(model)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            algo.compile_graph(jnp.zeros(3))
        assert descended_scan_length(algo.graph) == 23
