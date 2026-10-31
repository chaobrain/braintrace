# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""A learner refuses inputs that differ from the example inputs it was compiled
with (issues #227 and #241).

The update replays the jaxpr traced at ``compile_graph()`` on the leaves of its
inputs: it neither retraces nor casts, and a ``Quantity``'s unit is dropped.
Another dtype failed deep inside the replay or ran a program built for other
values (#227), and another unit of the same dimension ran at the wrong scale
(#241). Such an input now raises a ``TypeError`` that names the leaf, before
the replay.
"""
import warnings

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._input_data import LeafSignature, input_signature

_ALGOS = {'d_rtrl': {}, 'eprop': {}, 'pp_prop': {'decay_or_rank': 0.9}, 'ostl_recurrent': {}, 'uoro': {}}


class _Cell(brainstate.nn.Module):
    """``V`` of shape ``(4,)`` driven by ``x @ W``; a ``Quantity`` input is read in volts."""

    def __init__(self, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.W = brainstate.ParamState(jnp.full((3, 4), 0.5, dtype))

    def init_state(self, *args, **kwargs):
        self.V = brainstate.HiddenState(jnp.zeros(4, self.dtype))

    def update(self, x, gain=1.0):
        if isinstance(x, u.Quantity):
            x = x.to_decimal(u.volt)
        self.V.value = 0.9 * self.V.value + gain * braintrace.matmul(x, self.W.value)
        return self.V.value


class _Pair(brainstate.nn.Module):
    """A dict input ``{'a': (3,), 'b': (3,)}``."""

    def __init__(self):
        super().__init__()
        self.W = brainstate.ParamState(jnp.full((3, 4), 0.5))
        self.V = brainstate.HiddenState(jnp.zeros(4))

    def update(self, inputs):
        self.V.value = 0.9 * self.V.value + braintrace.matmul(inputs['a'] * inputs['b'], self.W.value)
        return self.V.value


def _learner(*example, algo='d_rtrl', model=None, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return braintrace.compile(_Cell() if model is None else model, algo, *example, **kw)


def _refused(learner, *args, match):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(TypeError, match=match) as err:
            learner(*args)
    return str(err.value)


@pytest.fixture
def _x64():
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', False)


class TestTheDtype:

    @pytest.mark.parametrize('algo', ['d_rtrl', 'eprop'])
    def test_the_model_of_issue_227(self, algo, _x64):
        # the model casts its input to float64 and writes it into a buffer;
        # traced with a float64 input the cast is a no-op. Before: a float32
        # input failed inside the replay, in lax.dynamic_update_slice
        class Delay(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.W = brainstate.ParamState(jnp.full((3, 4), 0.5))
                self.V = brainstate.HiddenState(jnp.zeros(4))

            def update(self, x):
                buf = jax.lax.dynamic_update_slice(jnp.zeros(4), x.astype(jnp.float64), (0,))
                self.V.value = 0.9 * self.V.value + braintrace.matmul(buf[:3], self.W.value)
                return self.V.value

        learner = _learner(jnp.ones(3, jnp.float64), algo=algo, model=Delay())
        assert learner(jnp.ones(3, jnp.float64)).dtype == jnp.float64
        message = _refused(learner, jnp.ones(3, jnp.float32), match='differ from the example inputs')
        assert f'{type(learner).__name__}.update()' in message
        assert 'args[0]: compiled with float64[3], given float32[3]; cast it with `.astype(float64)`.' in message
        assert 'neither retraces nor casts' in message

    def test_the_reverse_direction_is_refused(self, _x64):
        # before: it ran, on a program traced for float32
        learner = _learner(jnp.ones(3, jnp.float32), model=_Cell(jnp.float32))
        _refused(learner, jnp.ones(3, jnp.float64), match=r'compiled with float32\[3\], given float64\[3\]')

    @pytest.mark.parametrize('algo', sorted(_ALGOS))
    def test_every_algorithm_checks(self, algo):
        learner = _learner(jnp.ones(3), algo=algo, **_ALGOS[algo])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            learner(jnp.ones(3))
        _refused(learner, jnp.ones(3, jnp.float16), match=r'compiled with float32\[3\], given float16\[3\]')

    def test_a_python_scalar_has_its_canonical_dtype(self):
        learner = _learner(jnp.ones(3), 0.5)
        # a Python float is a weak float32; the weak type is not compared
        learner(jnp.ones(3), 2.0)
        learner(jnp.ones(3), jnp.float32(2.0))
        _refused(learner, jnp.ones(3), 2, match=r'args\[1\]: compiled with float32\[\], given int32\[\]')


class TestTheShape:

    def test_another_shape_is_refused(self):
        learner = _learner(jnp.ones(3))
        message = _refused(learner, jnp.ones(4), match=r'compiled with float32\[3\], given float32\[4\]')
        assert 'compile a new learner with an example input of this shape' in message


class TestTheUnit:
    """#241: ``jaxpr_call`` replays the jaxpr on the mantissas."""

    def test_the_same_unit_runs(self):
        x = jnp.ones(3) * u.volt
        model = _Cell()
        brainstate.nn.init_all_states(model)
        expected = model.update(x)
        learner = _learner(x)
        np.testing.assert_array_equal(learner(x), expected)

    def test_another_unit_of_the_same_dimension_is_refused(self):
        # before: 1000 mV ran as 1000 V, and the output and dL/dW were 1000 times too large
        learner = _learner(jnp.ones(3) * u.volt)
        message = _refused(learner, jnp.ones(3) * 1000.0 * u.mV, match='differ from the example inputs')
        assert 'args[0]: compiled with float32[3] in V, given float32[3] in mV; convert it with `.in_unit(V)`.' in message

    def test_the_gradient_is_refused_too(self):
        model = _Cell()
        learner = _learner(jnp.ones(3) * u.volt, model=model)
        grad = brainstate.transform.grad(lambda x: learner(x).sum(), model.states(brainstate.ParamState))
        with pytest.raises(TypeError, match='in V, given float32.3. in mV'):
            grad(jnp.ones(3) * 1000.0 * u.mV)

    def test_another_dimension_is_refused(self):
        # before: a UnitMismatchError from a retrace of the model
        learner = _learner(jnp.ones(3) * u.volt)
        _refused(learner, jnp.ones(3) * u.pA, match='in pA; it has another physical dimension than V')

    def test_a_plain_array_for_a_quantity_is_refused(self):
        learner = _learner(jnp.ones(3) * u.volt)
        _refused(learner, jnp.ones(3), match=r'compiled with float32\[3\] in V, given float32\[3\]; pass')

    def test_a_quantity_for_a_plain_array_is_refused(self):
        learner = _learner(jnp.ones(3))
        _refused(learner, jnp.ones(3) * u.volt, match=r'compiled with float32\[3\], given float32\[3\] in V')


class TestTheStructure:

    def test_another_structure_is_refused(self):
        learner = _learner(jnp.ones(3))
        message = _refused(learner, (jnp.ones(3),), match='inputs of structure')
        assert 'PyTreeDef(((*,),))' in message and 'PyTreeDef((*,))' in message

    def test_another_dict_key_is_refused(self):
        learner = _learner({'a': jnp.ones(3), 'b': jnp.ones(3)}, model=_Pair())
        _refused(learner, {'a': jnp.ones(3), 'c': jnp.ones(3)}, match='inputs of structure')

    def test_the_leaf_is_named_by_its_key_path(self):
        learner = _learner({'a': jnp.ones(3), 'b': jnp.ones(3)}, model=_Pair())
        message = _refused(learner, {'a': jnp.ones(3), 'b': jnp.ones(3, jnp.float16)}, match='differ')
        assert "args[0]['b']: compiled with float32[3], given float16[3]" in message
        assert "args[0]['a']" not in message


class TestStepData:
    """A ``MultiStepData`` is compared per step."""

    def test_a_multi_step_input_is_compared_per_step(self):
        learner = _learner(jnp.ones(3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert learner(braintrace.MultiStepData(jnp.ones((5, 3)))).shape == (5, 4)
            assert learner(braintrace.SingleStepData(jnp.ones(3))).shape == (4,)
        _refused(learner, braintrace.MultiStepData(jnp.ones((5, 4))),
                 match=r'given float32\[4\] per step of a MultiStepData')

    def test_a_multi_step_example_input_is_compiled_per_step(self):
        learner = _learner(braintrace.MultiStepData(jnp.ones((5, 3))))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert learner(jnp.ones(3)).shape == (4,)
        _refused(learner, jnp.ones((5, 3)), match=r'compiled with float32\[3\], given float32\[5,3\]')


class TestUnderTransforms:

    def test_a_loop_over_the_steps(self):
        learner = _learner(jnp.ones(3))
        with pytest.raises(TypeError, match=r'given float16\[3\]'):
            brainstate.transform.for_loop(learner, jnp.ones((4, 3), jnp.float16))

    def test_a_vmapped_learner_is_compared_per_example(self):
        learner = _learner(jnp.ones((2, 3)), batch_size=2, vmap=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert learner(jnp.ones((2, 3))).shape == (2, 4)
        _refused(learner, jnp.ones((2, 3), jnp.float16), match=r'compiled with float32\[3\], given float16\[3\]')

    def test_etrace_grad(self):
        # issue #227: etrace_grad over a sequence failed inside the replay as well
        model = _Cell()
        learner = _learner(jnp.ones(3), model=model)
        xs = jnp.ones((4, 3), jnp.float16)
        with pytest.raises(TypeError, match=r'given float16\[3\]'):
            learner.etrace_grad(xs, step_fn=lambda x: learner(x).sum())


class TestTheSignature:

    def test_it_is_recorded_at_compile_time(self):
        model = _Cell()
        brainstate.nn.init_all_states(model)
        x = jnp.ones(3) * u.volt
        minfo = braintrace.extract_module_info(model, x)
        assert minfo.input_signature == input_signature(x)
        assert minfo.add_jaxpr_outs([]).input_signature == minfo.input_signature

    def test_the_weak_type_is_dropped(self):
        assert input_signature(1.0) == input_signature(jnp.float32(3.0))
        assert input_signature(1) != input_signature(1.0)

    def test_a_multi_step_leaf_loses_its_leading_axis(self):
        multi = input_signature((braintrace.MultiStepData(jnp.ones((5, 3))), jnp.ones(2)))
        assert multi == input_signature((jnp.ones(3), jnp.ones(2)))
        assert [sig.per_step for _, sig in multi.leaves] == [True, False]

    def test_a_quantity_is_one_leaf_with_its_unit(self):
        sig = input_signature({'v': jnp.ones(3) * u.mV, 'n': jnp.ones(2)})
        assert sig.treedef == input_signature({'v': jnp.ones(3), 'n': jnp.ones(2)}).treedef
        assert dict(sig.leaves)["args[0]['v']"] == LeafSignature('Quantity', (3,), jnp.dtype('float32'), u.mV)
        assert dict(sig.leaves)["args[0]['n']"].unit is None

    def test_a_value_that_is_not_a_jax_type_is_compared_by_its_type(self):
        assert input_signature('a') == input_signature('b')
        assert dict(input_signature('a').leaves)['args[0]'].describe() == 'a str'
        assert input_signature('a') != input_signature(1.0)
