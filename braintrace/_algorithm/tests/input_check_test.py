# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""A learner conforms its inputs to the example inputs it was compiled with
(issues #227 and #241).

The update replays the jaxpr traced at ``compile_graph()`` on the leaves of its
inputs, and a ``Quantity``'s unit is dropped. Another dtype failed deep inside
the replay or ran a program built for other values (#227), and another unit of
the same dimension ran at the wrong scale (#241). Now one step of the inputs is
compared with the example inputs first:

* a lossless upcast within a category (``float16`` to ``float32``, ``int8`` to
  ``int32``) and a weak-typed value (a Python scalar) take the compiled dtype,
  as in JAX; a plain array and a ``Quantity`` in unit 1 stand for each other; a
  list stands for a tuple;
* any other difference raises a ``TypeError`` that names the leaf and every
  applicable remedy, before any state is read.
"""
import collections
import warnings

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintrace
from braintrace._input_data import LeafSignature, conform_inputs, input_signature

_ALGOS = {'d_rtrl': {}, 'eprop': {}, 'pp_prop': {'decay_or_rank': 0.9}, 'ostl_recurrent': {}, 'uoro': {}}


class _Cell(brainstate.nn.Module):
    """``V`` of shape ``(4,)`` driven by ``x @ W``; a ``Quantity`` input is read in
    volts (in unit 1 if it is dimensionless), and the input is cast to ``W``'s dtype.
    ``traces`` counts the traces of ``update``."""

    def __init__(self, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.W = brainstate.ParamState(jnp.full((3, 4), 0.5, dtype))
        self.traces = 0

    def init_state(self, *args, **kwargs):
        self.V = brainstate.HiddenState(jnp.zeros(4, self.dtype))

    def update(self, x, gain=1.0):
        self.traces += 1
        if isinstance(x, u.Quantity):
            x = u.get_mantissa(x) if x.unit == u.UNITLESS else x.to_decimal(u.volt)
        x = x.astype(self.W.value.dtype)
        self.V.value = 0.9 * self.V.value + gain * braintrace.matmul(x, self.W.value)
        return self.V.value


class _Index(brainstate.nn.Module):
    """``V`` driven by the row ``i`` of ``W``, for an integer input."""

    def __init__(self):
        super().__init__()
        self.W = brainstate.ParamState(jnp.full((3, 4), 0.5))

    def init_state(self, *args, **kwargs):
        self.V = brainstate.HiddenState(jnp.zeros(4))

    def update(self, i):
        self.V.value = 0.9 * self.V.value + braintrace.matmul(jax.nn.one_hot(i, 3), self.W.value)
        return self.V.value


class _Pair(brainstate.nn.Module):
    """A container input with the leaves ``a`` and ``b``, both ``(3,)``."""

    def __init__(self):
        super().__init__()
        self.W = brainstate.ParamState(jnp.full((3, 4), 0.5))

    def init_state(self, *args, **kwargs):
        self.V = brainstate.HiddenState(jnp.zeros(4))

    def update(self, inputs):
        a, b = (inputs['a'], inputs['b']) if isinstance(inputs, dict) else inputs
        self.V.value = 0.9 * self.V.value + braintrace.matmul(a + 2.0 * b, self.W.value)
        return self.V.value


def _learner(*example, algo='d_rtrl', model=None, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return braintrace.compile(_Cell() if model is None else model, algo, *example, **kw)


def _run(learner, *args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return learner(*args)


def _refused(learner, *args, match):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(TypeError, match=match) as err:
            learner(*args)
    return str(err.value)


def _array(leaf):
    leaf = u.get_mantissa(leaf)
    if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
        leaf = jax.random.key_data(leaf)
    return np.asarray(leaf)


def _values(learner):
    """Every state value of *learner*, in order: the model's, its traces and
    ``running_index``. (Trace paths hold object ids, so they are not compared.)"""
    values = [state.value for state in learner.states().values()]
    return [_array(leaf) for leaf in jax.tree.leaves(values, is_leaf=lambda x: isinstance(x, u.Quantity))]


def _assert_same(a, b):
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b)):
        assert x.dtype == y.dtype, i
        np.testing.assert_array_equal(x, y, err_msg=str(i))


def _steps(make, example, calls, algo='d_rtrl', **kw):
    """Outputs and final states of a fresh learner compiled with the example
    arguments *example* and called with each argument tuple of *calls*."""
    learner = _learner(*example, algo=algo, model=make(), **kw)
    outs = [np.asarray(_run(learner, *args)) for args in calls]
    return outs, _values(learner)


@pytest.fixture
def _x64():
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', False)


def _same_as_a_manual_cast(make, example, given, dtype, algo='d_rtrl', **kw):
    """Three calls with *given* equal, bit for bit, three calls with *given* cast by hand."""
    cast = jax.tree.map(lambda x: x.astype(dtype), given, is_leaf=lambda x: isinstance(x, u.Quantity))
    outs, states = _steps(make, (example,), [(given,)] * 3, algo=algo, **kw)
    ref_outs, ref_states = _steps(make, (example,), [(cast,)] * 3, algo=algo, **kw)
    for a, b in zip(outs, ref_outs):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)
    _assert_same(states, ref_states)


class TestTheCast:
    """A lossless upcast within a category, and a weak-typed value, take the compiled dtype."""

    @pytest.mark.parametrize('algo', ['d_rtrl', 'eprop'])
    def test_the_model_of_issue_227_runs_as_with_a_manual_cast(self, algo, _x64):
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

        x = jnp.asarray([0.3, -1.2, 2.1], jnp.float32)
        _same_as_a_manual_cast(Delay, jnp.ones(3, jnp.float64), x, jnp.float64, algo=algo)

    @pytest.mark.parametrize('given', [jnp.float16, jnp.bfloat16], ids=['float16', 'bfloat16'])
    def test_a_narrower_float_is_cast(self, given):
        x = jnp.asarray([0.3, -1.2, 2.1], given)
        _same_as_a_manual_cast(_Cell, jnp.ones(3), x, jnp.float32)

    @pytest.mark.parametrize('given', [jnp.float32, jnp.float16, jnp.bfloat16], ids=['float32', 'float16', 'bfloat16'])
    def test_a_narrower_float_is_cast_to_float64(self, given, _x64):
        x = jnp.asarray([0.3, -1.2, 2.1], given)
        _same_as_a_manual_cast(_Cell, jnp.ones(3, jnp.float64), x, jnp.float64)

    @pytest.mark.parametrize('algo', sorted(_ALGOS))
    def test_every_algorithm_casts(self, algo):
        x = jnp.asarray([0.3, -1.2, 2.1], jnp.float16)
        _same_as_a_manual_cast(_Cell, jnp.ones(3), x, jnp.float32, algo=algo, **_ALGOS[algo])

    @pytest.mark.parametrize('given', [jnp.int8, jnp.int16], ids=['int8', 'int16'])
    def test_a_narrower_int_is_cast(self, given):
        _same_as_a_manual_cast(_Index, jnp.int32(1), jnp.asarray(2, given), jnp.int32)

    def test_a_multi_step_input_is_cast_whole(self):
        xs = braintrace.MultiStepData(jnp.asarray(np.random.RandomState(0).rand(5, 3), jnp.float16))
        _same_as_a_manual_cast(_Cell, jnp.ones(3), xs, jnp.float32)

    def test_a_quantity_keeps_its_unit(self):
        x = jnp.asarray([0.3, -1.2, 2.1], jnp.float16) * u.volt
        _same_as_a_manual_cast(_Cell, jnp.ones(3) * u.volt, x, jnp.float32)

    def test_a_python_scalar_takes_the_compiled_dtype(self):
        # the example 0.5 is a weak float32; 2 is a weak int32, and becomes 2.0 as in JAX
        a = _steps(_Cell, (jnp.ones(3), 0.5), [(jnp.ones(3), 2)] * 2)
        b = _steps(_Cell, (jnp.ones(3), 0.5), [(jnp.ones(3), 2.0)] * 2)
        c = _steps(_Cell, (jnp.ones(3), 0.5), [(jnp.ones(3), jnp.float32(2.0))] * 2)
        for x, y in [(a, b), (a, c)]:
            for o, p in zip(x[0], y[0]):
                np.testing.assert_array_equal(o, p)
            _assert_same(x[1], y[1])

    def test_a_python_int_for_a_strong_float_example(self):
        learner = _learner(jnp.ones(3), jnp.float32(0.5))
        assert _run(learner, jnp.ones(3), 2).dtype == jnp.float32

    def test_under_x64_a_python_scalar_takes_the_compiled_dtype(self, _x64):
        # a strong float32 example: a Python float is a weak float64, narrowed as in JAX
        learner = _learner(jnp.ones(3, jnp.float32), jnp.float32(0.5), model=_Cell(jnp.float32))
        assert _run(learner, jnp.ones(3, jnp.float32), 2.0).dtype == jnp.float32
        # a Python-float example is a weak float64: a float32 scalar is an upcast
        learner = _learner(jnp.ones(3), 0.5, model=_Cell(jnp.float64))
        assert _run(learner, jnp.ones(3), jnp.float32(2.0)).dtype == jnp.float64

    def test_a_cast_in_a_loop_adds_a_convert_and_no_retrace(self, _x64):
        xs = jnp.asarray(np.random.RandomState(0).rand(4, 3), jnp.float32)
        runs = {}
        for name, inputs in [('cast', xs), ('manual', xs.astype(jnp.float64))]:
            model = _Cell(jnp.float64)
            learner = _learner(jnp.ones(3, jnp.float64), model=model)
            before = model.traces
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out = brainstate.transform.for_loop(learner, inputs)
                traces = model.traces - before
                eqns = brainstate.transform.make_jaxpr(lambda x: learner(x))(inputs[0])[0].eqns
            runs[name] = (np.asarray(out), traces, [e.primitive.name for e in eqns])
        np.testing.assert_array_equal(runs['cast'][0], runs['manual'][0])
        assert runs['cast'][1] == runs['manual'][1]  # the cast retraces nothing
        extra = collections.Counter(runs['cast'][2]) - collections.Counter(runs['manual'][2])
        assert extra == collections.Counter({'convert_element_type': 1})
        assert not collections.Counter(runs['manual'][2]) - collections.Counter(runs['cast'][2])


class TestTheRefusals:

    def test_a_downcast_is_refused(self, _x64):
        # before: it ran, on a program traced for float32
        learner = _learner(jnp.ones(3, jnp.float32), model=_Cell(jnp.float32))
        _refused(learner, jnp.ones(3, jnp.float64),
                 match=r'compiled with float32\[3\], given float64\[3\]; cast it with `.astype\(float32\)`')

    @pytest.mark.parametrize('algo', sorted(_ALGOS))
    def test_every_algorithm_refuses_a_change_of_category(self, algo):
        learner = _learner(jnp.ones(3), algo=algo, **_ALGOS[algo])
        _run(learner, jnp.ones(3))
        _refused(learner, jnp.ones(3, jnp.int32),
                 match=r'args\[0\]: compiled with float32\[3\], given int32\[3\]; cast it with `.astype\(float32\)`\.')

    @pytest.mark.parametrize('example, given', [
        (jnp.float16, jnp.bfloat16), (jnp.bfloat16, jnp.float16),
    ], ids=['bfloat16-for-float16', 'float16-for-bfloat16'])
    def test_float16_and_bfloat16_are_refused_for_each_other(self, example, given):
        learner = _learner(jnp.ones(3, example))
        _refused(learner, jnp.ones(3, given), match=rf'compiled with {jnp.dtype(example)}\[3\], given {jnp.dtype(given)}\[3\]')

    def test_an_unsigned_int_for_a_signed_one_is_refused(self):
        learner = _learner(jnp.int32(1), model=_Index())
        _refused(learner, jnp.uint8(2), match=r'compiled with int32\[\], given uint8\[\]')

    def test_a_narrower_int_example_refuses_a_wider_int(self):
        learner = _learner(jnp.int8(1), model=_Index())
        _refused(learner, jnp.int32(2), match=r'compiled with int8\[\], given int32\[\]')

    def test_a_python_bool_for_a_float_is_refused(self):
        # a Python bool is a strong bool in JAX. Before: it ran as 1.0
        learner = _learner(jnp.ones(3), 0.5)
        _refused(learner, jnp.ones(3), True,
                 match=r'args\[1\]: compiled with float32\[\], given bool\[\]; pass a Python float or a float32 scalar\.')

    def test_a_python_float_for_an_int_example_is_refused(self):
        learner = _learner(jnp.ones(3), 1)
        _refused(learner, jnp.ones(3), 2.5,
                 match=r'args\[1\]: compiled with int32\[\], given float32\[\]; pass a Python int or an int32 scalar\.')

    def test_another_shape_is_refused(self):
        learner = _learner(jnp.ones(3))
        message = _refused(learner, jnp.ones(4), match=r'compiled with float32\[3\], given float32\[4\]')
        assert 'compile a new learner with an example input of this shape' in message

    def test_a_refused_call_changes_no_state(self):
        learner = _learner(jnp.ones(3))
        for _ in range(2):
            _run(learner, jnp.asarray([0.3, -1.2, 2.1]))
        before = _values(learner)
        assert sum(np.abs(v).max() > 0 for v in before if v.dtype.kind == 'f') >= 2  # V and a trace
        _refused(learner, jnp.ones(4), match='differ')
        _refused(learner, jnp.ones(3) * u.mV, match='differ')
        _assert_same(_values(learner), before)

    def test_every_applicable_remedy_is_named(self):
        learner = _learner(jnp.ones(3) * u.volt)
        message = _refused(learner, jnp.ones(4, jnp.int32) * u.mV, match='differ')
        assert ('args[0]: compiled with float32[3] in V, given int32[4] in mV; compile a new learner with an '
                'example input of this shape; convert it with `.in_unit(V)`; cast it with `.astype(float32)`.') in message


class TestTheUnit:
    """#241: the replay reads the mantissas."""

    def test_the_same_unit_runs(self):
        x = jnp.ones(3) * u.volt
        model = _Cell()
        brainstate.nn.init_all_states(model)
        expected = model.update(x)
        learner = _learner(x)
        np.testing.assert_array_equal(_run(learner, x), expected)

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
        _refused(learner, jnp.ones(3), match=r'compiled with float32\[3\] in V, given float32\[3\]; pass a Quantity in V\.')

    def test_a_quantity_for_a_plain_array_is_refused(self):
        learner = _learner(jnp.ones(3))
        _refused(learner, jnp.ones(3) * u.volt,
                 match=r'compiled with float32\[3\], given float32\[3\] in V; pass a plain array\.')

    def test_a_dimensionless_unit_of_another_scale_is_refused(self):
        learner = _learner(jnp.ones(3))
        _refused(learner, jnp.ones(3) * (u.mV / u.volt), match=r'given float32\[3\] in mV / V; pass a plain array\.')

    def test_a_plain_array_and_unit_one_stand_for_each_other(self):
        x = jnp.asarray([0.3, -1.2, 2.1])
        plain = _steps(_Cell, (x,), [(x,)] * 2)
        for example, given in [(x, u.Quantity(x)), (u.Quantity(x), x), (u.Quantity(x), u.Quantity(x))]:
            outs, states = _steps(_Cell, (example,), [(given,)] * 2)
            for a, b in zip(outs, plain[0]):
                np.testing.assert_array_equal(a, b)
            _assert_same(states, plain[1])

    def test_a_multi_step_quantity_is_compared_per_step(self):
        learner = _learner(jnp.ones(3) * u.volt)
        assert _run(learner, braintrace.MultiStepData(jnp.ones((5, 3)) * u.volt)).shape == (5, 4)
        _refused(learner, braintrace.MultiStepData(jnp.ones((5, 3)) * u.mV),
                 match=r'compiled with float32\[3\] in V, given float32\[3\] in mV per step of a MultiStepData')


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
        message = _refused(learner, {'a': jnp.ones(3), 'b': jnp.ones(3, jnp.int32)}, match='differ')
        assert "args[0]['b']: compiled with float32[3], given int32[3]" in message
        assert "args[0]['a']" not in message

    def test_a_list_stands_for_a_tuple(self):
        a, b = jnp.asarray([0.3, -1.2, 2.1]), jnp.asarray([1.0, 0.5, -0.5])
        ref = _steps(_Pair, ((a, b),), [((a, b),)] * 2)
        for example, given in [((a, b), [a, b]), ([a, b], (a, b))]:
            outs, states = _steps(_Pair, (example,), [(given,)] * 2)
            for o, p in zip(outs, ref[0]):
                np.testing.assert_array_equal(o, p)
            _assert_same(states, ref[1])

    def test_an_ordered_dict_for_a_dict_is_refused(self):
        # a dict flattens in sorted key order, an OrderedDict in insertion order.
        # Before: OrderedDict(b=..., a=...) ran with its leaves swapped
        learner = _learner({'a': jnp.ones(3), 'b': jnp.zeros(3)}, model=_Pair())
        message = _refused(learner, collections.OrderedDict([('b', jnp.zeros(3)), ('a', jnp.ones(3))]),
                           match='inputs of structure')
        assert 'OrderedDict' in message and 'sorted key order' in message


class TestStepData:
    """A ``MultiStepData`` is compared per step."""

    def test_a_multi_step_input_is_compared_per_step(self):
        learner = _learner(jnp.ones(3))
        assert _run(learner, braintrace.MultiStepData(jnp.ones((5, 3)))).shape == (5, 4)
        assert _run(learner, braintrace.SingleStepData(jnp.ones(3))).shape == (4,)
        _refused(learner, braintrace.MultiStepData(jnp.ones((5, 4))),
                 match=r'given float32\[4\] per step of a MultiStepData')

    def test_a_multi_step_example_input_is_compiled_per_step(self):
        learner = _learner(braintrace.MultiStepData(jnp.ones((5, 3))))
        assert _run(learner, jnp.ones(3)).shape == (4,)
        _refused(learner, jnp.ones((5, 3)), match=r'compiled with float32\[3\], given float32\[5,3\]')


class TestUnderTransforms:

    def test_a_loop_over_the_steps(self):
        learner = _learner(jnp.ones(3))
        with pytest.raises(TypeError, match=r'given int32\[3\]'):
            brainstate.transform.for_loop(learner, jnp.ones((4, 3), jnp.int32))

    def test_a_vmapped_learner_is_compared_per_example(self):
        learner = _learner(jnp.ones((2, 3)), batch_size=2, vmap=True)
        assert _run(learner, jnp.ones((2, 3))).shape == (2, 4)
        assert _run(learner, jnp.ones((2, 3), jnp.float16)).dtype == jnp.float32
        _refused(learner, jnp.ones((2, 3), jnp.int32), match=r'compiled with float32\[3\], given int32\[3\]')

    def test_etrace_grad(self):
        # issue #227: etrace_grad over a sequence failed inside the replay as well
        learner = _learner(jnp.ones(3))
        with pytest.raises(TypeError, match=r'given int32\[3\]'):
            learner.etrace_grad(jnp.ones((4, 3), jnp.int32), step_fn=lambda x: learner(x).sum())


class TestModuleInfoJaxprCall:
    """The public replay conforms its inputs in the same way, at trace time."""

    @staticmethod
    def _minfo(x):
        model = _Cell()
        brainstate.nn.init_all_states(model)
        return braintrace.extract_module_info(model, x)

    def test_it_refuses(self):
        minfo = self._minfo(jnp.ones(3) * u.volt)
        with pytest.raises(TypeError, match=r'(?s)ModuleInfo.jaxpr_call\(\) was given .*in V, given float32\[3\] in mV'):
            minfo.jaxpr_call(jnp.ones(3) * u.mV)
        with pytest.raises(TypeError, match=r'given float32\[4\] in V'):
            jax.jit(lambda x: minfo.jaxpr_call(x)[0])(jnp.ones(4) * u.volt)

    def test_it_casts(self):
        minfo = self._minfo(jnp.ones(3))
        x = jnp.asarray([0.3, -1.2, 2.1], jnp.float16)
        cast, manual = minfo.jaxpr_call(x)[0], minfo.jaxpr_call(x.astype(jnp.float32))[0]
        assert cast.dtype == jnp.float32
        np.testing.assert_array_equal(cast, manual)


class TestTheSignature:

    def test_it_is_recorded_at_compile_time(self):
        model = _Cell()
        brainstate.nn.init_all_states(model)
        x = jnp.ones(3) * u.volt
        minfo = braintrace.extract_module_info(model, x)
        assert minfo.input_signature == input_signature(x)
        assert minfo.add_jaxpr_outs([]).input_signature == minfo.input_signature

    def test_the_weak_type_is_not_compared(self):
        assert input_signature(1.0) == input_signature(jnp.float32(3.0))
        assert input_signature(1) != input_signature(1.0)

    def test_a_multi_step_leaf_loses_its_leading_axis(self):
        multi = input_signature((braintrace.MultiStepData(jnp.ones((5, 3))), jnp.ones(2)))
        assert multi == input_signature((jnp.ones(3), jnp.ones(2)))
        assert [sig.per_step for _, sig in multi.leaves] == [True, False]

    def test_a_quantity_is_one_leaf_with_its_unit(self):
        sig = input_signature({'v': jnp.ones(3) * u.mV, 'n': jnp.ones(2)})
        assert sig.treedef == input_signature({'v': jnp.ones(3), 'n': jnp.ones(2)}).treedef
        assert dict(sig.leaves)["args[0]['v']"] == LeafSignature('array', (3,), jnp.dtype('float32'), u.mV)
        assert dict(sig.leaves)["args[0]['n']"].unit is None

    def test_a_list_and_a_tuple_have_one_structure(self):
        x = jnp.ones(3)
        assert input_signature([x, {'a': [x]}]).treedef == input_signature((x, {'a': (x,)})).treedef
        assert input_signature(collections.OrderedDict(a=x)).treedef != input_signature({'a': x}).treedef

    def test_a_value_that_is_not_a_jax_type_is_compared_by_its_type(self):
        assert input_signature('a') == input_signature('b')
        assert dict(input_signature('a').leaves)['args[0]'].describe() == 'a str'
        assert input_signature('a') != input_signature(1.0)


class TestConformInputs:
    """``conform_inputs`` on its own, for the leaves no learner test reaches."""

    def test_a_prng_key_is_compared_by_its_dtype(self):
        key = jax.random.key(0)
        args = (jax.random.key(1),)
        assert conform_inputs(input_signature(key), args, 'f') is args
        with pytest.raises(TypeError, match=r'compiled with key<fry>\[\], given float32\[\]; pass a key<fry> value\.'):
            conform_inputs(input_signature(key), (jnp.float32(1.0),), 'f')
        with pytest.raises(TypeError, match=r'compiled with float32\[\], given key<fry>\[\]; pass a float32 value\.'):
            conform_inputs(input_signature(jnp.float32(1.0)), (key,), 'f')

    def test_a_value_that_is_not_a_jax_type_is_kept(self):
        args = ('b',)
        assert conform_inputs(input_signature('a'), args, 'f') is args
        with pytest.raises(TypeError, match=r"compiled with a str, given float32\[\]; pass a str\."):
            conform_inputs(input_signature('a'), (1.0,), 'f')

    def test_nothing_is_rebuilt_when_nothing_changes(self):
        args = (jnp.ones(3), {'a': jnp.ones(2) * u.mV}, [1.0])
        assert conform_inputs(input_signature(*args), args, 'f') is args
