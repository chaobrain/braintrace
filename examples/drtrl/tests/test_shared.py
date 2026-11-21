import importlib.util
import pathlib

import numpy as np


def _load_shared():
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_drtrl_shared", root / "_shared.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_integrator_batch_shape():
    s = _load_shared()
    x, y = s.make_integrator_batch(num_step=25, num_batch=4, seed=195)
    assert x.shape == (25, 4, 1)
    assert y.shape == (25, 4, 1)
    # Compare against a high-precision reference, allowing float32 rounding
    # near zero rather than depending on JAX's accumulation order.
    expected = np.cumsum(np.asarray(x), axis=0, dtype=np.float64)
    np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-7)


def test_copy_batch_shape():
    s = _load_shared()
    x, y = s.make_copy_batch(time_lag=5, batch_size=4, seed=0)
    assert x.shape == (5 + 20, 4, 10)
    assert y.shape == (10, 4)


def test_xor_batch_shape_and_values():
    s = _load_shared()
    x, y = s.make_xor_batch(seq_len=12, delay=4, batch_size=8, seed=0)
    assert x.shape == (12, 8, 2)
    assert y.shape == (8,)
    assert set(np.asarray(y).tolist()).issubset({0, 1})


def test_sine_batch_shape():
    s = _load_shared()
    x, y = s.make_sine_batch(num_step=40, batch_size=4, seed=0)
    assert x.shape == (40, 4, 1)
    assert y.shape == (40, 4, 1)


def test_char_batches_shape():
    s = _load_shared()
    text = "abcdefghij" * 10
    vocab, x, y = s.make_char_batches(
        text=text, seq_len=8, batch_size=4, seed=0
    )
    assert len(vocab) > 0
    assert x.shape == (8, 4, len(vocab))
    assert y.shape == (8, 4)
