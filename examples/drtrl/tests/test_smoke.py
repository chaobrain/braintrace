"""Smoke tests: each example's main() must run one epoch end-to-end."""

import importlib.util
import pathlib

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[1]


def _load(fname: str):
    spec = importlib.util.spec_from_file_location(f"_drtrl_{fname}", EXAMPLES_DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("fname", [
    "01-basics-integrator.py",
    "02-batching-vmap.py",
    "03-batching-batched.py",
    "04-vjp-single-step.py",
    "05-vjp-multi-step.py",
    "07-operator-lora.py",
    "08-operator-conv.py",
    pytest.param("09-classification-mnist.py", marks=pytest.mark.skipif(
        True,
        reason="MNIST example is network-dependent; covered by __main__ only",
    )),
    "10-char-lm-generation.py",
    "11-knob-fast-solve.py",
    "12-knob-normalize-spectrum.py",
])
def test_example_runs(fname):
    mod = _load(fname)
    result = mod.main(n_epochs=1, batch_size=4, plot=False)
    assert "losses" in result
