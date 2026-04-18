"""Smoke tests: each example's main() runs one epoch end-to-end without exceptions."""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[1]

EXAMPLE_FILES = [
    "01-basics-lif-integrator.py",
    "02-neurons-alif-dms.py",
    "03-neurons-gif-working-memory.py",
]


def _load(fname: str):
    spec = importlib.util.spec_from_file_location(f"_pp_prop_{fname}", EXAMPLES_DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("fname", EXAMPLE_FILES)
def test_example_runs(fname):
    mod = _load(fname)
    result = mod.main(n_epochs=1, batch_size=4, plot=False)
    assert "losses" in result
    assert len(result["losses"]) >= 1
