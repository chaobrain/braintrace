"""Smoke tests: each self-contained example's main() must run a tiny config end-to-end.

Data-dependent examples (000, 003-batched, 003-vmap, 004) are skipped with a
documented reason — they require NMNIST or DVSGesture datasets.
"""

import importlib.util
import pathlib
import sys

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[1]


def _load(fname: str):
    # Ensure the examples directory is on sys.path so that sibling imports
    # (snn_models, utils) resolve correctly when the loader executes the module.
    examples_dir_str = str(EXAMPLES_DIR)
    if examples_dir_str not in sys.path:
        sys.path.insert(0, examples_dir_str)

    spec = importlib.util.spec_from_file_location(f"_example_{fname}", EXAMPLES_DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("fname,tiny_kwargs", [
    # 001: GIF SNN on DMS task — shorten windows heavily so each trial is fast
    (
        "001-gif-snn-for-dms.py",
        dict(
            batch_size=2,
            num_batch=1,
            n_rec=4,
            t_fixation=10.,   # ms (already tiny default)
            t_sample=10.,     # ms (shrunk from 500 ms)
            t_delay=10.,      # ms (shrunk from 1000 ms)
            t_test=10.,       # ms (shrunk from 500 ms)
            plot=False,
        ),
    ),
    # 002: COBA EI-RSNN — 1 epoch / batch, tiny network
    (
        "002-coba-ei-rsnn.py",
        dict(batch_size=2, n_rec=4, n_epochs=1, plot=False),
    ),
    # 003-all: synthetic tensors — tiny net, 1 data step, 1 batch
    (
        "003-snn-memory-and-speed-evaluation-all.py",
        dict(method='diag', n_rec=4, n_layer=1, data_length=2, batch_size=2, n_batches=1, plot=False),
    ),
    # 100: GRU copying task — 1 epoch, tiny sequence / batch / hidden
    (
        "100-gru-on-copying-task.py",
        dict(n_epochs=1, n_seq=8, batch_size=2, n_rec=4, run_bptt=False, plot=False),
    ),
    # 101: Integrator RNN — 1 epoch, tiny batch / hidden / steps per epoch
    (
        "101-integrator-rnn.py",
        dict(n_epochs=1, num_batch=8, num_hidden=8, n_batches_per_epoch=2, run_bptt=False, plot=False),
    ),
    # --- DATA-DEPENDENT examples: skipped (no dataset available in CI) ---
    pytest.param(
        "000-lif-snn-for-nmnist.py",
        {},
        marks=pytest.mark.skip(reason="needs NMNIST dataset; covered by __main__ only"),
    ),
    pytest.param(
        "003-snn-memory-and-speed-evaluation-batched.py",
        {},
        marks=pytest.mark.skip(reason="needs DVSGesture dataset; covered by __main__ only"),
    ),
    pytest.param(
        "003-snn-memory-and-speed-evaluation-vmap.py",
        {},
        marks=pytest.mark.skip(reason="needs DVSGesture dataset; covered by __main__ only"),
    ),
    pytest.param(
        "004-feedforward-conv-snn.py",
        {},
        marks=pytest.mark.skip(reason="needs NMNIST dataset; covered by __main__ only"),
    ),
])
def test_example_runs(fname, tiny_kwargs):
    mod = _load(fname)
    result = mod.main(**tiny_kwargs)
    assert isinstance(result, dict), f"main() must return a dict, got {type(result)}"
    assert "losses" in result, f"result must contain 'losses' key, got keys: {list(result.keys())}"
    assert len(result["losses"]) >= 1, f"losses must be non-empty, got: {result['losses']}"
