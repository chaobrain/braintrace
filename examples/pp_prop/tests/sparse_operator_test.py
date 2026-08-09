"""Native sparse-storage and backend tests for the pp-prop sparse example."""

import importlib.util
import pathlib
import sys

import brainevent
import brainstate
import braintrace
import brainunit as u
import jax
import jax.numpy as jnp


EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "09-operator-sparse.py"


def _load_example():
    """Load the sparse example without relying on its non-module filename."""
    sys.modules.pop("_shared", None)
    spec = importlib.util.spec_from_file_location("_pp_prop_sparse", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sparse_cell_stores_only_native_csr_edges():
    """The recurrent parameter is one value per edge, never an N-square mask."""
    module = _load_example()
    n_rec = 128
    density = 0.0625
    with brainstate.environ.context(dt=1.0 * u.ms):
        cell = module.SparseLIFCell(
            n_in=4,
            n_rec=n_rec,
            density=density,
            sparse_backend="jax_raw",
        )
    layer = cell.rec_syn.comm
    expected_edges = n_rec * round(n_rec * density)
    assert isinstance(layer, braintrace.nn.SparseLinear)
    assert isinstance(layer.spar_mat, brainevent.CSR)
    assert layer.weight.value["weight"].shape == (expected_edges,)
    assert layer.spar_mat.indices.size == expected_edges
    assert layer.spar_mat.indptr.size == n_rec + 1
    assert layer.weight.value["weight"].size < n_rec * n_rec
    assert not hasattr(layer, "w_mask")
    assert cell.ff_syn.comm.weight.value["weight"].shape == (4, n_rec)


def test_sparse_values_preserve_masked_dense_kaiming_variance():
    """Represented edges retain the original dense Kaiming value scale."""
    module = _load_example()
    n_rec = 256
    scale = 0.5
    matrix = module._fixed_degree_csr(
        n_rec=n_rec,
        density=0.25,
        scale=scale,
        seed=0,
        backend="jax_raw",
    )
    observed_std = float(jnp.std(matrix.data))
    expected_std = (scale / n_rec) ** 0.5
    assert expected_std * 0.95 < observed_std < expected_std * 1.05


def test_sparse_backend_override_reaches_batched_etp_primitive():
    """The explicit dependency-light backend supports native batched ETP."""
    module = _load_example()
    with brainstate.environ.context(dt=1.0 * u.ms):
        cell = module.SparseLIFCell(
            n_in=4,
            n_rec=32,
            density=0.125,
            sparse_backend="jax_raw",
        )
    layer = cell.rec_syn.comm
    jaxpr = jax.make_jaxpr(layer)(jnp.ones((3, 32), dtype=jnp.float32))
    primitives = {str(eqn.primitive) for eqn in jaxpr.jaxpr.eqns}
    assert layer.spar_mat.backend == "jax_raw"
    assert "etp_sp_mm" in primitives
    assert layer(jnp.ones((3, 32), dtype=jnp.float32)).shape == (3, 32)


def test_sparse_example_trains_with_jax_raw_backend():
    """A tiny native-CSR pp-prop epoch produces a finite loss."""
    module = _load_example()
    result = module.main(
        n_epochs=1,
        batch_size=2,
        num_step=3,
        plot=False,
        sparse_backend="jax_raw",
    )
    assert len(result["losses"]) == 1
    assert jnp.isfinite(jnp.asarray(result["losses"][0]))
