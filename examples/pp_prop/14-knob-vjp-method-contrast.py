# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""14 · single-step vs multi-step VJP vs BPTT head-to-head on DMS.

Trains three identical LIF RSNNs on the same DMS data:
one with vjp_method='single-step', one with 'multi-step', one with BPTT.
Reports per-epoch loss and final accuracy for all three.
"""

import pathlib
import sys
from typing import Dict

import brainstate
import braintools
import jax.numpy as jnp
import brainunit as u

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shared  # noqa: E402


class Net(brainstate.nn.Module):
    def __init__(self, n_in: int, n_rec: int, n_out: int):
        super().__init__()
        self.cell = _shared.LIFCell(n_in=n_in, n_rec=n_rec)
        self.readout = _shared.LeakyReadout(n_rec=n_rec, n_out=n_out)

    def update(self, x):
        return self.readout(self.cell(x))


def _accuracy(outputs_seq, labels):
    mean_out = outputs_seq.mean(axis=0)
    return float(jnp.mean(jnp.argmax(mean_out, axis=-1) == labels))


def _eval(model, inputs, labels):
    # kept manual: eval re-init, no online construction
    @brainstate.transform.vmap_new_states(state_tag="new", axis_size=inputs.shape[1])
    def init():
        brainstate.nn.init_all_states(model)

    init()
    vmap_model = brainstate.nn.Vmap(model, vmap_states="new")
    outs = brainstate.transform.for_loop(lambda x: vmap_model(x), inputs)
    return _accuracy(outs, labels)


def main(n_epochs: int = 4, batch_size: int = 32, num_step: int = 40, plot: bool = True) -> Dict:
    n_in = 16
    with brainstate.environ.context(dt=1.0 * u.ms):
        def make():
            m = Net(n_in=n_in, n_rec=64, n_out=2)
            w = m.states(brainstate.ParamState)
            o = braintools.optim.Adam(lr=1e-3)
            o.register_trainable_weights(w)
            return m, o

        m_ss, o_ss = make()
        m_ms, o_ms = make()
        m_bp, o_bp = make()

        @brainstate.transform.jit
        def train_ss(inputs, labels):
            return _shared.online_train_epoch_fixed_target(
                m_ss, o_ss, inputs, labels, decay_or_rank=0.97, vjp_method="single-step"
            )

        @brainstate.transform.jit
        def train_ms(inputs, labels):
            return _shared.online_train_epoch_fixed_target(
                m_ms, o_ms, inputs, labels, decay_or_rank=0.97, vjp_method="multi-step"
            )

        @brainstate.transform.jit
        def train_bp(inputs, labels):
            return _shared.bptt_train_epoch_fixed_target(m_bp, o_bp, inputs, labels)

        ss_l, ms_l, bp_l = [], [], []
        for epoch in range(n_epochs):
            xs, ys = _shared.make_dms_spikes(
                num_step=num_step, num_batch=batch_size, n_in=n_in, seed=epoch,
            )
            ss_l.append(float(train_ss(xs, ys)))
            ms_l.append(float(train_ms(xs, ys)))
            bp_l.append(float(train_bp(xs, ys)))
            print(
                f"[14-vjp-contrast] epoch {epoch}  "
                f"single={ss_l[-1]:.4f}  multi={ms_l[-1]:.4f}  bptt={bp_l[-1]:.4f}"
            )

        xs_e, ys_e = _shared.make_dms_spikes(
            num_step=num_step, num_batch=batch_size, n_in=n_in, seed=9999,
        )
        acc_ss = _eval(m_ss, xs_e, ys_e)
        acc_ms = _eval(m_ms, xs_e, ys_e)
        acc_bp = _eval(m_bp, xs_e, ys_e)
        print(f"[14-vjp-contrast] acc  single={acc_ss:.3f}  multi={acc_ms:.3f}  bptt={acc_bp:.3f}")

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ss_l, label="single-step")
        ax.plot(ms_l, label="multi-step")
        ax.plot(bp_l, label="BPTT", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("14 · vjp_method contrast on DMS")
        ax.legend()
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    for val in (ss_l[-1], ms_l[-1], bp_l[-1]):
        assert jnp.isfinite(jnp.asarray(val))
    return {
        "losses": ss_l, "multi_step_losses": ms_l, "bptt_losses": bp_l,
        "acc_single": acc_ss, "acc_multi": acc_ms, "acc_bptt": acc_bp,
    }


if __name__ == "__main__":
    result = main()
    # Full-run chance-level bounds (smoke test skips these since it uses n_epochs=1):
    assert result["acc_single"] > 0.5, f"single-step acc {result['acc_single']} <= 0.5"
    assert result["acc_multi"] > 0.5, f"multi-step acc {result['acc_multi']} <= 0.5"
