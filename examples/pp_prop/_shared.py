# Copyright 2026 BrainX Ecosystem Limited. Licensed under the Apache License, 2.0.
"""Shared SNN data generators, cell wrappers, and train-step helpers for examples/pp_prop."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import brainpy.state
import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import saiunit as u


DEFAULT_DT = 1.0 * u.ms
DEFAULT_SEED = 42


# --- Data generators -----------------------------------------------------


def make_integrator_spikes(
    num_step: int = 50,
    num_batch: int = 32,
    rate_hz: float = 50.0,
    dt: float = 1e-3,
    seed: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Poisson input spikes. Target = cumulative mean rate (continuous)."""
    rng = np.random.default_rng(seed)
    p = rate_hz * dt
    spikes = (rng.random((num_step, num_batch, 1)) < p).astype(np.float32)
    targets = np.cumsum(spikes, axis=0) / num_step
    return jnp.asarray(spikes), jnp.asarray(targets)


def make_dms_spikes(
    num_step: int = 40,
    num_batch: int = 32,
    n_in: int = 16,
    fr_hz: float = 80.0,
    dt: float = 1e-3,
    seed: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Delayed match-to-sample: sample window + delay + test window, binary label."""
    rng = np.random.default_rng(seed)
    t_sample = num_step // 4
    t_delay = num_step // 2
    labels = rng.integers(0, 2, size=(num_batch,)).astype(np.int32)
    sample_dir = rng.integers(0, n_in, size=(num_batch,))
    test_dir = np.where(labels == 1, sample_dir, (sample_dir + n_in // 2) % n_in)
    fr = fr_hz * dt
    xs = np.zeros((num_step, num_batch, n_in), dtype=np.float32)
    for b in range(num_batch):
        xs[:t_sample, b, sample_dir[b]] = fr
        xs[t_sample + t_delay:, b, test_dir[b]] = fr
    xs = (rng.random(xs.shape) < xs).astype(np.float32)
    return jnp.asarray(xs), jnp.asarray(labels)


def make_memory_pattern(
    num_step: int = 30,
    num_batch: int = 32,
    n_in: int = 8,
    cue_frac: float = 0.2,
    seed: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Working memory: binary pattern during cue window, silent delay, recall target at end."""
    rng = np.random.default_rng(seed)
    cue_steps = max(1, int(num_step * cue_frac))
    pattern = rng.integers(0, 2, size=(num_batch, n_in)).astype(np.float32)
    xs = np.zeros((num_step, num_batch, n_in), dtype=np.float32)
    xs[:cue_steps] = pattern[None, :, :]
    return jnp.asarray(xs), jnp.asarray(pattern)


def make_poisson_mnist(
    num_step: int = 25,
    num_batch: int = 32,
    rate_hz: float = 80.0,
    dt: float = 1e-3,
    digits: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    seed: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Poisson-encoded sklearn 8x8 digits. Falls back to synthetic patterns if sklearn missing."""
    try:
        from sklearn.datasets import load_digits
        data = load_digits()
        imgs = data.images / data.images.max()
        labels = data.target
        mask = np.isin(labels, list(digits))
        imgs = imgs[mask]
        labels = labels[mask]
    except ImportError:
        rng_fallback = np.random.default_rng(seed)
        n_per = 50
        imgs = rng_fallback.random((n_per * len(digits), 8, 8)).astype(np.float32)
        labels = np.repeat(np.arange(len(digits)), n_per)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(labels), size=(num_batch,))
    flat = imgs[idx].reshape(num_batch, 64)
    label_idx = np.array([list(digits).index(int(l)) for l in labels[idx]], dtype=np.int32)
    p = flat[None, :, :] * rate_hz * dt
    spikes = (rng.random((num_step, num_batch, 64)) < p).astype(np.float32)
    return jnp.asarray(spikes), jnp.asarray(label_idx)


# --- SNN cells -----------------------------------------------------------
#
# All cells use the AlignPostProj + Expon + CUBA pattern with consistent
# saiunit units. Pattern:
#
#     linear (mA) -> Expon syn (mA) -> CUBA (scales to current) -> LIF/ALIF/GIF (mV)
#
# `braintrace.nn.Linear` is the ETP-tracked op; CUBA + LIF share the expected
# mV/mA/ohm unit algebra inside the neuron model.


class LIFCell(brainstate.nn.Module):
    """LIF recurrent block: concat(input, last-spike) -> Linear -> Expon -> CUBA -> LIF -> spikes."""

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        tau_mem: u.Quantity = 20.0 * u.ms,
        tau_syn: u.Quantity = 10.0 * u.ms,
        V_th: u.Quantity = 1.0 * u.mV,
        ff_scale: float = 2.0,
        rec_scale: float = 1.0,
    ):
        super().__init__()
        import braintrace
        self.neu = brainpy.state.LIF(
            n_rec, R=1. * u.ohm, tau=tau_mem, V_th=V_th,
            V_reset=0. * u.mV, V_rest=0. * u.mV,
            V_initializer=braintools.init.ZeroInit(unit=u.mV),
        )
        ff = braintools.init.KaimingNormal(ff_scale, unit=u.mA)
        rec = braintools.init.KaimingNormal(rec_scale, unit=u.mA)
        w = u.math.concatenate([ff((n_in, n_rec)), rec((n_rec, n_rec))], axis=0)
        self.syn = brainpy.state.AlignPostProj(
            comm=braintrace.nn.Linear(
                n_in + n_rec, n_rec, w_init=w,
                b_init=braintools.init.ZeroInit(unit=u.mA),
            ),
            syn=brainpy.state.Expon(
                n_rec, tau=tau_syn,
                g_initializer=braintools.init.ZeroInit(unit=u.mA),
            ),
            out=brainpy.state.CUBA(scale=1.),
            post=self.neu,
        )

    def update(self, x):
        self.syn(u.math.concatenate([x, self.neu.get_spike()], axis=-1))
        self.neu(0. * u.mA)
        return self.neu.get_spike()


class ALIFCell(brainstate.nn.Module):
    """ALIF (adaptive threshold) recurrent block. Same interface as LIFCell."""

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        tau_mem: u.Quantity = 20.0 * u.ms,
        tau_syn: u.Quantity = 10.0 * u.ms,
        tau_a: u.Quantity = 100.0 * u.ms,
        V_th: u.Quantity = 1.0 * u.mV,
        beta: u.Quantity = 1.8 * u.mV,
        ff_scale: float = 2.0,
        rec_scale: float = 1.0,
    ):
        super().__init__()
        import braintrace
        self.neu = brainpy.state.ALIF(
            n_rec, R=1. * u.ohm, tau=tau_mem, tau_a=tau_a, V_th=V_th, beta=beta,
            V_reset=0. * u.mV, V_rest=0. * u.mV,
            V_initializer=braintools.init.ZeroInit(unit=u.mV),
        )
        ff = braintools.init.KaimingNormal(ff_scale, unit=u.mA)
        rec = braintools.init.KaimingNormal(rec_scale, unit=u.mA)
        w = u.math.concatenate([ff((n_in, n_rec)), rec((n_rec, n_rec))], axis=0)
        self.syn = brainpy.state.AlignPostProj(
            comm=braintrace.nn.Linear(
                n_in + n_rec, n_rec, w_init=w,
                b_init=braintools.init.ZeroInit(unit=u.mA),
            ),
            syn=brainpy.state.Expon(
                n_rec, tau=tau_syn,
                g_initializer=braintools.init.ZeroInit(unit=u.mA),
            ),
            out=brainpy.state.CUBA(scale=1.),
            post=self.neu,
        )

    def update(self, x):
        self.syn(u.math.concatenate([x, self.neu.get_spike()], axis=-1))
        self.neu(0. * u.mA)
        return self.neu.get_spike()


class GIFCell(brainstate.nn.Module):
    """GIF (generalized integrate-and-fire) with heterogeneous tau_I2. Same interface."""

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        tau_mem: u.Quantity = 20.0 * u.ms,
        tau_syn: u.Quantity = 10.0 * u.ms,
        V_th: u.Quantity = 1.0 * u.mV,
        A2: u.Quantity = -0.5 * u.mA,
        tau_I2_low: u.Quantity = 80.0 * u.ms,
        tau_I2_high: u.Quantity = 200.0 * u.ms,
        ff_scale: float = 2.0,
        rec_scale: float = 1.0,
    ):
        super().__init__()
        import pathlib
        import sys
        # lazy import of local GIF class (tau_I2 heterogeneity)
        repo_examples = pathlib.Path(__file__).resolve().parent.parent
        if str(repo_examples) not in sys.path:
            sys.path.insert(0, str(repo_examples))
        from snn_models import GIF  # type: ignore
        import braintrace
        self.neu = GIF(
            n_rec,
            V_th_inf=V_th,
            tau=tau_mem,
            tau_I2=brainstate.random.uniform(tau_I2_low, tau_I2_high, n_rec),
            A2=A2,
        )
        ff = braintools.init.KaimingNormal(ff_scale, unit=u.mA)
        rec = braintools.init.KaimingNormal(rec_scale, unit=u.mA)
        w = u.math.concatenate([ff((n_in, n_rec)), rec((n_rec, n_rec))], axis=0)
        self.syn = brainpy.state.AlignPostProj(
            comm=braintrace.nn.Linear(
                n_in + n_rec, n_rec, w_init=w,
                b_init=braintools.init.ZeroInit(unit=u.mA),
            ),
            syn=brainpy.state.Expon(
                n_rec, tau=tau_syn,
                g_initializer=braintools.init.ZeroInit(unit=u.mA),
            ),
            out=brainpy.state.CUBA(scale=1.),
            post=self.neu,
        )

    def update(self, x):
        self.syn(u.math.concatenate([x, self.neu.get_spike()], axis=-1))
        self.neu(0. * u.mA)
        return self.neu.get_spike()


class COBAEICell(brainstate.nn.Module):
    """Dale-law E/I recurrent block using signed init on braintrace.nn.Linear."""

    def __init__(
        self,
        n_in: int,
        n_exc: int,
        n_inh: int,
        tau_mem: u.Quantity = 20.0 * u.ms,
        tau_syn: u.Quantity = 10.0 * u.ms,
        V_th: u.Quantity = 1.0 * u.mV,
        ff_scale: float = 2.0,
        rec_scale: float = 1.0,
    ):
        super().__init__()
        import braintrace
        n_rec = n_exc + n_inh
        self.n_exc = n_exc
        ff = braintools.init.KaimingNormal(ff_scale, unit=u.mA)((n_in, n_rec))
        rec_pos = u.math.abs(braintools.init.KaimingNormal(rec_scale, unit=u.mA)((n_exc, n_rec)))
        rec_neg = -u.math.abs(braintools.init.KaimingNormal(rec_scale, unit=u.mA)((n_inh, n_rec)))
        w = u.math.concatenate([ff, rec_pos, rec_neg], axis=0)
        self.neu = brainpy.state.LIF(
            n_rec, R=1. * u.ohm, tau=tau_mem, V_th=V_th,
            V_reset=0. * u.mV, V_rest=0. * u.mV,
            V_initializer=braintools.init.ZeroInit(unit=u.mV),
        )
        self.syn = brainpy.state.AlignPostProj(
            comm=braintrace.nn.Linear(
                n_in + n_rec, n_rec, w_init=w,
                b_init=braintools.init.ZeroInit(unit=u.mA),
            ),
            syn=brainpy.state.Expon(
                n_rec, tau=tau_syn,
                g_initializer=braintools.init.ZeroInit(unit=u.mA),
            ),
            out=brainpy.state.CUBA(scale=1.),
            post=self.neu,
        )

    def update(self, x):
        self.syn(u.math.concatenate([x, self.neu.get_spike()], axis=-1))
        self.neu(0. * u.mA)
        return self.neu.get_spike()


class LeakyReadout(brainstate.nn.Module):
    """Leaky-rate readout for regression / classification."""

    def __init__(self, n_rec: int, n_out: int, tau_o: u.Quantity = 10.0 * u.ms):
        super().__init__()
        import braintrace
        self.readout = braintrace.nn.LeakyRateReadout(
            in_size=n_rec, out_size=n_out, tau=tau_o,
            w_init=braintools.init.KaimingNormal(),
        )

    def update(self, spikes):
        return self.readout(spikes)


# --- Training helpers ----------------------------------------------------


def online_train_epoch(
    model: brainstate.nn.Module,
    opt,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    loss_fn: Callable,
    decay_or_rank=0.95,
    vjp_method: str = "single-step",
) -> float:
    """Run one online-training epoch using pp_prop. Returns mean step loss."""
    import braintrace
    weights = model.states(brainstate.ParamState)
    online_model = braintrace.IODimVjpAlgorithm(
        model, decay_or_rank=decay_or_rank, vjp_method=vjp_method
    )

    @brainstate.transform.vmap_new_states(state_tag="new", axis_size=inputs.shape[1])
    def init():
        brainstate.nn.init_all_states(model)
        online_model.compile_graph(inputs[0, 0])

    init()
    vmap_model = brainstate.nn.Vmap(online_model, vmap_states="new")

    def step_loss(inp, tar):
        out = vmap_model(inp)
        return loss_fn(out, tar), out

    def grad_step(prev_grads, pair):
        inp, tar = pair
        f_grad = brainstate.transform.grad(step_loss, weights, has_aux=True, return_value=True)
        cur_grads, local_loss, _ = f_grad(inp, tar)
        return jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads), local_loss

    init_grads = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})
    grads, step_losses = brainstate.transform.scan(grad_step, init_grads, (inputs, targets))
    grads = brainstate.nn.clip_grad_norm(grads, 1.0)
    opt.update(grads)
    return float(step_losses.mean())


def online_train_epoch_fixed_target(
    model: brainstate.nn.Module,
    opt,
    inputs: jnp.ndarray,
    target_labels: jnp.ndarray,
    decay_or_rank=0.95,
    vjp_method: str = "single-step",
) -> float:
    """Classification variant: fixed label per batch, softmax-xent loss applied each step."""
    import braintrace
    weights = model.states(brainstate.ParamState)
    online_model = braintrace.IODimVjpAlgorithm(
        model, decay_or_rank=decay_or_rank, vjp_method=vjp_method
    )

    @brainstate.transform.vmap_new_states(state_tag="new", axis_size=inputs.shape[1])
    def init():
        brainstate.nn.init_all_states(model)
        online_model.compile_graph(inputs[0, 0])

    init()
    vmap_model = brainstate.nn.Vmap(online_model, vmap_states="new")

    def step_loss(inp):
        out = vmap_model(inp)
        loss = braintools.metric.softmax_cross_entropy_with_integer_labels(
            out, target_labels
        ).mean()
        return loss, out

    def grad_step(prev_grads, inp):
        f_grad = brainstate.transform.grad(step_loss, weights, has_aux=True, return_value=True)
        cur_grads, local_loss, _ = f_grad(inp)
        return jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads), local_loss

    init_grads = jax.tree.map(jnp.zeros_like, {k: v.value for k, v in weights.items()})
    grads, step_losses = brainstate.transform.scan(grad_step, init_grads, inputs)
    grads = brainstate.nn.clip_grad_norm(grads, 1.0)
    opt.update(grads)
    return float(step_losses.mean())


def bptt_train_epoch_fixed_target(
    model: brainstate.nn.Module,
    opt,
    inputs: jnp.ndarray,
    target_labels: jnp.ndarray,
) -> float:
    """BPTT baseline with per-step softmax-cross-entropy over a fixed label."""
    weights = model.states(brainstate.ParamState)

    @brainstate.transform.vmap_new_states(state_tag="new", axis_size=inputs.shape[1])
    def init():
        brainstate.nn.init_all_states(model)

    init()
    vmap_model = brainstate.nn.Vmap(model, vmap_states="new")

    def run_step(inp):
        out = vmap_model(inp)
        loss = braintools.metric.softmax_cross_entropy_with_integer_labels(
            out, target_labels
        ).mean()
        return out, loss

    def bptt_body():
        _, losses = brainstate.transform.for_loop(run_step, inputs)
        return losses.mean()

    grads, loss = brainstate.transform.grad(bptt_body, weights, return_value=True)()
    grads = brainstate.nn.clip_grad_norm(grads, 1.0)
    opt.update(grads)
    return float(loss)


def plot_loss_curve(losses, title: str, save_path: str | None = None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    else:
        plt.show()
    plt.close(fig)
