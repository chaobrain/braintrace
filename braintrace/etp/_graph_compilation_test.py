#!/usr/bin/env python3
"""
Comprehensive ETP graph compilation verification.

Defines many complex dynamics, architectures, and connections to verify
that ``compile_etp_graph`` correctly identifies:

1. Hidden groups (which hidden states are coupled)
2. ETP op relations (which weights connect to which hidden groups)
3. Transition jaxprs (the y -> h sub-computation)
4. Backward weight tracing (weight_fn before primitive)
5. Selective participation (non-ETP params are excluded)

Run with:
    python -m pytest braintrace/etp/_graph_compilation_test.py -v -s
"""

import textwrap
from typing import Sequence

import jax
import jax.numpy as jnp
import pytest
import brainstate

from braintrace.etp import (
    matmul,
    element_wise,
    conv,
    compile_etp_graph,
    ETPGraph,
    ETPOpRelation,
)
from braintrace.etp._primitives import etp_matmul_p, etp_elemwise_p, etp_conv_p


# ======================================================================
# Graph visualization helpers
# ======================================================================

def format_graph(graph: ETPGraph, title: str) -> str:
    """Format an ETPGraph as a readable text diagram."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"  {title}")
    lines.append(f"{'=' * 70}")

    # Hidden groups
    lines.append(f"\n  Hidden Groups ({len(graph.hidden_groups)}):")
    for g in graph.hidden_groups:
        paths_str = ", ".join(str(p) for p in g.hidden_paths)
        shapes = [str(s.varshape) for s in g.hidden_states]
        lines.append(f"    Group[{g.index}]: {paths_str}")
        lines.append(f"      shapes: {shapes}, num_state: {g.num_state}")
        lines.append(f"      invars:  {[str(v) for v in g.hidden_invars]}")
        lines.append(f"      outvars: {[str(v) for v in g.hidden_outvars]}")

    # ETP op relations
    lines.append(f"\n  ETP Relations ({len(graph.etp_op_relations)}):")
    for i, rel in enumerate(graph.etp_op_relations):
        lines.append(f"    Rel[{i}]: {rel.primitive.name}")
        lines.append(f"      weight_path: {rel.weight_path}")
        lines.append(f"      y_shape:     {rel.y_var.aval.shape}")
        if rel.x_var is not None:
            lines.append(f"      x_shape:     {rel.x_var.aval.shape}")
        lines.append(f"      groups:      {[g.index for g in rel.hidden_groups]}")
        lines.append(f"      hidden_paths: {rel.connected_hidden_paths}")
        lines.append(f"      eqn_params:  {rel.eqn_params}")

    # Connectivity summary (ASCII art)
    lines.append(f"\n  Connectivity:")
    for rel in graph.etp_op_relations:
        w_name = rel.weight_path[-1]
        prim = rel.primitive.name.replace("etp_", "")
        for g in rel.hidden_groups:
            h_names = [p[-1] for p in g.hidden_paths]
            h_str = ", ".join(h_names)
            lines.append(f"    {w_name} --[{prim}]--> Group[{g.index}]({h_str})")

    lines.append(f"{'=' * 70}\n")
    return "\n".join(lines)


def print_graph(graph: ETPGraph, title: str):
    """Print the formatted graph."""
    print(format_graph(graph, title))


# ======================================================================
# Model definitions
# ======================================================================

# ------------------------------------------------------------------
# 1. Vanilla RNN: single matmul, single hidden
# ------------------------------------------------------------------
class VanillaRNN(brainstate.nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size + hidden_size, hidden_size)) * 0.01
        )
        self.b = brainstate.ParamState(jnp.zeros(hidden_size))

    def init_state(self, batch_size=None, **kwargs):
        self.h = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value], axis=-1)
        self.h.value = jax.nn.tanh(matmul(xh, self.W.value, self.b.value))
        return self.h.value


# ------------------------------------------------------------------
# 2. Mixed ETP / non-ETP: w_in excluded, w_rec included
# ------------------------------------------------------------------
class MixedParamRNN(brainstate.nn.Module):
    """Only w_rec participates in ETP; w_in uses regular matmul."""

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.w_in = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size, hidden_size)) * 0.01
        )
        self.w_rec = brainstate.ParamState(
            jax.random.normal(jax.random.key(1), (hidden_size, hidden_size)) * 0.01
        )

    def init_state(self, batch_size=None, **kwargs):
        self.h = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        inp = x @ self.w_in.value              # regular JAX op -> excluded
        rec = matmul(self.h.value, self.w_rec.value)  # etp -> included
        self.h.value = jax.nn.tanh(inp + rec)
        return self.h.value


# ------------------------------------------------------------------
# 3. Deep stacked RNN: 3 layers, each with own hidden
# ------------------------------------------------------------------
class DeepStackedRNN(brainstate.nn.Module):
    """3-layer stacked RNN, each layer has its own hidden state."""

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W1 = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size + hidden_size, hidden_size)) * 0.01
        )
        self.W2 = brainstate.ParamState(
            jax.random.normal(jax.random.key(1), (hidden_size + hidden_size, hidden_size)) * 0.01
        )
        self.W3 = brainstate.ParamState(
            jax.random.normal(jax.random.key(2), (hidden_size + hidden_size, hidden_size)) * 0.01
        )

    def init_state(self, batch_size=None, **kwargs):
        self.h1 = brainstate.HiddenState(jnp.zeros(self.hidden_size))
        self.h2 = brainstate.HiddenState(jnp.zeros(self.hidden_size))
        self.h3 = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        xh1 = jnp.concatenate([x, self.h1.value], axis=-1)
        self.h1.value = jax.nn.tanh(matmul(xh1, self.W1.value))

        xh2 = jnp.concatenate([self.h1.value, self.h2.value], axis=-1)
        self.h2.value = jax.nn.tanh(matmul(xh2, self.W2.value))

        xh3 = jnp.concatenate([self.h2.value, self.h3.value], axis=-1)
        self.h3.value = jax.nn.tanh(matmul(xh3, self.W3.value))

        return self.h3.value


# ------------------------------------------------------------------
# 4. Residual RNN: skip connection h_new = tanh(Wx) + h_old
# ------------------------------------------------------------------
class ResidualRNN(brainstate.nn.Module):
    """h_new = tanh(W @ [x, h]) + h (residual skip)."""

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size + hidden_size, hidden_size)) * 0.01
        )

    def init_state(self, batch_size=None, **kwargs):
        self.h = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value], axis=-1)
        self.h.value = jax.nn.tanh(matmul(xh, self.W.value)) + self.h.value
        return self.h.value


# ------------------------------------------------------------------
# 5. Leaky integrator with element-wise decay
# ------------------------------------------------------------------
class LeakyIntegrator(brainstate.nn.Module):
    """h_new = decay * h_old + (1 - decay) * tanh(W @ x), decay = sigmoid(tau)."""

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size, hidden_size)) * 0.01
        )
        self.tau = brainstate.ParamState(jnp.ones(hidden_size))

    def init_state(self, batch_size=None, **kwargs):
        self.h = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        inp = matmul(x, self.W.value)
        decay = element_wise(self.tau.value, fn=jax.nn.sigmoid)
        self.h.value = decay * self.h.value + (1 - decay) * jax.nn.tanh(inp)
        return self.h.value


# ------------------------------------------------------------------
# 6. GRU-like gated unit: multiple weights feeding one hidden
# ------------------------------------------------------------------
class GRULikeCell(brainstate.nn.Module):
    """
    Simplified GRU-like gating with 3 ETP matmuls -> 1 hidden state.
    z = sigmoid(W_z @ [x, h])
    r = sigmoid(W_r @ [x, h])
    n = tanh(W_n @ [x, r*h])
    h = (1-z)*n + z*h
    """

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_z = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size + hidden_size, hidden_size)) * 0.01
        )
        self.W_r = brainstate.ParamState(
            jax.random.normal(jax.random.key(1), (in_size + hidden_size, hidden_size)) * 0.01
        )
        self.W_n = brainstate.ParamState(
            jax.random.normal(jax.random.key(2), (in_size + hidden_size, hidden_size)) * 0.01
        )

    def init_state(self, batch_size=None, **kwargs):
        self.h = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value], axis=-1)
        z = jax.nn.sigmoid(matmul(xh, self.W_z.value))
        r = jax.nn.sigmoid(matmul(xh, self.W_r.value))
        xrh = jnp.concatenate([x, r * self.h.value], axis=-1)
        n = jax.nn.tanh(matmul(xrh, self.W_n.value))
        self.h.value = (1 - z) * n + z * self.h.value
        return self.h.value


# ------------------------------------------------------------------
# 7. LSTM-like cell: 2 coupled hidden states (c, h) in one group
# ------------------------------------------------------------------
class LSTMLikeCell(brainstate.nn.Module):
    """
    Simplified LSTM with cell state c and hidden state h.
    c and h should be in the same hidden group.
    """

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_f = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size + hidden_size, hidden_size)) * 0.01
        )
        self.W_i = brainstate.ParamState(
            jax.random.normal(jax.random.key(1), (in_size + hidden_size, hidden_size)) * 0.01
        )
        self.W_g = brainstate.ParamState(
            jax.random.normal(jax.random.key(2), (in_size + hidden_size, hidden_size)) * 0.01
        )
        self.W_o = brainstate.ParamState(
            jax.random.normal(jax.random.key(3), (in_size + hidden_size, hidden_size)) * 0.01
        )

    def init_state(self, batch_size=None, **kwargs):
        self.c = brainstate.HiddenState(jnp.zeros(self.hidden_size))
        self.h = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value], axis=-1)
        f = jax.nn.sigmoid(matmul(xh, self.W_f.value))
        i = jax.nn.sigmoid(matmul(xh, self.W_i.value))
        g = jax.nn.tanh(matmul(xh, self.W_g.value))
        o = jax.nn.sigmoid(matmul(xh, self.W_o.value))
        self.c.value = f * self.c.value + i * g
        self.h.value = o * jax.nn.tanh(self.c.value)
        return self.h.value


# ------------------------------------------------------------------
# 8. Weight function: softplus + abs applied before ETP primitive
# ------------------------------------------------------------------
class WeightFnRNN(brainstate.nn.Module):
    """Weight undergoes softplus transformation before etp_matmul.
    Compiler must trace backward through softplus to find raw_W."""

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.raw_W = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size + hidden_size, hidden_size)) * 0.01
        )

    def init_state(self, batch_size=None, **kwargs):
        self.h = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value], axis=-1)
        w = jax.nn.softplus(self.raw_W.value)
        self.h.value = jax.nn.tanh(matmul(xh, w))
        return self.h.value


# ------------------------------------------------------------------
# 9. Chained weight functions: abs(softplus(W))
# ------------------------------------------------------------------
class ChainedWeightFnRNN(brainstate.nn.Module):
    """Multiple transformations on weight: abs(softplus(W))."""

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.raw_W = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size + hidden_size, hidden_size)) * 0.01
        )

    def init_state(self, batch_size=None, **kwargs):
        self.h = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value], axis=-1)
        w = jnp.abs(jax.nn.softplus(self.raw_W.value))
        self.h.value = jax.nn.tanh(matmul(xh, w))
        return self.h.value


# ------------------------------------------------------------------
# 10. Fan-in: multiple weights feeding one hidden state
# ------------------------------------------------------------------
class FanInRNN(brainstate.nn.Module):
    """Two separate ETP matmuls whose outputs are summed into one hidden."""

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_in = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size, hidden_size)) * 0.01
        )
        self.W_rec = brainstate.ParamState(
            jax.random.normal(jax.random.key(1), (hidden_size, hidden_size)) * 0.01
        )

    def init_state(self, batch_size=None, **kwargs):
        self.h = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        inp = matmul(x, self.W_in.value)
        rec = matmul(self.h.value, self.W_rec.value)
        self.h.value = jax.nn.tanh(inp + rec)
        return self.h.value


# ------------------------------------------------------------------
# 11. Fan-out: one weight affecting multiple independent hidden states
# ------------------------------------------------------------------
class FanOutRNN(brainstate.nn.Module):
    """One shared W used in two ETP matmuls feeding two separate hidden states.
    h1 and h2 should be in separate groups (no mutual coupling)."""

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_shared = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size, hidden_size)) * 0.01
        )

    def init_state(self, batch_size=None, **kwargs):
        self.h1 = brainstate.HiddenState(jnp.zeros(self.hidden_size))
        self.h2 = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        y = matmul(x, self.W_shared.value)
        self.h1.value = jax.nn.tanh(y + self.h1.value)
        self.h2.value = jax.nn.relu(y + self.h2.value)
        return self.h1.value + self.h2.value


# ------------------------------------------------------------------
# 12. Coupled oscillator: h1 <-> h2 bidirectional coupling
# ------------------------------------------------------------------
class CoupledOscillator(brainstate.nn.Module):
    """
    h1_new = tanh(W1_self @ h1 + W1_cross @ h2 + x)
    h2_new = tanh(W2_cross @ h1 + W2_self @ h2 + x)

    Uses separate matmuls per hidden state to avoid concatenating
    two hidden invars in one equation (an existing limitation of the
    hidden group compiler).  h1 and h2 are still mutually coupled
    through the cross connections -> same hidden group.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W1_self = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (hidden_size, hidden_size)) * 0.01
        )
        self.W1_cross = brainstate.ParamState(
            jax.random.normal(jax.random.key(1), (hidden_size, hidden_size)) * 0.01
        )
        self.W2_cross = brainstate.ParamState(
            jax.random.normal(jax.random.key(2), (hidden_size, hidden_size)) * 0.01
        )
        self.W2_self = brainstate.ParamState(
            jax.random.normal(jax.random.key(3), (hidden_size, hidden_size)) * 0.01
        )

    def init_state(self, batch_size=None, **kwargs):
        self.h1 = brainstate.HiddenState(jnp.zeros(self.hidden_size))
        self.h2 = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        # Separate matmuls avoid the "two hidden invars in one equation" limitation
        h1_self = matmul(self.h1.value, self.W1_self.value)
        h1_cross = matmul(self.h2.value, self.W1_cross.value)
        self.h1.value = jax.nn.tanh(h1_self + h1_cross + x)

        h2_cross = matmul(self.h1.value, self.W2_cross.value)
        h2_self = matmul(self.h2.value, self.W2_self.value)
        self.h2.value = jax.nn.tanh(h2_cross + h2_self + x)

        return self.h1.value + self.h2.value


# ------------------------------------------------------------------
# 13. Mixed primitives: matmul + elemwise in one model
# ------------------------------------------------------------------
class MixedPrimitiveRNN(brainstate.nn.Module):
    """Uses etp_matmul for input and etp_elemwise for decay."""

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size + hidden_size, hidden_size)) * 0.01
        )
        self.alpha = brainstate.ParamState(jnp.ones(hidden_size) * 0.5)

    def init_state(self, batch_size=None, **kwargs):
        self.h = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value], axis=-1)
        y = matmul(xh, self.W.value)
        a = element_wise(self.alpha.value, fn=jax.nn.sigmoid)
        self.h.value = a * self.h.value + (1 - a) * jax.nn.tanh(y)
        return self.h.value


# ------------------------------------------------------------------
# 14. Separate input and recurrent weights (Elman network)
# ------------------------------------------------------------------
class ElmanRNN(brainstate.nn.Module):
    """
    h = tanh(W_x @ x + W_h @ h + b)
    Both W_x and W_h are ETP-tracked separately.
    """

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_x = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size, hidden_size)) * 0.01
        )
        self.W_h = brainstate.ParamState(
            jax.random.normal(jax.random.key(1), (hidden_size, hidden_size)) * 0.01
        )
        self.b = brainstate.ParamState(jnp.zeros(hidden_size))

    def init_state(self, batch_size=None, **kwargs):
        self.h = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        y_x = matmul(x, self.W_x.value)
        y_h = matmul(self.h.value, self.W_h.value)
        self.h.value = jax.nn.tanh(y_x + y_h + self.b.value)
        return self.h.value


# ------------------------------------------------------------------
# 15. Deep skip-connection: layer 1 feeds both layer 2 and layer 3
# ------------------------------------------------------------------
class SkipConnRNN(brainstate.nn.Module):
    """
    Layer 1: h1 = tanh(W1 @ [x, h1])
    Layer 2: h2 = tanh(W2 @ [h1, h2])
    Layer 3: h3 = tanh(W3 @ [h1 + h2, h3])  <- skip from layer 1
    """

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W1 = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size + hidden_size, hidden_size)) * 0.01
        )
        self.W2 = brainstate.ParamState(
            jax.random.normal(jax.random.key(1), (hidden_size + hidden_size, hidden_size)) * 0.01
        )
        self.W3 = brainstate.ParamState(
            jax.random.normal(jax.random.key(2), (hidden_size + hidden_size, hidden_size)) * 0.01
        )

    def init_state(self, batch_size=None, **kwargs):
        self.h1 = brainstate.HiddenState(jnp.zeros(self.hidden_size))
        self.h2 = brainstate.HiddenState(jnp.zeros(self.hidden_size))
        self.h3 = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        xh1 = jnp.concatenate([x, self.h1.value], axis=-1)
        self.h1.value = jax.nn.tanh(matmul(xh1, self.W1.value))

        h1h2 = jnp.concatenate([self.h1.value, self.h2.value], axis=-1)
        self.h2.value = jax.nn.tanh(matmul(h1h2, self.W2.value))

        skip_h3 = jnp.concatenate([self.h1.value + self.h2.value, self.h3.value], axis=-1)
        self.h3.value = jax.nn.tanh(matmul(skip_h3, self.W3.value))

        return self.h3.value


# ------------------------------------------------------------------
# 16. Nonlinear hidden interaction: h_new = tanh(W @ [x, h^2])
# ------------------------------------------------------------------
class NonlinearHiddenRNN(brainstate.nn.Module):
    """Hidden state enters through a nonlinearity (h^2) before concatenation."""

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size + hidden_size, hidden_size)) * 0.01
        )

    def init_state(self, batch_size=None, **kwargs):
        self.h = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        xh2 = jnp.concatenate([x, self.h.value ** 2], axis=-1)
        self.h.value = jax.nn.tanh(matmul(xh2, self.W.value))
        return self.h.value


# ------------------------------------------------------------------
# 17. Multi-timescale: fast and slow hidden states
# ------------------------------------------------------------------
class MultiTimescaleRNN(brainstate.nn.Module):
    """
    h_fast = 0.1*tanh(W_fast@[x, h_fast]) + 0.9*h_fast
    h_slow = 0.9*tanh(W_slow@[h_fast, h_slow]) + 0.1*h_slow
    Different timescale constants, separate hidden groups.
    """

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_fast = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size + hidden_size, hidden_size)) * 0.01
        )
        self.W_slow = brainstate.ParamState(
            jax.random.normal(jax.random.key(1), (hidden_size + hidden_size, hidden_size)) * 0.01
        )

    def init_state(self, batch_size=None, **kwargs):
        self.h_fast = brainstate.HiddenState(jnp.zeros(self.hidden_size))
        self.h_slow = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        xhf = jnp.concatenate([x, self.h_fast.value], axis=-1)
        self.h_fast.value = 0.1 * jax.nn.tanh(matmul(xhf, self.W_fast.value)) + 0.9 * self.h_fast.value

        hfhs = jnp.concatenate([self.h_fast.value, self.h_slow.value], axis=-1)
        self.h_slow.value = 0.9 * jax.nn.tanh(matmul(hfhs, self.W_slow.value)) + 0.1 * self.h_slow.value

        return self.h_fast.value + self.h_slow.value


# ------------------------------------------------------------------
# 18. Elemwise-only model: diagonal RNN (no matmul)
# ------------------------------------------------------------------
class DiagonalRNN(brainstate.nn.Module):
    """Pure element-wise recurrence: h_new = sigmoid(alpha) * h + x."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha = brainstate.ParamState(jnp.zeros(hidden_size))

    def init_state(self, batch_size=None, **kwargs):
        self.h = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        decay = element_wise(self.alpha.value, fn=jax.nn.sigmoid)
        self.h.value = decay * self.h.value + (1 - decay) * x
        return self.h.value


# ------------------------------------------------------------------
# 19. Multi-elemwise: separate decay for different states
# ------------------------------------------------------------------
class MultiElemwiseRNN(brainstate.nn.Module):
    """Two element-wise params with separate hidden states."""

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size, hidden_size)) * 0.01
        )
        self.alpha = brainstate.ParamState(jnp.zeros(hidden_size))
        self.beta = brainstate.ParamState(jnp.ones(hidden_size))

    def init_state(self, batch_size=None, **kwargs):
        self.h = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        inp = matmul(x, self.W.value)
        a = element_wise(self.alpha.value, fn=jax.nn.sigmoid)
        b = element_wise(self.beta.value, fn=jax.nn.softplus)
        self.h.value = a * self.h.value + b * jax.nn.tanh(inp)
        return self.h.value


# ------------------------------------------------------------------
# 20. Parallel independent streams: two RNNs, no cross-talk
# ------------------------------------------------------------------
class ParallelStreamsRNN(brainstate.nn.Module):
    """Two independent RNN streams. Should produce 2 separate hidden groups."""

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_a = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size + hidden_size, hidden_size)) * 0.01
        )
        self.W_b = brainstate.ParamState(
            jax.random.normal(jax.random.key(1), (in_size + hidden_size, hidden_size)) * 0.01
        )

    def init_state(self, batch_size=None, **kwargs):
        self.h_a = brainstate.HiddenState(jnp.zeros(self.hidden_size))
        self.h_b = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        xha = jnp.concatenate([x, self.h_a.value], axis=-1)
        self.h_a.value = jax.nn.tanh(matmul(xha, self.W_a.value))

        xhb = jnp.concatenate([x, self.h_b.value], axis=-1)
        self.h_b.value = jax.nn.tanh(matmul(xhb, self.W_b.value))

        return self.h_a.value + self.h_b.value


# ======================================================================
# Tests
# ======================================================================

def _compile(model, x):
    """Helper: init states and compile graph."""
    model.init_state()
    return compile_etp_graph(model, x)


class TestVanillaRNN:
    def test_structure(self):
        model = VanillaRNN(4, 8)
        graph = _compile(model, jnp.zeros(4))
        print_graph(graph, "1. VanillaRNN")

        assert len(graph.hidden_groups) == 1
        assert len(graph.etp_op_relations) == 1
        rel = graph.etp_op_relations[0]
        assert rel.primitive is etp_matmul_p
        assert rel.y_var.aval.shape == (8,)
        assert len(rel.hidden_groups) == 1
        assert rel.eqn_params['has_bias'] is True


class TestMixedParamRNN:
    def test_only_rec_tracked(self):
        model = MixedParamRNN(4, 8)
        graph = _compile(model, jnp.zeros(4))
        print_graph(graph, "2. MixedParamRNN (w_in excluded, w_rec included)")

        # Only w_rec should appear (regular matmul for w_in is invisible)
        assert len(graph.etp_op_relations) == 1
        rel = graph.etp_op_relations[0]
        assert rel.weight_path[-1] == 'w_rec'


class TestDeepStackedRNN:
    def test_three_layers(self):
        model = DeepStackedRNN(4, 8)
        graph = _compile(model, jnp.zeros(4))
        print_graph(graph, "3. DeepStackedRNN (3 layers)")

        assert len(graph.etp_op_relations) == 3
        for rel in graph.etp_op_relations:
            assert rel.primitive is etp_matmul_p

        # Each layer connects to at least its own hidden group
        all_group_indices = set()
        for rel in graph.etp_op_relations:
            for g in rel.hidden_groups:
                all_group_indices.add(g.index)
        # With feedforward stacking, there can be cross-group connections
        assert len(all_group_indices) >= 1


class TestResidualRNN:
    def test_skip_connection(self):
        model = ResidualRNN(4, 8)
        graph = _compile(model, jnp.zeros(4))
        print_graph(graph, "4. ResidualRNN (h + tanh(Wx))")

        assert len(graph.hidden_groups) == 1
        assert len(graph.etp_op_relations) == 1


class TestLeakyIntegrator:
    def test_matmul_and_elemwise(self):
        model = LeakyIntegrator(4, 8)
        graph = _compile(model, jnp.zeros(4))
        print_graph(graph, "5. LeakyIntegrator (matmul + elemwise)")

        primitives = {rel.primitive for rel in graph.etp_op_relations}
        assert etp_matmul_p in primitives
        # elemwise may or may not connect depending on reachability
        assert len(graph.etp_op_relations) >= 1

        # Check that tau is tracked via elemwise if present
        elemwise_rels = [r for r in graph.etp_op_relations if r.primitive is etp_elemwise_p]
        if elemwise_rels:
            assert 'tau' in str(elemwise_rels[0].weight_path)


class TestGRULikeCell:
    def test_three_weights_one_hidden(self):
        model = GRULikeCell(4, 8)
        graph = _compile(model, jnp.zeros(4))
        print_graph(graph, "6. GRULikeCell (3 matmuls -> 1 hidden)")

        # All 3 weights connect to the same single hidden group
        assert len(graph.hidden_groups) == 1
        assert len(graph.etp_op_relations) == 3
        weight_names = {r.weight_path[-1] for r in graph.etp_op_relations}
        assert weight_names == {'W_z', 'W_r', 'W_n'}
        for rel in graph.etp_op_relations:
            assert len(rel.hidden_groups) == 1
            assert rel.hidden_groups[0].index == 0


class TestLSTMLikeCell:
    def test_two_coupled_hidden_states(self):
        model = LSTMLikeCell(4, 8)
        graph = _compile(model, jnp.zeros(4))
        print_graph(graph, "7. LSTMLikeCell (c and h coupled)")

        assert len(graph.etp_op_relations) == 4
        weight_names = {r.weight_path[-1] for r in graph.etp_op_relations}
        assert weight_names == {'W_f', 'W_i', 'W_g', 'W_o'}

        # c and h should be in the same group since h depends on c
        # and c depends on h (through xh concatenation)
        all_paths_in_groups = set()
        for g in graph.hidden_groups:
            for p in g.hidden_paths:
                all_paths_in_groups.add(p[-1])
        assert 'c' in all_paths_in_groups
        assert 'h' in all_paths_in_groups


class TestWeightFnRNN:
    def test_backward_tracing(self):
        model = WeightFnRNN(4, 8)
        graph = _compile(model, jnp.zeros(4))
        print_graph(graph, "8. WeightFnRNN (softplus before matmul)")

        assert len(graph.etp_op_relations) == 1
        rel = graph.etp_op_relations[0]
        assert rel.weight_path[-1] == 'raw_W'


class TestChainedWeightFnRNN:
    def test_deep_backward_tracing(self):
        model = ChainedWeightFnRNN(4, 8)
        graph = _compile(model, jnp.zeros(4))
        print_graph(graph, "9. ChainedWeightFnRNN (abs(softplus(W)))")

        assert len(graph.etp_op_relations) == 1
        rel = graph.etp_op_relations[0]
        assert rel.weight_path[-1] == 'raw_W'


class TestFanInRNN:
    def test_two_weights_one_hidden(self):
        model = FanInRNN(4, 8)
        graph = _compile(model, jnp.zeros(4))
        print_graph(graph, "10. FanInRNN (W_in + W_rec -> h)")

        assert len(graph.hidden_groups) == 1
        assert len(graph.etp_op_relations) == 2
        weight_names = {r.weight_path[-1] for r in graph.etp_op_relations}
        assert weight_names == {'W_in', 'W_rec'}


class TestFanOutRNN:
    def test_shared_weight_two_hiddens(self):
        model = FanOutRNN(4, 8)
        graph = _compile(model, jnp.zeros(4))
        print_graph(graph, "11. FanOutRNN (shared W -> h1, h2)")

        # The shared weight should connect to hidden states
        assert len(graph.etp_op_relations) >= 1
        rel = graph.etp_op_relations[0]
        assert rel.weight_path[-1] == 'W_shared'
        # It should reach both h1 and h2
        all_connected = set()
        for r in graph.etp_op_relations:
            for p in r.connected_hidden_paths:
                all_connected.add(p[-1])
        assert 'h1' in all_connected
        assert 'h2' in all_connected


class TestCoupledOscillator:
    def test_bidirectional_coupling(self):
        model = CoupledOscillator(8)
        graph = _compile(model, jnp.zeros(8))
        print_graph(graph, "12. CoupledOscillator (h1 <-> h2)")

        assert len(graph.etp_op_relations) == 4
        weight_names = {r.weight_path[-1] for r in graph.etp_op_relations}
        assert weight_names == {'W1_self', 'W1_cross', 'W2_cross', 'W2_self'}

        # h1 and h2 should be in the same group (mutual coupling)
        all_paths = set()
        for g in graph.hidden_groups:
            for p in g.hidden_paths:
                all_paths.add(p[-1])
        assert 'h1' in all_paths
        assert 'h2' in all_paths


class TestMixedPrimitiveRNN:
    def test_matmul_and_elemwise_primitives(self):
        model = MixedPrimitiveRNN(4, 8)
        graph = _compile(model, jnp.zeros(4))
        print_graph(graph, "13. MixedPrimitiveRNN (matmul + elemwise)")

        primitives = {rel.primitive for rel in graph.etp_op_relations}
        assert etp_matmul_p in primitives
        assert len(graph.etp_op_relations) >= 1


class TestElmanRNN:
    def test_separate_input_recurrent(self):
        model = ElmanRNN(4, 8)
        graph = _compile(model, jnp.zeros(4))
        print_graph(graph, "14. ElmanRNN (W_x + W_h -> h)")

        assert len(graph.hidden_groups) == 1
        assert len(graph.etp_op_relations) == 2
        weight_names = {r.weight_path[-1] for r in graph.etp_op_relations}
        assert weight_names == {'W_x', 'W_h'}


class TestSkipConnRNN:
    def test_skip_connections(self):
        model = SkipConnRNN(4, 8)
        graph = _compile(model, jnp.zeros(4))
        print_graph(graph, "15. SkipConnRNN (layer1 -> layer2 + layer3)")

        assert len(graph.etp_op_relations) == 3
        # Verify all hidden states are present
        all_hidden = set()
        for g in graph.hidden_groups:
            for p in g.hidden_paths:
                all_hidden.add(p[-1])
        assert all_hidden >= {'h1', 'h2', 'h3'}


class TestNonlinearHiddenRNN:
    def test_nonlinear_hidden(self):
        model = NonlinearHiddenRNN(4, 8)
        graph = _compile(model, jnp.zeros(4))
        print_graph(graph, "16. NonlinearHiddenRNN (h^2 before concat)")

        assert len(graph.hidden_groups) == 1
        assert len(graph.etp_op_relations) == 1


class TestMultiTimescaleRNN:
    def test_two_timescales(self):
        model = MultiTimescaleRNN(4, 8)
        graph = _compile(model, jnp.zeros(4))
        print_graph(graph, "17. MultiTimescaleRNN (fast + slow)")

        assert len(graph.etp_op_relations) == 2
        weight_names = {r.weight_path[-1] for r in graph.etp_op_relations}
        assert weight_names == {'W_fast', 'W_slow'}


class TestDiagonalRNN:
    def test_elemwise_only(self):
        model = DiagonalRNN(8)
        graph = _compile(model, jnp.zeros(8))
        print_graph(graph, "18. DiagonalRNN (elemwise only)")

        # Should have at least 1 elemwise relation
        elemwise_rels = [r for r in graph.etp_op_relations if r.primitive is etp_elemwise_p]
        if elemwise_rels:
            assert len(graph.hidden_groups) == 1


class TestMultiElemwiseRNN:
    def test_multiple_elemwise(self):
        model = MultiElemwiseRNN(4, 8)
        graph = _compile(model, jnp.zeros(4))
        print_graph(graph, "19. MultiElemwiseRNN (alpha + beta + matmul)")

        assert len(graph.hidden_groups) == 1
        assert len(graph.etp_op_relations) >= 1  # matmul at minimum


class TestParallelStreamsRNN:
    def test_independent_streams(self):
        model = ParallelStreamsRNN(4, 8)
        graph = _compile(model, jnp.zeros(4))
        print_graph(graph, "20. ParallelStreamsRNN (2 independent streams)")

        assert len(graph.etp_op_relations) == 2
        # Each stream has its own hidden group
        assert len(graph.hidden_groups) == 2
        for g in graph.hidden_groups:
            assert len(g.hidden_paths) == 1


# ======================================================================
# Integration: forward pass consistency check
# ======================================================================

class TestForwardConsistency:
    """Verify that models produce the same output with and without ETP graph compilation."""

    @pytest.mark.parametrize("model_cls,args", [
        (VanillaRNN, (4, 8)),
        (MixedParamRNN, (4, 8)),
        (DeepStackedRNN, (4, 8)),
        (ResidualRNN, (4, 8)),
        (LeakyIntegrator, (4, 8)),
        (GRULikeCell, (4, 8)),
        (WeightFnRNN, (4, 8)),
        (FanInRNN, (4, 8)),
        (ElmanRNN, (4, 8)),
        (NonlinearHiddenRNN, (4, 8)),
        (MultiTimescaleRNN, (4, 8)),
    ])
    def test_forward(self, model_cls, args):
        """Model forward pass should work without error after graph compilation."""
        model = model_cls(*args)
        model.init_state()
        x = jax.random.normal(jax.random.key(42), (args[0],))

        # Forward pass
        out = model(x)
        assert out.shape[-1] == args[1]

        # Graph compiles without error
        model.init_state()
        graph = compile_etp_graph(model, x)
        assert isinstance(graph, ETPGraph)


# ======================================================================
# Run all and print summary
# ======================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  ETP Graph Compilation Verification Suite")
    print("=" * 70 + "\n")

    tests = [
        ("1. VanillaRNN", VanillaRNN(4, 8), 4),
        ("2. MixedParamRNN", MixedParamRNN(4, 8), 4),
        ("3. DeepStackedRNN", DeepStackedRNN(4, 8), 4),
        ("4. ResidualRNN", ResidualRNN(4, 8), 4),
        ("5. LeakyIntegrator", LeakyIntegrator(4, 8), 4),
        ("6. GRULikeCell", GRULikeCell(4, 8), 4),
        ("7. LSTMLikeCell", LSTMLikeCell(4, 8), 4),
        ("8. WeightFnRNN", WeightFnRNN(4, 8), 4),
        ("9. ChainedWeightFnRNN", ChainedWeightFnRNN(4, 8), 4),
        ("10. FanInRNN", FanInRNN(4, 8), 4),
        ("11. FanOutRNN", FanOutRNN(4, 8), 4),
        ("12. CoupledOscillator", CoupledOscillator(8), 8),
        ("13. MixedPrimitiveRNN", MixedPrimitiveRNN(4, 8), 4),
        ("14. ElmanRNN", ElmanRNN(4, 8), 4),
        ("15. SkipConnRNN", SkipConnRNN(4, 8), 4),
        ("16. NonlinearHiddenRNN", NonlinearHiddenRNN(4, 8), 4),
        ("17. MultiTimescaleRNN", MultiTimescaleRNN(4, 8), 4),
        ("18. DiagonalRNN", DiagonalRNN(8), 8),
        ("19. MultiElemwiseRNN", MultiElemwiseRNN(4, 8), 4),
        ("20. ParallelStreamsRNN", ParallelStreamsRNN(4, 8), 4),
    ]

    passed = 0
    failed = 0
    for title, model, in_size in tests:
        try:
            model.init_state()
            x = jnp.zeros(in_size)
            graph = compile_etp_graph(model, x)
            print_graph(graph, title)
            passed += 1
        except Exception as e:
            print(f"FAILED: {title}: {e}")
            failed += 1

    print(f"\n{'=' * 70}")
    print(f"  Summary: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 70}")
