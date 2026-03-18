#!/usr/bin/env python3
"""
Minimal example for the braintrace.etp package.

Demonstrates:
  1. Defining a recurrent model with etp_matmul (no ETraceParam needed)
  2. Compiling the ETP graph
  3. Inspecting the compiled graph
  4. Composability: jit, grad, vmap all work automatically
"""

import brainstate
import jax
import jax.numpy as jnp

import braintrace.etp as etp


# ──────────────────────────────────────────────────────────
# 1. Define a model using etp_matmul
# ──────────────────────────────────────────────────────────

class VanillaRNN(brainstate.nn.Module):
    """
    Single-layer vanilla RNN built with etp_matmul.

    No ETraceParam, no MatMulOp — just plain ParamState + etp_matmul.
    """

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Weights are plain ParamState — not ETraceParam
        self.W = brainstate.ParamState(
            jax.random.normal(jax.random.key(0), (in_size + hidden_size, hidden_size)) * 0.01
        )
        self.b = brainstate.ParamState(jnp.zeros(hidden_size))

    def init_state(self, batch_size=None, **kwargs):
        self.h = brainstate.HiddenState(jnp.zeros(self.hidden_size))

    def update(self, x):
        xh = jnp.concatenate([x, self.h.value], axis=-1)
        # etp_matmul marks this as an ETP operation
        self.h.value = jax.nn.tanh(etp.matmul(xh, self.W.value, self.b.value))
        return self.h.value


# ──────────────────────────────────────────────────────────
# 2. Compile the ETP graph
# ──────────────────────────────────────────────────────────

in_size = 4
hidden_size = 8

model = VanillaRNN(in_size, hidden_size)
model.init_all_states()

sample_x = jnp.zeros(in_size)
graph = etp.compile_etp_graph(model, sample_x)

print("=== Compiled ETP Graph ===")
print(f"  Hidden groups : {len(graph.hidden_groups)}")
print(f"  ETP relations : {len(graph.etp_op_relations)}")
for rel in graph.etp_op_relations:
    print(f"    {rel.primitive.name}: weight={rel.weight_path}, "
          f"y_shape={rel.y_var.aval.shape}, "
          f"groups={[g.index for g in rel.hidden_groups]}")
print()

# ──────────────────────────────────────────────────────────
# 3. Forward pass — etp_matmul is transparent at runtime
# ──────────────────────────────────────────────────────────

model.init_all_states()
x = jax.random.normal(jax.random.key(1), (in_size,))
h = model(x)
print(f"Forward: x.shape={x.shape} → h.shape={h.shape}")
print(f"  h = {h}")
print()

# ──────────────────────────────────────────────────────────
# 4. Composability: grad works through etp_matmul
# ──────────────────────────────────────────────────────────

model.init_all_states()


def loss_fn(x):
    h = model(x)
    return jnp.sum(h ** 2)


grad_x = jax.grad(loss_fn)(x)
print(f"Grad wrt input: shape={grad_x.shape}")
print(f"  grad_x = {grad_x}")
print()

# ──────────────────────────────────────────────────────────
# 5. Composability: jit wraps everything
# ──────────────────────────────────────────────────────────

model.init_all_states()


@jax.jit
def jit_forward(x):
    return model(x)


h_jit = jit_forward(x)
print(f"JIT forward: h_jit.shape={h_jit.shape}")
print()

# ──────────────────────────────────────────────────────────
# 6. Composability: vmap over batch
# ──────────────────────────────────────────────────────────

batch_x = jax.random.normal(jax.random.key(2), (16, in_size))


# vmap over batch — batching rule on etp_matmul handles this
@brainstate.transform.vmap_new_states(state_tag='batch', axis_size=16)
def init_batched():
    model.init_all_states()


init_batched()
batched_model = brainstate.nn.Vmap(model, vmap_states='batch')
h_batch = batched_model(batch_x)
print(f"Vmap forward: batch_x.shape={batch_x.shape} → h_batch.shape={h_batch.shape}")
print()

# ──────────────────────────────────────────────────────────
# 7. Jaxpr inspection — etp_matmul appears as a primitive
# ──────────────────────────────────────────────────────────

model.init_all_states()
jaxpr = jax.make_jaxpr(model.update)(x)
print("=== Jaxpr ===")
print(jaxpr)
print()

etp_eqns = [eq for eq in jaxpr.jaxpr.eqns if etp.is_etp_primitive(eq.primitive)]
print(f"ETP primitives found in jaxpr: {len(etp_eqns)}")
for eq in etp_eqns:
    print(f"  {eq.primitive.name}: {[str(v) for v in eq.invars]} → {[str(v) for v in eq.outvars]}")
print()


# ──────────────────────────────────────────────────────────
# 8. weight_fn before primitive — chain rule is automatic
# ──────────────────────────────────────────────────────────

class WeightFnRNN(brainstate.nn.Module):
    """RNN with softplus(w) applied before etp_matmul."""

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
        # weight_fn applied as normal JAX op — chain rule handled automatically
        w = jax.nn.softplus(self.raw_W.value)
        self.h.value = jax.nn.tanh(etp.matmul(xh, w))
        return self.h.value


wfn_model = WeightFnRNN(in_size, hidden_size)
wfn_model.init_all_states()

graph_wfn = etp.compile_etp_graph(wfn_model, sample_x)
print("=== WeightFnRNN Graph ===")
print(f"  ETP relations: {len(graph_wfn.etp_op_relations)}")
for rel in graph_wfn.etp_op_relations:
    print(f"    {rel.primitive.name}: weight={rel.weight_path}")

# grad flows through softplus → etp_matmul
wfn_model.init_all_states()
grad_x_wfn = jax.grad(lambda x: wfn_model(x).sum())(x)
print(f"  Grad through weight_fn: shape={grad_x_wfn.shape}")
print()

print("All examples completed successfully.")
