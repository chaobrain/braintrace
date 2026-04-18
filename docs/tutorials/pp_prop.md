# pp_prop — Online Gradient Estimation for Spiking Networks

`braintrace.pp_prop` (exposed also as `braintrace.ES_D_RTRL` and
`braintrace.IODimVjpAlgorithm`) is an online gradient estimator based on
eligibility traces with input-output dimensional complexity. It trains
recurrent spiking networks one time step at a time, without
backpropagation through time.

This tutorial is a narrative companion to `examples/pp_prop/`. Open the
examples side-by-side as you read.

## 1. What pp_prop solves

Backpropagation through time (BPTT) gives exact gradients for recurrent
networks but needs to store the entire forward trajectory in memory. For
long sequences, deep networks, or streaming inputs, the memory cost blows
up.

pp_prop replaces the full Jacobian product by an **eligibility trace** —
a low-pass-filtered approximation of the parameter-to-hidden-state
Jacobian — plus a per-step VJP from the loss to the hidden state:

$$
\nabla_{\boldsymbol\theta} \mathcal L
\approx
\sum_t \frac{\partial \mathcal L^t}{\partial h^t} \odot \boldsymbol\epsilon^t,
\qquad
\boldsymbol\epsilon^t \approx \boldsymbol\epsilon_f^t \otimes \boldsymbol\epsilon_x^t.
$$

The trace factorises into an input-side term and a hidden-side term, each
updated by a low-pass filter with decay $\alpha$:

$$
\boldsymbol\epsilon_x^t = \alpha\,\boldsymbol\epsilon_x^{t-1} + x^t
\qquad
\boldsymbol\epsilon_f^t = \alpha\,\mathrm{diag}(D^t)\circ \boldsymbol\epsilon_f^{t-1}
  + (1-\alpha)\,\mathrm{diag}(D_f^t)
$$

where $D^t, D_f^t$ are local Jacobians produced by the rule registered
for each ETP primitive.

### Complexity

| Quantity | pp_prop (I/O dim) | D_RTRL (param dim) | BPTT |
|----------|-------------------|---------------------|------|
| Memory   | $O(BI + BO)$      | $O(BIO)$            | $O(TBIO)$ |
| Compute  | $O(TBIO)$         | $O(TBIO)$           | $O(TBIO)$ |

For a linear recurrent layer with $n$ hidden units:
pp_prop is $O(Bn)$ memory and $O(Bn^2)$ compute per step.

## 2. When pp_prop beats D_RTRL

Both algorithms are online and avoid BPTT's trajectory storage. pp_prop
saves the extra factor-of-IO memory by replacing the outer-product
eligibility tensor with its rank-1 factorisation. The price is a diagonal
approximation: correlations between rows of the parameter-to-hidden
Jacobian are discarded.

Use pp_prop when:

- Hidden states are large (memory savings matter).
- Online streaming is required (long, unbounded sequences).
- The task's gradient signal does not rely on fine-grained cross-parameter
  correlations.

Fall back to D_RTRL when memory is not the bottleneck and a tighter
gradient estimate is needed. Fall back to BPTT when the sequence is short
enough to fit in memory and you need exact gradients for a published
benchmark.

## 3. Walk-through of the example series

Each paragraph below points at the example and the single axis it is
demonstrating.

### 3.1 Basics (file 01)

`01-basics-lif-integrator.py` is the smallest working pp_prop call.
A one-layer LIF RSNN receives a stream of Poisson spikes; its readout is
trained to match the cumulative spike rate.

### 3.2 Neuron models (files 02-04)

`02-neurons-alif-dms.py` swaps LIF for an ALIF neuron on DMS.
`03-neurons-gif-working-memory.py` introduces GIF with per-neuron slow
adaptation currents. `04-neurons-coba-ei-rsnn.py` builds a Dale-law E/I
block where recurrent signs are fixed by the initialiser. In all three
cases, pp_prop needs no modification — the per-primitive rules in
`braintrace/_etrace_op/` cover the dense matmul gate, and the neuron
dynamics are transparent to the algorithm.

### 3.3 Batching (files 05-06)

`05-batching-vmap.py` uses `brainstate.nn.Vmap(vmap_states='new')` to
replicate the unbatched model across the batch dimension.
`06-batching-batched.py` bypasses vmap by making the network natively
batched — `braintrace.matmul` dispatches to the batched primitive
`etp_mm_p` when the input already has a batch axis.

### 3.4 VJP methods (files 07-08)

`07-vjp-single-step.py` uses the default `vjp_method='single-step'`:
$\partial L^t / \partial h^t$ is computed from the loss at time $t$ only.
`08-vjp-multi-step.py` uses `vjp_method='multi-step'`, which multiplies
the loss gradient by the window of recent hidden-to-hidden Jacobians
before folding it into the trace. Multi-step is strictly more expressive
but runs slower.

### 3.5 Operators (files 09-11)

`09-operator-sparse.py` uses a fixed sparse connectivity mask on the
recurrent matrix. Because `saiunit.sparse` COO/CSR primitives lack JAX
batching rules today, the file uses a masked-dense fallback via
`braintrace.nn.Linear(..., w_mask=...)` that still exercises pp_prop's
per-primitive trace rule.
`10-operator-lora.py` parameterises the recurrent matrix as
$W = \alpha B A$ with rank $r \ll n$, via `braintrace.lora_matmul`.
`11-operator-conv.py` swaps matmul for a 2D convolution via
`braintrace.nn.Conv2d`, demonstrating that pp_prop's eligibility trace is
per-primitive — adding a new operator means writing a new ETP rule, not
changing the algorithm.

### 3.6 Flagship comparison (file 12)

`12-classification-neuromorphic.py` trains two identical LIF RSNNs on
Poisson-encoded 10-class digits, one with pp_prop, one with BPTT. It
reports per-epoch losses and final accuracies.

### 3.7 Knob sweeps (files 13-14)

`13-knob-decay-vs-rank.py` sweeps `decay_or_rank` across both float and
integer parameterisations ($n_{\text{rank}} = 2/(1-\alpha)-1$).
`14-knob-vjp-method-contrast.py` runs single-step pp_prop, multi-step
pp_prop, and BPTT on the same DMS task and plots three loss curves on
one axis.

## 4. Limitations

1. **Diagonal approximation.** The trace factorises as
   $\epsilon_f \otimes \epsilon_x$, which drops off-diagonal couplings.
   Tasks that depend critically on those couplings may learn slower than
   with BPTT.
2. **Single hidden-group assumption per primitive.** If an ETP primitive
   feeds multiple disjoint hidden groups, pp_prop allocates one trace per
   group but relies on the per-primitive rule to handle the summation.
3. **Operator-invariant rule.** Each new operator needs a hand-written
   `xy_to_dw` / `init_pp` rule in `braintrace/_etrace_op/`. See
   `CLAUDE.md` for the "adding a new primitive" recipe.
4. **Weight-through-weight pathways are not supported.** If a trainable
   ETP weight $W_1$ feeds another trainable ETP weight $W_2$ before
   reaching a hidden state, the compiler correctly excludes $W_1$
   (see `CLAUDE.md` "no weight → weight → hidden pathway" invariant).
   Cells like GRU hit this: some of their inner linears are not trained
   by pp_prop and remain non-temporal.

## 5. When to reach for BPTT instead

If your sequences fit in memory, if you can afford the compute, and if
you need exact gradients for a published benchmark — use BPTT. pp_prop is
for the regime where BPTT's memory is the bottleneck. File 12 shows that
on a mid-sized task pp_prop reaches competitive accuracy, but on tasks
where the diagonal approximation loses signal it will lag.

## 6. Further reading

- The `ES-D-RTRL` manuscript:
  [https://www.biorxiv.org/content/10.1101/2024.09.24.614728v2](https://www.biorxiv.org/content/10.1101/2024.09.24.614728v2)
- `braintrace/_etrace_vjp/pp_prop.py` — full docstrings and mathematical
  derivation of the update rules.
- `docs/tutorials/drtrl.md` — the parameter-dimensional dual algorithm
  (D_RTRL / `ParamDimVjpAlgorithm`).
