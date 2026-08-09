# P4 — UORO, three-factor and DNI

Status: spec, revised after review
Roadmap: [`2026-07-25-algorithm-axes-roadmap.md`](2026-07-25-algorithm-axes-roadmap.md) § P4
Baseline: commit `77e44cd` (P3 landed)
Target release: 0.2.5

## Goal

Deliver the three axis values the vocabulary reserves but `_UNIMPLEMENTED`
rejects:

| Coordinate | Rule | What it adds that nothing in-tree has |
|---|---|---|
| `trace_factorization='random_projection'` | UORO (Tallec & Ollivier 2018) | an **unbiased** trace — every existing coordinate is biased |
| `learning_signal='modulatory'` | three-factor | a learning signal that is *not* a loss derivative |
| `learning_signal='bootstrapped'` | DNI (Jaderberg 2017) | a *learned* estimate of the future-loss term the truncation drops |

The three are independent in the axis space and nearly independent in the code:
UORO is a new trace engine, `modulatory` is a new branch of
`_compute_learning_signal`, `bootstrapped` is a new *separate* hook on the window
cotangent. They ship in one phase because they share the statistical and
degenerate-equality acceptance machinery, not because they interact. Internally
they land in the order Part 1 → Part 2 → Part 3, each behind its own acceptance
set, because Part 3 depends on Part 2's audit of what the multi-step path already
adds to an ETP gradient.

**All three new branches live in `vjp_base.py`, not in the presets.**
`_compile.py` resolves an engine class from `trace_factorization` alone
(`_FACTORIZATION_TO_ENGINE`), so `ETraceConfig(learning_signal='modulatory')`
selects `ParamDimVjpAlgorithm`. If the branch lived in a preset,
config-based construction would silently run `symmetric`. The presets are
convenience wrappers over configs and nothing more.

---

## Part 1 — UORO

### The design question the roadmap deferred

The roadmap claims `random_projection` is "O(P+H), **unbiased**" without saying
*unbiased for what*. That is the whole design decision, because UORO is unbiased
for whatever influence recursion it rolls, and the repository has two candidates:

- the **anchored / block-diagonal** transition every `per_param` coordinate uses
  today (`HiddenGroup.diagonal_jacobian`), or
- the **full within-group** transition `D_full[p,a,q,b] = ∂h^t[p,a]/∂h^{t-1}[q,b]`.

Measured on a 4-unit tanh RNN, `T = 8`, ETP weight = the recurrent matmul
(prototype: exact influence recursion vs. its UORO estimator, seeds `0..N-1`,
relative Frobenius deviation):

| N seeds | UORO rolling `D_full` vs BPTT | UORO rolling `D_blk` vs BPTT | UORO rolling `D_blk` vs the `D_blk` trace |
|---|---|---|---|
| 1 | 1.40e+00 | 1.12e+00 | 5.79e-01 |
| 16 | 2.47e-01 | 1.23e+00 | 2.98e-01 |
| 64 | 1.59e-01 | 1.14e+00 | 1.46e-01 |
| 256 | 1.26e-01 | 1.13e+00 | 9.04e-02 |
| 1024 | **6.72e-02** | **1.11e+00** | **2.70e-02** |

Rolling the block-diagonal transition converges — cleanly, at the same 1/√N —
onto the *biased* trace, and never onto the exact one. Reference points from the
same run: the exact full influence recursion equals BPTT to `1.40e-16`, and the
block-diagonal recursion sits `1.14e+00` away from it.

**It settles as: `random_projection` rolls the full within-group Jacobian, and
requires `recurrence_scope='coupled'`.** A rank-1 unbiased estimator of an
already-biased trace would be strictly worse than the biased trace itself — more
variance, the same asymptotic error, and no memory saved, since the anchored
`per_param` trace is *already* smaller than UORO's two factors (see the memory
table). The only coordinate where the random projection buys something is the one
where the quantity being estimated is otherwise unaffordable.

`coupled` is required because it is the cheapest scope whose
`include_recurrent_mixing` is true, which is what puts hidden→hidden ETP mixing
into the transition at all. UORO then keeps the whole transition instead of
extracting its block diagonal, so **the recursion UORO estimates is the
saturating end of P3's SnAp scale, at SnAp-1 memory, in expectation.** That is
the claim to sell, and § Acceptance pins it against exactly that reference.

### Unbiased for *what*: the three approximations UORO does not fix

This is where the roadmap's one-word claim needs three sentences.

1. **Cross-group coupling.** The compiler splits hidden states into groups and
   drops inter-group connections (`hidden_group.py`); one factor pair per group
   cannot carry the omitted `D_gk` terms. UORO is unbiased *within* a group.
2. **The instantaneous term's tail.** `_compute_hid2weight_jacobian` obtains `df`
   from a single all-ones-tangent JVP of the `y → hidden` map, which is exact
   only for a position-preserving elementwise tail and becomes a row sum
   otherwise — **F-31, pre-existing, unchanged by P4**.
3. **The primitive's own solve regime.** The projector is built from the same
   per-primitive rules the framework already uses (below), so it inherits each
   primitive's supported regime exactly — no better, no worse.

So: **UORO is an unbiased estimator of the exact within-group influence
recursion that the compiled transition defines.** On the model class where that
recursion is itself exact — a single hidden group with a position-preserving
elementwise tail, the class in which P3 measured saturated SnAp-n at `8.03e-08`
from BPTT — it is therefore unbiased for BPTT. On any other model it is unbiased
for the block-local recursion and biased against BPTT, exactly like every other
coordinate in this repository.

UORO is **not** restricted to that class. Refusing multi-group models would make
it the only coordinate in the repo that refuses them, and would forbid the
genericity sweep in U6. It runs everywhere; the docstring says what it is
unbiased for, and the BPTT-equality assertions run only where they mean
something.

### The representation

Per hidden group `g`, one rank-1 factor pair. Hidden units of group `g` are
`u = (p, a)` over `varshape × num_state`, `H_g = P_g · S_g`:

```
eps_tilde_g[j, u]  ~=  s_tilde_g[u] * theta_tilde_g[j]
```

- `s_tilde_g`: shape `(*varshape_g, num_state_g)` — the hidden-side factor.
- `theta_tilde_g`: **parameter-shaped**, one per `(relation, group)` pair, with
  *no* hidden axis and *no* trailing state axis.

The step, with `nu_g` a Rademacher draw of `s_tilde_g`'s shape:

```
proj_g          = nu_g^T J_f                       # parameter-shaped
rho0_g          = sqrt((||theta_tilde_g|| + eps) / (||D_full_g s_tilde_g|| + eps))
rho1_g          = sqrt((||proj_g||        + eps) / (||nu_g||               + eps))
s_tilde_g      <- rho0_g * (D_full_g s_tilde_g) + rho1_g * nu_g
theta_tilde_g  <- theta_tilde_g / rho0_g + proj_g / rho1_g
```

and the weight gradient contributed at a window boundary is

```
dL/dtheta = sum_g (learning_signal_g . s_tilde_g) * theta_tilde_g
```

— a scalar per group times a parameter-shaped array.

**Why it is unbiased, correctly stated.** Write `a = D s_tilde`, `b =
theta_tilde`, `c = nu`, `d = nu^T J_f`. The updated outer product is

```
a b^T  +  c d^T  +  (rho0/rho1) a d^T  +  (rho1/rho0) c b^T
```

- `a b^T = D eps_tilde` — `rho0` cancels exactly; no `nu` dependence at all.
- `c d^T = nu nu^T J_f` — quadratic in `nu`, and both `rho`s have cancelled.
- the two cross terms are **odd** in `nu_t`: `d` and `c` are odd, `rho1` is
  *even* (both norms are invariant under `nu -> -nu`), and `a`, `b`, `rho0`
  do not depend on `nu_t` at all.

So conditionally on `nu_{<t}`, any draw distribution that is symmetric under
negation and satisfies `E[nu nu^T] = I` gives `E[eps_tilde_new] = D eps_tilde +
J_f` exactly, and induction over `t` gives unbiasedness. The spec's earlier
"bilinear in nu" wording was wrong — `rho1` is nonlinear in `nu`; the argument
that survives is the parity one, and it is what U1's enumeration exploits.

A corollary worth writing down because it makes the tests robust: **`rho0` may be
any positive `nu_t`-independent scalar and `rho1` any positive even function of
`nu_t`** without affecting unbiasedness. The normaliser choice is a *variance*
decision, not a correctness one. In particular, for a parameter reached by two
relations of the same group the spec accumulates `||theta_tilde||` as the root
sum of the pieces' squared norms rather than the norm of their routed sum; those
differ, but only in variance.

### `nu^T J_f`: the same composition the framework already trusts

`nu^T J_f` is the instantaneous term contracted with the learning-signal rule:

```
nu^T J_f  ==  solve_rule(nu, instant_rule(x, df, weights))
```

where `instant_rule` is `instant_drtrl` if registered else `xy_to_dw`, and
`solve_rule` is `solve_drtrl` if registered else `dt_to_t` — with the closed-form
`FastPathRules` (`fp.instant` / `fp.solve`) substituted when
`fp.applicable(eqn_params)`, exactly as `_apply_relation_step` and
`_solve_param_dim_weight_gradients` choose today. It is *not* universally
`fp.*`: `_registries.py` registers no fast path for conv, sparse or LoRA, and
those go through their dedicated rules.

This is the composition the framework already uses for the instantaneous part of
a gradient, so **no `_op/` file changes** — and, equally, **no genericity claim
beyond each primitive's own documented regime**. The known sharp edges, to be
pinned rather than assumed:

- **`einsum` with a shared axis.** `_einsum_dt_to_t` sums the signal over shared
  letters *before* multiplying by a trace that was itself already summed over
  them, so `sum_t nu_t * x_t` and `(sum_t nu_t)(sum_t x_t)` are conflated. That
  is the documented regime restriction in `einsum.py`'s docstring, inherited
  unchanged.
- **conv bias** needs the explicit spatial reduction of roadmap lesson 7.
- **LoRA** chains through the current weights in `solve_drtrl`, so the projector
  is in raw-parameter coordinates only after that chaining.

U6 therefore pins the projector **per primitive against an independent
`jax.vjp`**, and does not settle for finite gradients.

### Units

`theta_tilde` leaves may carry `brainunit` units; `s_tilde`, `nu` and `eps` are
dimensionless. `||theta_tilde|| + eps` is therefore not a legal expression.
Contract: every `rho` is computed over **mantissas only** and is a dimensionless
scalar; each `theta_tilde` leaf keeps its own unit through the divide and the
add; the final `(signal . s_tilde) * theta_tilde` carries the parameter's unit.
A heterogeneous-unit model is a required test case, not an afterthought.

### Memory: state the win honestly

Sizes for `tanh_rnn(n_in=3, n_rec=4)`, batch 1, ETP weight `(4, 4)`, group
`varshape=(1, 4)`, `num_state=1` (measured trace shapes):

| Carrier | Elements | Formula |
|---|---|---|
| `per_param`, `diagonal` or `coupled` (anchored) | 16 | `B·\|θ\|·S` |
| `per_param`, `sparse_n` at saturation (P3) | 64 | `B·\|θ\|·S·n_neigh` |
| full within-group influence matrix | 64 | `B·\|θ\|·P·S` |
| **`random_projection`** | **20** | `\|θ\| + B·P·S` |

UORO is **not** a memory win over the anchored per-parameter trace — it is a
*bias* win at comparable carrier size, and a memory win against saturated
SnAp-n / the full influence matrix, which is what unbiasedness would otherwise
cost. Two further honesty requirements: the roadmap's "O(P+H)" describes
**carrier storage, not peak memory** (`D_full` is an `O((P·S)²)` transient, see
below), and the docstring says so.

### Scope

**In:**
- `random_projection` as a third `trace_factorization`, on a new engine
  `RandomProjectionVjpAlgorithm` in `braintrace/_algorithm/random_projection_vjp.py`
  (naming after `param_dim_vjp.py` / `io_dim_vjp.py`), registered in
  `_FACTORIZATION_TO_ENGINE`.
- A `UORO` preset in `braintrace/_algorithm/uoro.py`, exported from
  `braintrace._algorithm` **and** `braintrace` (P3 lesson: the top-level
  `__all__` was missed for `SnAp`).
- `HiddenGroup.full_jacobian(...)` returning the full `(*V, S, *V, S)` Jacobian,
  plus one `ETraceVjpGraphExecutor` constructor flag selecting it in
  `_compute_hid2hid_jacobian`. `block_diagonal_last_dim` already materialises
  this array before extracting blocks, so the new mode is that minus the
  extraction. The flag still needs threading from `vjp_base.py` through the
  executor constructor — one argument, no compiler metadata, unlike `sparse_n`
  (which needs `group.snap` built at compile time).
- Relation-local helpers extracted from `param_dim_vjp.py` so both engines share
  one spelling of "build the instantaneous term" and "apply the solve rule and
  reduce to parameter shape".
- Statistical acceptance infrastructure in `oracle.py` (§ Acceptance).
- Matrix rules 11 (`random_projection` requires `coupled`) and 12
  (`random_projection` excludes `trace_filter='kappa'`, for rule 1's reason: the
  filtered trace is not rank-1).

**Out, with reasons:**
- **KF-RTRL / OK.** Different factorisations (Kronecker, rank-r), each needing
  its own carrier and its own unbiasedness proof. Additional *coordinates*, not
  variants of this one.
- **A matrix-free `D_full` JVP.** UORO only needs `D_full @ s_tilde`. Deferred
  because `block_diagonal_last_dim` already materialises `(P·S)²` today for
  `coupled`, so this is not a regression, and the fused-stepper requirement keeps
  it transient per step rather than stacked over `T`. Follow-up **F-32**.
- **Cross-group influence.** Would change the compiler's grouping contract, not
  the trace.
- **`random_projection` inside a descended `scan`.** The descent path forces
  `include_recurrent_mixing=False`, so a full Jacobian computed in a descended
  body would silently omit exactly the mixing UORO needs. Lessons 14/15: raise.
- **`random_projection` under `io_factorized`** — the factorised x-side carries
  no hidden index to project (rule 9's argument).

### API surface

```python
ETraceConfig(trace_factorization='random_projection', recurrence_scope='coupled')
UORO(model, vjp_method='multi-step', projection_key=42, projection_eps=1e-12)
```

**PRNG.** The key is carried in the etrace-data pytree and advanced with
`brainstate.random` (AGENTS.md rule 11) — an explicit
`brainstate.random.RandomState(carried_key)` whose split feeds
`brainstate.random.bernoulli(..., key=draw_key)`, mapped to `±1`. Not the global
counter: the stepper runs inside `jax.custom_vjp` under
`brainstate.transform.scan`, where a global draw is neither replayable across the
fwd/bwd pair nor stable under `reset_state`. `reset_state` re-derives the stream
from `projection_key`, so a reset run is bit-identical. The backward residual's
`old_etrace_vals` needs no replay — it carries the pre-window key alongside the
pre-window factors, and the solve draws nothing.

**The test seam must be functional.** `_draw_projection(key, shape, dtype)` is a
documented protected hook, but a test override must **not** consume a Python
iterator: the scan body is traced once, so `next(it)` would reuse a single draw
for every step. U1 supplies a stacked `(n_steps, *shape)` array and the carry
holds the step index into it.

UORO must use the **fused stepper** (`_make_etrace_stepper`), not the
stack-then-scan path, so `D_full` is never stacked over `T`.

### The ε guard is load-bearing

At the first step `s_tilde = 0` and `theta_tilde = 0`, so both `rho` ratios are
`0/0`. Measured with `eps = 0`: **NaN at every `T`**. `projection_eps` defaults
to `1e-12`, is documented as the reason the exactness pins carry a tolerance
above float64 epsilon, and is asserted to produce finite factors on the first
step. As delivered it is also *validated*: a value that is zero, negative, or
underflows to zero in float32 is refused at construction rather than allowed to
produce the documented NaN.

The same degeneracy has a testing consequence that took a review round to see.
At step 1 `rho0 = sqrt(eps/eps) = 1` whatever its formula says, so a one-step
hand computation pins `rho1` alone; U1's enumeration cannot pin either, since
unbiasedness holds for *any* draw-independent positive `rho0`. The delivered
suite therefore hand-computes **two** steps on a saturating fixture where `rho0`
lands near 16, and asserts that separation so the fixture cannot drift back to the
degenerate regime (roadmap lesson 47).

### Precision: the factors are scan carries

Both factors ride a `jax.lax.scan` carry, so their dtype is a contract, not a
preference — a mismatch is a hard `carry input and carry output must have equal
types` on the first `MultiStepData` call, not a silent downcast. Two independent
ways to break it, both of which did:

* allocating `s_tilde` at a hard-coded `float32`, which fails for any float16
  model and for any model under `jax_enable_x64`. Fixed by reading the dtype off
  the group's own hidden states (`_group_dtype`); `theta_tilde` already followed
  each parameter.
* letting the normalisers set the output dtype. `_tree_sq_norm` accumulates at
  float32-or-wider *deliberately* (a sum of squares in float16 underflows below
  about `1e-3`), so `rho0 * d_s + rho1 * nu` comes back wider than the carry.
  Fixed by narrowing the result, not the reduction.

Pinned by a parametrised test over `float16 / bfloat16 / float32`, which also
asserts the factors come back at the model's dtype rather than merely not
crashing.

---

## Part 2 — `modulatory`

Replace each group's learning signal with a user-supplied modulator:

```
learning_signal_g = expand_to(m, (*varshape_g, num_state_g))
```

**Replace, not multiply.** Multiplying (`m ⊙ ∂L/∂h`) would be a four-factor rule
and would make the roadmap's degenerate criterion — "must equal `symmetric`
element-wise when the modulatory signal is set to `∂L/∂h`" — unsatisfiable.

**`expand_to`, not bare broadcasting.** NumPy broadcasting aligns trailing axes,
so a `(1, n_rec)` modulator does **not** broadcast to a `(1, n_rec, 1)` signal.
The contract, matching AGENTS.md's SNN learning-signal trailing-axis rule: a
modulator whose shape equals the group's `varshape` gains a trailing size-1 state
axis; a scalar is used as-is; anything else must broadcast against
`(*varshape, num_state)` after that expansion, and otherwise raises naming the
group and both shapes.

**`ThreeFactor` requires `vjp_method='single-step'`, and raises otherwise.**
Under multi-step, `_solve_weight_gradients` adds `dl_to_etws_at_t` — the
within-window reverse-AD gradient of the ETP parameters
(`param_dim_vjp.py:1309`) — on top of the trace contraction. Replacing the
*boundary* signal therefore yields a hybrid whose in-window half is still an
unmodulated loss gradient: not a three-factor rule, and a rule nobody asked for.
Single-step makes every ETP contribution flow through the replaced signal, which
is the rule. This also makes the modulator per *step*, which is what a
neuromodulator is. Consequence: `update_schedule` stays out of scope — it would
only be load-bearing if ThreeFactor were multi-step.

**The anti-OSTTP contract** (roadmap risk 5). The modulator is **one array,
expanded to every group** — never a per-group sequence. A scalar reward is valid
for any model whatever its HiddenGroup count. There is deliberately no length-`n_groups`
spelling: that binding is what made OSTTP non-general. Passing a list or tuple
raises with that explanation.

Delivery: `ThreeFactor` preset in `braintrace/_algorithm/three_factor.py`; the
`modulatory` branch itself in `vjp_base._compute_learning_signal`. The modulator
arrives through the retained `_get_update_aux` side channel — read
*synchronously* at the top of `update()`, per that hook's docstring — via
`learner.update(*inputs, modulator=m)` or an assignable `learner.modulator`, with
the keyword taking precedence. A missing modulator raises; it does not fall back
to `symmetric` (lesson 15). The stash is cleared in a `finally` so an exception
mid-update cannot leak a stale modulator into the next call — on the successful
path as well as the failing one, which has its own test.

**Where the shape check happens, and for which groups.** The expansion that feeds
the rule runs inside the `custom_vjp` *backward* pass, so validating only there
would let a malformed modulator through a forward-only `update()` and fail later
from inside JAX. The refusal is therefore raised eagerly at the top of `update()`,
against every group's declared signal shape — **including descended groups**. An
earlier revision skipped those, reasoning that a descended group's signal carries
a leading substep axis; measured on `snn_scan_rnn(loops=40)` it does not, because
`scan_descent` folds the per-substep Jacobians inside the body and the reverse
pass hands out one array per group of exactly `(*varshape, num_state)`. The skip
bought nothing and cost the pre-flight on every descended model (roadmap
lesson 43); it is gone, with a test asserting the fixture still descends so the
pin cannot go vacuous.

---

## Part 3 — `bootstrapped` (DNI)

### Where the synthetic gradient goes, and where it must *not* go

`_update_fn` returns the **window-exit** hidden values, so `grads[1]` in
`_update_fn_bwd` is the cotangent arriving at `h^exit` from outside the window —
zero under a truncated-window loss. That is where DNI's estimate belongs, ahead
of the in-window reverse pass, so the window's own reverse-AD propagates it.

But it must reach only *some* of the consumers of that pass, and getting this
wrong is the central correctness trap of Part 3.

Index windows `[a_k, b_k)` with `b_k = a_{k+1}`, and let `l_t` be the loss
produced by the step that writes `h^{t+1}`. The estimate injected at window `k`'s
exit is `g_k ≈ ∂(Σ_{t ≥ b_k} l_t)/∂h^{b_k}` — **strictly future, half-open**, so
the exit step's own loss is inside window `k` and is not counted twice.

- **Plain (non-ETP) parameters, inputs, other states: inject.** Their credit is
  truncated at the window edge, and injecting `g_k` makes the sum over windows
  telescope to the exact gradient: θ's occurrence inside window `k` reaches
  future losses exactly once, through `g_k`.
- **ETP parameters: do NOT inject.** Their cross-window credit is *already*
  carried by the eligibility trace: an occurrence at `t ∈ window k` reaching a
  loss in window `j > k` is counted by window `j`'s boundary term, because
  `eps_tilde(a_j)` contains that occurrence (`t < a_j`) and `∂l/∂h^{a_j}` is in
  window `j`'s signal. Adding `g_k`'s contribution to the ETP gradient would
  count the same path a second time. **This is what an eligibility trace is for**
  — DNI's job here is to give the *plain* parameters the cross-window credit the
  trace already gives the ETP ones.

### Implementation: two linear passes, not one

`eval_jaxpr` on the residual jaxpr is linear in the cotangents, so:

1. **Pass 1** — today's call, unmodified. Yields the clean
   `dg_last_hiddens`, `dg_non_etrace_params`, `dg_etrace_params`, `dg_args`,
   `dg_oth_states`.
2. **Pass 2** — only when a synthesiser is active: the same `eval_jaxpr` with
   every cotangent zero except the exit-hidden one, set to
   `stop_gradient(M(h^exit))`.

Then:

| Consumer | Value |
|---|---|
| ETP boundary learning signal (`dl2h`) | pass 1 only |
| `dg_etrace_params` | pass 1 only |
| `dg_non_etrace_params`, `dg_args`, `dg_oth_states` | pass 1 + pass 2 |
| returned `dg_last_hiddens` | pass 1 + pass 2 — it is both the previous window's cotangent and the synthesiser's regression target, and both want the future term |

Cost: one extra transposed-jaxpr evaluation per window when DNI is on, zero
otherwise. Correctness is by linearity, and it is *checkable* — see B2.

`M(h^exit)` is computed inside `_update_fn_fwd` (the only place `h^exit` exists)
and stashed in the residuals. The synthesiser's parameter **values** are threaded
in through `_get_update_aux` as an explicit argument rather than closed over, so
`jax.custom_vjp` never captures a tracer it might be asked to differentiate;
`stop_gradient` on the output keeps the online loss from training the synthesiser
through the wrong path. Group-shaped output is mapped back to the per-hidden-path
cotangent tree with `group.split_hidden`, and JAX zero cotangents and units are
handled explicitly.

This is a **new hook, `_inject_exit_cotangent`**, not a reuse of
`_compute_learning_signal`: replacing a boundary signal and adding an exit
cotangent are different operations on different tensors at different times.

### Training the synthesiser

The regression target for `M(h^{a_k})` is `∂L_{≥a_k}/∂h^{a_k}`, which the
returned `dg_last_hiddens` already is — available through public API with no new
side channel, by putting the hidden states in the differentiation set. Measured
on `tanh_rnn`: `brainstate.transform.grad(loss, {**params, **hiddens})` returns
`{('h',): (1, 4), ('w',): (4, 4), ('win',): (3, 4)}`, and `('h',)` is the target.

Delivery in `braintrace/_algorithm/dni.py`:
- `SyntheticGradient` — per group, mapping the concatenated hidden value to that
  group's signal shape, with its own `ParamState`s that are **not** ETP routed,
  plus a functional `apply(param_values, h)` for the threaded call.
- `DNI` preset wiring `learning_signal='bootstrapped'`.
- The training recipe as a documented, tested helper, with the model parameters
  frozen, the target detached, and the auxiliary optimiser left to the caller as
  it is in the paper.

The **oracle** target helper `future_hidden_gradients` must snapshot and restore
every model state it touches: it rolls the model forward to each boundary and
differentiates a suffix loss, and leaking that rollout into the states under test
would corrupt every subsequent assertion.

---

## Acceptance

Every gradient assertion whose subject is a learning-rule axis goes through a
finite-window oracle — `chunked_online_param_gradients` with `chunk_size < T`,
or `online_param_gradients_singlestep_naive` for the single-step rules — and
each equality criterion **names its negative control pair and the parameter keys
it compares**. "Guarded by `assert_gradients_differ`" alone is not enough: a
whole-tree comparison dominated by plain parameters can pass while the axis under
test does nothing (roadmap lesson 2). `vjp_method='multi-step'` is mandatory on
the chunked path — `single-step` + `MultiStepData` raises `NotImplementedError`
from `vjp_graph_executor.solve_h2w_h2h_l2h_jacobian`.

**UORO's reference is not BPTT.** It is `SnAp(n)` at a saturating order, run
through the *same* chunked path with the *same* `chunk_size` — the exact
within-group influence recursion, which is precisely what UORO estimates. That
isolates the estimator from the cross-group and F-31 tail approximations it does
not fix. On single-group elementwise-tail fixtures that reference also equals
BPTT (P3: `8.03e-08`), and both comparisons are asserted there.

### UORO

**U1 — exhaustive exactness (the primary pin).** Conditionally on earlier draws,
the update's `nu_t` dependence splits into an even part `nu nu^T J_f` and odd
cross terms. A draw set closed under negation with `(1/|S|) Σ nu nu^T = I` kills
the odd part pairwise and maps the even part to `J_f` — both *exactly*. The `2^H`
Rademacher sign patterns are such a set, so the mean over an exhaustive
enumeration is the exact recursion at machine precision, with no statistics.

Measured (prototype), `H = 2`, `T = 3`, one-step windows, **non-zero initial
hidden state**, exhaustive over the two draws that reach a boundary
(`4 × 4 = 16` runs):

| Rolled Jacobian | exhaustive mean vs BPTT |
|---|---|
| `D_full` | **2.07e-16** |
| `D_blk` | 2.59e-04 |

Two fixture facts, both measured, both load-bearing:

- **The initial hidden state must be non-zero.** With `h⁰ = 0` the recurrent
  weight's first instantaneous term vanishes, so `D` is applied to a zero
  influence and `D_full`/`D_blk` become indistinguishable (`1.0e-16` vs
  `3.3e-16`) — the pin passes for the wrong implementation.
- **`T ≥ 3` with one-step windows.** At `T ≤ 2` the boundary trace holds only
  instantaneous terms and `D` never enters.

U1 alone is passable by a *deterministic* implementation that simply computes the
exact influence matrix, and by wrong normalisation or wrong tied-key handling, so
it ships with four companions:

- **U1a** single runs must deviate from the mean (measured `1.4e-02 … 3.1e-02`),
  so the sample variance is non-zero and the enumeration is doing work;
- **U1b** hand-computed `s_tilde` / `theta_tilde` values after two steps on a
  2-unit model, asserted leaf by leaf — the only test that pins the normalisers;
- **U1c** the same enumeration on `tied_weight_rnn` (two relations, one group,
  one parameter path) and on `stacked_tanh_rnn` (two groups), pinning the keying
  and the shared-`nu`/shared-`rho` composition;
- **U1d** the draw table is indexed by a carried step counter, and a regression
  test asserts consecutive steps consume *different* rows (the trace-once trap).

**U2 — deleted.** The original degenerate pin (`H = 1`, `P = 1`, exactness for
all `T`) is false. For `D = J_f = 1` the estimate after two steps is
`(nu_1 + nu_2)² ∈ {0, 4}` while the truth is `2`; the `1.6e-09` measured on the
tanh fixture at `T = 3` is a cancellation accident of that fixture. A pin that
holds by accident is worse than no pin, and a correct implementation would fail
it on a different fixture.

**U3 — a real confidence interval, not a decay bound.** The earlier proposal
(`deviation(N) ≤ 4/√N`) is not a bias test: a fixed bias of 0.1 satisfies it at
every `N` used. Instead: fix `K = 8` unit directions (one fixed seed, drawn
once), project each seed's ETP gradient onto them to get scalars `v_s[k]`, and
for each `k` assert

```
|mean_k - reference_k|  <=  z * sample_std_k / sqrt(N)     (z = 3.5, Bonferroni over K)
sample_std_k / sqrt(N)  <=  0.25 * |reference_k|           (non-vacuity: the interval must be tight)
```

Both halves are needed: the first is the unbiasedness test, the second stops a
huge sample variance from admitting anything. Fixed seeds, `slow` marker
(roadmap risk 4). Failure prints the whole per-direction table.

**U4 — negative controls, plural.** UORO's mean must differ from `D_RTRL`'s
gradient (it is not the biased trace) **and** a single UORO run must differ from
the mean (it is not deterministic). The first alone is passed by exact RTRL.

**U5 — structural.** `full_jacobian` returns `(*V, S, *V, S)` and equals
`jax.jacrev` of the group transition; `theta_tilde` carries no state and no batch
axis; the carrier totals 20 elements on `tanh_rnn(3, 4)`; **the factor count is
one `s_tilde` per group and one `theta_tilde` per `(group, ETP parameter path)`
pair** — *not* per `(relation, group)`, which is what an earlier draft of this
spec said. The revision is deliberate and matters twice over: it is the
mathematically faithful layout (one parameter-tree-shaped factor per group), and
it removes the root-sum-of-squares approximation of the norm that a per-relation
layout would have forced. `tied_weight_rnn` is the fixture that separates the
two: two relations, one group, one `ParamState` path, and therefore exactly
**one** `theta_tilde`, into which both relations' projections are summed.

Two group-count facts have to be pinned alongside it, because the obvious fixture
does not have the shape one would assume. Hidden grouping follows the
*transition*, so `recurrence_scope='coupled'` — which UORO requires — merges every
set of mutually reachable hidden states into one group. Measured on
`stacked_tanh_rnn(4, 4)`: `diagonal` gives 2 groups, `coupled` gives **1** holding
both `h1` and `h2`. Every pre-existing spec in `oracle_models.py` collapses to a
single group at coupled scope, so the multi-group clause needs the new
`two_island_rnn` fixture (two severed subnetworks) to be non-vacuous.

Also: heterogeneous units survive a full step and gradient; and
`random_projection` + `diagonal`, + `sparse_n`, + `kappa`, + `io_factorized`,
+ a descended `scan` each raise with a message naming the coordinate. The memory
table is asserted as *carrier storage* and separately documented as not being
peak memory.

**U6 — the projector, per primitive, against `jax.vjp`.** For each anchored
primitive, assert `solve_rule(nu, instant_rule(x, df, w)) == vjp(y_fn, w)(nu ⊙ df)`
reduced to parameter shape, on an independent JAX reference. Coverage must include
the shared-axis `einsum`, a transformed (`weight_fn`) case, `conv` with bias, the
`embedding`/`sparse` paths, and a tied parameter. Finite gradients do not test a
projector.

**U7 — reproducibility.** `reset_state` reproduces a run bit-for-bit; two
different `projection_key`s give different gradients; the first step's factors are
finite with the default `eps`.

### `modulatory`

All `modulatory` criteria run on `online_param_gradients_singlestep_naive`,
because `ThreeFactor` is single-step by construction.

**M1 — degenerate equality with a negative control.** Capture the `symmetric`
per-group signals via a `_compute_learning_signal` override on a **single-group**
model, feed them back as the modulator, assert element-wise equality with
`symmetric` on the ETP keys. Then assert a *different* modulator (scaled by 2)
does **not** produce that gradient — otherwise "ignore the modulator and return
the symmetric signal" passes. Single group deliberately: an exactly-`∂L/∂h`
modulator for a multi-group model would need a per-group sequence, the binding
this axis refuses.

**M2 — the anti-OSTTP pin, by captured signal.** On `stacked_tanh_rnn` (two
groups) with a scalar modulator, capture what each group's signal *actually* is
and assert it equals the explicit `expand_to(m, group_shape)` — not merely that
gradients are finite, which a symmetric fallback also achieves. Repeat with a
`(1, n_rec)` modulator on a model whose group count differs from `n_rec`, pinning
that the expansion is shape-driven and not group-indexed.

**M3 — refusals and lifecycle.** Non-broadcastable modulator raises naming the
group and both shapes; missing modulator raises rather than falling back; a
list/tuple of per-group arrays raises with the OSTTP explanation; `vjp_method='multi-step'`
raises naming the in-window direct term; the keyword overrides the attribute; an
exception mid-`update()` does not leave a stale modulator for the next call; two
consecutive calls with different modulators give different gradients.

**M4 — it does something, measurably.** On the single-step harness (where no
unmodulated in-window term exists to mask the axis), a reward-like scalar
modulator must give a gradient that differs from `symmetric`, differs from a
zero modulator (which must give exactly zero ETP gradient), and flips sign with
the modulator's sign. Plus a descent smoke test on a reward-modulated objective.

The single-step restriction has a consequence worth stating plainly, because it
is easy to read the modulatory axis as affecting only the ETP parameters:
**`ThreeFactor` is single-step-only, and single-step zeroes every plain
(non-ETP) parameter's gradient exactly** (F-33 in
`docs/specs/2026-07-25-known-limitations.md`). So a `ThreeFactor` learner trains
its ETP-routed weights and *nothing else* — a readout or input projection that is
not in a `hidden_param_op_relation` receives an exact zero, not a truncated
approximation. This is a property of `vjp_method`, not of the modulator, but the
ctor refusal in `ETraceVjpAlgorithm` (modulatory requires single-step, § Part 2)
makes the two inseparable in practice, so M4 asserts the zero rather than leaving
a caller to discover it.

**Status 2026-08-08:** F-33 is resolved. The compiled graph now partitions
parameter paths before VJP: plain-only paths receive exact current-step
reverse-mode gradients, while any path owned by an ETP relation remains wholly
ETP-routed. The paragraph above records the behavior when this historical phase
was accepted.

### `bootstrapped`

**B1 — `M ≡ 0` is a bit-exact no-op**, and a *live* non-zero synthesiser must
change the plain-parameter gradients. Without the second half, an entirely
ignored synthesiser passes.

**B2 — an oracle synthesiser, both halves element-wise.** Pin
`M(h^{b_k}) = ∂(Σ_{t ≥ b_k} l_t)/∂h^{b_k}` from `future_hidden_gradients`. Then
on a fixture with **both** plain and ETP parameters:
- **plain keys equal full-sequence BPTT** element-wise — the telescoping claim;
  with a negative control that `M ≡ 0` does *not* equal BPTT on those same keys
  (otherwise the fixture has no cross-window plain credit and the test is
  vacuous);
- **ETP keys are bit-identical to the non-DNI run** — the no-double-counting
  claim, and the sharpest available check that pass 2 is routed correctly.

The earlier "ETP keys equal BPTT at `recurrence_scope='coupled'`" clause is
**deleted twice over**: it double-counts (§ Part 3), and `coupled` is SnAp-1, not
full RTRL — P3 measured `6.18e-02` deviation there, against `8e-08` for the
saturating order.

**B3 — a learned synthesiser helps, honestly.** Freeze the model parameters,
train only the synthesiser's, detach the targets, and evaluate the deviation from
BPTT on a **held-out** sequence. Must beat the `M ≡ 0` deviation on that same
held-out sequence and seed.

**B4 — end-to-end RL smoke test** (roadmap requirement): a **delayed-reward
recurrent** task whose credit spans several windows — not a bandit, which has no
temporal credit for DNI to supply. Controls: an `M ≡ 0` run and a
stopped-gradient-synthesiser run must both do worse. `slow` marker.

Delivered with a **fourth arm the spec did not ask for**, and it is the one that
makes the other three readable: an *oracle* synthesiser holding the true
`dL_{≥ b}/dh^b` of the training objective, recomputed against the current
parameters every epoch. It is the ceiling — the estimate a learned `M` is
approximating — so the criterion becomes an ordering rather than a single
comparison. Measured over three seeds at 15 epochs of Adam(`3e-3`):
`oracle < trained < frozen ≈ M ≡ 0` on every one (seed 0: `0.131 < 0.221 <
0.271`). The arm was added because the criterion failed three separate times for
reasons that had nothing to do with DNI, and without an exact-estimate arm there
is no way to tell a wrong routing from a weak predictor. Three harness properties
turned out to be load-bearing, each measured failing before being fixed:
`train_synthetic_gradient`'s `loss_fn` must be the objective actually descended
(F-35); the synthesiser must be refitted as the model moves, since its target is a
function of the parameters; and the fixture must be conditioned for an optimiser
to run on it — `delayed_reward_rnn` needed its convex `(1 - leak)` factor, without
which every arm reached `nan`.

**B5 — structure.** The synthesiser's parameters appear in no
`graph.hidden_param_op_relations` entry; the injected cotangent is provably
non-zero; the online loss produces zero gradient w.r.t. the synthesiser's
parameters; the auxiliary loss produces non-zero gradient w.r.t. them; the
per-group mapping is correct on a two-group model; a shape mismatch raises naming
both shapes; `bootstrapped` without a synthesiser raises.

### Statistical infrastructure (new, in `oracle.py`)

The roadmap flags this as missing and asks for it to be counted. Deterministic
given the seed list:

- `seed_gradient_samples(spec, inputs, algo_factory, seeds, chunk_size)` — the
  per-seed gradient trees, `algo_factory(model, seed)`.
- `project_gradients(trees, directions)` — fixed-direction scalar projections.
- `assert_unbiased(samples, reference, directions, z=3.5, tightness=0.25)` — the
  two-sided interval test above, printing the full per-direction table on
  failure, because a statistical failure that prints one number is unactionable.
- `future_hidden_gradients(model_factory, inputs, boundaries)` — the DNI target
  oracle, state-snapshotting.

---

## Test plan

Sibling `*_test.py`, per AGENTS.md — never a `tests/` directory, never `test_*`.

| File | Adds |
|---|---|
| `braintrace/_algorithm/uoro_test.py` | U1, U1a–U1d, U3–U7 |
| `braintrace/_algorithm/random_projection_vjp_test.py` | U6's per-primitive projector sweep; factor bookkeeping |
| `braintrace/_algorithm/three_factor_test.py` | M1–M4 |
| `braintrace/_algorithm/dni_test.py` | B1–B5 |
| `braintrace/_algorithm/axes_test.py` | rules 11/12; the three values no longer rejected by rule 8; `describe()` round-trip |
| `braintrace/_algorithm/oracle_test.py` | the four helpers, incl. a deliberately-biased estimator the interval test must reject |
| `braintrace/_algorithm/vjp_base_test.py` | `_inject_exit_cotangent` two-pass linearity; `expand_to`; `_get_update_aux` lifecycle |
| `braintrace/_algorithm/param_dim_vjp_test.py` | the extracted relation-local helpers, unchanged behaviour for `per_param` |
| `braintrace/_compiler/hidden_group_test.py` | `full_jacobian` vs `jax.jacrev`; shape/dtype/units |
| `braintrace/_algorithm/vjp_graph_executor_test.py` | the full-Jacobian executor flag |
| `braintrace/_algorithm/oracle_models.py` (+ `_test`) | `nonzero_init_rnn` (U1), `unit_weight_rnn` (units), `plain_and_etp_rnn` (B2, with real cross-window plain credit), `delayed_reward_rnn` (B4) |
| `braintrace/_compile_test.py` | config- and string-based construction of all three coordinates |
| `braintrace/_algorithm/tests/axis_acceptance_test.py` | field-by-field preset coordinates for `UORO`, `ThreeFactor`, `DNI` |
| `braintrace/__init___test.py` | the three presets in both `__all__`s |

TDD order: `axes_test.py` and `oracle_test.py` first (no engine needed), then
U5/U6 structural and projector pins, then U1 and its companions, then Part 2,
then Part 3 — with B1 written and passing before anything else in Part 3.

---

## Risks

1. **U1's enumeration is 16 algorithm constructions**, each recompiling. If it
   exceeds a few minutes it moves behind the `slow` marker, and U1b (hand-computed
   factors) becomes the fast pin.
2. **Antithetic variance reduction does not work here, and looks like it should.**
   Measured: flipping the sign of the *entire* `nu` sequence leaves the estimate
   **bit-identical** — `rho1` is even and both factors flip, so their product is
   invariant. Pairing `(nu, -nu)` halves the effective sample count and measured
   *worse* than plain sampling at equal cost (`3.54e-01` vs `1.78e-01` at 64
   runs). Recorded so nobody "optimises" U3 into a weaker test.
3. **UORO's variance grows with the number of boundaries.** Test sequences stay
   short, and the docstring says unbiased, not low-variance.
4. **DNI's two-pass split is the riskiest single change.** B1 (`M ≡ 0` bit-exact)
   and B2's ETP half (bit-identical to non-DNI) are written and passing before
   the plain-parameter half is attempted.
5. **The fwd-side synthesiser call reads state inside `jax.custom_vjp`.**
   Mitigation: parameter values threaded as an explicit argument;
   `stop_gradient` on the output.
6. **`modulatory` recreating OSTTP's binding.** M2/M3 are the guard, and the API
   has no per-group spelling to misuse.
7. **Extracting relation-local helpers from `param_dim_vjp.py` touches the
   working engine.** Mitigation: the extraction is behaviour-preserving and
   `param_dim_vjp_test.py` plus the existing suite run green before UORO consumes
   the helpers.

## Out of scope

KF-RTRL/OK; a matrix-free `D_full` JVP (F-32); cross-group influence;
`random_projection` under `io_factorized` or inside descended scans; per-group
modulator sequences; multi-step `ThreeFactor`; a built-in synthesiser training
loop; `update_schedule` values.
