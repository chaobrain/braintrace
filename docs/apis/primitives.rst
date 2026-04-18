Custom Primitives
=================

.. currentmodule:: braintrace

.. contents::
   :local:
   :depth: 2

This page documents how to register a new ETP primitive and the four
ETP-specific rules every primitive must provide. The built-in primitives
(``etp_mm``, ``etp_mv``, ``etp_conv``, ``etp_elemwise``, ``etp_sp_mm``,
``etp_sp_mv``, ``etp_lora_mm``, ``etp_lora_mv``) are themselves registered
through this same machinery — adding a custom op uses the same surface.


Two registration styles
-----------------------

There are two equivalent ways to register a primitive. Pick whichever
fits your codebase:

* **Class-based** — call :func:`register_primitive` to obtain an
  :class:`ETPPrimitive`, then attach the four rules via its
  ``register_*`` methods. Best for incremental development where you
  want to register rules close to where they are defined.
* **Spec-based** — declare an :class:`ETPPrimitiveSpec` (a single
  dataclass-like value), then pass it to
  :func:`register_primitive_spec`. Best when you want a single object
  that fully describes the primitive (useful for testing and for the
  compiler's ``get_primitive_spec`` query).

Both styles populate the same four global registries
(:data:`ETP_RULES_YW_TO_W`, :data:`ETP_RULES_XY_TO_DW`,
:data:`ETP_RULES_INIT_DRTRL`, :data:`ETP_RULES_INIT_PP`) and result in a
fully-functional ``ETPPrimitive``.


Registration entry points
-------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   register_primitive
   register_primitive_spec
   get_primitive_spec


Primitive class & spec
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ETPPrimitive
   ETPPrimitiveSpec


The four ETP rules
------------------

Every ETP primitive must supply four rules. They are stored in four
global ``dict`` registries keyed by primitive — the compiler and the
online-learning algorithms look up the rule for a primitive at compile
time.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Registry
     - Signature
     - Purpose
   * - ``ETP_RULES_YW_TO_W``
     - ``(hidden_dim, trace, **params) -> trace``
     - D-RTRL trace propagation: combine an upstream hidden-state
       Jacobian factor with the trace through the current weight.
   * - ``ETP_RULES_XY_TO_DW``
     - ``(x, hidden_dim, weight, **params) -> dW``
     - Weight-gradient rule: produce ``dL/dW`` given ``x``, the
       hidden-state Jacobian factor, and the current weight value.
   * - ``ETP_RULES_INIT_DRTRL``
     - ``(x_var, y_var, weight, num_hidden_state) -> zeros``
     - D-RTRL trace initialiser. Returns a zero array (or pytree of
       arrays) shaped to hold the parameter-dim trace.
   * - ``ETP_RULES_INIT_PP``
     - ``(x_var, y_var, weight, num_hidden_state) -> zeros``
     - pp_prop / ES-D-RTRL trace initialiser. Returns a zero array
       shaped to hold the IO-dim trace.

The four registries live in :mod:`braintrace._etrace_op` and are
populated by both registration styles.


Class-based example
-------------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import braintrace

   def _my_impl(x, w, *, scale=1.0):
       return scale * (x @ w)

   my_p = braintrace.register_primitive('etp_my_op', _my_impl, batched=True)

   # Rules can be registered one-by-one, or in a single call via
   # ``register_etp_rules(yw_to_w=..., xy_to_dw=..., ...)``.
   my_p.register_yw_to_w(
       lambda hidden, trace, **params: trace * hidden[None, :]
   )
   my_p.register_xy_to_dw(
       lambda x, hidden, w, **params:
           jax.vjp(lambda w_: _my_impl(x, w_, **params), w)[1](hidden)[0]
   )
   my_p.register_init_drtrl(
       lambda x_var, y_var, w, ns:
           jnp.zeros((x_var.aval.shape[0], *jnp.shape(w.value), ns))
   )
   my_p.register_init_pp(
       lambda x_var, y_var, w, ns:
           jnp.zeros((*y_var.aval.shape, ns), dtype=y_var.aval.dtype)
   )

After registration the primitive is ready to use:

.. code-block:: python

   x = jnp.ones((4, 3))
   w = jnp.ones((3, 5))
   y = my_p.bind(x, w, scale=0.5)         # all standard JAX rules work
   gw = jax.grad(lambda w_: my_p.bind(x, w_, scale=0.5).sum())(w)


Spec-based example
------------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import braintrace

   def _impl(x, w, *, scale=1.0):
       return scale * (x @ w)

   spec = braintrace.ETPPrimitiveSpec(
       name='etp_my_op',
       impl=_impl,
       yw_to_w=lambda hidden, trace, **p: trace * hidden[None, :],
       xy_to_dw=lambda x, hidden, w, **p:
           jax.vjp(lambda w_: _impl(x, w_, **p), w)[1](hidden)[0],
       init_drtrl=lambda x_var, y_var, w, ns:
           jnp.zeros((x_var.aval.shape[0], *jnp.shape(w.value), ns)),
       init_pp=lambda x_var, y_var, w, ns:
           jnp.zeros((*y_var.aval.shape, ns), dtype=y_var.aval.dtype),
       weight_invar_index=1,
       x_invar_index=0,
       batched=True,
   )

   my_p = braintrace.register_primitive_spec(spec)

   # Later, the compiler can query the spec it was built from:
   assert braintrace.get_primitive_spec(my_p) is spec


Auto-derived JAX rules
----------------------

You **only** need to write the four ETP rules above. All standard JAX
machinery is derived automatically from your ``impl``:

* ``abstract_eval`` — via ``jax.eval_shape(impl)``
* MLIR lowering — via ``mlir.lower_fun(impl)``
* JVP — via ``jax.jvp(impl)``
* transpose — derived by JAX from the JVP
* batching — via ``jax.vmap(impl)``

This means a custom primitive immediately works under
``jit``/``grad``/``vmap``/``jvp`` without any extra code.


The ``gradient_enabled`` flag
-----------------------------

Pass ``gradient_enabled=True`` only for **identity-like** primitives
that may sit on the tail of the ``y -> h`` walk (the only built-in is
``etp_elemwise_p``).

For *trainable* ops, leave the default ``gradient_enabled=False``. The
compiler treats such a primitive as a tail boundary: a preceding ETP
weight whose only path to ``h`` passes through it is correctly excluded
from ETP, because per-primitive ETP rules cannot express the
"weight-then-weight-then-hidden" composition.

See :doc:`/tutorial/etp_primitives` for a full walk-through and
:doc:`/advanced/compiler_internals` for the underlying invariant.
