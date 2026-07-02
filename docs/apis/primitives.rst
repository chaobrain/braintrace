Custom ETP Primitives
=====================

.. currentmodule:: braintrace

.. contents::
   :local:
   :depth: 2

An ETP primitive is a thin marker around a standard JAX op. All standard JAX
rules (abstract eval, lowering, JVP, transpose, batching) are **auto-derived**
from the op's implementation — you only hand-write the small set of
*ETP-specific* rules that describe trace propagation. This page shows how to
register your own primitive; the built-ins (``etp_mm``, ``etp_mv``,
``etp_gmm``, ``etp_gmv``, ``etp_emb``, ``etp_emb_v``, ``etp_einsum``,
``etp_conv``, ``etp_elemwise``, ``etp_sp_mm``, ``etp_sp_mv``,
``etp_lora_mm``, ``etp_lora_mv``) are registered through the very same
machinery.


Registration
-----------

Call :func:`register_primitive` to obtain an :class:`ETPPrimitive`, then attach
the four ETP rules via its ``register_*`` methods (or all at once with
``register_etp_rules``).

Beyond the implementation function, :func:`register_primitive` accepts a few
keyword arguments that record the primitive's invar / outvar layout for the
compiler:

* ``trainable_invars_fn`` — ``eqn.params -> {key: invar_index}``, declaring
  every trainable input (e.g. ``{'weight': 1}`` for a plain matmul, or
  ``{'weight': 1, 'bias': 2}`` when a bias is present). Defaults to the
  single-weight ``{'weight': 1}`` layout when omitted.
* ``x_invar_index`` — position of the input ``x`` in ``eqn.invars``
  (``None`` for input-free primitives such as ``etp_elemwise_p``).
* ``y_outvar_index`` — position of the output ``y`` in ``eqn.outvars``.

The call populates the four global rule registries
(:data:`ETP_RULES_YW_TO_W`, :data:`ETP_RULES_XY_TO_DW`,
:data:`ETP_RULES_INIT_DRTRL`, :data:`ETP_RULES_INIT_PP`) and returns a
fully-functional :class:`ETPPrimitive`.


Registration entry points
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   register_primitive


Primitive class
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ETPPrimitive


The four ETP rules
------------------

Every ETP primitive must supply four rules. They are stored in four global
``dict`` registries keyed by primitive; the compiler and the online-learning
algorithms look up the rule for a primitive at compile time.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Registry
     - Signature
     - Purpose
   * - ``ETP_RULES_YW_TO_W``
     - ``(hidden_dim, trace, **params) -> trace``
     - D-RTRL trace propagation: combine an upstream hidden-state Jacobian
       factor with the trace through the current weight.
   * - ``ETP_RULES_XY_TO_DW``
     - ``(x, hidden_dim, weight, **params) -> dW``
     - Weight-gradient rule: produce ``dL/dW`` given ``x``, the hidden-state
       Jacobian factor, and the current weight value.
   * - ``ETP_RULES_INIT_DRTRL``
     - ``(x_var, y_var, weight, num_hidden_state) -> zeros``
     - D-RTRL trace initialiser. Returns a zero array (or pytree of arrays)
       shaped to hold the parameter-dim trace.
   * - ``ETP_RULES_INIT_PP``
     - ``(x_var, y_var, weight, num_hidden_state) -> zeros``
     - pp_prop / ES-D-RTRL trace initialiser. Returns a zero array shaped to
       hold the IO-dim trace.

The four registries live in :mod:`braintrace._op` and are populated by
:func:`register_primitive`.


Registration example
--------------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import braintrace

   def _my_impl(x, w, *, scale=1.0):
       return scale * (x @ w)

   my_p = braintrace.register_primitive(
       'etp_my_op',
       _my_impl,
       batched=True,
       trainable_invars_fn=lambda params: {'weight': 1},
       x_invar_index=0,
   )

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


Auto-derived JAX rules
----------------------

You **only** write the four ETP rules above. All standard JAX machinery is
derived automatically from your ``impl``:

* ``abstract_eval`` — via ``jax.eval_shape(impl)``
* MLIR lowering — via ``mlir.lower_fun(impl)``
* JVP — via ``jax.jvp(impl)``
* transpose — derived by JAX from the JVP
* batching — via ``jax.vmap(impl)``

So a custom primitive immediately works under ``jit`` / ``grad`` / ``vmap`` /
``jvp`` without any extra code.


The ``gradient_enabled`` flag
-----------------------------

Pass ``gradient_enabled=True`` only for **identity-like** primitives that may
sit on the tail of the ``y -> h`` walk (the only built-in is
``etp_elemwise_p``).

For *trainable* ops, leave the default ``gradient_enabled=False``. The compiler
treats such a primitive as a tail boundary: a preceding ETP weight whose only
path to ``h`` passes through it is correctly excluded from ETP, because
per-primitive ETP rules cannot express the "weight-then-weight-then-hidden"
composition.

See :doc:`/tutorials/etp_primitives` for a full walk-through and
:doc:`/advanced/compiler_internals` for the underlying invariant.
