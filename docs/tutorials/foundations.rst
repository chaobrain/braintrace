Foundations
===========

Build the conceptual path from an ETP-marked operation to the hidden state it
influences. These chapters explain the contracts that determine whether a
parameter participates in online learning.

Choose a foundation
-------------------

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: ETP Operator Fundamentals
      :link: five_primitive_functions
      :link-type: doc

      Use the five public ETP operators with batching, physical units, JIT,
      gradients, and ``vmap``.

   .. grid-item-card:: braintrace.nn Layers
      :link: neural_network_layers
      :link-type: doc

      Compose ETP-aware layers and understand operation-based parameter
      selection and relation boundaries.

   .. grid-item-card:: Hidden State Management
      :link: hidden_states
      :link-type: doc

      Define, initialize, reset, and inspect the hidden-state groups discovered
      by the compiler.

Recommended sequence
--------------------

.. list-table::
   :header-rows: 1
   :widths: 12 30 58

   * - Step
     - Chapter
     - Use it to
   * - 1
     - :doc:`ETP Operator Fundamentals <five_primitive_functions>`
     - Select trainable operations and verify unit and JAX-transform behavior.
   * - 2
     - :doc:`braintrace.nn Layers <neural_network_layers>`
     - Assemble recurrent models without violating parameter-to-hidden relation
       boundaries.
   * - 3
     - :doc:`Hidden State Management <hidden_states>`
     - Control state structure and understand how the compiler forms hidden
       groups.

Where to look first
-------------------

- Unexpected units, batching, or transform behavior: start with
  :doc:`five_primitive_functions`.
- A weight is excluded or marked non-temporal: inspect the relation-boundary
  discussion in :doc:`neural_network_layers`.
- State shapes, grouping, initialization, or reset behavior are unclear: use
  :doc:`hidden_states`.

.. toctree::
   :hidden:
   :maxdepth: 1

   five_primitive_functions.ipynb
   neural_network_layers.ipynb
   hidden_states.ipynb
