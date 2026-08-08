Online Learning Networks
========================

Build an online-learning-ready recurrent model in three connected steps. First,
ETP operators mark the parameterized computations that participate in online
learning. Next, ``braintrace.nn`` layers compose those operators into reusable
network blocks. Finally, hidden states provide the temporal destinations that
the compiler connects to the marked parameter paths.

Choose a foundation
-------------------

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: Operators for Online Learning
      :link: five_primitive_functions
      :link-type: doc

      Mark trainable computation paths with the five public ETP operators and
      verify their batching, unit, and JAX-transform contracts.

   .. grid-item-card:: Neural Network Layers for Online Learning
      :link: neural_network_layers
      :link-type: doc

      Compose the marked operators through ``braintrace.nn`` layers while
      preserving operation-based parameter selection and relation boundaries.

   .. grid-item-card:: Hidden States for Online Learning
      :link: hidden_states
      :link-type: doc

      Define and initialize the recurrent state that makes a model temporal,
      then inspect the hidden groups discovered by the compiler.

Recommended sequence
--------------------

.. list-table::
   :header-rows: 1
   :widths: 12 30 58

   * - Step
     - Chapter
     - Use it to
   * - 1
     - :doc:`Operators for Online Learning <five_primitive_functions>`
     - Mark parameter operations for online learning and verify their unit and
       JAX-transform behavior.
   * - 2
     - :doc:`Neural Network Layers for Online Learning <neural_network_layers>`
     - Assemble marked operations into recurrent models without violating
       parameter-to-hidden relation boundaries.
   * - 3
     - :doc:`Hidden States for Online Learning <hidden_states>`
     - Make the model temporal and understand how the compiler forms hidden
       groups from recurrent state.

Where to look first
-------------------

- Unexpected units, batching, or transform behavior: start with
  :doc:`five_primitive_functions`.
- A weight is excluded or marked non-temporal: inspect the relation-boundary
  discussion in :doc:`neural_network_layers`.
- State shapes, grouping, initialization, or reset behavior are unclear: use
  :doc:`hidden_states`.

Related API reference
---------------------

- Operator signatures and core ETP types: :doc:`../apis/concepts`.
- ETP-aware layer classes: :doc:`../apis/nn`.
- Hidden-group discovery, compilation, and diagnostics: :doc:`../apis/compiler`.

.. toctree::
   :hidden:
   :maxdepth: 1

   five_primitive_functions.ipynb
   neural_network_layers.ipynb
   hidden_states.ipynb
