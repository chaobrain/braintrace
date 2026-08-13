Compiler & Runtime
==================

Follow the compiler workflow from a minimal recurrent model to a stacked one.
Each chapter combines model compilation with report and graph inspection so
that discovered hidden groups and ETP relations stay tied to the model that
produced them.

Compile and inspect
-------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Single-Layer RNN
      :link: single_layer_rnn
      :link-type: doc

      Compile one recurrent layer, read its compilation report, inspect its
      ETP graph, and compare temporal and non-temporal parameters.

   .. grid-item-card:: Two-Layer RNN
      :link: two_layer_rnn
      :link-type: doc

      Compile stacked recurrent layers and inspect how the compiler separates
      their hidden groups, retained relations, and excluded parameter paths.

Diagnostic workflow
-------------------

1. Define the recurrent model and compile it with a representative input.
2. Read ``learner.report`` for included, excluded, and diagnostic decisions.
3. Inspect ``learner.graph`` to verify each parameter-primitive-hidden
   relation behind the report.
4. Compare the single-layer and two-layer results before generalizing an
   expected grouping rule.
5. Resolve structural diagnostics before evaluating learning behavior.

.. important::

   Graphs and reports establish what the compiler selected. They do not, by
   themselves, demonstrate that an approximate learning rule matches BPTT or
   produces a valid descent direction.

.. toctree::
   :hidden:
   :maxdepth: 1

   single_layer_rnn.ipynb
   two_layer_rnn.ipynb
