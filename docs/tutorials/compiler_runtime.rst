Online Learning Compilers
=========================

Follow one compiler workflow from a minimal recurrent model to a stacked one.
The walkthrough keeps report and graph inspection beside the model that
produced them, then compares the two compiled structures directly.

Compile and inspect
-------------------

.. grid:: 1
   :gutter: 2

   .. grid-item-card:: RNN Compiler Walkthrough
      :link: rnn_compiler
      :link-type: doc

      Compile single-layer and two-layer RNNs in one page, inspect their
      hidden groups and ETP relations, and compare the compiler decisions.

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

   rnn_compiler.ipynb
