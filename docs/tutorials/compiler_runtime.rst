Online Learning Compilers
=========================

Inspect what BrainTrace compiled before interpreting an online-learning result.
These chapters connect model structure to hidden groups, ETP relations, and
compiler diagnostics.

Inspect the runtime
-------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Graph Compilation
      :link: graph_compilation
      :link-type: doc

      Compile a recurrent model, inspect its ETP graph, and use
      :func:`braintrace.compile_etrace_graph` when an algorithm wrapper is not
      required.

   .. grid-item-card:: Visualization
      :link: visualization
      :link-type: doc

      Read ``learner.report`` and ``learner.graph`` to inspect included,
      excluded, and grouped model state.

Diagnostic workflow
-------------------

1. Compile the smallest representative input and confirm that compilation
   completes without errors.
2. Check the discovered hidden groups and their state paths.
3. Compare ETP weights with excluded or non-temporal weights.
4. Inspect each weight-primitive-hidden relation rather than relying only on
   relation counts.
5. Resolve structural diagnostics before evaluating an algorithm's learning
   behavior.

.. important::

   Graphs and reports establish what the compiler selected. They do not, by
   themselves, demonstrate that an approximate learning rule matches BPTT or
   produces a valid descent direction.

.. toctree::
   :hidden:
   :maxdepth: 1

   graph_compilation.ipynb
   visualization.ipynb
