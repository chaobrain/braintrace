Online Learning Algorithms
==========================

Study the approximation carried by each algorithm before treating its output
as a gradient estimate. The examples deliberately use different model
structures: a matched task would be easier to compare visually, but it would
not exercise the structural assumptions that distinguish these estimators.

.. note::

   Read :doc:`../quickstart/concepts` first. The chapters assume that you
   understand ETP selection and the role of hidden-state recurrences.

Choose an algorithm
-------------------

.. grid:: 1 1 2 3
   :gutter: 2

   .. grid-item-card:: D-RTRL
      :link: drtrl
      :link-type: doc

      Follow diagonal recurrent traces across a sequence and examine when the
      approximation can differ from BPTT.

   .. grid-item-card:: pp-prop
      :link: pp_prop
      :link-type: doc

      Follow input/output-factorized traces and the contraction that turns
      those factors into parameter gradients.

   .. grid-item-card:: e-prop
      :link: eprop
      :link-type: doc

      Separate local eligibility traces from symmetric or random-feedback
      learning signals in a recurrent spiking network.

   .. grid-item-card:: OSTL
      :link: ostl
      :link-type: doc

      Compare the with-H recurrent rule with the without-H feedforward rule.

   .. grid-item-card:: SnAp
      :link: snap
      :link-type: doc

      Widen a trace over an explicitly sparse recurrent dependency graph.

Select by retained structure
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 29 25 28

   * - Chapter
     - Trace structure
     - Appropriate regime
     - Important boundary
   * - :doc:`D-RTRL <drtrl>`
     - Parameter-shaped trace with a diagonal hidden-Jacobian approximation
     - General recurrent ETP models
     - Cross-position recurrence is approximated
   * - :doc:`pp-prop <pp_prop>`
     - Separate input and output factors
     - Linear-memory recurrent SNN training
     - Factorization error is model dependent
   * - :doc:`e-prop <eprop>`
     - Local trace times a broadcast learning signal
     - Recurrent LIF/ALIF networks
     - Random feedback is not symmetric feedback
   * - :doc:`OSTL <ostl>`
     - With-H or without-H temporal factor
     - Recurrent or feedforward SNNs, respectively
     - The two regimes are not interchangeable
   * - :doc:`SnAp <snap>`
     - Sparse n-step recurrent neighborhood
     - Structurally sparse recurrence
     - Dense recurrence saturates too early to be informative

For a first pass, read D-RTRL before pp-prop, then choose the SNN-specific or
sparsity-specific chapter that matches the model. No chapter establishes
universal equality with BPTT; each states the regime and checks appropriate to
its approximation.

.. toctree::
   :hidden:
   :maxdepth: 1

   drtrl.ipynb
   pp_prop.ipynb
   eprop.ipynb
   ostl.ipynb
   snap.ipynb
