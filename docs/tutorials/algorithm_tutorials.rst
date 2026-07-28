Algorithm Tutorials
===================

Study the approximation carried by each algorithm before treating its output
as a gradient estimate. Both chapters use a matched MiniGRU task so the
algorithmic difference remains the main variable.

.. note::

   Read :doc:`../quickstart/concepts` first. The chapters assume that you
   understand ETP selection and the role of hidden-state recurrences.

Choose an algorithm
-------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: D-RTRL
      :link: drtrl
      :link-type: doc

      Follow diagonal recurrent traces across a sequence and examine when the
      approximation can differ from BPTT.

   .. grid-item-card:: pp_prop
      :link: pp_prop
      :link-type: doc

      Follow input/output-factorized traces and the contraction that turns
      those factors into parameter gradients.

Compare the algorithms
----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Chapter
     - Trace structure
     - Primary question
   * - :doc:`D-RTRL <drtrl>`
     - Parameter-shaped trace with a diagonal hidden-Jacobian approximation
     - What recurrent influence is retained, and why can it differ from BPTT?
   * - :doc:`pp_prop <pp_prop>`
     - Separate input and output factors
     - How does factorization change trace propagation and gradient assembly?

For a first pass, read D-RTRL before pp_prop. Neither chapter establishes
universal equality with BPTT; each states the regime and checks appropriate to
its approximation.

.. toctree::
   :hidden:
   :maxdepth: 1

   drtrl.ipynb
   pp_prop.ipynb
