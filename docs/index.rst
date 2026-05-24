``braintrace`` documentation
============================

`braintrace <https://github.com/chaobrain/braintrace>`_ implements scalable online learning for recurrent neural networks (RNNs) and spiking neural networks (SNNs) using eligibility trace propagation (ETP).

The key idea: mark weight operations with **ETP primitives** (``braintrace.matmul``, ``braintrace.conv``, etc.) to include them in online learning. Regular JAX operations are automatically excluded — no special parameter classes needed.

----


Basic Usage
^^^^^^^^^^^

.. code-block:: python

   import braintrace
   import brainstate

   class MyRNN(brainstate.nn.Module):
       def __init__(self):
           super().__init__()
           self.rnn = braintrace.nn.GRUCell(10, 64)
           self.out = braintrace.nn.Linear(64, 10)

       def update(self, x):
           return self.out(self.rnn(x))

   model = MyRNN()
   model.init_all_states()

   # Wrap with an online learning algorithm (just 2 lines)
   trainer = braintrace.D_RTRL(model)
   trainer.compile_graph(example_input)

   # Now use brainstate.transform.grad as usual — gradients are
   # computed online via eligibility traces, not BPTT.


----


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U braintrace[cpu]

    .. tab-item:: GPU (CUDA 12.0)

       .. code-block:: bash

          pip install -U braintrace[cuda12]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U braintrace[tpu]


----


See also the ecosystem
^^^^^^^^^^^^^^^^^^^^^^

``braintrace`` is part of the `brain simulation ecosystem <https://brainx.chaobrain.com/>`_.


----


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Quickstart

   quickstart/concepts.ipynb
   quickstart/rnn_online_learning.ipynb
   quickstart/snn_online_learning.ipynb


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorial

   tutorials/etp_primitives.ipynb
   tutorials/hidden_states.ipynb
   tutorials/graph_visualization.ipynb
   tutorials/batching.ipynb


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Advanced

   advanced/compiler_internals.ipynb
   advanced/custom_algorithms.ipynb
   advanced/limitations.ipynb


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Examples

   examples/core_examples.rst


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   changelog.md
   apis/concepts.rst
   apis/primitives.rst
   apis/compiler.rst
   apis/algorithms.rst
   apis/nn.rst
