``braintrace`` documentation
============================

`braintrace <https://github.com/chaobrain/braintrace>`_ is designed for the scalable online learning of biological neural networks.

----



Basic Usage
^^^^^^^^^^^


Here we show how easy it is to use `braintrace` to build and train a simple SNN/RNN model.



.. code-block::

   import braintrace
   import brainstate

   # define models as usual
   model = brainstate.nn.Sequential(
       braintrace.nn.GRU(2, 2),
       braintrace.nn.GRU(2, 1),
   )

   # initialize the model
   brainstate.nn.init_all_states(model)

   # the only thing you need to do just two lines of code
   model = braintrace.ParamDimVjpAlgorithm(model)
   model.compile_graph(your_inputs)

   # train your model as usual
   ...

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


``braintrace`` is of part of our `brain simulation ecosystem <https://brainmodeling.readthedocs.io/>`_.


----


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Quickstart

   quickstart/concepts-en.ipynb
   quickstart/concepts-zh.ipynb
   quickstart/snn_online_learning-en.ipynb
   quickstart/snn_online_learning-zh.ipynb
   quickstart/rnn_online_learning-en.ipynb
   quickstart/rnn_online_learning-zh.ipynb



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorial

   tutorial/show_graph-en.ipynb
   tutorial/show_graph-zh.ipynb
   tutorial/etraceop-en.ipynb
   tutorial/etraceop-zh.ipynb
   tutorial/etracestate-en.ipynb
   tutorial/etracestate-zh.ipynb
   tutorial/batching-en.ipynb
   tutorial/batching-zh.ipynb


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Advanced Tutorial

   advanced/IR_analysis-en.ipynb
   advanced/IR_analysis-zh.ipynb
   advanced/limitations-en.ipynb
   advanced/limitations-zh.ipynb
   advanced/online_algorithm_customization-en.ipynb
   advanced/online_algorithm_customization-zh.ipynb




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
   apis/compiler.rst
   apis/algorithms.rst
   apis/nn.rst

