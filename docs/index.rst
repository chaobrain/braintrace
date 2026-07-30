BrainTrace documentation
========================

`BrainTrace <https://github.com/chaobrain/braintrace>`_ implements scalable
online learning for recurrent neural networks (RNNs) and spiking neural
networks (SNNs) using eligibility trace propagation (ETP). Choose an entry
point below to begin.

----


Installation
^^^^^^^^^^^^

Choose the platform that will run JAX. BrainTrace requires Python 3.11 or newer.

.. container:: braintrace-installation

   .. tab-set::

      .. tab-item:: CPU

         .. code-block:: bash

            pip install -U "braintrace[cpu]"

      .. tab-item:: NVIDIA GPU

         CUDA 12:

         .. code-block:: bash

            pip install -U "braintrace[cuda12]"

         CUDA 13:

         .. code-block:: bash

            pip install -U "braintrace[cuda13]"

         The NVIDIA driver, CUDA version, and JAX wheel must be compatible.

      .. tab-item:: TPU

         .. code-block:: bash

            pip install -U "braintrace[tpu]"

      .. tab-item:: Development

         .. code-block:: bash

            pip install -r requirements.txt

For verification and platform troubleshooting, see
:doc:`Installation <quickstart/installation>`.

----


Learn more
^^^^^^^^^^

Choose the entry point that matches your next task.

.. grid:: 1 2 3 3
   :gutter: 2
   :class-container: braintrace-learn-grid

   .. grid-item-card:: :material-regular:`rocket_launch;2em` Installation
      :link: quickstart/installation
      :link-type: doc
      :class-card: braintrace-learn-card

   .. grid-item-card:: :material-regular:`play_circle;2em` Quickstart
      :link: quickstart/quickstart
      :link-type: doc
      :class-card: braintrace-learn-card

   .. grid-item-card:: :material-regular:`hub;2em` Core Concepts
      :link: quickstart/concepts
      :link-type: doc
      :class-card: braintrace-learn-card

   .. grid-item-card:: :material-regular:`memory;2em` RNN Online Learning
      :link: tutorials/rnn_online_learning
      :link-type: doc
      :class-card: braintrace-learn-card

   .. grid-item-card:: :material-regular:`bolt;2em` SNN Online Learning
      :link: tutorials/snn_online_learning
      :link-type: doc
      :class-card: braintrace-learn-card

   .. grid-item-card:: :material-regular:`route;2em` Algorithm Guide
      :link: apis/algorithms
      :link-type: doc
      :class-card: braintrace-learn-card

   .. grid-item-card:: :material-regular:`settings;2em` Advanced Topics
      :link: advanced/compiler_internals
      :link-type: doc
      :class-card: braintrace-learn-card

   .. grid-item-card:: :material-regular:`explore;2em` Examples
      :link: examples/snn_examples
      :link-type: doc
      :class-card: braintrace-learn-card

   .. grid-item-card:: :material-regular:`data_exploration;2em` API Reference
      :link: apis/concepts
      :link-type: doc
      :class-card: braintrace-learn-card


----


Learning path
^^^^^^^^^^^^^

Follow the central path from top to bottom. Branches mark the RNN/SNN workflow
and the two algorithm tutorials.

.. image:: _static/braintrace-learning-map.svg
   :alt: Numbered dependency map of the BrainTrace documentation modules
   :align: center
   :width: 430px

.. container:: text-center

   *A dependency guide, not a required linear syllabus.*

.. container:: braintrace-path-nav

   Foundation
      **1** :doc:`Quickstart <quickstart/quickstart>` · **2**
      :doc:`Core Concepts <quickstart/concepts>`

   Online Training
      **3** :doc:`RNN Online Learning <tutorials/rnn_online_learning>` · **4**
      :doc:`SNN Online Learning <tutorials/snn_online_learning>`

   Algorithm selection
      **5** :doc:`Choose an Algorithm <apis/algorithms>` · **6**
      :doc:`D-RTRL <tutorials/drtrl>` · **7**
      :doc:`pp-prop <tutorials/pp_prop>`

   Internals and application
      **8** :doc:`Operators for Online Learning <tutorials/five_primitive_functions>` · **9**
      :doc:`Graph Compilation <tutorials/graph_compilation>` · **10**
      :doc:`Examples and Advanced Topics <examples/snn_examples>`


----


See also the BrainX ecosystem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BrainTrace is part of `BrainX <https://brainx.chaobrain.com/>`_, an open,
differentiable ecosystem for full-scale brain simulation.


----


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Get started

   quickstart/installation.ipynb
   quickstart/quickstart.ipynb
   quickstart/concepts.ipynb


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorial

   tutorials/online_training.rst
   tutorials/algorithm_tutorials.rst
   tutorials/foundations.rst
   tutorials/compiler_runtime.rst


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Advanced

   advanced/batching.ipynb
   advanced/etp_primitives.ipynb
   advanced/customizing_primitive_transforms.ipynb
   advanced/compiler_internals.ipynb
   advanced/custom_algorithms.ipynb
   advanced/limitations.ipynb


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Examples

   examples/snn_examples.rst
   examples/rnn_examples.rst
   examples/pp_prop_examples.rst
   examples/drtrl_examples.rst


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   Release Notes <changelog.md>
   ETP Operators <apis/concepts.rst>
   Compiler and Executor <apis/compiler.rst>
   Algorithms <apis/algorithms.rst>
   Neural Network Layers <apis/nn.rst>
   Others <apis/primitives.rst>
