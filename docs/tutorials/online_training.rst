Online Learning Routines
========================

Take a model from definition to an executable online-learning workflow. These
chapters use compact tasks to make state handling, trace updates, and training
behavior inspectable.

.. note::

   Complete :doc:`../quickstart/quickstart` and
   :doc:`../quickstart/concepts` first if you have not yet compiled a
   BrainTrace model.

Choose a workflow
-----------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: RNN Online Learning
      :link: rnn_online_learning
      :link-type: doc

      Train a GRU on the copying task with D-RTRL, then compare the online
      workflow with a BPTT baseline.

      **Best for:** continuous hidden states and sequence-memory tasks.

   .. grid-item-card:: SNN Online Learning
      :link: snn_online_learning
      :link-type: doc

      Build a recurrent LIF network, train it with pp-prop, and inspect how
      factorized traces differ from D-RTRL.

      **Best for:** spike-based dynamics, surrogate gradients, and physical
      units.

What both workflows establish
-----------------------------

- how model state is initialized and reset between sequences;
- where :func:`braintrace.compile` enters the training pipeline;
- how repeated updates are executed with compiled stateful transforms; and
- which conclusions are specific to the demonstrated task and approximation.

.. toctree::
   :hidden:
   :maxdepth: 1

   rnn_online_learning.ipynb
   snn_online_learning.ipynb
