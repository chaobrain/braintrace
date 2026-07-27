Spiking Neural Network Examples
===============================

Start here for task-level SNN training and performance evaluation. The runnable
scripts live in the repository's
`examples directory <https://github.com/chaobrain/braintrace/tree/main/examples>`__.

Learning tasks
--------------

* ``000-lif-snn-for-nmnist.py`` trains a recurrent LIF-delta SNN on framed
  N-MNIST events.
* ``001-gif-snn-for-dms.py`` trains a GIF recurrent SNN on delayed
  matching-to-sample.
* ``002-coba-ei-rsnn.py`` trains an excitatory/inhibitory recurrent SNN on an
  evidence-accumulation task with configurable current- or conductance-based
  synapses.
* ``004-feedforward-conv-snn.py`` builds a feed-forward convolutional SNN and
  includes online and BPTT trainers.

Performance examples
--------------------

The three ``003-snn-memory-and-speed-evaluation-*.py`` scripts compare time and
memory under synthetic, batched-state, and per-sample ``vmap`` execution. Use
them for implementation benchmarking, not as evidence that two algorithms
have equivalent gradient accuracy.

Begin with the :doc:`SNN Online Learning </quickstart/snn_online_learning>`
workflow before moving to dataset-scale scripts.

Related API
-----------

* :doc:`pp-prop </apis/generated/braintrace.pp_prop>`
* :doc:`EProp </apis/generated/braintrace.EProp>`
* :doc:`OSTLRecurrent </apis/generated/braintrace.OSTLRecurrent>`
* :doc:`Neural Network Layers </apis/nn>`

