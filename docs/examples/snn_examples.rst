Spiking Neural Network Examples
===============================

Start here for task-level SNN training and performance evaluation. The runnable
scripts live in the repository's
`examples directory <https://github.com/chaobrain/braintrace/tree/main/examples>`__.

Learning tasks
--------------

* `000-lif-snn-for-nmnist.py <https://github.com/chaobrain/braintrace/blob/main/examples/000-lif-snn-for-nmnist.py>`__
  trains a recurrent LIF-delta SNN on framed N-MNIST events. API:
  :doc:`pp-prop </apis/algorithm_details/braintrace.pp_prop>`.
* `001-gif-snn-for-dms.py <https://github.com/chaobrain/braintrace/blob/main/examples/001-gif-snn-for-dms.py>`__
  trains a GIF recurrent SNN on delayed matching-to-sample. API:
  :doc:`pp-prop </apis/algorithm_details/braintrace.pp_prop>`.
* `002-coba-ei-rsnn.py <https://github.com/chaobrain/braintrace/blob/main/examples/002-coba-ei-rsnn.py>`__
  trains an excitatory/inhibitory recurrent SNN on an evidence-accumulation
  task with configurable current- or conductance-based synapses. API:
  :doc:`SignedWLinear </apis/generated/braintrace.nn.SignedWLinear>`.
* `004-feedforward-conv-snn.py <https://github.com/chaobrain/braintrace/blob/main/examples/004-feedforward-conv-snn.py>`__
  builds a feed-forward convolutional SNN with online and BPTT trainers. API:
  :doc:`Conv2d </apis/generated/braintrace.nn.Conv2d>`.

Performance examples
--------------------

These scripts compare time and memory under synthetic, batched-state, and
per-sample ``vmap`` execution:

* `003-snn-memory-and-speed-evaluation-all.py <https://github.com/chaobrain/braintrace/blob/main/examples/003-snn-memory-and-speed-evaluation-all.py>`__
  runs the complete comparison. API:
  :doc:`D-RTRL </apis/algorithm_details/braintrace.D_RTRL>`.
* `003-snn-memory-and-speed-evaluation-batched.py <https://github.com/chaobrain/braintrace/blob/main/examples/003-snn-memory-and-speed-evaluation-batched.py>`__
  uses batched state. API:
  :doc:`compile </apis/algorithm_details/braintrace.compile>`.
* `003-snn-memory-and-speed-evaluation-vmap.py <https://github.com/chaobrain/braintrace/blob/main/examples/003-snn-memory-and-speed-evaluation-vmap.py>`__
  uses per-sample ``vmap`` execution. API:
  :doc:`compile </apis/algorithm_details/braintrace.compile>`.

Use these scripts for implementation benchmarking, not as evidence that two
algorithms have equivalent gradient accuracy.

Begin with the :doc:`SNN Online Learning </tutorials/snn_online_learning>`
workflow before moving to dataset-scale scripts.

Related API
-----------

* :doc:`pp-prop </apis/algorithm_details/braintrace.pp_prop>`
* :doc:`EProp </apis/algorithm_details/braintrace.EProp>`
* :doc:`OSTLRecurrent </apis/algorithm_details/braintrace.OSTLRecurrent>`
* :doc:`Neural Network Layers </apis/nn>`
