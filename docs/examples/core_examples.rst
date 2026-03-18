Examples
========

Full examples are hosted on GitHub in the
`examples <https://github.com/chaobrain/braintrace/tree/main/examples>`__ directory.
Each example is self-contained and designed to be easily forkable.


Spiking Neural Networks
-----------------------

- **000-lif-snn-for-nmnist.py** — LIF spiking neural network trained on N-MNIST
  using ``ES_D_RTRL`` (IODimVjpAlgorithm) for online learning.

- **001-gif-snn-for-dms.py** — GIF spiking neural network on the Delayed
  Matching-to-Sample cognitive task.

- **002-coba-ei-rsnn.py** — COBA/CUBA excitatory-inhibitory recurrent spiking
  network for evidence accumulation.

- **004-feedforward-conv-snn.py** — Feedforward convolutional SNN on N-MNIST
  using ``braintrace.nn.Conv2d``.


Rate-Based Recurrent Neural Networks
-------------------------------------

- **100-gru-on-copying-task.py** — GRU network on the copying task comparing
  online learning (``D_RTRL``) vs BPTT.

- **101-integrator-rnn.py** — Vanilla RNN as a white noise integrator.


Memory & Speed Benchmarks
-------------------------

- **003-snn-memory-and-speed-evaluation-\*.py** — Benchmarking scripts comparing
  memory usage and speed across different backend strategies (all, batched, vmap).
