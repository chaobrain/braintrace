<h1 align="center">BrainTrace</h1>
<h2 align="center">Eligibility Trace-based Online Learning for Brain Dynamics</h2>

<p align="center">
  	<img alt="Header image of braintrace." src="https://brainx.chaobrain.com/images/braintrace.webp" width=40%>
</p>

<p align="center">
	<a href="https://pypi.org/project/braintrace/"><img alt="Supported Python Version" src="https://img.shields.io/pypi/pyversions/braintrace"></a>
	<a href="https://github.com/chaobrain/braintrace/blob/main/LICENSE"><img alt="LICENSE" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  	<a href="https://brainx.chaobrain.com/braintrace/"><img alt="Documentation" src="https://readthedocs.org/projects/braintrace/badge/?version=latest"></a>
  	<a href="https://badge.fury.io/py/braintrace"><img alt="PyPI version" src="https://badge.fury.io/py/braintrace.svg"></a>
    <a href="https://github.com/chaobrain/braintrace/actions/workflows/CI.yml"><img alt="Continuous Integration" src="https://github.com/chaobrain/braintrace/actions/workflows/CI.yml/badge.svg"></a>
    <a href="https://codecov.io/gh/chaobrain/braintrace"><img alt="Code Coverage" src="https://codecov.io/gh/chaobrain/braintrace/branch/main/graph/badge.svg"></a>
</p>

[``braintrace``](https://github.com/chaobrain/braintrace) provides online learning algorithms for biological neural networks.
It has been integrated into our establishing [brain modeling ecosystem](https://brainx.chaobrain.com/).

## Installation

``braintrace`` requires Python 3.11 or newer and supports Linux, macOS, and Windows. You can install ``braintrace`` via pip:

```bash
pip install braintrace --upgrade
```

Alternatively, you can install `BrainX`, which bundles `braintrace` with other compatible packages for a comprehensive brain modeling ecosystem:

```bash
pip install BrainX -U
```

## Quickstart

Mark the trainable operations with an ETP primitive (here, via `braintrace.nn`
layers, which use them internally), then compile the model into a learner and
drive a sequence. There is no scan to write and no special parameter class —
weights stay plain `brainstate.ParamState`.

```python
import brainstate
import jax.numpy as jnp
import braintrace

class GRUNet(brainstate.nn.Module):
    def __init__(self, n_in, n_rec, n_out):
        super().__init__()
        self.rnn = braintrace.nn.GRUCell(n_in, n_rec)
        self.out = braintrace.nn.Linear(n_rec, n_out)

    def update(self, x):
        return self.out(self.rnn(x))

model = GRUNet(3, 8, 1)

# One call: initialise states, compile the eligibility-trace graph,
# return a ready learner.
learner = braintrace.compile(model, braintrace.D_RTRL, jnp.zeros((1, 3)), batch_size=1)

def step_loss(x, y):
    return jnp.mean((learner(x) - y) ** 2)

xs = brainstate.random.randn(100, 1, 3)   # (time, batch, features)
ys = brainstate.random.randn(100, 1, 1)

# Drives the sequence and accumulates online gradients -- O(1) memory in T.
grads, losses = learner.etrace_grad(xs, ys, step_fn=step_loss, return_value=True)
```

Swap `braintrace.D_RTRL` for `pp_prop`, `SnAp`, `UORO`, `ThreeFactor`, `DNI`,
`EProp`, the OSTL variants, or an explicit `braintrace.ETraceConfig` to change
the learning rule; nothing else in the snippet changes.

## Documentation

The official documentation is hosted on Read the Docs: [https://brainx.chaobrain.com/braintrace](https://brainx.chaobrain.com/braintrace)

## Citation

If you use this package in your research, please cite:

```bibtex

@Article{Wang2026,
  author={Wang, Chaoming
          and Dong, Xingsi
          and Ji, Zilong
          and Xiao, Mingqing
          and Jiang, Jiedong
          and Liu, Xiao
          and Huan, Yuxiang
          and Wu, Si},
  title={Model-agnostic linear-memory online learning in spiking neural networks},
  journal={Nature Communications},
  year={2026},
  month={Jan},
  day={19},
  abstract={Spiking neural networks (SNNs) offer a promising paradigm for modeling brain dynamics and developing neuromorphic intelligence, yet an online learning system capable of training rich spiking dynamics over long horizons with low memory footprints has been missing. Existing online approaches either incur quadratic memory growth, sacrifice biological fidelity through oversimplified models, or lack end-to-end automated tooling. Here, we introduce BrainTrace, a model-agnostic, linear-memory, and automated online learning system for spiking neural networks. BrainTrace standardizes model specification to encompass diverse neuronal and synaptic dynamics; implements a linear-memory online learning rule by exploiting intrinsic properties of spiking dynamics; and provides a compiler that automatically generates optimized online-learning code for arbitrary user-defined models. Across diverse dynamics and tasks, BrainTrace achieves strong learning performance with a low memory footprint and high computational throughput. Critically, these properties enable online fitting of a whole-brain-scale Drosophila SNN that recapitulates region-level functional activity. By reconciling generality, efficiency, and usability, BrainTrace establishes a foundation for spiking network modeling at scale.},
  issn={2041-1723},
  doi={10.1038/s41467-026-68453-w},
  url={https://doi.org/10.1038/s41467-026-68453-w},
  publisher={Nature Publishing Group UK London}
}

```


## See also the ecosystem

``braintrace`` is one part of our brain simulation ecosystem: https://brainx.chaobrain.com/
