<h1 align="center">BrainTrace</h1>
<h2 align="center">Eligibility Trace-based Online Learning for Brain Dynamics</h2>

<p align="center">
  	<img alt="Header image of braintrace." src="https://raw.githubusercontent.com/chaobrain/braintrace/main/docs/_static/braintrace.png" width=40%>
</p>

<p align="center">
	<a href="https://pypi.org/project/braintrace/"><img alt="Supported Python Version" src="https://img.shields.io/pypi/pyversions/braintrace"></a>
	<a href="https://github.com/chaobrain/braintrace/blob/main/LICENSE"><img alt="LICENSE" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  	<a href="https://braintrace.readthedocs.io/?badge=latest"><img alt="Documentation" src="https://readthedocs.org/projects/braintrace/badge/?version=latest"></a>
  	<a href="https://badge.fury.io/py/braintrace"><img alt="PyPI version" src="https://badge.fury.io/py/braintrace.svg"></a>
    <a href="https://github.com/chaobrain/braintrace/actions/workflows/CI.yml"><img alt="Continuous Integration" src="https://github.com/chaobrain/braintrace/actions/workflows/CI.yml/badge.svg"></a>
</p>

[``braintrace``](https://github.com/chaobrain/braintrace) provides online learning algorithms for biological neural networks.
It has been integrated into our establishing [brain modeling ecosystem](https://brainmodeling.readthedocs.io/).

## Installation

``braintrace`` can run on Python 3.10+ installed on Linux, MacOS, and Windows. You can install ``braintrace`` via pip:

```bash
pip install braintrace --upgrade
```

Alternatively, you can install `BrainX`, which bundles `braintrace` with other compatible packages for a comprehensive brain modeling ecosystem:

```bash
pip install BrainX -U
```

## Documentation

The official documentation is hosted on Read the Docs: [https://braintrace.readthedocs.io](https://braintrace.readthedocs.io)

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
  url={https://doi.org/10.1038/s41467-026-68453-w}
}

```


## See also the ecosystem

``braintrace`` is one part of our brain simulation ecosystem: https://brainmodeling.readthedocs.io/
