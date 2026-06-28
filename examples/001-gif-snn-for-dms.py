# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# see braintrace documentations for more details.

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import brainstate
import braintools
import matplotlib
matplotlib.use('Agg')  # headless backend: render to file, no display needed
import matplotlib.pyplot as plt
import numpy as np
import brainunit as u

from snn_models import DMSDataset, GifNet, OnlineTrainer


def main(
    *,
    batch_size: int = 128,
    num_batch: int = 100,
    n_rec: int = 200,
    t_fixation: float = 10.,   # ms
    t_sample: float = 500.,    # ms
    t_delay: float = 1000.,    # ms
    t_test: float = 500.,      # ms
    plot: bool = True,
) -> dict:
    with brainstate.environ.context(dt=1. * u.ms):
        data = DMSDataset(
            bg_fr=1. * u.Hz,
            t_fixation=t_fixation * u.ms,
            t_sample=t_sample * u.ms,
            t_delay=t_delay * u.ms,
            t_test=t_test * u.ms,
            n_input=100,
            firing_rate=100. * u.Hz,
            batch_size=batch_size,
            num_batch=num_batch,
        )

        net = GifNet(
            n_in=data.num_inputs,
            n_rec=n_rec,
            n_out=data.num_outputs,
            tau_neu=100. * u.ms,
            tau_syn=100. * u.ms,
            tau_I2=1500. * u.ms,
            A2=1. * u.mA,
        )

        onliner = OnlineTrainer(
            target=net,
            opt=braintools.optim.Adam(lr=1e-3),
            dataset=data,
            n_sim=data.n_sim,
            x_fun=lambda x_local: np.transpose(x_local, (1, 0, 2))
        )
        losses, accs = onliner.f_train()

    if plot:
        fig, gs = braintools.visualize.get_figure(1, 2, 4., 5.)
        fig.add_subplot(gs[0, 0])
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        fig.add_subplot(gs[0, 1])
        plt.plot(accs)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()

    return {"losses": list(losses), "accs": list(accs)}


if __name__ == '__main__':
    main()
