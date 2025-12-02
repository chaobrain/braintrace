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
# -*- coding: utf-8 -*-

from typing import Callable, Union

import brainstate
import braintools
import brainunit as u

from braintrace._etrace_concepts import (
    ETraceParam,
    ElemWiseParam,
)
from braintrace._typing import ArrayLike
from ._linear import Linear

__all__ = [
    'ValinaRNNCell',
    'GRUCell',
    'MGUCell',
    'LSTMCell',
    'URLSTMCell',
    'MinimalRNNCell',
    'MiniGRU',
    'MiniLSTM',
    'LRUCell',
]


class ValinaRNNCell(brainstate.nn.RNNCell):
    """Vanilla RNN cell.

    A basic recurrent neural network cell that applies a simple recurrent transformation
    to the input and previous hidden state.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        The number of input units.
    out_size : brainstate.typing.Size
        The number of hidden units.
    state_init : Callable or ArrayLike, optional
        The state initializer. Default is ZeroInit().
    w_init : Callable or ArrayLike, optional
        The input weight initializer. Default is XavierNormal().
    b_init : Callable or ArrayLike, optional
        The bias weight initializer. Default is ZeroInit().
    activation : str or Callable, optional
        The activation function. It can be a string or a callable function. Default is 'relu'.
    name : str or None, optional
        The name of the module. Default is None.
    param_type : type, optional
        The type of the parameter. Default is ETraceParam.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create a Vanilla RNN cell
        >>> rnn_cell = braintrace.nn.ValinaRNNCell(in_size=32, out_size=64)
        >>> rnn_cell.init_state(batch_size=8)
        >>>
        >>> # Process a sequence of inputs
        >>> x = brainstate.random.randn(8, 32)
        >>> h = rnn_cell(x)
        >>> print(h.shape)
        (8, 64)
    """
    __module__ = 'braintrace.nn'

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        out_size: brainstate.typing.Size,
        state_init: Union[ArrayLike, Callable] = braintools.init.ZeroInit(),
        w_init: Union[ArrayLike, Callable] = braintools.init.XavierNormal(),
        b_init: Union[ArrayLike, Callable] = braintools.init.ZeroInit(),
        activation: str | Callable = 'relu',
        name: str = None,
        param_type: type = ETraceParam,
    ):
        super().__init__(name=name)

        # parameters
        self._state_initializer = state_init
        self.out_size = out_size
        self.in_size = in_size

        # activation function
        if isinstance(activation, str):
            self.activation = getattr(brainstate.functional, activation)
        else:
            assert callable(activation), "The activation function should be a string or a callable function. "
            self.activation = activation

        # weights
        self.W = Linear(
            self.in_size[-1] + self.out_size[-1], self.out_size[-1],
            w_init=w_init,
            b_init=b_init,
            param_type=param_type
        )

    def init_state(self, batch_size: int = None, **kwargs):
        self.h = brainstate.HiddenState(
            braintools.init.param(self._state_initializer, self.out_size, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = braintools.init.param(self._state_initializer, self.out_size, batch_size)

    def update(self, x):
        xh = u.math.concatenate([x, self.h.value], axis=-1)
        self.h.value = self.activation(self.W(xh))
        return self.h.value


class GRUCell(brainstate.nn.RNNCell):
    r"""Gated Recurrent Unit (GRU) cell.

    Gated Recurrent Unit (GRU) cell, implemented as in
    `Learning Phrase Representations using RNN Encoder-Decoder for
    Statistical Machine Translation <https://arxiv.org/abs/1406.1078>`_.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        The number of input units.
    out_size : brainstate.typing.Size
        The number of hidden units.
    w_init : Callable or ArrayLike, optional
        The input weight initializer. Default is Orthogonal().
    b_init : Callable or ArrayLike, optional
        The bias weight initializer. Default is ZeroInit().
    state_init : Callable or ArrayLike, optional
        The state initializer. Default is ZeroInit().
    activation : str or Callable, optional
        The activation function. It can be a string or a callable function. Default is 'tanh'.
    name : str or None, optional
        The name of the module. Default is None.
    param_type : type, optional
        The type of the parameter. Default is ETraceParam.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create a GRU cell
        >>> gru_cell = braintrace.nn.GRUCell(in_size=128, out_size=256)
        >>> gru_cell.init_state(batch_size=16)
        >>>
        >>> # Process a sequence of inputs
        >>> x = brainstate.random.randn(16, 128)
        >>> h = gru_cell(x)
        >>> print(h.shape)
        (16, 256)
    """
    __module__ = 'braintrace.nn'

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        out_size: brainstate.typing.Size,
        w_init: Union[ArrayLike, Callable] = braintools.init.Orthogonal(),
        b_init: Union[ArrayLike, Callable] = braintools.init.ZeroInit(),
        state_init: Union[ArrayLike, Callable] = braintools.init.ZeroInit(),
        activation: str | Callable = 'tanh',
        name: str = None,
        param_type: type = ETraceParam,
    ):
        super().__init__(name=name)

        # parameters
        self._state_initializer = state_init
        self.out_size = out_size
        self.in_size = in_size

        # activation function
        if isinstance(activation, str):
            self.activation = getattr(brainstate.functional, activation)
        else:
            assert callable(activation), "The activation function should be a string or a callable function. "
            self.activation = activation

        # weights
        params = dict(w_init=w_init, b_init=b_init, param_type=param_type)
        self.Wz = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], **params)
        self.Wr = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], **params)
        self.Wh = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], **params)

    def init_state(self, batch_size: int = None, **kwargs):
        self.h = brainstate.HiddenState(braintools.init.param(self._state_initializer, self.out_size, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = braintools.init.param(self._state_initializer, self.out_size, batch_size)

    def update(self, x):
        old_h = self.h.value
        xh = u.math.concatenate([x, old_h], axis=-1)
        z = brainstate.nn.sigmoid(self.Wz(xh))
        r = brainstate.nn.sigmoid(self.Wr(xh))
        rh = r * old_h
        h = self.activation(self.Wh(u.math.concatenate([x, rh], axis=-1)))
        h = (1 - z) * old_h + z * h
        self.h.value = h
        return h


class CFNCell(brainstate.nn.RNNCell):
    r"""Chaos Free Networks (CFN) cell.

    Chaos Free Networks (CFN) cell, implemented as in
    `A recurrent neural network without chaos <https://arxiv.org/abs/1612.06212>`_.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        The number of input units.
    out_size : brainstate.typing.Size
        The number of hidden units.
    w_init : Callable or ArrayLike, optional
        The input weight initializer. Default is Orthogonal().
    b_init : Callable or ArrayLike, optional
        The bias weight initializer. Default is ZeroInit().
    state_init : Callable or ArrayLike, optional
        The state initializer. Default is ZeroInit().
    activation : str or Callable, optional
        The activation function. It can be a string or a callable function. Default is 'tanh'.
    name : str or None, optional
        The name of the module. Default is None.
    param_type : type, optional
        The type of the parameter. Default is ETraceParam.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create a CFN cell
        >>> cfn_cell = braintrace.nn.CFNCell(in_size=64, out_size=128)
        >>> cfn_cell.init_state(batch_size=10)
        >>>
        >>> # Process a sequence of inputs
        >>> x = brainstate.random.randn(10, 64)
        >>> h = cfn_cell(x)
        >>> print(h.shape)
        (10, 128)
    """
    __module__ = 'braintrace.nn'

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        out_size: brainstate.typing.Size,
        w_init: Union[ArrayLike, Callable] = braintools.init.Orthogonal(),
        b_init: Union[ArrayLike, Callable] = braintools.init.ZeroInit(),
        state_init: Union[ArrayLike, Callable] = braintools.init.ZeroInit(),
        activation: str | Callable = 'tanh',
        name: str = None,
        param_type: type = ETraceParam,
    ):
        super().__init__(name=name)

        # parameters
        self._state_initializer = state_init
        self.out_size = out_size
        self.in_size = in_size

        # activation function
        if isinstance(activation, str):
            self.activation = getattr(brainstate.functional, activation)
        else:
            assert callable(activation), "The activation function should be a string or a callable function. "
            self.activation = activation

        # weights
        params = dict(w_init=w_init, b_init=b_init, param_type=param_type)
        self.Wf = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], **params)
        self.Wi = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], **params)
        self.Wh = Linear(self.out_size[-1], self.out_size[-1], **params)

    def init_state(self, batch_size: int = None, **kwargs):
        self.h = brainstate.HiddenState(braintools.init.param(self._state_initializer, self.out_size, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = braintools.init.param(self._state_initializer, self.out_size, batch_size)

    def update(self, x):
        old_h = self.h.value
        xh = u.math.concatenate([x, old_h], axis=-1)
        f = brainstate.nn.sigmoid(self.Wf(xh))
        i = brainstate.nn.sigmoid(self.Wi(xh))
        h = f * self.activation(old_h) + i * self.activation(self.Wh(x))
        self.h.value = h
        return h


class MGUCell(brainstate.nn.RNNCell):
    r"""Minimal Gated Recurrent Unit (MGU) cell.

    Minimal Gated Recurrent Unit (MGU) cell, implemented as in
    `Minimal Gated Unit for Recurrent Neural Networks <https://arxiv.org/abs/1603.09420>`_.

    .. math::

       \begin{aligned}
       f_{t}&=\sigma (W_{f}x_{t}+U_{f}h_{t-1}+b_{f})\\
       {\hat {h}}_{t}&=\phi (W_{h}x_{t}+U_{h}(f_{t}\odot h_{t-1})+b_{h})\\
       h_{t}&=(1-f_{t})\odot h_{t-1}+f_{t}\odot {\hat {h}}_{t}
       \end{aligned}

    where:

    - :math:`x_{t}`: input vector
    - :math:`h_{t}`: output vector
    - :math:`{\hat {h}}_{t}`: candidate activation vector
    - :math:`f_{t}`: forget vector
    - :math:`W, U, b`: parameter matrices and vector

    Parameters
    ----------
    in_size : brainstate.typing.Size
        The number of input units.
    out_size : brainstate.typing.Size
        The number of hidden units.
    w_init : Callable or ArrayLike, optional
        The input weight initializer. Default is Orthogonal().
    b_init : Callable or ArrayLike, optional
        The bias weight initializer. Default is ZeroInit().
    state_init : Callable or ArrayLike, optional
        The state initializer. Default is ZeroInit().
    activation : str or Callable, optional
        The activation function. It can be a string or a callable function. Default is 'tanh'.
    name : str or None, optional
        The name of the module. Default is None.
    param_type : type, optional
        The type of the parameter. Default is ETraceParam.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create an MGU cell
        >>> mgu_cell = braintrace.nn.MGUCell(in_size=96, out_size=192)
        >>> mgu_cell.init_state(batch_size=12)
        >>>
        >>> # Process a sequence of inputs
        >>> x = brainstate.random.randn(12, 96)
        >>> h = mgu_cell(x)
        >>> print(h.shape)
        (12, 192)
    """
    __module__ = 'braintrace.nn'

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        out_size: brainstate.typing.Size,
        w_init: Union[ArrayLike, Callable] = braintools.init.Orthogonal(),
        b_init: Union[ArrayLike, Callable] = braintools.init.ZeroInit(),
        state_init: Union[ArrayLike, Callable] = braintools.init.ZeroInit(),
        activation: str | Callable = 'tanh',
        name: str = None,
        param_type: type = ETraceParam,
    ):
        super().__init__(name=name)

        # parameters
        self._state_initializer = state_init
        self.out_size = out_size
        self.in_size = in_size

        # activation function
        if isinstance(activation, str):
            self.activation = getattr(brainstate.functional, activation)
        else:
            assert callable(activation), "The activation function should be a string or a callable function. "
            self.activation = activation

        # weights
        params = dict(w_init=w_init, b_init=b_init, param_type=param_type)
        self.Wf = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], **params)
        self.Wh = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], **params)

    def init_state(self, batch_size: int = None, **kwargs):
        self.h = brainstate.HiddenState(braintools.init.param(self._state_initializer, self.out_size, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = braintools.init.param(self._state_initializer, self.out_size, batch_size)

    def update(self, x):
        old_h = self.h.value
        xh = u.math.concatenate([x, old_h], axis=-1)
        f = brainstate.nn.sigmoid(self.Wf(xh))
        fh = f * old_h
        h = self.activation(self.Wh(u.math.concatenate([x, fh], axis=-1)))
        self.h.value = (1 - f) * self.h.value + f * h
        return self.h.value


class LSTMCell(brainstate.nn.RNNCell):
    r"""Long short-term memory (LSTM) RNN core.

    The implementation is based on (zaremba, et al., 2014) [1]_. Given
    :math:`x_t` and the previous state :math:`(h_{t-1}, c_{t-1})` the core
    computes

    .. math::

       \begin{array}{ll}
       i_t = \sigma(W_{ii} x_t + W_{hi} h_{t-1} + b_i) \\
       f_t = \sigma(W_{if} x_t + W_{hf} h_{t-1} + b_f) \\
       g_t = \tanh(W_{ig} x_t + W_{hg} h_{t-1} + b_g) \\
       o_t = \sigma(W_{io} x_t + W_{ho} h_{t-1} + b_o) \\
       c_t = f_t c_{t-1} + i_t g_t \\
       h_t = o_t \tanh(c_t)
       \end{array}

    where :math:`i_t`, :math:`f_t`, :math:`o_t` are input, forget and
    output gate activations, and :math:`g_t` is a vector of cell updates.

    The output is equal to the new hidden, :math:`h_t`.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        The dimension of the input vector.
    out_size : brainstate.typing.Size
        The number of hidden unit in the node.
    w_init : Callable or ArrayLike, optional
        The input weight initializer. Default is XavierNormal().
    b_init : Callable or ArrayLike, optional
        The bias weight initializer. Default is ZeroInit().
    state_init : Callable or ArrayLike, optional
        The state initializer. Default is ZeroInit().
    activation : str or Callable, optional
        The activation function. It can be a string or a callable function. Default is 'tanh'.
    name : str or None, optional
        The name of the module. Default is None.
    param_type : type, optional
        The type of the parameter. Default is ETraceParam.

    Notes
    -----
    Forget gate initialization: Following (Jozefowicz, et al., 2015) [2]_ we add 1.0
    to :math:`b_f` after initialization in order to reduce the scale of forgetting in
    the beginning of the training.

    References
    ----------
    .. [1] Zaremba, Wojciech, Ilya Sutskever, and Oriol Vinyals. "Recurrent neural
           network regularization." arXiv preprint arXiv:1409.2329 (2014).
    .. [2] Jozefowicz, Rafal, Wojciech Zaremba, and Ilya Sutskever. "An empirical
           exploration of recurrent network architectures." In International conference
           on machine learning, pp. 2342-2350. PMLR, 2015.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create an LSTM cell
        >>> lstm_cell = braintrace.nn.LSTMCell(in_size=256, out_size=512)
        >>> lstm_cell.init_state(batch_size=20)
        >>>
        >>> # Process a sequence of inputs
        >>> x = brainstate.random.randn(20, 256)
        >>> h = lstm_cell(x)
        >>> print(h.shape)
        (20, 512)
    """
    __module__ = 'braintrace.nn'

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        out_size: brainstate.typing.Size,
        w_init: Union[ArrayLike, Callable] = braintools.init.XavierNormal(),
        b_init: Union[ArrayLike, Callable] = braintools.init.ZeroInit(),
        state_init: Union[ArrayLike, Callable] = braintools.init.ZeroInit(),
        activation: str | Callable = 'tanh',
        name: str = None,
        param_type: type = ETraceParam,
    ):
        super().__init__(name=name)

        # parameters
        self.out_size = out_size
        self.in_size = in_size

        # initializers
        self._state_initializer = state_init

        # activation function
        if isinstance(activation, str):
            self.activation = getattr(brainstate.functional, activation)
        else:
            assert callable(activation), "The activation function should be a string or a callable function. "
            self.activation = activation

        # weights
        params = dict(w_init=w_init, b_init=b_init, param_type=param_type)
        self.Wi = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], **params)
        self.Wg = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], **params)
        self.Wf = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], **params)
        self.Wo = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], **params)

    def init_state(self, batch_size: int = None, **kwargs):
        self.c = brainstate.HiddenState(braintools.init.param(self._state_initializer, self.out_size, batch_size))
        self.h = brainstate.HiddenState(braintools.init.param(self._state_initializer, self.out_size, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.c.value = braintools.init.param(self._state_initializer, self.out_size, batch_size)
        self.h.value = braintools.init.param(self._state_initializer, self.out_size, batch_size)

    def update(self, x):
        h, c = self.h.value, self.c.value
        xh = u.math.concatenate([x, h], axis=-1)
        i = self.Wi(xh)
        g = self.Wg(xh)
        f = self.Wf(xh)
        o = self.Wo(xh)
        c = brainstate.nn.sigmoid(f + 1.) * c + brainstate.nn.sigmoid(i) * self.activation(g)
        h = brainstate.nn.sigmoid(o) * self.activation(c)
        self.h.value = h
        self.c.value = c
        return h


class URLSTMCell(brainstate.nn.RNNCell):
    """Update-Reset LSTM (URLSTM) cell.

    A variant of LSTM that uses update and reset gates for more flexible
    control over the cell state dynamics.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        The dimension of the input vector.
    out_size : brainstate.typing.Size
        The number of hidden units in the node.
    w_init : Callable or ArrayLike, optional
        The input weight initializer. Default is XavierNormal().
    state_init : Callable or ArrayLike, optional
        The state initializer. Default is ZeroInit().
    activation : str or Callable, optional
        The activation function. It can be a string or a callable function. Default is 'tanh'.
    name : str or None, optional
        The name of the module. Default is None.
    param_type : type, optional
        The type of the parameter. Default is ETraceParam.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create a URLSTM cell
        >>> urlstm_cell = braintrace.nn.URLSTMCell(in_size=128, out_size=256)
        >>> urlstm_cell.init_state(batch_size=16)
        >>>
        >>> # Process a sequence of inputs
        >>> x = brainstate.random.randn(16, 128)
        >>> h = urlstm_cell(x)
        >>> print(h.shape)
        (16, 256)
    """
    __module__ = 'braintrace.nn'

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        out_size: brainstate.typing.Size,
        w_init: Union[ArrayLike, Callable] = braintools.init.XavierNormal(),
        state_init: Union[ArrayLike, Callable] = braintools.init.ZeroInit(),
        activation: str | Callable = 'tanh',
        name: str = None,
        param_type: type = ETraceParam,
    ):
        super().__init__(name=name)

        # parameters
        self.out_size = out_size
        self.in_size = in_size

        # initializers
        self._state_initializer = state_init

        # activation function
        if isinstance(activation, str):
            self.activation = getattr(brainstate.functional, activation)
        else:
            assert callable(activation), "The activation function should be a string or a callable function. "
            self.activation = activation

        # weights
        params = dict(w_init=w_init, b_init=None, param_type=param_type)
        self.Wu = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], **params)
        self.Wf = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], **params)
        self.Wr = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], **params)
        self.Wo = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], **params)
        self.bias = param_type(self._forget_bias(), op=u.math.add, grad='full')

    def _forget_bias(self):
        u = brainstate.random.uniform(1 / self.out_size[-1], 1 - 1 / self.out_size[1], (self.out_size[-1],))
        return -u.math.log(1 / u - 1)

    def init_state(self, batch_size: int = None, **kwargs):
        self.c = brainstate.HiddenState(
            braintools.init.param(self._state_initializer, self.out_size, batch_size))
        self.h = brainstate.HiddenState(
            braintools.init.param(self._state_initializer, self.out_size, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.c.value = braintools.init.param(self._state_initializer, self.out_size, batch_size)
        self.h.value = braintools.init.param(self._state_initializer, self.out_size, batch_size)

    def update(self, x: ArrayLike) -> ArrayLike:
        h, c = self.h.value, self.c.value
        xh = u.math.concatenate([x, h], axis=-1)
        f = self.Wf(xh)
        r = self.Wr(xh)
        u_ = self.Wu(xh)
        o = self.Wo(xh)
        f_ = brainstate.nn.sigmoid(self.bias.execute(f))
        r_ = brainstate.nn.sigmoid(-self.bias.execute(-r))
        g = 2 * r_ * f_ + (1 - 2 * r_) * f_ ** 2
        next_cell = g * c + (1 - g) * self.activation(u_)
        next_hidden = brainstate.nn.sigmoid(o) * self.activation(next_cell)
        self.h.value = next_hidden
        self.c.value = next_cell
        return next_hidden


class MinimalRNNCell(brainstate.nn.RNNCell):
    r"""Minimal RNN Cell.

    Minimal RNN Cell, implemented as in
    `MinimalRNN: Toward More Interpretable and Trainable Recurrent Neural Networks <https://arxiv.org/abs/1711.06788>`_

    At each step :math:`t`, the model first maps its input :math:`\mathbf{x}_t` to a
    latent space through :math:`\mathbf{z}_t=\Phi(\mathbf{x}_t)`. :math:`\Phi(\cdot)`
    here can be any highly flexible functions such as neural networks. By default,
    we take :math:`\Phi(\cdot)` as a fully connected layer with tanh activation.

    Given the latent representation :math:`\mathbf{z}_t` of the input, MinimalRNN
    then updates its states simply as:

    .. math::

        \mathbf{h}_t=\mathbf{u}_t\odot\mathbf{h}_{t-1}+(\mathbf{1}-\mathbf{u}_t)\odot\mathbf{z}_t

    where :math:`\mathbf{u}_t=\sigma(\mathbf{U}_h\mathbf{h}_{t-1}+\mathbf{U}_z\mathbf{z}_t+\mathbf{b}_u)`
    is the update gate.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        The number of input units.
    out_size : brainstate.typing.Size
        The number of hidden units.
    w_init : Callable or ArrayLike, optional
        The input weight initializer. Default is Orthogonal().
    b_init : Callable or ArrayLike, optional
        The bias weight initializer. Default is ZeroInit().
    state_init : Callable or ArrayLike, optional
        The state initializer. Default is ZeroInit().
    phi : Callable or None, optional
        The input activation function. Default is None.
    name : str or None, optional
        The name of the module. Default is None.
    param_type : type, optional
        The type of the parameter. Default is ETraceParam.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create a Minimal RNN cell
        >>> minrnn_cell = braintrace.nn.MinimalRNNCell(in_size=100, out_size=200)
        >>> minrnn_cell.init_state(batch_size=24)
        >>>
        >>> # Process a sequence of inputs
        >>> x = brainstate.random.randn(24, 100)
        >>> h = minrnn_cell(x)
        >>> print(h.shape)
        (24, 200)
    """
    __module__ = 'braintrace.nn'

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        out_size: brainstate.typing.Size,
        w_init: Union[ArrayLike, Callable] = braintools.init.Orthogonal(),
        b_init: Union[ArrayLike, Callable] = braintools.init.ZeroInit(),
        state_init: Union[ArrayLike, Callable] = braintools.init.ZeroInit(),
        phi: Callable = None,
        name: str = None,
        param_type: type = ETraceParam,
    ):
        super().__init__(name=name)

        # parameters
        self._state_initializer = state_init
        self.out_size = out_size
        self.in_size = in_size

        # functions
        params = dict(w_init=w_init, b_init=b_init, param_type=param_type)
        if phi is None:
            phi = Linear(self.in_size[-1], self.out_size[-1], **params)
        assert callable(phi), f"The phi function should be a callable function. But got {phi}"
        self.phi = phi

        # weights
        self.W_u = Linear(self.out_size[-1] * 2, self.out_size[-1], **params)

    def init_state(self, batch_size: int = None, **kwargs):
        self.h = brainstate.HiddenState(braintools.init.param(self._state_initializer, self.out_size, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = braintools.init.param(self._state_initializer, self.out_size, batch_size)

    def update(self, x):
        z = self.phi(x)
        f = brainstate.nn.sigmoid(self.W_u(u.math.concatenate([z, self.h.value], axis=-1)))
        self.h.value = f * self.h.value + (1 - f) * z
        return self.h.value


class MiniGRU(brainstate.nn.RNNCell):
    r"""Minimal GRU cell.

    Minimal GRU Cell, a simplified version of GRU implemented as in
    `MinimalRNN: Toward More Interpretable and Trainable Recurrent Neural Networks <https://arxiv.org/abs/1711.06788>`_

    At each step :math:`t`, the model processes the input through a gating mechanism
    that controls information flow. The hidden state is updated as:

    .. math::

        \mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \mathbf{W}_x \mathbf{x}_t

    where :math:`\mathbf{z}_t=\sigma(\mathbf{W}_z[\mathbf{x}_t; \mathbf{h}_{t-1}])`
    is the update gate.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        The number of input units.
    out_size : brainstate.typing.Size
        The number of hidden units.
    w_init : Callable or ArrayLike, optional
        The input weight initializer. Default is Orthogonal().
    b_init : Callable or ArrayLike, optional
        The bias weight initializer. Default is ZeroInit().
    state_init : Callable or ArrayLike, optional
        The state initializer. Default is ZeroInit().
    name : str or None, optional
        The name of the module. Default is None.
    param_type : type, optional
        The type of the parameter. Default is ETraceParam.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create a Mini GRU cell
        >>> minigru_cell = braintrace.nn.MiniGRU(in_size=80, out_size=160)
        >>> minigru_cell.init_state(batch_size=32)
        >>>
        >>> # Process a sequence of inputs
        >>> x = brainstate.random.randn(32, 80)
        >>> h = minigru_cell(x)
        >>> print(h.shape)
        (32, 160)
    """
    __module__ = 'braintrace.nn'

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        out_size: brainstate.typing.Size,
        w_init: Union[ArrayLike, Callable] = braintools.init.Orthogonal(),
        b_init: Union[ArrayLike, Callable] = braintools.init.ZeroInit(),
        state_init: Union[ArrayLike, Callable] = braintools.init.ZeroInit(),
        name: str = None,
        param_type: type = ETraceParam,
    ):
        super().__init__(name=name)

        # parameters
        self._state_initializer = state_init
        self.out_size = out_size
        self.in_size = in_size

        # functions
        params = dict(w_init=w_init, b_init=b_init, param_type=param_type)
        self.W_x = Linear(self.in_size[-1], self.out_size[-1], **params)

        # weights
        self.W_z = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], **params)

    def init_state(self, batch_size: int = None, **kwargs):
        self.h = brainstate.HiddenState(braintools.init.param(self._state_initializer, self.out_size, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = braintools.init.param(self._state_initializer, self.out_size, batch_size)

    def update(self, x):
        z = brainstate.nn.sigmoid(self.W_z(u.math.concatenate([x, self.h.value], axis=-1)))
        self.h.value = (1 - z) * self.h.value + z * self.W_x(x)
        return self.h.value


class MiniLSTM(brainstate.nn.RNNCell):
    r"""Minimal LSTM cell.

    Minimal LSTM Cell, a simplified version of LSTM implemented as in
    `MinimalRNN: Toward More Interpretable and Trainable Recurrent Neural Networks <https://arxiv.org/abs/1711.06788>`_

    This simplified LSTM uses forget and input gates to control the flow of information,
    updating the hidden state as:

    .. math::

        \mathbf{h}_t = \mathbf{f}_t \odot \mathbf{h}_{t-1} + \mathbf{i}_t \odot \mathbf{W}_x \mathbf{x}_t

    where :math:`\mathbf{f}_t` and :math:`\mathbf{i}_t` are the forget and input gates,
    respectively.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        The number of input units.
    out_size : brainstate.typing.Size
        The number of hidden units.
    w_init : Callable or ArrayLike, optional
        The input weight initializer. Default is Orthogonal().
    b_init : Callable or ArrayLike, optional
        The bias weight initializer. Default is ZeroInit().
    state_init : Callable or ArrayLike, optional
        The state initializer. Default is ZeroInit().
    name : str or None, optional
        The name of the module. Default is None.
    param_type : type, optional
        The type of the parameter. Default is ETraceParam.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create a Mini LSTM cell
        >>> minilstm_cell = braintrace.nn.MiniLSTM(in_size=150, out_size=300)
        >>> minilstm_cell.init_state(batch_size=40)
        >>>
        >>> # Process a sequence of inputs
        >>> x = brainstate.random.randn(40, 150)
        >>> h = minilstm_cell(x)
        >>> print(h.shape)
        (40, 300)
    """
    __module__ = 'braintrace.nn'

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        out_size: brainstate.typing.Size,
        w_init: Union[ArrayLike, Callable] = braintools.init.Orthogonal(),
        b_init: Union[ArrayLike, Callable] = braintools.init.ZeroInit(),
        state_init: Union[ArrayLike, Callable] = braintools.init.ZeroInit(),
        name: str = None,
        param_type: type = ETraceParam,
    ):
        super().__init__(name=name)

        # parameters
        self._state_initializer = state_init
        self.out_size = out_size
        self.in_size = in_size

        # functions
        params = dict(w_init=w_init, b_init=b_init, param_type=param_type)
        self.W_x = Linear(self.in_size[-1], self.out_size[-1], **params)

        # weights
        self.W_f = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], **params)
        self.W_i = Linear(self.in_size[-1] + self.out_size[-1], self.out_size[-1], **params)

    def init_state(self, batch_size: int = None, **kwargs):
        self.h = brainstate.HiddenState(braintools.init.param(self._state_initializer, self.out_size, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h.value = braintools.init.param(self._state_initializer, self.out_size, batch_size)

    def update(self, x):
        xh = u.math.concatenate([x, self.h.value], axis=-1)
        f = brainstate.nn.sigmoid(self.W_f(xh))
        i = brainstate.nn.sigmoid(self.W_i(xh))
        self.h.value = f * self.h.value + i * self.W_x(x)
        return self.h.value


def glorot_init(s):
    return brainstate.random.randn(*s) / u.math.sqrt(s[0])


class LRUCell(brainstate.nn.Module):
    r"""Linear Recurrent Unit (LRU) layer.

    `Linear Recurrent Unit <https://arxiv.org/abs/2303.06349>`_ (LRU) layer, which
    uses diagonal complex-valued state transitions for efficient sequence modeling.

    .. math::

       h_{t+1} = \lambda * h_t + \exp(\gamma^{\mathrm{log}}) B x_{t+1} \\
       \lambda = \text{diag}(\exp(-\exp(\nu^{\mathrm{log}}) + i \exp(\theta^\mathrm{log}))) \\
       y_t = Re[C h_t + D x_t]

    Parameters
    ----------
    d_model : int
        Input and output dimensions.
    d_hidden : int
        Hidden state dimension.
    r_min : float, optional
        Smallest lambda norm. Default is 0.0.
    r_max : float, optional
        Largest lambda norm. Default is 1.0.
    max_phase : float, optional
        Max phase lambda. Default is 6.28.

    Examples
    --------
    .. code-block:: python

        >>> import braintrace
        >>> import brainstate
        >>>
        >>> # Create an LRU cell
        >>> lru_cell = braintrace.nn.LRUCell(d_model=64, d_hidden=128)
        >>> lru_cell.init_state(batch_size=16)
        >>>
        >>> # Process a sequence of inputs
        >>> x = brainstate.random.randn(16, 64)
        >>> y = lru_cell(x)
        >>> print(y.shape)
        (16, 64)
    """

    def __init__(
        self,
        d_model: int,  # input and output dimensions
        d_hidden: int,  # hidden state dimension
        r_min: float = 0.0,  # smallest lambda norm
        r_max: float = 1.0,  # largest lambda norm
        max_phase: float = 6.28,  # max phase lambda
    ):
        super().__init__()

        self.in_size = d_model
        self.out_size = d_hidden

        self.d_hidden = d_hidden
        self.d_model = d_model
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase

        # -------- recurrent weight matrix --------

        # theta parameter
        theta_log = u.math.log(max_phase * brainstate.random.uniform(size=d_hidden))
        self.theta_log = ElemWiseParam(theta_log)

        # nu parameter
        nu_log = u.math.log(
            -0.5 * u.math.log(
                brainstate.random.uniform(size=d_hidden) * (r_max ** 2 - r_min ** 2) + r_min ** 2
            )
        )
        self.nu_log = ElemWiseParam(nu_log)

        # -------- input weight matrix --------

        # gamma parameter
        diag_lambda = u.math.exp(-u.math.exp(nu_log) + 1j * u.math.exp(theta_log))
        gamma_log = u.math.log(u.math.sqrt(1 - u.math.abs(diag_lambda) ** 2))
        self.gamma_log = ElemWiseParam(gamma_log)

        # Glorot initialized Input/Output projection matrices
        self.B_re = Linear(d_model, d_hidden, w_init=glorot_init, b_init=None)
        self.B_im = Linear(d_model, d_hidden, w_init=glorot_init, b_init=None)

        # -------- output weight matrix --------

        self.C_re = Linear(d_hidden, d_model, w_init=glorot_init, b_init=None)
        self.C_im = Linear(d_hidden, d_model, w_init=glorot_init, b_init=None)

        # Parameter for skip connection
        self.D = ElemWiseParam(brainstate.random.randn(d_model))

    def init_state(self, batch_size: int = None, **kwargs):
        self.h_re = brainstate.HiddenState(braintools.init.param(braintools.init.ZeroInit(), self.d_hidden, batch_size))
        self.h_im = brainstate.HiddenState(braintools.init.param(braintools.init.ZeroInit(), self.d_hidden, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.h_re.value = braintools.init.param(braintools.init.ZeroInit(), self.d_hidden, batch_size)
        self.h_im.value = braintools.init.param(braintools.init.ZeroInit(), self.d_hidden, batch_size)

    def update(self, inputs):
        a = u.math.exp(-u.math.exp(self.nu_log.execute()))
        b = u.math.exp(self.theta_log.execute())
        c = u.math.exp(self.gamma_log.execute())
        a_cos_b = a * u.math.cos(b)
        a_sin_b = a * u.math.sin(b)
        self.h_re.value = a_cos_b * self.h_re.value - a_sin_b * self.h_im.value + c * self.B_re(inputs)
        self.h_im.value = a_sin_b * self.h_re.value + a_cos_b * self.h_im.value + c * self.B_im(inputs)
        r = self.C_re(self.h_re.value) - self.C_im(self.h_im.value) + inputs * self.D.execute()
        return r
