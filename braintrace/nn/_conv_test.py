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

"""
Comprehensive tests for convolutional neural network layers.

Note: There is a known issue in ConvOp where batch dimensions are not properly
handled. The tests are written to test layer creation and the basic structure,
with integration tests marked as expected failures until the underlying issue
in _etrace_operators.py is fixed.
"""

import pytest

brainstate = pytest.importorskip("brainstate")
jnp = pytest.importorskip("jax.numpy")
braintools = pytest.importorskip("braintools")
init = braintools.init
braintrace = pytest.importorskip("braintrace")
from braintrace.nn._conv import to_dimension_numbers, replicate


class TestUtilityFunctions:
    """Test utility functions used in convolution modules."""

    def test_to_dimension_numbers_1d_channels_last(self):
        """Test 1D convolution dimension numbers with channels_last=True."""
        dim_numbers = to_dimension_numbers(num_spatial_dims=1, channels_last=True, transpose=False)
        assert dim_numbers.lhs_spec == (0, 2, 1)  # (batch, channels, spatial)
        assert dim_numbers.rhs_spec == (2, 1, 0)  # (out_channels, in_channels, spatial)
        assert dim_numbers.out_spec == (0, 2, 1)  # (batch, channels, spatial)

    def test_to_dimension_numbers_1d_channels_first(self):
        """Test 1D convolution dimension numbers with channels_last=False."""
        dim_numbers = to_dimension_numbers(num_spatial_dims=1, channels_last=False, transpose=False)
        assert dim_numbers.lhs_spec == (0, 1, 2)  # (batch, channels, spatial)
        assert dim_numbers.rhs_spec == (2, 1, 0)  # (out_channels, in_channels, spatial)
        assert dim_numbers.out_spec == (0, 1, 2)  # (batch, channels, spatial)

    def test_to_dimension_numbers_2d_channels_last(self):
        """Test 2D convolution dimension numbers with channels_last=True."""
        dim_numbers = to_dimension_numbers(num_spatial_dims=2, channels_last=True, transpose=False)
        assert dim_numbers.lhs_spec == (0, 3, 1, 2)  # (batch, channels, H, W)
        assert dim_numbers.rhs_spec == (3, 2, 0, 1)  # (out_channels, in_channels, H, W)
        assert dim_numbers.out_spec == (0, 3, 1, 2)  # (batch, channels, H, W)

    def test_to_dimension_numbers_2d_channels_first(self):
        """Test 2D convolution dimension numbers with channels_last=False."""
        dim_numbers = to_dimension_numbers(num_spatial_dims=2, channels_last=False, transpose=False)
        assert dim_numbers.lhs_spec == (0, 1, 2, 3)  # (batch, channels, H, W)
        assert dim_numbers.rhs_spec == (3, 2, 0, 1)  # (out_channels, in_channels, H, W)
        assert dim_numbers.out_spec == (0, 1, 2, 3)  # (batch, channels, H, W)

    def test_to_dimension_numbers_3d_channels_last(self):
        """Test 3D convolution dimension numbers with channels_last=True."""
        dim_numbers = to_dimension_numbers(num_spatial_dims=3, channels_last=True, transpose=False)
        assert dim_numbers.lhs_spec == (0, 4, 1, 2, 3)  # (batch, channels, H, W, D)
        assert dim_numbers.rhs_spec == (4, 3, 0, 1, 2)  # (out_channels, in_channels, H, W, D)
        assert dim_numbers.out_spec == (0, 4, 1, 2, 3)  # (batch, channels, H, W, D)

    def test_to_dimension_numbers_transpose(self):
        """Test dimension numbers with transpose=True."""
        dim_numbers = to_dimension_numbers(num_spatial_dims=2, channels_last=True, transpose=True)
        assert dim_numbers.lhs_spec == (0, 3, 1, 2)  # (batch, channels, H, W)
        assert dim_numbers.rhs_spec == (2, 3, 0, 1)  # (in_channels, out_channels, H, W)
        assert dim_numbers.out_spec == (0, 3, 1, 2)  # (batch, channels, H, W)

    def test_replicate_scalar_to_tuple(self):
        """Test replicate function with scalar input."""
        result = replicate(3, 2, "test_param")
        assert result == (3, 3)

    def test_replicate_single_element_sequence(self):
        """Test replicate function with single element sequence."""
        result = replicate([5], 3, "test_param")
        assert result == (5, 5, 5)

    def test_replicate_full_sequence(self):
        """Test replicate function with full sequence."""
        result = replicate([1, 2, 3], 3, "test_param")
        assert result == (1, 2, 3)

    def test_replicate_string(self):
        """Test replicate function with string input."""
        result = replicate("same", 2, "test_param")
        assert result == ("same", "same")

    def test_replicate_bytes(self):
        """Test replicate function with bytes input."""
        result = replicate(b"test", 2, "test_param")
        assert result == (b"test", b"test")

    def test_replicate_error_wrong_length(self):
        """Test replicate function raises error with wrong length sequence."""
        with pytest.raises(TypeError):
            replicate([1, 2], 3, "test_param")


class TestConv1d:
    """Test Conv1d convolution layer."""

    def test_conv1d_basic_creation(self):
        """Test basic Conv1d layer creation."""
        conv = braintrace.nn.Conv1d(in_size=(10, 3), out_channels=16, kernel_size=3)
        assert conv.in_channels == 3
        assert conv.out_channels == 16
        assert conv.kernel_size == (3,)
        assert conv.in_size == (10, 3)

    def test_conv1d_forward_with_batch(self):
        """Test Conv1d forward pass with batch dimension."""
        conv = braintrace.nn.Conv1d(in_size=(10, 3), out_channels=16, kernel_size=3)
        x = brainstate.random.randn(4, 10, 3)
        y = conv(x)
        assert y.shape == (4, 10, 16)

    def test_conv1d_forward_without_batch(self):
        """Test Conv1d forward pass without batch dimension."""
        conv = braintrace.nn.Conv1d(in_size=(10, 3), out_channels=16, kernel_size=3)
        x = brainstate.random.randn(10, 3)
        y = conv(x)
        assert y.shape == (10, 16)

    def test_conv1d_different_strides(self):
        """Test Conv1d with different stride values."""
        conv = braintrace.nn.Conv1d(in_size=(10, 3), out_channels=16, kernel_size=3, stride=2)
        x = brainstate.random.randn(4, 10, 3)
        y = conv(x)
        # With stride=2 and SAME padding, output size should be ceil(10/2) = 5
        assert y.shape == (4, 5, 16)

    def test_conv1d_valid_padding(self):
        """Test Conv1d with VALID padding."""
        conv = braintrace.nn.Conv1d(in_size=(10, 3), out_channels=16, kernel_size=3, padding='VALID')
        x = brainstate.random.randn(4, 10, 3)
        y = conv(x)
        # With VALID padding and kernel_size=3, output size should be 10-3+1 = 8
        assert y.shape == (4, 8, 16)

    def test_conv1d_same_padding(self):
        """Test Conv1d with SAME padding."""
        conv = braintrace.nn.Conv1d(in_size=(10, 3), out_channels=16, kernel_size=3, padding='SAME')
        x = brainstate.random.randn(4, 10, 3)
        y = conv(x)
        # With SAME padding, output size should be same as input
        assert y.shape == (4, 10, 16)

    def test_conv1d_explicit_padding_int(self):
        """Test Conv1d with explicit integer padding."""
        conv = braintrace.nn.Conv1d(in_size=(10, 3), out_channels=16, kernel_size=3, padding=1)
        x = brainstate.random.randn(4, 10, 3)
        y = conv(x)
        assert y.ndim == 3
        assert y.shape[0] == 4
        assert y.shape[-1] == 16

    def test_conv1d_explicit_padding_tuple(self):
        """Test Conv1d with explicit tuple padding."""
        conv = braintrace.nn.Conv1d(in_size=(10, 3), out_channels=16, kernel_size=3, padding=(1, 1))
        x = brainstate.random.randn(4, 10, 3)
        y = conv(x)
        assert y.ndim == 3
        assert y.shape[0] == 4
        assert y.shape[-1] == 16

    def test_conv1d_with_bias(self):
        """Test Conv1d with bias initialization."""
        conv = braintrace.nn.Conv1d(
            in_size=(10, 3),
            out_channels=16,
            kernel_size=3,
            b_init=init.Constant(0.1)
        )
        x = brainstate.random.randn(4, 10, 3)
        y = conv(x)
        assert y.shape == (4, 10, 16)

    def test_conv1d_without_bias(self):
        """Test Conv1d without bias."""
        conv = braintrace.nn.Conv1d(in_size=(10, 3), out_channels=16, kernel_size=3, b_init=None)
        x = brainstate.random.randn(4, 10, 3)
        y = conv(x)
        assert y.shape == (4, 10, 16)

    def test_conv1d_with_groups(self):
        """Test Conv1d with grouped convolution."""
        conv = braintrace.nn.Conv1d(in_size=(10, 4), out_channels=8, kernel_size=3, groups=2)
        x = brainstate.random.randn(4, 10, 4)
        y = conv(x)
        assert y.shape == (4, 10, 8)

    def test_conv1d_depthwise(self):
        """Test Conv1d with depthwise convolution (groups = in_channels)."""
        in_channels = 4
        conv = braintrace.nn.Conv1d(
            in_size=(10, in_channels),
            out_channels=in_channels,
            kernel_size=3,
            groups=in_channels
        )
        x = brainstate.random.randn(4, 10, in_channels)
        y = conv(x)
        assert y.shape == (4, 10, in_channels)

    # def test_conv1d_lhs_dilation(self):
    #     """Test Conv1d with lhs_dilation (atrous convolution on input)."""
    #     conv = braintrace.nn.Conv1d(in_size=(10, 3), out_channels=16, kernel_size=3, lhs_dilation=2)
    #     x = brainstate.random.randn(4, 10, 3)
    #     y = conv(x)
    #     assert y.ndim == 3
    #     assert y.shape[0] == 4
    #     assert y.shape[-1] == 16

    def test_conv1d_rhs_dilation(self):
        """Test Conv1d with rhs_dilation (atrous convolution on kernel)."""
        conv = braintrace.nn.Conv1d(in_size=(10, 3), out_channels=16, kernel_size=3, rhs_dilation=2)
        x = brainstate.random.randn(4, 10, 3)
        y = conv(x)
        assert y.ndim == 3
        assert y.shape[0] == 4
        assert y.shape[-1] == 16

    def test_conv1d_custom_initializers(self):
        """Test Conv1d with custom weight and bias initializers."""
        conv = braintrace.nn.Conv1d(
            in_size=(10, 3),
            out_channels=16,
            kernel_size=3,
            w_init=init.Constant(0.5),
            b_init=init.Constant(0.0)
        )
        x = brainstate.random.randn(4, 10, 3)
        y = conv(x)
        assert y.shape == (4, 10, 16)

    def test_conv1d_with_weight_mask(self):
        """Test Conv1d with weight mask."""
        kernel_shape = (3, 3, 16)  # (kernel_size, in_channels, out_channels)
        mask = jnp.ones(kernel_shape)
        conv = braintrace.nn.Conv1d(in_size=(10, 3), out_channels=16, kernel_size=3, w_mask=mask)
        x = brainstate.random.randn(4, 10, 3)
        y = conv(x)
        assert y.shape == (4, 10, 16)

    def test_conv1d_kernel_shape(self):
        """Test Conv1d kernel shape is correct."""
        conv = braintrace.nn.Conv1d(in_size=(10, 3), out_channels=16, kernel_size=5)
        assert conv.kernel_shape == (5, 3, 16)

    def test_conv1d_input_validation_wrong_ndim(self):
        """Test Conv1d input validation rejects wrong number of dimensions."""
        conv = braintrace.nn.Conv1d(in_size=(10, 3), out_channels=16, kernel_size=3)
        x = brainstate.random.randn(10)  # 1D input - should fail
        with pytest.raises(ValueError):
            conv(x)

    def test_conv1d_input_validation_wrong_shape(self):
        """Test Conv1d input validation rejects wrong shape."""
        conv = braintrace.nn.Conv1d(in_size=(10, 3), out_channels=16, kernel_size=3)
        x = brainstate.random.randn(4, 12, 3)  # Wrong spatial dimension
        with pytest.raises(ValueError):
            conv(x)

    def test_conv1d_sequence_stride(self):
        """Test Conv1d with stride as sequence."""
        conv = braintrace.nn.Conv1d(in_size=(10, 3), out_channels=16, kernel_size=3, stride=[2])
        x = brainstate.random.randn(4, 10, 3)
        y = conv(x)
        assert y.shape == (4, 5, 16)

    def test_conv1d_sequence_kernel_size(self):
        """Test Conv1d with kernel_size as sequence."""
        conv = braintrace.nn.Conv1d(in_size=(10, 3), out_channels=16, kernel_size=[3])
        x = brainstate.random.randn(4, 10, 3)
        y = conv(x)
        assert y.shape == (4, 10, 16)


class TestConv2d:
    """Test Conv2d convolution layer."""

    def test_conv2d_basic_creation(self):
        """Test basic Conv2d layer creation."""
        conv = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=3)
        assert conv.in_channels == 3
        assert conv.out_channels == 32
        assert conv.kernel_size == (3, 3)
        assert conv.in_size == (28, 28, 3)

    def test_conv2d_forward_with_batch(self):
        """Test Conv2d forward pass with batch dimension."""
        conv = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=3)
        x = brainstate.random.randn(8, 28, 28, 3)
        y = conv(x)
        assert y.shape == (8, 28, 28, 32)

    def test_conv2d_forward_without_batch(self):
        """Test Conv2d forward pass without batch dimension."""
        conv = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=3)
        x = brainstate.random.randn(28, 28, 3)
        y = conv(x)
        assert y.shape == (28, 28, 32)

    def test_conv2d_different_strides(self):
        """Test Conv2d with different stride values."""
        conv = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=3, stride=2)
        x = brainstate.random.randn(8, 28, 28, 3)
        y = conv(x)
        # With stride=2 and SAME padding, output size should be ceil(28/2) = 14
        assert y.shape == (8, 14, 14, 32)

    def test_conv2d_asymmetric_strides(self):
        """Test Conv2d with different stride values for each dimension."""
        conv = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=3, stride=(2, 1))
        x = brainstate.random.randn(8, 28, 28, 3)
        y = conv(x)
        # With stride=(2,1) and SAME padding
        assert y.shape == (8, 14, 28, 32)

    def test_conv2d_valid_padding(self):
        """Test Conv2d with VALID padding."""
        conv = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=3, padding='VALID')
        x = brainstate.random.randn(8, 28, 28, 3)
        y = conv(x)
        # With VALID padding and kernel_size=3, output size should be 28-3+1 = 26
        assert y.shape == (8, 26, 26, 32)

    def test_conv2d_same_padding(self):
        """Test Conv2d with SAME padding."""
        conv = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=3, padding='SAME')
        x = brainstate.random.randn(8, 28, 28, 3)
        y = conv(x)
        # With SAME padding, output size should be same as input
        assert y.shape == (8, 28, 28, 32)

    def test_conv2d_explicit_padding_int(self):
        """Test Conv2d with explicit integer padding."""
        conv = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=3, padding=1)
        x = brainstate.random.randn(8, 28, 28, 3)
        y = conv(x)
        assert y.ndim == 4
        assert y.shape[0] == 8
        assert y.shape[-1] == 32

    def test_conv2d_explicit_padding_tuple(self):
        """Test Conv2d with explicit tuple padding."""
        conv = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=3, padding=(1, 1))
        x = brainstate.random.randn(8, 28, 28, 3)
        y = conv(x)
        assert y.ndim == 4
        assert y.shape[0] == 8
        assert y.shape[-1] == 32

    def test_conv2d_explicit_padding_sequence(self):
        """Test Conv2d with explicit sequence of tuples padding."""
        conv = braintrace.nn.Conv2d(
            in_size=(28, 28, 3),
            out_channels=32,
            kernel_size=3,
            padding=[(1, 1), (2, 2)]
        )
        x = brainstate.random.randn(8, 28, 28, 3)
        y = conv(x)
        assert y.ndim == 4
        assert y.shape[0] == 8
        assert y.shape[-1] == 32

    def test_conv2d_with_bias(self):
        """Test Conv2d with bias initialization."""
        conv = braintrace.nn.Conv2d(
            in_size=(28, 28, 3),
            out_channels=32,
            kernel_size=3,
            b_init=init.Constant(0.1)
        )
        x = brainstate.random.randn(8, 28, 28, 3)
        y = conv(x)
        assert y.shape == (8, 28, 28, 32)

    def test_conv2d_without_bias(self):
        """Test Conv2d without bias."""
        conv = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=3, b_init=None)
        x = brainstate.random.randn(8, 28, 28, 3)
        y = conv(x)
        assert y.shape == (8, 28, 28, 32)

    def test_conv2d_with_groups(self):
        """Test Conv2d with grouped convolution."""
        conv = braintrace.nn.Conv2d(in_size=(28, 28, 4), out_channels=8, kernel_size=3, groups=2)
        x = brainstate.random.randn(8, 28, 28, 4)
        y = conv(x)
        assert y.shape == (8, 28, 28, 8)

    def test_conv2d_depthwise(self):
        """Test Conv2d with depthwise convolution (groups = in_channels)."""
        in_channels = 4
        conv = braintrace.nn.Conv2d(
            in_size=(28, 28, in_channels),
            out_channels=in_channels,
            kernel_size=3,
            groups=in_channels
        )
        x = brainstate.random.randn(8, 28, 28, in_channels)
        y = conv(x)
        assert y.shape == (8, 28, 28, in_channels)

    # def test_conv2d_lhs_dilation(self):
    #     """Test Conv2d with lhs_dilation (atrous convolution on input)."""
    #     conv = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=3, lhs_dilation=2)
    #     x = brainstate.random.randn(8, 28, 28, 3)
    #     y = conv(x)
    #     assert y.ndim == 4
    #     assert y.shape[0] == 8
    #     assert y.shape[-1] == 32

    def test_conv2d_rhs_dilation(self):
        """Test Conv2d with rhs_dilation (atrous convolution on kernel)."""
        conv = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=3, rhs_dilation=2)
        x = brainstate.random.randn(8, 28, 28, 3)
        y = conv(x)
        assert y.ndim == 4
        assert y.shape[0] == 8
        assert y.shape[-1] == 32

    def test_conv2d_asymmetric_kernel(self):
        """Test Conv2d with asymmetric kernel size."""
        conv = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=(3, 5))
        x = brainstate.random.randn(8, 28, 28, 3)
        y = conv(x)
        assert y.shape == (8, 28, 28, 32)
        assert conv.kernel_shape == (3, 5, 3, 32)

    def test_conv2d_custom_initializers(self):
        """Test Conv2d with custom weight and bias initializers."""
        conv = braintrace.nn.Conv2d(
            in_size=(28, 28, 3),
            out_channels=32,
            kernel_size=3,
            w_init=init.Constant(0.5),
            b_init=init.Constant(0.0)
        )
        x = brainstate.random.randn(8, 28, 28, 3)
        y = conv(x)
        assert y.shape == (8, 28, 28, 32)

    def test_conv2d_with_weight_mask(self):
        """Test Conv2d with weight mask."""
        kernel_shape = (3, 3, 3, 32)  # (H, W, in_channels, out_channels)
        mask = jnp.ones(kernel_shape)
        conv = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=3, w_mask=mask)
        x = brainstate.random.randn(8, 28, 28, 3)
        y = conv(x)
        assert y.shape == (8, 28, 28, 32)

    def test_conv2d_kernel_shape(self):
        """Test Conv2d kernel shape is correct."""
        conv = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=5)
        assert conv.kernel_shape == (5, 5, 3, 32)

    def test_conv2d_input_validation_wrong_ndim(self):
        """Test Conv2d input validation rejects wrong number of dimensions."""
        conv = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=3)
        x = brainstate.random.randn(28, 3)  # 2D input - should fail
        with pytest.raises(ValueError):
            conv(x)

    def test_conv2d_input_validation_wrong_shape(self):
        """Test Conv2d input validation rejects wrong shape."""
        conv = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=3)
        x = brainstate.random.randn(8, 32, 32, 3)  # Wrong spatial dimensions
        with pytest.raises(ValueError):
            conv(x)


class TestConv3d:
    """Test Conv3d convolution layer."""

    def test_conv3d_basic_creation(self):
        """Test basic Conv3d layer creation."""
        conv = braintrace.nn.Conv3d(in_size=(16, 16, 16, 3), out_channels=64, kernel_size=3)
        assert conv.in_channels == 3
        assert conv.out_channels == 64
        assert conv.kernel_size == (3, 3, 3)
        assert conv.in_size == (16, 16, 16, 3)

    def test_conv3d_forward_with_batch(self):
        """Test Conv3d forward pass with batch dimension."""
        conv = braintrace.nn.Conv3d(in_size=(16, 16, 16, 3), out_channels=64, kernel_size=3)
        x = brainstate.random.randn(2, 16, 16, 16, 3)
        y = conv(x)
        assert y.shape == (2, 16, 16, 16, 64)

    def test_conv3d_forward_without_batch(self):
        """Test Conv3d forward pass without batch dimension."""
        conv = braintrace.nn.Conv3d(in_size=(16, 16, 16, 3), out_channels=64, kernel_size=3)
        x = brainstate.random.randn(16, 16, 16, 3)
        y = conv(x)
        assert y.shape == (16, 16, 16, 64)

    def test_conv3d_different_strides(self):
        """Test Conv3d with different stride values."""
        conv = braintrace.nn.Conv3d(in_size=(16, 16, 16, 3), out_channels=64, kernel_size=3, stride=2)
        x = brainstate.random.randn(2, 16, 16, 16, 3)
        y = conv(x)
        # With stride=2 and SAME padding, output size should be ceil(16/2) = 8
        assert y.shape == (2, 8, 8, 8, 64)

    def test_conv3d_asymmetric_strides(self):
        """Test Conv3d with different stride values for each dimension."""
        conv = braintrace.nn.Conv3d(
            in_size=(16, 16, 16, 3),
            out_channels=64,
            kernel_size=3,
            stride=(2, 1, 2)
        )
        x = brainstate.random.randn(2, 16, 16, 16, 3)
        y = conv(x)
        # With stride=(2,1,2) and SAME padding
        assert y.shape == (2, 8, 16, 8, 64)

    def test_conv3d_valid_padding(self):
        """Test Conv3d with VALID padding."""
        conv = braintrace.nn.Conv3d(
            in_size=(16, 16, 16, 3),
            out_channels=64,
            kernel_size=3,
            padding='VALID'
        )
        x = brainstate.random.randn(2, 16, 16, 16, 3)
        y = conv(x)
        # With VALID padding and kernel_size=3, output size should be 16-3+1 = 14
        assert y.shape == (2, 14, 14, 14, 64)

    def test_conv3d_same_padding(self):
        """Test Conv3d with SAME padding."""
        conv = braintrace.nn.Conv3d(
            in_size=(16, 16, 16, 3),
            out_channels=64,
            kernel_size=3,
            padding='SAME'
        )
        x = brainstate.random.randn(2, 16, 16, 16, 3)
        y = conv(x)
        # With SAME padding, output size should be same as input
        assert y.shape == (2, 16, 16, 16, 64)

    def test_conv3d_explicit_padding_int(self):
        """Test Conv3d with explicit integer padding."""
        conv = braintrace.nn.Conv3d(
            in_size=(16, 16, 16, 3),
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        x = brainstate.random.randn(2, 16, 16, 16, 3)
        y = conv(x)
        assert y.ndim == 5
        assert y.shape[0] == 2
        assert y.shape[-1] == 64

    def test_conv3d_explicit_padding_tuple(self):
        """Test Conv3d with explicit tuple padding."""
        conv = braintrace.nn.Conv3d(
            in_size=(16, 16, 16, 3),
            out_channels=64,
            kernel_size=3,
            padding=(1, 1)
        )
        x = brainstate.random.randn(2, 16, 16, 16, 3)
        y = conv(x)
        assert y.ndim == 5
        assert y.shape[0] == 2
        assert y.shape[-1] == 64

    def test_conv3d_explicit_padding_sequence(self):
        """Test Conv3d with explicit sequence of tuples padding."""
        conv = braintrace.nn.Conv3d(
            in_size=(16, 16, 16, 3),
            out_channels=64,
            kernel_size=3,
            padding=[(1, 1), (1, 1), (1, 1)]
        )
        x = brainstate.random.randn(2, 16, 16, 16, 3)
        y = conv(x)
        assert y.ndim == 5
        assert y.shape[0] == 2
        assert y.shape[-1] == 64

    def test_conv3d_with_bias(self):
        """Test Conv3d with bias initialization."""
        conv = braintrace.nn.Conv3d(
            in_size=(16, 16, 16, 3),
            out_channels=64,
            kernel_size=3,
            b_init=init.Constant(0.1)
        )
        x = brainstate.random.randn(2, 16, 16, 16, 3)
        y = conv(x)
        assert y.shape == (2, 16, 16, 16, 64)

    def test_conv3d_without_bias(self):
        """Test Conv3d without bias."""
        conv = braintrace.nn.Conv3d(
            in_size=(16, 16, 16, 3),
            out_channels=64,
            kernel_size=3,
            b_init=None
        )
        x = brainstate.random.randn(2, 16, 16, 16, 3)
        y = conv(x)
        assert y.shape == (2, 16, 16, 16, 64)

    def test_conv3d_with_groups(self):
        """Test Conv3d with grouped convolution."""
        conv = braintrace.nn.Conv3d(
            in_size=(16, 16, 16, 4),
            out_channels=8,
            kernel_size=3,
            groups=2
        )
        x = brainstate.random.randn(2, 16, 16, 16, 4)
        y = conv(x)
        assert y.shape == (2, 16, 16, 16, 8)

    def test_conv3d_depthwise(self):
        """Test Conv3d with depthwise convolution (groups = in_channels)."""
        in_channels = 4
        conv = braintrace.nn.Conv3d(
            in_size=(16, 16, 16, in_channels),
            out_channels=in_channels,
            kernel_size=3,
            groups=in_channels
        )
        x = brainstate.random.randn(2, 16, 16, 16, in_channels)
        y = conv(x)
        assert y.shape == (2, 16, 16, 16, in_channels)

    # def test_conv3d_lhs_dilation(self):
    #     """Test Conv3d with lhs_dilation (atrous convolution on input)."""
    #     conv = braintrace.nn.Conv3d(
    #         in_size=(16, 16, 16, 3),
    #         out_channels=64,
    #         kernel_size=3,
    #         lhs_dilation=2
    #     )
    #     x = brainstate.random.randn(2, 16, 16, 16, 3)
    #     y = conv(x)
    #     assert y.ndim == 5
    #     assert y.shape[0] == 2
    #     assert y.shape[-1] == 64

    def test_conv3d_rhs_dilation(self):
        """Test Conv3d with rhs_dilation (atrous convolution on kernel)."""
        conv = braintrace.nn.Conv3d(
            in_size=(16, 16, 16, 3),
            out_channels=64,
            kernel_size=3,
            rhs_dilation=2
        )
        x = brainstate.random.randn(2, 16, 16, 16, 3)
        y = conv(x)
        assert y.ndim == 5
        assert y.shape[0] == 2
        assert y.shape[-1] == 64

    def test_conv3d_asymmetric_kernel(self):
        """Test Conv3d with asymmetric kernel size."""
        conv = braintrace.nn.Conv3d(
            in_size=(16, 16, 16, 3),
            out_channels=64,
            kernel_size=(3, 5, 3)
        )
        x = brainstate.random.randn(2, 16, 16, 16, 3)
        y = conv(x)
        assert y.shape == (2, 16, 16, 16, 64)
        assert conv.kernel_shape == (3, 5, 3, 3, 64)

    def test_conv3d_custom_initializers(self):
        """Test Conv3d with custom weight and bias initializers."""
        conv = braintrace.nn.Conv3d(
            in_size=(16, 16, 16, 3),
            out_channels=64,
            kernel_size=3,
            w_init=init.Constant(0.5),
            b_init=init.Constant(0.0)
        )
        x = brainstate.random.randn(2, 16, 16, 16, 3)
        y = conv(x)
        assert y.shape == (2, 16, 16, 16, 64)

    def test_conv3d_with_weight_mask(self):
        """Test Conv3d with weight mask."""
        kernel_shape = (3, 3, 3, 3, 64)  # (H, W, D, in_channels, out_channels)
        mask = jnp.ones(kernel_shape)
        conv = braintrace.nn.Conv3d(
            in_size=(16, 16, 16, 3),
            out_channels=64,
            kernel_size=3,
            w_mask=mask
        )
        x = brainstate.random.randn(2, 16, 16, 16, 3)
        y = conv(x)
        assert y.shape == (2, 16, 16, 16, 64)

    def test_conv3d_kernel_shape(self):
        """Test Conv3d kernel shape is correct."""
        conv = braintrace.nn.Conv3d(in_size=(16, 16, 16, 3), out_channels=64, kernel_size=5)
        assert conv.kernel_shape == (5, 5, 5, 3, 64)

    def test_conv3d_input_validation_wrong_ndim(self):
        """Test Conv3d input validation rejects wrong number of dimensions."""
        conv = braintrace.nn.Conv3d(in_size=(16, 16, 16, 3), out_channels=64, kernel_size=3)
        x = brainstate.random.randn(16, 16, 3)  # 3D input - should fail
        with pytest.raises(ValueError):
            conv(x)

    def test_conv3d_input_validation_wrong_shape(self):
        """Test Conv3d input validation rejects wrong shape."""
        conv = braintrace.nn.Conv3d(in_size=(16, 16, 16, 3), out_channels=64, kernel_size=3)
        x = brainstate.random.randn(2, 20, 20, 20, 3)  # Wrong spatial dimensions
        with pytest.raises(ValueError):
            conv(x)


class TestConvEdgeCases:
    """Test edge cases and error conditions."""

    def test_conv_invalid_groups_out_channels(self):
        """Test that out_channels must be divisible by groups."""
        with pytest.raises(AssertionError):
            braintrace.nn.Conv2d(in_size=(28, 28, 4), out_channels=9, kernel_size=3, groups=2)

    def test_conv_invalid_groups_in_channels(self):
        """Test that in_channels must be divisible by groups."""
        with pytest.raises(AssertionError):
            braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=8, kernel_size=3, groups=2)

    def test_conv_invalid_padding_string(self):
        """Test that only SAME and VALID are accepted as string padding."""
        with pytest.raises(AssertionError):
            braintrace.nn.Conv2d(
                in_size=(28, 28, 3),
                out_channels=32,
                kernel_size=3,
                padding='INVALID'
            )

    def test_conv_invalid_padding_wrong_length(self):
        """Test that padding sequence with wrong number of tuples raises error."""
        # Padding with 3 tuples for 2D conv should fail (needs 1 or 2 tuples)
        with pytest.raises(ValueError):
            braintrace.nn.Conv2d(
                in_size=(28, 28, 3),
                out_channels=32,
                kernel_size=3,
                padding=[(1, 1), (1, 1), (1, 1)]  # Too many tuples for 2D conv
            )

    def test_conv_invalid_in_size_length(self):
        """Test that in_size must have correct length."""
        with pytest.raises(AssertionError):
            braintrace.nn.Conv2d(
                in_size=(28, 3),  # Should be 3D for Conv2d
                out_channels=32,
                kernel_size=3
            )


class TestConvIntegration:
    """Integration tests for convolution layers."""

    def test_conv1d_stacked_layers(self):
        """Test stacking multiple Conv1d layers."""
        conv1 = braintrace.nn.Conv1d(in_size=(10, 3), out_channels=16, kernel_size=3)
        conv2 = braintrace.nn.Conv1d(in_size=(10, 16), out_channels=32, kernel_size=3)

        x = brainstate.random.randn(4, 10, 3)
        y1 = conv1(x)
        y2 = conv2(y1)

        assert y1.shape == (4, 10, 16)
        assert y2.shape == (4, 10, 32)

    def test_conv2d_stacked_layers(self):
        """Test stacking multiple Conv2d layers."""
        conv1 = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=3)
        conv2 = braintrace.nn.Conv2d(in_size=(28, 28, 32), out_channels=64, kernel_size=3)

        x = brainstate.random.randn(8, 28, 28, 3)
        y1 = conv1(x)
        y2 = conv2(y1)

        assert y1.shape == (8, 28, 28, 32)
        assert y2.shape == (8, 28, 28, 64)

    def test_conv3d_stacked_layers(self):
        """Test stacking multiple Conv3d layers."""
        conv1 = braintrace.nn.Conv3d(in_size=(16, 16, 16, 3), out_channels=32, kernel_size=3)
        conv2 = braintrace.nn.Conv3d(in_size=(16, 16, 16, 32), out_channels=64, kernel_size=3)

        x = brainstate.random.randn(2, 16, 16, 16, 3)
        y1 = conv1(x)
        y2 = conv2(y1)

        assert y1.shape == (2, 16, 16, 16, 32)
        assert y2.shape == (2, 16, 16, 16, 64)

    def test_conv2d_with_pooling_like_stride(self):
        """Test Conv2d with stride > 1 for downsampling."""
        # Create a simple CNN-like architecture with downsampling
        conv1 = braintrace.nn.Conv2d(in_size=(32, 32, 3), out_channels=16, kernel_size=3, stride=1)
        conv2 = braintrace.nn.Conv2d(in_size=(32, 32, 16), out_channels=32, kernel_size=3, stride=2)
        conv3 = braintrace.nn.Conv2d(in_size=(16, 16, 32), out_channels=64, kernel_size=3, stride=2)

        x = brainstate.random.randn(4, 32, 32, 3)
        y1 = conv1(x)
        y2 = conv2(y1)
        y3 = conv3(y2)

        assert y1.shape == (4, 32, 32, 16)
        assert y2.shape == (4, 16, 16, 32)
        assert y3.shape == (4, 8, 8, 64)

    def test_conv_output_shape_computation(self):
        """Test that output shape is correctly computed during initialization."""
        conv = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=3)
        assert conv.out_size is not None
        assert len(conv.out_size) == 3
        assert conv.out_size[-1] == 32

    def test_conv_with_jit_compilation(self):
        """Test that convolution works with JAX JIT compilation."""
        conv = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=3)

        @brainstate.transform.jit
        def forward(x):
            return conv(x)

        x = brainstate.random.randn(8, 28, 28, 3)
        y = forward(x)
        assert y.shape == (8, 28, 28, 32)

    def test_conv_deterministic_with_same_seed(self):
        """Test that convolution is deterministic with same random seed."""
        # Create two identical convolutions with the same seed
        brainstate.random.seed(42)
        conv1 = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=3)

        brainstate.random.seed(42)
        conv2 = braintrace.nn.Conv2d(in_size=(28, 28, 3), out_channels=32, kernel_size=3)

        x = brainstate.random.randn(8, 28, 28, 3)
        y1 = conv1(x)
        y2 = conv2(x)

        assert jnp.allclose(y1, y2)
