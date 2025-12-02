# Release Notes


## Version 0.1.1

### Major Changes

#### Project Rename: BrainScale → BrainTrace
- **Renamed the entire project from `brainscale` to `braintrace`**: This change reflects the project's focus on eligibility trace-based learning algorithms
  - Package directory renamed from `brainscale/` to `braintrace/`
  - All internal imports updated from `brainscale` to `braintrace`
  - Updated all 95 files including source code, tests, documentation, and examples
  - Updated `pyproject.toml` with new project name and metadata
  - Updated README with new project branding and citation information

#### VJP-Based Eligibility Trace Algorithms
- **Added new VJP-based eligibility trace module** (`_etrace_vjp/`): Comprehensive implementation of vector-Jacobian product based algorithms
  - `base.py`: Core base classes and utilities for VJP operations (671 lines)
  - `d_rtrl.py`: Diagonal Real-Time Recurrent Learning implementation (756 lines)
  - `esd_rtrl.py`: Efficient Sparse Diagonal RTRL implementation (847 lines)
  - `hybrid.py`: Hybrid approaches combining multiple techniques (604 lines)
  - `graph_executor.py`: Graph-based execution for VJP computations
  - `misc.py`: Miscellaneous utilities including matrix spectrum normalization

- **Refactored VJP algorithm structure**: Migrated from monolithic `_etrace_vjp_algorithms.py` (2,888 lines) to modular architecture
  - Better separation of concerns
  - Improved testability with dedicated test files (`d_rtrl_test.py`, `esd_rtrl_test.py`, `graph_executor_test.py`)

#### Logo and Branding
- Updated logo format from JPG to PNG for consistency
- Updated logo across documentation

### Breaking Changes

**Package Rename:**
1. **Import path change**: All imports must now use `braintrace` instead of `brainscale`

```python
# Old (0.1.0)
import brainscale
from brainscale import EligibilityTrace
from brainscale.nn import Linear, GRUCell

# New (0.1.1)
import braintrace
from braintrace import EligibilityTrace
from braintrace.nn import Linear, GRUCell
```

2. **Installation**: Package name changed from `brainscale` to `braintrace`

```bash
# Old
pip install brainscale

# New
pip install braintrace
```

### Migration Guide

#### Update Import Statements
Replace all occurrences of `brainscale` with `braintrace`:

```python
# Find and replace in your codebase
# brainscale → braintrace
```

#### VJP Algorithm Usage
The new VJP-based algorithms are now available through the modular interface:

```python
from braintrace._etrace_vjp import d_rtrl, esd_rtrl, hybrid
```

### Version
- Bumped version from `0.1.0` to `0.1.1`


## Version 0.1.0

### Major Changes

#### State Management Refactoring
- **Renamed `ETraceState` to `HiddenState`**: All eligibility trace state management now uses the more general `HiddenState` naming convention
  - Updated across `_etrace_algorithms.py`, `_etrace_concepts.py`, `_state_managment.py`
  - Added deprecation warnings for `ETraceState` to guide users to `brainstate.HiddenState`
  - Updated all documentation and examples to reflect the new naming

- **Renamed `ETraceGroupState` to `HiddenGroupState`**: Improved consistency in hidden state handling
  - Updated in `_etrace_compiler_hidden_group.py`
  - Added deprecation warnings for backward compatibility

- **Added deprecation handling**: Implemented `__getattr__` in main `__init__.py` to provide helpful warnings when using deprecated names:
  - `ETraceState` → `brainstate.HiddenState`
  - `ETraceGroupState` → `brainstate.HiddenGroupState`
  - `ETraceTreeState` → `brainstate.HiddenTreeState`

#### Neural Network Module Reorganization

- **Consolidated neural network modules**: Removed standalone neuron, synapse, and activation modules, migrating them to `brainstate` and `brainpy` ecosystems
  - **Deleted files**:
    - `brainscale/nn/_neurons.py` (IF, LIF, ALIF now in `brainpy.state`)
    - `brainscale/nn/_synapses.py` (Expon, Alpha, DualExpon, STP, STD now in `brainpy.state`)
    - `brainscale/nn/_elementwise.py` (activation functions now in `brainstate.nn`)
    - `brainscale/nn/_poolings.py` (pooling layers now in `brainstate.nn`)

- **Renamed `_rate_rnns.py` to `_rnn.py`**: Simplified module naming for better clarity

- **Added comprehensive deprecation warnings in `nn.__getattr__`**: Automatically redirects users to the correct modules:
  - Neuron models (IF, LIF, ALIF) → `brainpy.state`
  - Synapse models (Expon, Alpha, DualExpon, STP, STD) → `brainpy.state`
  - Activation functions (ReLU, Sigmoid, etc.) → `brainstate.nn`
  - Pooling layers (MaxPool, AvgPool, etc.) → `brainstate.nn`
  - Dropout layers → `brainstate.nn`

#### API Improvements

- **Normalization parameter standardization**: Renamed `normalized_shape` to `in_size` across all normalization layers for consistency
  - Updated in `_normalizations.py` for LayerNorm, GroupNorm, InstanceNorm, etc.
  - Improved clarity and consistency with other layer APIs

- **Enhanced input dimension validation**: Improved error checking in convolutional layers to catch dimension mismatches early

- **Refactored imports for consistency**: Updated all internal imports to use `braintools` for optimization and initialization utilities consistently across the codebase

#### Testing Infrastructure

- **Added comprehensive unit tests** for neural network modules:
  - `_conv_test.py`: 868 lines of tests for convolutional layers (Conv1d, Conv2d, Conv3d, ConvTranspose)
  - `_linear_test.py`: 658 lines of tests for linear layers (Linear, Identity)
  - `_normalizations_test.py`: 695 lines of tests for normalization layers (LayerNorm, BatchNorm, GroupNorm, etc.)
  - `_readout_test.py`: 763 lines of tests for readout layers (LeakyRateReadout, LeakySpikeReadout)
  - `_rnn_test.py`: 710 lines of tests for RNN cells (VanillaRNNCell, GRUCell, LSTMCell, MGUCell, etc.)
  - Total: 3,694 lines of new test coverage

#### Documentation Updates

- **Streamlined API documentation**: Updated `docs/apis/nn.rst` to remove redundant sections and enhance RNN overview
- **Updated tutorials and examples**: All 16 tutorial notebooks and 11 example scripts updated to reflect new APIs:
  - Concepts tutorials (en/zh)
  - RNN and SNN online learning guides
  - Batching strategies documentation
  - ETrace state management examples
  - Graph visualization tutorials

#### Code Quality Improvements

- **Removed redundant docstrings**: Cleaned up duplicate documentation in `LeakyRateReadout` and `LeakySpikeReadout`
- **Improved code organization**: Streamlined `__all__` definitions across all modules
- **Enhanced readability**: Consistent import structure and better code formatting throughout

#### Dependency Updates

- **Updated `requirements.txt`**: Refined dependency specifications to ensure compatibility with latest `brainstate` and `brainpy` versions
- **Updated `pyproject.toml`**: Bumped version to 0.1.0 and updated project metadata


### Breaking Changes

**API Changes:**
1. **State class renaming** (with deprecation warnings):
   - `ETraceState` → Use `brainstate.HiddenState` instead
   - `ETraceGroupState` → Use `brainstate.HiddenGroupState` instead
   - `ETraceTreeState` → Use `brainstate.HiddenTreeState` instead

2. **Neural network component migration** (with deprecation warnings):
   - Neuron models (IF, LIF, ALIF) → Use `brainpy.state` module
   - Synapse models (Expon, Alpha, etc.) → Use `brainpy.state` module
   - Activation functions → Use `brainstate.nn` module
   - Pooling layers → Use `brainstate.nn` module

3. **Normalization parameter rename**:
   - `normalized_shape` → `in_size` (for LayerNorm, GroupNorm, etc.)

4. **Module file reorganization**:
   - `nn/_rate_rnns.py` → `nn/_rnn.py`
   - Removed: `_neurons.py`, `_synapses.py`, `_elementwise.py`, `_poolings.py`

### Migration Guide

#### For State Management:
```python
# Old (0.0.11)
from brainscale import ETraceState, ETraceGroupState

# New (0.1.0)
from brainstate import HiddenState, HiddenGroupState
```

#### For Neural Network Components:
```python
# Old (0.0.11)
from brainscale.nn import IF, LIF, Expon, ReLU, MaxPool2d

# New (0.1.0)
from brainpy.state import IF, LIF, Expon
from brainstate.nn import ReLU, MaxPool2d
```

#### For Normalization Layers:
```python
# Old (0.0.11)
norm = LayerNorm(normalized_shape=(128,))

# New (0.1.0)
norm = LayerNorm(in_size=128)
```

**Note**: All deprecated APIs include automatic warnings that will guide you to the correct replacements. The old APIs will continue to work in 0.1.0 but will be removed in a future release.

### Version
- Bumped version from `0.0.11` to `0.1.0`



## Version 0.0.11

### Major Changes

#### Import Refactoring
- **Migrated imports from `brainstate` to `braintools`**: All initialization-related imports now use `braintools.init` instead of `brainstate.init`
  - Updated imports in:
    - `brainscale/nn/_neurons.py`: Changed `from brainstate import init` to `from braintools import init`
    - `brainscale/nn/_linear.py`: Changed `from brainstate import init` to `from braintools import init`
    - `brainscale/nn/_conv.py`: Updated initialization imports
    - `brainscale/nn/_synapses.py`: Updated initialization imports
    - `brainscale/nn/_readout.py`: Updated initialization imports

- **Migrated neural network model imports from `brainstate.nn` to `brainpy`**: Updated base classes for neuron models
  - `IF`, `LIF`, `ALIF` now inherit from `brainpy` instead of `brainstate.nn`
  - Maintained API compatibility while using the new `brainpy` backend

- **Updated functional API calls**: Changed from `brainstate.functional.sigmoid` to `brainstate.nn.sigmoid` in RNN cells

#### Dependency Updates
- **Added `brainpy` as a required dependency** in `requirements.txt`

#### Documentation Enhancements
- **Improved docstring formatting across the codebase**:
  - Enhanced parameter documentation with proper type annotations using NumPy-style docstrings
  - Added missing "Returns" sections to property and method docstrings
  - Converted inline examples to proper "Examples" sections with code blocks
  - Updated documentation in:
    - `brainscale/_etrace_algorithms.py`: Enhanced `EligibilityTrace` and `ETraceAlgorithm` documentation
    - `brainscale/_etrace_compiler_base.py`: Improved parameter and return type documentation
    - `brainscale/_etrace_compiler_module_info.py`: Enhanced module documentation

#### Core Algorithm Updates
- **RNN State Management**: Updated all RNN cells to use `braintools.init.param` for state initialization and reset
  - `ValinaRNNCell`: Updated `init_state()` and `reset_state()` methods
  - `GRUCell`: Updated state management and activation functions
  - `CFNCell`: Updated forget and input gate implementations
  - `MGUCell`: Updated minimal gated unit state handling

#### Test Updates
- **Refactored test imports**: Updated test files to use new import paths
  - `brainscale/_etrace_model_test.py`: Updated with new import structure
  - `brainscale/_etrace_vjp_algorithms_test.py`: Aligned with new API

#### Version
- Bumped version from `0.0.10` to `0.0.11`

### Files Changed (17 files)
- `.gitignore`: Added new patterns
- `brainscale/__init__.py`: Updated version number
- `brainscale/_etrace_algorithms.py`: Enhanced documentation and imports
- `brainscale/_etrace_compiler_base.py`: Improved documentation
- `brainscale/_etrace_compiler_graph.py`: Minor updates
- `brainscale/_etrace_compiler_hidden_group.py`: Minor updates
- `brainscale/_etrace_compiler_module_info.py`: Enhanced documentation
- `brainscale/_etrace_model_test.py`: Updated test imports
- `brainscale/_etrace_vjp_algorithms_test.py`: Updated test imports
- `brainscale/_etrace_vjp_graph_executor.py`: Updated imports
- `brainscale/nn/_conv.py`: Migrated to braintools imports
- `brainscale/nn/_linear.py`: Migrated to braintools imports
- `brainscale/nn/_neurons.py`: Migrated to brainpy and braintools
- `brainscale/nn/_rate_rnns.py`: Migrated to braintools and updated functional APIs
- `brainscale/nn/_readout.py`: Updated imports
- `brainscale/nn/_synapses.py`: Updated imports
- `requirements.txt`: Added brainpy dependency

### Breaking Changes
None. All changes maintain backward compatibility at the API level.

### Migration Guide
If you have custom code using brainscale:
- No changes required for end users
- If extending brainscale internally, note that initialization utilities now come from `braintools` instead of `brainstate`


