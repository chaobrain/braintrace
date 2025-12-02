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


import warnings
from enum import Enum
from typing import Sequence

import brainstate
import brainunit as u
import jax.tree

from ._compatible_imports import Var
from ._typing import Path, ETraceDF_Key, ETraceWG_Key


__all__ = [
    'NotSupportedError',
    'CompilationError',
]


def _remove_quantity(tree):
    """
    Remove the quantity from the tree.

    Args:
      tree: The tree.

    Returns:
      The tree without the quantity.
    """

    def fn(x):
        if isinstance(x, u.Quantity):
            return x.magnitude
        return x

    return jax.tree.map(fn, tree, is_leaf=lambda x: isinstance(x, u.Quantity))


def check_dict_keys(
    d1: dict,
    d2: dict,
):
    """
    Check the keys of two dictionaries.

    Parameters
    ----------
    d1 : dict
      The first dictionary.
    d2 : dict
      The second dictionary.

    Raises
    ------
    ValueError
      If the keys of the two dictionaries are not the same.
    """
    if d1.keys() != d2.keys():
        raise ValueError(f'The keys of the two dictionaries are not the same: {d1.keys()} != {d2.keys()}.')


def hid_group_key(hidden_group_id: int) -> str:
    """
    Generate a key for a hidden group based on its ID.

    Parameters
    ----------
    hidden_group_id : int
        The ID of the hidden group.

    Returns
    -------
    str
        A string key representing the hidden group.
    """
    assert isinstance(hidden_group_id, int), f'hidden_group_id must be an int, but got {hidden_group_id}.'
    return f'hidden_group_{hidden_group_id}'


def etrace_x_key(
    x_key: Var,
) -> int:
    """
    Generate a key for the eligibility trace based on a variable key.

    Parameters
    ----------
    x_key : Var
        The variable key associated with the trace.

    Returns
    -------
    int
        An integer identifier derived from the variable key.
    """
    return id(x_key)


def etrace_df_key(
    y_key: Var,
    hidden_group_id: int,
) -> ETraceDF_Key:
    """
    Generate a key for the eligibility trace dataframe.

    Parameters
    ----------
    y_key : Var
        The variable key associated with the trace.
    hidden_group_id : int
        The ID of the hidden group.

    Returns
    -------
    tuple
        A tuple containing the variable key and a string key representing the hidden group.
    """
    assert isinstance(y_key, Var), f'y_key must be a Var, but got {y_key}.'
    return (id(y_key), hid_group_key(hidden_group_id))


def etrace_param_key(
    weight_path: Path,
    y_key: Var,
    hidden_group_id: int,
) -> ETraceWG_Key:
    """
    Generate a key for the eligibility trace parameter.

    Parameters
    ----------
    weight_path : Path
        The path to the weight, represented as a list or tuple of strings.
    y_key : Var
        The variable key associated with the trace.
    hidden_group_id : int
        The ID of the hidden group.

    Returns
    -------
    tuple
        A tuple containing the weight path, variable key, and a string key representing the hidden group.
    """
    assert isinstance(weight_path, (list, tuple)), f'weight_path must be a list or tuple, but got {weight_path}.'
    assert all(isinstance(x, (str, int)) for x in weight_path), \
        f'weight_path must be a list of str, but got {weight_path}.'
    assert isinstance(y_key, Var), f'y_key must be a Var, but got {y_key}.'
    return (weight_path, id(y_key), hid_group_key(hidden_group_id))


def unknown_state_path(i: int) -> Path:
    """
    Generate a path for an unknown state.

    Parameters
    ----------
    i : int
        An integer representing the index of the unknown state.

    Returns
    -------
    Path
        A tuple containing a string that represents the path of the unknown state.
    """
    return (f'_unknown_path_{i}',)


def _dimensionless(x):
    if isinstance(x, u.Quantity):
        return x.mantissa
    else:
        return x


def remove_units(xs):
    """
    Remove units from a tree structure of quantities.

    This function traverses a tree structure and removes the units from any
    quantities found, leaving only the dimensionless values.

    Parameters
    ----------
    xs : Any
        The tree structure containing quantities with units.

    Returns
    -------
    Any
        A tree structure with the same shape as `xs`, but with units removed
        from any quantities.
    """
    return jax.tree.map(
        _dimensionless,
        xs,
        is_leaf=u.math.is_quantity
    )


git_issue_addr = 'https://github.com/chaobrain/braintrace/issues'


def deprecation_getattr(module, deprecations):
    """
    Create a custom getattr function to handle deprecated attributes.

    This function generates a custom getattr function for a module, which
    checks if an attribute is deprecated and handles it accordingly by
    raising an AttributeError or issuing a warning.

    Parameters
    ----------
    module : str
        The name of the module for which the custom getattr function is created.
    deprecations : dict
        A dictionary where keys are attribute names and values are tuples
        containing a deprecation message and an optional function. If the
        function is None, accessing the attribute will raise an AttributeError.

    Returns
    -------
    function
        A custom getattr function that handles deprecated attributes.
    """

    def getattr(name):
        if name in deprecations:
            message, fn = deprecations[name]
            if fn is None:  # Is the deprecation accelerated?
                raise AttributeError(message)
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return fn
        raise AttributeError(f"module {module!r} has no attribute {name!r}")

    return getattr


class NotSupportedError(Exception):
    """
    Exception raised for operations that are not supported.

    This exception is used to indicate that a particular operation or
    functionality is not supported within the context of the application.

    """
    __module__ = 'braintrace'


class CompilationError(Exception):
    """
    Exception raised for errors that occur during the compilation process.

    This exception is used to indicate that a compilation error has occurred
    within the context of the application.
    """
    __module__ = 'braintrace'


def state_traceback(states: Sequence[brainstate.State]):
    """
    Generate a traceback string for a sequence of brain model states.

    This function iterates over a sequence of brain model states and constructs
    a string that contains detailed traceback information for each state. The
    traceback includes the index of the state, its representation, and the
    source information where it was defined.

    Parameters
    ----------
    states : Sequence[brainstate.State]
        A sequence of states from the brain model. Each state should be an
        instance of `brainstate.State` and contain source information for traceback.

    Returns
    -------
    str
        A string containing the traceback information for each state in the
        sequence. Each state's traceback includes its index, representation,
        and source definition details.
    """
    state_info = []
    for i, state in enumerate(states):
        state_info.append(
            f'State {i}: {state}\n'
            f'defined at \n'
            f'{state.source_info.traceback}\n'
        )
    return '\n'.join(state_info)


def set_module_as(module: str = 'braintrace'):
    """
    Decorator to set the module attribute of a function.

    This function returns a decorator that sets the `__module__` attribute
    of a function to the specified module name.

    Parameters
    ----------
    module : str, optional
        The name of the module to set for the function, by default 'braintrace'.

    Returns
    -------
    function
        A decorator function that sets the `__module__` attribute of the
        decorated function to the specified module name.
    """

    def wrapper(fun: callable):
        fun.__module__ = module
        return fun

    return wrapper


class BaseEnum(Enum):
    """
    Base class for creating enumerations with additional utility methods.

    This class extends the standard Enum class to provide additional
    methods for retrieving enumeration members by name or directly
    from an instance.
    """

    @classmethod
    def get_by_name(cls, name: str):
        """
        Retrieve an enumeration member by its name.

        This method searches for an enumeration member within the class
        that matches the provided name and returns it.

        Parameters
        ----------
        name : str
            The name of the enumeration member to retrieve.

        Returns
        -------
        Enum
            The enumeration member corresponding to the provided name.

        Raises
        ------
        ValueError
            If no enumeration member with the specified name is found.
        """
        all_names = []
        for item in cls:
            all_names.append(item.name)
            if item.name == name:
                return item
        raise ValueError(f'Cannot find the {cls.__name__} type {name}. Only support {all_names}.')

    @classmethod
    def get(cls, item: str | Enum):
        """
        Retrieve an enumeration member by its name or directly if it is an Enum.

        This method returns the enumeration member if the provided item is
        already an instance of the enumeration. If the item is a string, it
        attempts to find the corresponding enumeration member by name.

        Parameters
        ----------
        item : str | Enum
            The name of the enumeration member to retrieve, or an instance
            of the enumeration.

        Returns
        -------
        Enum
            The enumeration member corresponding to the provided item.

        Raises
        ------
        ValueError
            If the item is a string and no enumeration member with the
            specified name is found, or if the item is neither a string
            nor an instance of the enumeration.
        """
        if isinstance(item, cls):
            return item
        elif isinstance(item, str):
            return cls.get_by_name(item)
        else:
            raise ValueError(f'Cannot find the {cls.__name__} type {item}.')
