"""
Custom typing extension.

Classes:
    ConstantHolder: Base class for storing constants and avoiding to hardcode everything.
    SaveableBaseModel: Child class of pydantic.BaseModel which enables saving and loading that BaseModel.
    TypedNamedTuple: Child class of SaveableBaseModel, can be used similarly to a NamedTuple and has some
        tensor handling utilities.
    ConfigClass: Base class for storing configuration fields that appear in the configuration YAML files.
"""

from __future__ import annotations

import inspect
import json
from collections import Iterable, Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast, KeysView, ItemsView, ValuesView

import numpy as np
import torch as th
from pydantic import BaseModel

INF = 32752  # infinity expressed in float16, this is large enough s.t. exp(-INF) == 0
TENSOR_TYPES = (th.Tensor, np.ndarray)
PathType = Union[str, Path]


class ConfigClass:
    """
    Base class for config storage classes. Defines representation for printing.
    """

    def __repr__(self) -> str:
        """
        Represent class attributes as key, value pairs.

        Returns:
            String representation of the config.
        """
        str_repr = ["", "-" * 10 + " " + type(self).__name__]
        for key, value in vars(self).items():
            if key in ["config_orig"]:
                continue
            if isinstance(value, ConfigClass):
                # str_repr += ["-" * 10 + " " + key, str(value)]
                str_repr += [str(value)]
            else:
                str_repr += [f"    {key} = {value}"]
        return "\n".join(str_repr)


# ---------- SaveableBaseModel: Class for modeling and storing states. ----------

class SaveableBaseModel(BaseModel):
    """
    Saveable version of pydantic BaseModel class.
    """

    def save(self, file: Union[str, Path]) -> None:
        """
        Save the model.

        Args:
            file: Target json file.
        """
        try:
            json.dump(self.dict(), Path(file).open("wt", encoding="utf8"))
        except TypeError as e:
            # something in the object is probably not JSON serializable.
            print("---------- JSON encoding error! ----------")
            for key, val in self.dict().items():
                print(f"{key}: {type(val)}")
            raise TypeError(f"See console output. JSON save to {file} failed.") from e

    def load(self, file: Union[str, Path]) -> SaveableBaseModel:
        """
        Load model values from file.

        Args:
            file: Source json file.

        Returns:
            Class instance with values set from the file.
        """
        for key, val in json.load(Path(file).open("rt", encoding="utf8")).items():
            self.__setattr__(key, val)
        return self

    @classmethod
    def create_from_file(cls, file: Union[str, Path]) -> SaveableBaseModel:
        """
        Instantiate model from file.

        Args:
            file: Source json file.

        Returns:
            Class instance with values set from the file.
        """
        return cls(**json.load(Path(file).open("rt", encoding="utf8")))


    class Config:
        # configure pydantic.BaseModel to fail on assigning wrongly typed values
        validate_assignment = True


# ---------- TypedNamedTuple: Class for explicit data modeling. ----------

def _nested_shape_check(field_name: str, tensor_container: Any,
                        shape: [List[Optional[int]]]) -> None:
    """
    Check if input tensor matches the given shape. If input is iterable or mapping, recurse into it and check
    if all contained tensors match the given shape.

    Args:
        field_name: Used to give a more verbose error.
        tensor_container: Input tensor or container of tensors.
        shape: Target shape to check.

    Raises:
        AssertionError (wrong shape), TypeError (wrong input type)
    """
    if isinstance(tensor_container, TENSOR_TYPES):
        value_shape = tensor_container.shape
        err_msg = f"Shape mismatch, input {value_shape} defined {shape} on field {field_name}"
        # check same number of dimensions
        assert len(value_shape) == len(shape), err_msg
        # check each dimension
        for s1, s2 in zip(value_shape, shape):
            # either target shape is arbitrary (None) or it matches input shape
            assert s2 is None or s1 == s2, err_msg
    elif isinstance(tensor_container, Iterable):
        for tensor_subcontainer in tensor_container:
            _nested_shape_check(field_name, tensor_subcontainer, shape)
    elif isinstance(tensor_container, Mapping):
        for _, tensor_subcontainer in tensor_container.items():
            _nested_shape_check(field_name, tensor_subcontainer, shape)
    else:
        raise TypeError(
                f"Tensor shape check on class {type(tensor_container)} not supported, field {field_name}.")


class TypedNamedTuple(BaseModel):
    """
    Behaves similar to NamedTuple. Includes type and shape validation.

    Notes:
        Implementation of pydantic BaseModel that can be instantiated with args instead of kwargs
        Define class field _shape_dict to check shapes.

    Args:
        *args: Values for the model with same order as defined.

    Examples:
        >>> class ExampleTuple(TypedNamedTuple):
        >>>     key: str
        >>>     data: th.Tensor
        >>>     # shape check: first dimension arbitrary, second must match exactly
        >>>     _shapes_dict = {"key": (None, 6)}
        >>> t = ExampleTuple("key", th.zeros(4, 6))
        >>> t.key # access with field attribute
        >>> t.dict()["key"] # access like a dict
        >>> t.tuple()[0] # access like a tuple
    """
    _shapes_dict: Dict[str, List[Optional[int]]] = {}

    def __init__(self, *args, **kwargs):
        assert len(args) <= len(self.__fields__), (f"Too many ({len(args)}) arguments "
                                                   f"for class {self.__class__.__name__}")
        if len(args) > 0:
            # fill the kwargs dict with (name, value) entries from args
            for (field, _model_field), arg in zip(self.__fields__.items(), args):
                assert field not in kwargs, f"Duplicate argument '{field}' for class {self.__class__.__name__}."
                kwargs[field] = arg
        # instantiate the model with that dict
        super().__init__(**kwargs)
        self.validate_shapes()

    def __len__(self) -> int:
        """
        Convenience function: length of the tuple

        Returns:
            Length.
        """
        return len(self.__fields__)

    def tuple(self) -> Tuple[Any]:
        """
        Access the model values as tuple.

        Returns:
            Model values as tuple.
        """
        return tuple(self.dict().values())

    def dict(self, **kwargs) -> Dict[str, Any]:  # pylint: disable=useless-super-delegation
        """
        Overwrite this function for proper type hints.

        Returns:
            Model fields and values as dict.
        """
        return super().dict(**kwargs)

    def keys(self) -> KeysView:
        """
        Get list of constant keys.

        Returns:
            Constant keys.
        """
        return self.dict().keys()

    def items(self) -> ItemsView:
        """
        Get list of constant keys.

        Returns:
            Constant keys.
        """
        return self.dict().items()

    def values(self) -> ValuesView:
        """
        Return constant values.

        Returns:
            Constant values.
        """
        return self.dict().values()

    def validate_shapes(self):
        """
        Use class field _shapes_dict to check if input tensors match the target shapes.

        Returns:
        """
        # loop all defined shapes
        for key, shape in self._shapes_dict.items():
            # get the field value with defined name
            value = self.dict()[key]
            # compare to target shape
            _nested_shape_check(key, value, shape)

    def to_cuda(self, *, non_blocking: bool = True) -> None:
        """
        Convenience function: Move all tensors in the model to cuda.

        Args:
            non_blocking: Some PyTorch internal parameter, has something to do with pin_memory in dataloader.
                Usually shouldn't hurt to keep it at True.
        """
        # loop all tensors in the model
        for name, value in self.dict().items():
            if isinstance(value, th.Tensor):
                # update pydantic BaseModel with setattr
                setattr(self, name, value.cuda(non_blocking=non_blocking))


    class Config:
        # allow torch tensors etc.
        arbitrary_types_allowed = True


# ---------- ConstantHolder: Container for constants ----------

class _StringRepr(type):
    """
    Metaclass for overwriting result of str(Class).
    """

    def __str__(cls) -> str:
        """
        When calling str(Class), call Class._get_string_repr method.

        Returns:
            Custom class string representation.
        """
        return cls._get_string_repr()  # pylint: disable=no-value-for-parameter

    def _get_string_repr(cls) -> str:
        """
        Override this to return string representation of the class.

        Returns:
            Class representation as string.
        """
        raise NotImplementedError


class ConstantHolder(metaclass=_StringRepr):
    """
    Class to hold constants. Attributes must be uppercase. Cannot be instanced.

    Notes:
        There is some magic happening here:
        The properties of this base class (_keys, _values, _dict) will hold all constants including those of inherited
        classes. The interface will then dynamically return the correct things given the current cls.__name__.

    Examples:
        Instantiate the class and set constants as class attributes.
        Set allowed_types to single type or list of types for value checks.
        >>> class MyConstants(allowed_types=str):
        >>>     FIELD = "value"


    Methods:
        keys: Get list of constant keys.
        values: Get list of constant values.
        items: Get list of constant key/value tuples.
        dict: Get dictionary of constant key/value pairs.
        get: Get value given key, error if not found.
        get_safe: Get value given key, return default if not found.
        check_has_key: Returns bool whether or not the key is in the class.
        assert_has_key: Raise error if the key is not found.
        check_has_value: Returns bool whether or not the value is in the class.
        assert_has_value: Raise error if the value is not found.

    Notes:
        Public interface: Methods keys, values, items, dict, get. Supports __getitem__ syntax (using []).

        This class is introduced because enum.Enum has lots of unnecessary restrictions and is clumsy to use.
        Public methods resemble those of a dict but return lists, not e.g. instances of dict_keys.
    """
    # create the class properties with empty entries for the root parent
    _keys: Dict[str, List[str]] = {"ConstantHolder": []}
    _values: Dict[str, List[Any]] = {"ConstantHolder": []}
    _dict: Dict[str, Dict[str, Any]] = {"ConstantHolder": {}}

    # ---------- Public interface ----------
    @classmethod
    def keys(cls) -> List[str]:
        """
        Get list of constant keys.

        Returns:
            Constant keys.
        """
        return cls._keys[cls.__name__]

    @classmethod
    def values(cls) -> List[Any]:
        """
        Return constant values.

        Returns:
            Constant values.
        """
        return cls._values[cls.__name__]

    @classmethod
    def dict(cls) -> Dict[str, Any]:
        """
        Return constant key-value pairs as dict.

        Returns:
            Constant keys.
        """
        return cls._dict[cls.__name__]

    @classmethod
    def items(cls) -> List[Tuple[str, Any]]:
        """
        Return constant key-value pairs as list of tuples like dict items.

        Returns:
            Constant keys.
        """
        return list(zip(cls._keys[cls.__name__], cls._values[cls.__name__]))

    @classmethod
    def get(cls, key: str) -> Any:
        """
        Get constant value given the key. Raise error if not found.

        Args:
            key: Constant key.

        Returns:
            Constant value.
        """
        if key not in cls.keys():
            raise IndexError(f"No key: {key} in {cls}")
        return cast(Any, getattr(cls, key))

    @classmethod
    def get_safe(cls, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Get constant value given the key. Return default if not found.

        Args:
            key: Constant key.
            default: Value to return if key is not found, default None.

        Returns:
            Constant value or default.
        """
        if key not in cls.keys():
            return default
        return cls.get(key)

    @classmethod
    def check_has_key(cls, key: str) -> bool:
        """
        Check if the key is in the class.

        Args:
            key: Constant key.

        Returns:
            Whether or not the key is defined in the class.
        """
        return key in cls.keys()

    @classmethod
    def assert_has_key(cls, key: str) -> None:
        """
        Throw error if the key is not found in the class.

        Args:
            key: Constant key.
        """
        assert cls.check_has_key(key), f"Key not found: {key} in {cls}"

    @classmethod
    def check_has_value(cls, value: Any) -> bool:
        """
        Check if the value is in the class.

        Args:
            value: Constant value.

        Returns:
            Whether or not the key is defined in the class.
        """
        return value in cls.values()

    @classmethod
    def assert_has_value(cls, value: str) -> None:
        """
        Throw error if the value is not found in the class.

        Args:
            value: Constant value.
        """
        assert cls.check_has_value(value), f"Value not found: {value} in {cls}"

    # ---------- Private setup methods ----------

    @classmethod
    def _get_string_repr(cls) -> str:
        """
        Return class name and content as string for better error messages.

        Returns:
            String representation.
        """
        return f"ConstantHolder {cls.__name__}: {cls.items()}"

    @classmethod
    def __init_subclass__(
            cls, allowed_types: Optional[Union[type, List[type], Tuple[type, ...]]] = None) -> None:
        """
        Setup properties for the public interface when this class is inherited.

        This will be called on nested inheritance as well.

        Args:
            allowed_types: Optionally specify a type or list of types that are allowed for values.
                By default all values are allowed.
        """
        cls._keys[cls.__name__] = []
        cls._values[cls.__name__] = []
        cls._dict[cls.__name__] = {}

        # add parent fields
        for parent_cls in cls.__bases__:
            cls._keys[cls.__name__] += cls._keys[parent_cls.__name__]
            cls._values[cls.__name__] += cls._values[parent_cls.__name__]
            cls._dict[cls.__name__].update(cls._dict[parent_cls.__name__])

        # loop attributes, check correctness and extend the parent's class properties _keys, _values, _dict.
        for key in cls.__dict__:
            # ignore non-public fields
            if key[0] == "_":
                continue

            # get the value of the constant
            value = getattr(cls, key)

            # ignore classmethods
            if inspect.ismethod(value) and value.__self__ is cls:
                continue

            # make sure all constants are uppercase
            assert key == key.upper(), f"Constant: {key} in class: {cls.__name__} must be uppercase."

            # if allowed types is specified, make sure the value types are allowed
            if allowed_types is not None:
                # isinstance errors when fed lists instead of tuple, so convert lists to tuples
                if isinstance(allowed_types, list):
                    allowed_types = tuple(allowed_types)
                assert isinstance(value, allowed_types), (
                        f"Constant: {key} in class: {cls.__name__} must be of type {allowed_types}")

            # update class properties
            cls._keys[cls.__name__].append(key)
            cls._values[cls.__name__].append(value)
            cls._dict[cls.__name__][key] = value

    def __init__(self) -> None:
        """
        Raise error when trying to instance a ConstantHolder class.
        """
        raise RuntimeError(
                f"Do not instance this class, it's a ConstantHolder: {type(self).__name__}")
