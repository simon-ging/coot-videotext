"""
Test types.
"""

import json
import tempfile
from pathlib import Path

import pydantic
import pytest
import torch as th

from nntrainer import typext


def test_typednamedtuple() -> None:
    """
    Test typed named tuples class based on pydantic.BaseModel
    """

    class ExampleTuple(typext.TypedNamedTuple):
        key: str
        data: th.Tensor

        # data tensor should have shape (X, 4) where X is arbitrary.
        _shapes_dict = {
            "data": (None, 4)}

    # correct input
    t = ExampleTuple("key", th.zeros(7, 4))

    # correct input with mixed args, kwargs
    ExampleTuple("key", data=th.zeros(3, 4))
    ExampleTuple(key="key", data=th.zeros(3, 4))

    # not enough input
    with pytest.raises(pydantic.ValidationError):
        ExampleTuple("key")

    # duplicate argument
    with pytest.raises(AssertionError):
        ExampleTuple("key", key="key", data=th.zeros(3, 4))

    # too many arguments
    with pytest.raises(AssertionError):
        ExampleTuple("key", th.zeros(3, 4), None)

    # wrong type
    with pytest.raises(pydantic.ValidationError):
        ExampleTuple(False, 0)

    # wrong shape
    with pytest.raises(AssertionError):
        ExampleTuple("key", th.zeros(3, 6))
    with pytest.raises(AssertionError):
        ExampleTuple("key", th.zeros(6))
    with pytest.raises(AssertionError):
        ExampleTuple("key", th.zeros(4, 1, 1))

    # test dict and tuple access
    assert isinstance(t.dict(), dict)
    assert t.dict()["key"] == "key"
    assert isinstance(t.tuple(), tuple)
    assert t.tuple()[0] == "key"


def test_saveablebasemodel() -> None:
    """
    Test saveable base model based on pydantic.BaseModel
    """

    class TestState(typext.SaveableBaseModel):
        test_field: int = 1

    input_dict = {"test_field": 7}
    t1 = TestState(**input_dict)
    print(t1)
    tf = Path(tempfile.gettempdir()) / "temp_nntrainer.tmp"
    t1.save(tf)
    file_content = json.load(tf.open(encoding="utf8"))
    assert file_content == input_dict, f"{file_content} vs {input_dict}"
    t2 = TestState().load(tf)
    assert t1 == t2
    t3 = TestState.create_from_file(tf)
    assert t1 == t3

    wrong_dict = {"test_field": "str"}
    json.dump(wrong_dict, tf.open("wt", encoding="utf8"))
    with pytest.raises(pydantic.ValidationError):
        TestState.create_from_file(tf)
    with pytest.raises(pydantic.ValidationError):
        TestState().load(tf)


if __name__ == "__main__":
    test_typednamedtuple()
    test_saveablebasemodel()
