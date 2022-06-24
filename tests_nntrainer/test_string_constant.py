"""
Test ConstantHolder.
"""
import pytest

from nntrainer.typext import ConstantHolder


# noinspection PyUnusedLocal,PyStatementEffect
# pylint: disable=unused-variable
def test_string_constant() -> None:
    """
    Test ConstantHolder.
    """

    # define some test classes
    class NewConst(ConstantHolder):
        FIELD = "some value"
        ANOTHER_FIELD = "another value"

    class DerivedConst(NewConst):
        THIRD_FIELD = "yet another value"

    class DerivedConstTwo(NewConst):
        FOURTH_FIELD = "yet another value"

    # test keys() method, make sure each element in the inheritance tree has the correct fields
    assert ConstantHolder.keys() == [], str(ConstantHolder)
    assert NewConst.keys() == ["FIELD", "ANOTHER_FIELD"], str(NewConst)
    assert DerivedConst.keys() == NewConst.keys() + ["THIRD_FIELD"], str(DerivedConst)
    assert DerivedConstTwo.keys() == NewConst.keys() + ["FOURTH_FIELD"], str(DerivedConstTwo)

    # check multiple inheritance works
    class ParentA(ConstantHolder):
        A = "A"

    class ParentB(ConstantHolder):
        B = "B"

    class ChildAB(ParentA, ParentB):
        C = "C"

    assert ChildAB.keys() == ["A", "B", "C"], str(ChildAB)

    # check allowed_types keyword arg works
    class MixedConst(ConstantHolder, allowed_types=[str, int]):
        VAL = "str"
        ANOTHERVAL = 8

    with pytest.raises(AssertionError):
        class IntConst(ConstantHolder, allowed_types=int):
            VAL = 7
            FAIL = "hi"

    # test values() method
    assert DerivedConstTwo.values() == ["some value", "another value", "yet another value"], str(DerivedConstTwo)

    # test dict() method
    assert DerivedConstTwo.dict() == {
        "FIELD": "some value", "ANOTHER_FIELD": "another value", "FOURTH_FIELD": "yet another value"}, str(
        DerivedConstTwo)

    # check dict() result is properly sorted
    assert list(DerivedConstTwo.dict().keys()) == DerivedConstTwo.keys()
    assert list(DerivedConstTwo.dict().values()) == DerivedConstTwo.values()

    # test __class_getitem__ method
    assert DerivedConst.get("FIELD") == "some value"
    with pytest.raises(IndexError):
        DerivedConst.get("FAIL")

    # test get method
    assert DerivedConst.get_safe("FIELD") == "some value"
    assert DerivedConst.get_safe("FAIL") is None
    assert DerivedConst.get_safe("FAIL", "default") == "default"

    # test str method
    assert str(DerivedConst) == ("ConstantHolder DerivedConst: [('FIELD', 'some value'), "
                                 "('ANOTHER_FIELD', 'another value'), ('THIRD_FIELD', 'yet another value')]")

    # instantiation must error
    with pytest.raises(RuntimeError):
        NewConst()

    # lowercase fields must error
    with pytest.raises(AssertionError):
        class ErrorConst(NewConst):
            THIRD_FIELD = "yet another value"
            lowercase_field = "Bla"


if __name__ == "__main__":
    test_string_constant()
