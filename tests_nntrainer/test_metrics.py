"""
Test metric utilities.
"""

from nntrainer import metric


def test_averagemeter() -> None:
    """
    Test averagemeter.
    """
    meter = metric.AverageMeter()
    meter.update(4, 2)
    assert meter.value == 4
    assert meter.sum == 8
    assert meter.count == 2
    assert meter.avg == 4

    meter.update(1)
    assert meter.value == 1
    assert meter.sum == 9
    assert meter.count == 3
    assert meter.avg == 3

    meter.reset()
    assert meter.value == 0
    assert meter.sum == 0
    assert meter.count == 0
    assert meter.avg == 0


if __name__ == "__main__":
    test_averagemeter()
