import pytest

from mdlab.exercise import check


def test_unattempted_passes_silently():
    assert check(None, 1.0) is None


def test_correct_answer():
    assert check(1.0005, 1.0, rtol=1e-3) is True


def test_wrong_answer_raises():
    with pytest.raises(AssertionError):
        check(1.1, 1.0)


def test_answer_of_the_wrong_shape_raises():
    with pytest.raises(AssertionError, match="shape"):
        check([1.0, 1.0], 1.0)
