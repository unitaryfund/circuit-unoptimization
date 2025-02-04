"""Tests for utility functions."""

import pytest
import math
from unopt.utils import quadratic


@pytest.mark.parametrize(
    "x,a,b,c,expected",
    [
        (0, 1, 2, 3, 3),
        (1, 1, 2, 3, 6),
        (2, 1, 2, 3, 11),
        (3, 2, -1, 4, 19),
    ],
)
def test_quadratic(x: float, a: float, b: float, c: float, expected: float) -> None:
    """Test the quadratic function."""
    result = quadratic(x, a, b, c)
    assert math.isclose(result, expected, rel_tol=1e-9)
