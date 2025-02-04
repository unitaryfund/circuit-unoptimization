def quadratic(x: float, a: float, b: float, c: float) -> float:
    """Evaluate a quadratic function of the form ax^2 + bx + c.

    Args:
        x: The input value.
        a: The coefficient of the quadratic term (x^2).
        b: The coefficient of the linear term (x).
        c: The constant term.

    Returns:
        The result of the quadratic equation for the given x, a, b, and c.
    """
    return a * x**2 + b * x + c
