"""Core implementation of the Gaussian hypergeometric function 2F1.

The implementation uses Euler's hypergeometric transformation and
series acceleration to evaluate 2F1 on the real interval (-inf, 1).
It avoids third-party dependencies by using only the standard library's
``math`` module.
"""

from __future__ import annotations

import math
from typing import Union

Number = Union[int, float]


def _gamma_ratio(x: float, y: float) -> float:
    """Return gamma(x)/gamma(y) for positive x, y using log-gamma.

    This avoids overflow for large arguments, which is important when
    a, b, or c are moderate to large and z approaches the radius of
    convergence.
    """
    return math.exp(math.lgamma(x) - math.lgamma(y))


def hypergeometric2f1(a: Number, b: Number, c: Number, z: Number) -> float:
    """Evaluate the Gaussian hypergeometric function 2F1(a,b;c;z).

    Parameters
    ----------
    a, b : float
        Numerator parameters.
    c : float
        Denominator parameter. Must not be a non-positive integer.
    z : float
        Argument. Real values less than 1 are supported. The branch cut
        is taken to be [1, inf); values on the cut raise ``ValueError``.

    Returns
    -------
    float
        Value of 2F1(a,b;c;z).

    Raises
    ------
    ValueError
        If c is a non-positive integer, or z is on the branch cut
        (z >= 1).
    ZeroDivisionError
        If c is a non-positive integer and the denominator becomes zero.
    """
    a = float(a)
    b = float(b)
    c = float(c)
    z = float(z)

    if z >= 1.0:
        raise ValueError("z must be less than 1; z=1 is a singular point")

    # Handle denominator singularities explicitly. Non-positive integer c
    # makes the series undefined.
    if c <= 0.0 and c.is_integer():
        raise ValueError("c must not be a non-positive integer")

    # Special case: if a or b is a non-positive integer, the series
    # terminates and the function is a polynomial. Direct summation is
    # exact in this case.
    if a <= 0.0 and a.is_integer():
        n = int(-a)
        term = 1.0
        total = 1.0
        for k in range(1, n + 1):
            term *= (a + k - 1.0) * (b + k - 1.0) / ((c + k - 1.0) * k) * z
            total += term
        return total
    if b <= 0.0 and b.is_integer():
        return hypergeometric2f1(b, a, c, z)

    # For z <= -1, the standard series converges very slowly. Use Euler's
    # transformation to improve convergence. This is exact for all
    # parameter values where c-a-b is not a non-positive integer.
    if z <= -1.0:
        # Euler transformation:
        # 2F1(a,b;c;z) = (1-z)^(-a) * 2F1(a, c-b; c; z/(z-1))
        # The new argument z/(z-1) lies in [0, 1), and the series
        # converges much faster.
        transformed_z = z / (z - 1.0)
        return (1.0 - z) ** (-a) * hypergeometric2f1(a, c - b, c, transformed_z)

    # For z in (-1, 1), use the direct series. Convergence is acceptable
    # for most parameters; for large parameters near z=1 it can be slow,
    # but the standard series is the most transparent and stable choice
    # without a continued fraction implementation.
    term = 1.0
    total = 1.0
    k = 1
    max_iter = 100000
    tol = 1e-15
    while k < max_iter:
        term *= (a + k - 1.0) * (b + k - 1.0) / ((c + k - 1.0) * k) * z
        total += term
        if abs(term) < tol * max(1.0, abs(total)):
            break
        k += 1
    return total
