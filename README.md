# Hypergeometric Function

Evaluates the Gaussian hypergeometric function 2F1(a,b;c;z) for real arguments z < 1 using convergent series and Euler's transformation.

## Usage

```python
from hypergeometric_function import hypergeometric2f1

# 2F1(1,1;2;0.5) = 1.3862943611198906
value = hypergeometric2f1(1.0, 1.0, 2.0, 0.5)
print(value)
```

## Why this library exists

The Gaussian hypergeometric function appears in many areas of applied mathematics, from special functions to probability distributions. This library provides a small, dependency-free implementation that covers the most common real-valued use cases. The trade-off is that it does not support complex arguments or the branch cut z >= 1, which keeps the code simple and predictable.

## Edge cases

- `c` must not be a non-positive integer; otherwise a `ValueError` is raised.
- `z >= 1` is the branch cut and raises `ValueError`.
- For `z <= -1`, Euler's transformation is used automatically to speed convergence.
