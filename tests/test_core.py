"""Tests for hypergeometric_function.core."""

import math
import unittest

from hypergeometric_function import hypergeometric2f1


class TestHypergeometric2f1(unittest.TestCase):
    def test_polynomial_case_zero_a(self):
        # 2F1(0, b; c; z) = 1
        self.assertAlmostEqual(hypergeometric2f1(0, 2.5, 1.5, 0.3), 1.0)

    def test_polynomial_case_negative_integer_a(self):
        # 2F1(-n, b; c; z) is a polynomial of degree n.
        # 2F1(-2, 1; 1; z) = 1 - 2z + z^2
        self.assertAlmostEqual(
            hypergeometric2f1(-2, 1, 1, 0.4),
            1 - 2 * 0.4 + 0.4**2,
            places=12,
        )

    def test_polynomial_case_negative_integer_b(self):
        # Symmetry: 2F1(a,b;c;z) = 2F1(b,a;c;z)
        self.assertAlmostEqual(
            hypergeometric2f1(1, -2, 1, 0.4),
            hypergeometric2f1(-2, 1, 1, 0.4),
            places=12,
        )

    def test_known_value_z_zero(self):
        # 2F1(a,b;c;0) = 1
        self.assertAlmostEqual(hypergeometric2f1(0.5, 1.5, 2.0, 0.0), 1.0)

    def test_known_value_arctanh(self):
        # 2F1(1/2, 1; 3/2; z^2) * z = arctanh(z)
        z = 0.5
        expected = math.atanh(z)
        actual = z * hypergeometric2f1(0.5, 1.0, 1.5, z * z)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_known_value_log(self):
        # 2F1(1, 1; 2; z) = -log(1-z)/z
        z = 0.3
        expected = -math.log(1 - z) / z
        actual = hypergeometric2f1(1, 1, 2, z)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_known_value_arcsin(self):
        # 2F1(1/2, 1/2; 3/2; z^2) * z = arcsin(z)
        z = 0.7
        expected = math.asin(z)
        actual = z * hypergeometric2f1(0.5, 0.5, 1.5, z * z)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_euler_transformation_for_large_negative_z(self):
        # Direct series for z <= -1 is slow; Euler transformation is used.
        # Compare against a known value: 2F1(1, 2; 3; -2) = ?
        # Using the formula 2F1(1,2;3;x) = 2*(1-x)^(-2) - 2/(1-x)
        # Not simple; instead compare with direct high-precision summation
        # using a small tolerance but the transformation should give a
        # stable result. We can use the identity with log:
        # 2F1(1,1;2;z) = -log(1-z)/z, valid for z < 1
        z = -3.0
        expected = -math.log(1 - z) / z
        actual = hypergeometric2f1(1.0, 1.0, 2.0, z)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_negative_z_less_than_minus_one_converges(self):
        # Ensure that the transformed series converges and returns a value
        # for a non-trivial parameter set.
        result = hypergeometric2f1(0.5, 0.5, 1.5, -2.0)
        self.assertTrue(math.isfinite(result))

    def test_raises_on_branch_cut(self):
        with self.assertRaises(ValueError):
            hypergeometric2f1(0.5, 0.5, 1.5, 1.0)
        with self.assertRaises(ValueError):
            hypergeometric2f1(0.5, 0.5, 1.5, 2.0)

    def test_raises_on_nonpositive_integer_c(self):
        with self.assertRaises(ValueError):
            hypergeometric2f1(0.5, 0.5, 0, 0.5)
        with self.assertRaises(ValueError):
            hypergeometric2f1(0.5, 0.5, -1, 0.5)

    def test_negative_integer_c_not_allowed(self):
        with self.assertRaises(ValueError):
            hypergeometric2f1(1.0, 1.0, -2.0, 0.5)

    def test_symmetry_a_b(self):
        a, b, c, z = 0.3, 1.7, 2.5, 0.4
        self.assertAlmostEqual(
            hypergeometric2f1(a, b, c, z),
            hypergeometric2f1(b, a, c, z),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
