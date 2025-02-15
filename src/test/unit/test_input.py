import unittest
from numpy import *

class TestPyInputFiles(unittest.TestCase):

    '''
    Testing:
    All .py input files from ../test/input/ can compute and run as python scripts
    '''

    def test_add(self):

        # ARRANGE
        def f(x): return x + 2.3 + x
        def g(x): return (x + 2.1, 2 + x)

        # ACT
        f_expect = (6.3)
        f_result = f(2.0)

        g_expect = (4.1, 4.0)
        g_result = g(2.0)

        # ASSERT
        self.assertEqual(f_result, f_expect)
        self.assertEqual(g_result, g_expect)


    def test_cos(self):

        # ARRANGE
        def f(x): return cos(x)

        # ACT
        f_result = f(4.2)
        f_expect = -0.4902608213
        
        # ASSERT
        self.assertAlmostEqual(f_result, f_expect, 6)


    def test_exp(self):

        # ARRANGE
        def f(x): return exp(x)

        # ACT
        f_result = f(3.0)
        f_expect = 20.08553692
        
        # ASSERT
        self.assertAlmostEqual(f_result, f_expect, 6)


    def test_ln(self):

        # ARRANGE
        def f1(x): return log(x)
        def f2(x): return log(x * log(x))
        def f3(x): return log(x[0] * sin(x[1]))

        # ACT
        f1_result = f1(3.0)
        f1_expect = 1.098612289
        f2_result = f2(3.0)
        f2_expect = 1.192660117
        f3_result = f3([3.0, 2.0])
        f3_expect = 1.003529252

        # ASSERT
        self.assertAlmostEqual(f1_result, f1_expect, 6)
        self.assertAlmostEqual(f2_result, f2_expect, 6)
        self.assertAlmostEqual(f3_result, f3_expect, 6)


    def test_mul(self):

        # ARRANGE
        def f(x): return 3.4 * x
        def g(y): return 2 * y * f(y)

        # ACT
        f_result = f(2.1)
        f_expect = 7.14
        g_result = g(2.1)
        g_expect = 29.988

        # ASSERT
        self.assertAlmostEqual(f_result, f_expect, 6)
        self.assertAlmostEqual(g_result, g_expect, 6)


    def test_neg(self):

        # ARRANGE
        def f(x): return -x

        # ACT
        f_result = f(2.15)
        f_expect = -2.15

        # ASSERT
        self.assertAlmostEqual(f_result, f_expect, 6)


    def test_pow(self):

        # ARRANGE
        def f(x): return pow(x, 3.0)
        def g(x): return pow(f(x), 2.5)

        # ACT
        f_result = f(4.4)
        f_expect = 85.184
        g_result = g(1.32)
        g_expect = 8.022403173

        # ASSERT
        self.assertAlmostEqual(f_result, f_expect, 6)
        self.assertAlmostEqual(g_result, g_expect, 6)

    def test_proj(self):

        # ARRANGE
        def f(x): return x[0] + x[1]
        def g(x): return (x[0] + x[1], x[2])

        # ACT
        f_result = f([1.4, 3.1])
        f_expect = 4.5
        g_result = g([1.2, 2.1, 3.0])
        g_expect = (3.3, 3.0)

        # ASSERT
        self.assertAlmostEqual(f_result, f_expect, 6)
        self.assertAlmostEqual(g_result, g_expect, 6)

    def test_simple1(self):

        # ARRANGE
        def f(x): return log(sin(x))

        # ACT
        f_result = f(2.4)
        f_expect = -0.3923566300

        # ASSERT
        self.assertAlmostEqual(f_result, f_expect, 6)

    def test_simple1(self):

        # ARRANGE
        def f(x): return log(sin(x))

        # ACT
        f_result = f(2.4)
        f_expect = -0.3923566300

        # ASSERT
        self.assertAlmostEqual(f_result, f_expect, 6)

    def test_simple2(self):

        # ARRANGE
        def f(x): x3 = x[0]; return x3 + x[1]

        # ACT
        f_result = f([1.2, 5.9])
        f_expect = 7.1

        # ASSERT
        self.assertAlmostEqual(f_result, f_expect, 6)

    def test_simple3(self):

        # ARRANGE
        def f(x): x1 = 2; x2 = x; return x1 + x1 * x2 - sin(x2)

        # ACT
        f_result = f(1.2)
        f_expect = 3.467960914

        # ASSERT
        self.assertAlmostEqual(f_result, f_expect, 6)

    def test_sin(self):

        # ARRANGE
        def f(x): return sin(x)
        def g(x): return 0 - sin(x[0])

        # ACT
        f_result = f(2.7)
        f_expect = 0.4273798802
        g_result = g([2.2])
        g_expect = -0.8084964038

        # ASSERT
        self.assertAlmostEqual(f_result, f_expect, 6)
        self.assertAlmostEqual(g_result, g_expect, 6)

    def test_sub(self):

        # ARRANGE
        def f(x): return x - 2.3 - 40
        def g(x): return (x - 2.1, 2 - x)

        # ACT
        f_result = f(1.7)
        f_expect = -40.6
        g_result = g(2.3)
        g_expect = (0.2, -0.3)

        # ASSERT
        self.assertAlmostEqual(f_result, f_expect, 6)
        for i in range(len(g_expect)):
            self.assertAlmostEqual(g_result[i], g_expect[i], 6)

if __name__ == '__main__':
    unittest.main()