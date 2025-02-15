import os
import sys
import unittest
from numpy import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import input.add as add
import input.ln as ln

""" class TestPyInputFiles(unittest.TestCase):

    '''
    Testing:
    All .py input files from ../test/input/ can compute and run as python scripts
    '''

    def test_add(self):
        
        self.assertEqual(
            add.f(2.0),
            (4.1, 4.0)
        )

        self.assertEqual(
            add.g(2.0),
            (0.1, 0)
        ) """

if __name__ == '__main__':
    unittest.main()