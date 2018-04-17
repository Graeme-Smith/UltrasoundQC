"""Contains tests for curvilinearQC.py and associate scripts"""

import unittest2 as unittest
from helpers import *


class CurvilinearTest(unittest.TestCase):
    """Tests for curvilinearQC.py."""

    def test_pixel_intensities(self):
        """Test that pixel_intensities() returns correct output"""
        test_array = np.array([[0, 0, 0], [1, 1, 1], [1, 2, 3]])
        col_mean, row_mean = pixel_intensities(test_array)
        message = "pixel_intensities() returning incorrect values"
        self.assertAlmostEqual(col_mean[0], 0.66666667, msg=message)
        self.assertAlmostEqual(col_mean[1], 1., msg=message)
        self.assertAlmostEqual(col_mean[2], 1.33333333, msg=message)
        self.assertAlmostEqual(row_mean[0], 0., msg=message)
        self.assertAlmostEqual(row_mean[1], 1., msg=message)
        self.assertAlmostEqual(row_mean[2], 2., msg=message)


if __name__ == '__main__':
    unittest.main()
