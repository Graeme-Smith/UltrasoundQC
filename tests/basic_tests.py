"""Contains tests for array_spiker.py and associate scripts"""

import unittest2 as unittest
from helpers import *


class CurvilinearTest(unittest.TestCase):
    """Tests for curvilinearQC.py."""

    def test_pixel_intensities(self):
        """Test that pixel_intensities() returns correct output"""
        test_array = np.array([[0, 0, 0], [1, 1, 1], [1, 2, 3]])
        col_mean, row_mean = pixel_intensities(test_array)
        self.assertEquals(cols,
                          [0.66666667, 1., 1.33333333],
                          msg="TODO:")
        self.assertEquals(row_mean,
                          [0., 1., 2.],
                          msg="TODO:")


if __name__ == '__main__':
    unittest.main()
