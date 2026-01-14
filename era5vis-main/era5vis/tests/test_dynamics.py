import numpy as np
import pytest
from era5vis.dynamics import geopotential_height 

def test_geopotential_height_single_value():
    z = np.array(9.81)
    result = geopotential_height(z)
    assert result == 1
    

def test_geopotential_height_array():
   # Test with a numpy array
     z = np.array([0, 9810, 19620])
    expected = np.array([0, 1000, 2000], dtype=np.uint16)
    result = geopotential_height(z)
    np.testing.assert_array_equal(result, expected)
