import numpy as np
import pytest
from era5vis.dynamics import geopotential_height 

def test_geopotential_height_conversion():
    # 9.81 / 9.81 = 1
    assert geopotential_height(9.81) == 1
    # 0 should stay 0
    assert geopotential_height(0) == 0

def test_geopotential_height_rounding():
   # Testing how the .astype(np.uint16) handles decimals (it rounds down)
    assert geopotential_height(15.0) == 1
