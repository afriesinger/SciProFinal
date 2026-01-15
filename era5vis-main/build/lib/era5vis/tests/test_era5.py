''' Test functions for era5.py '''

import numpy as np
import xarray as xr
import pytest
from era5vis import era5
from era5vis.era5 import check_alpine_bounds


def test_horiz_cross_section(retrieve_param_level_from_ds):

    # extract the horizontal cross section
    param, level = retrieve_param_level_from_ds
    da = era5.horiz_cross_section(param, level, 0)

    # check that the correct parameter is extracted
    assert da.GRIB_shortName == param

    # check that the DataArray has the correct type and dimensions
    assert isinstance(da, xr.DataArray)
    assert da.dims == ('latitude', 'longitude')

    # check that pressure_level and valid_time are indeed scalars
    da.pressure_level.item()
    da.valid_time.item()

def test_check_alpine_bounds_valid():
    #Inside the Alps
    area = [47.5, 11.0, 47.0, 11.5]
    assert check_alpine_bounds(area) is None

def test_check_alpine_bounds_invalid():
    # Outside the Alps 
    area = [52.5, 13.0, 52.0, 13.5]
    with pytest.raises(ValueError, match="Area outside terrain bounds"):
        check_alpine_bounds(area)
