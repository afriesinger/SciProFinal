''' Test functions for era5.py '''

import numpy as np
import xarray as xr

from era5vis import era5


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
