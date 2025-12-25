''' fixtures used in tests '''

from datetime import datetime

import pytest
import xarray as xr

from era5vis import cfg

@pytest.fixture
def retrieve_param_level_from_ds():

    # retrieve variable name and level from the dataset to make sure 
    # that we don't call the function with bad arguments
    with xr.open_dataset(cfg.datafile) as ds:
        param = [variable for variable in ds.variables if
                 ('pressure_level' in ds[variable].dims) and ('longitude' in ds[variable].dims)][0]
        level = ds.pressure_level.to_numpy()[0].astype(int)

    return param, level


@pytest.fixture
def retrieve_param_level_time_from_ds():

    # retrieve variable name, level, and time from the dataset to make sure 
    # that we don't call the function with bad arguments
    with xr.open_dataset(cfg.datafile) as ds:
        param = [variable for variable in ds.variables if
                 ('pressure_level' in ds[variable].dims) and ('longitude' in ds[variable].dims)][0]
        level = ds.pressure_level.to_numpy()[0].astype(int)
        time = ds.valid_time.to_numpy()[0].astype(
               'datetime64[ms]').astype(datetime).strftime('%Y%m%d%H%M')

    return param, level, time
