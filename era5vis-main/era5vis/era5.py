"""Functions interacting with the ERA5 dataset. """

import xarray as xr

from era5vis import cfg


def check_data_availability(param, level=None, time=None, time_ind=None):
    pass


def horiz_cross_section(param, lvl, time):
    """Extract a horizontal cross section from the ERA5 data.
    
    Parameters
    ----------
    param: str
        ERA5 variable
    lvl : integer
        model pressure level (hPa)
    time : str or integer
        time string or time index

    Returns
    -------
    da: xarray.DataArray
        2D DataArray of param
    """

    # use either sel or sel depending on the type of time (index or date format)
    with xr.open_dataset(cfg.datafile).load() as ds:
        if isinstance(time, str):
            da = ds[param].sel(pressure_level=lvl).sel(valid_time=time)
        elif isinstance(time, int):
            da = ds[param].sel(pressure_level=lvl).isel(valid_time=time)
        else:
            raise TypeError('time must be a time format string or integer')

    return da
