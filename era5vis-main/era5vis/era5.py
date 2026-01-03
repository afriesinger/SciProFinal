"""Functions interacting with the ERA5 dataset."""

import os
import sys
import cdsapi
import datetime
import xarray as xr
from era5vis import cfg

def check_data_availability(param, level=None, time=None):
    """
    Checks if the requested parameter, level, and time are present in the dataset.
    
    Returns
    -------
    bool: True if data is available, raises ValueError otherwise.
    """
    if not os.path.exists(cfg.datafile):
        raise FileNotFoundError(f"Data file {cfg.datafile} not found. Please download it first.")

    with xr.open_dataset(cfg.datafile) as ds:
        # 1. Check if the variable (param) exists
        if param not in ds.data_vars:
            raise ValueError(f"Variable '{param}' not found in dataset. Available: {list(ds.data_vars)}")

        # 2. Check if the pressure level exists (if provided)
        if level is not None:
            if level not in ds.pressure_level.values:
                raise ValueError(f"Level {level} hPa not found. Available: {ds.pressure_level.values}")

        # 3. Check if the time exists (if provided)
        if time is not None:
            try:
                if isinstance(time, str):
                    ds.sel(valid_time=time)
                elif isinstance(time, int):
                    ds.isel(valid_time=time)
            except (KeyError, IndexError):
                raise ValueError(f"Time '{time}' is out of range for this dataset.")

    return True

def check_alpine_bounds(area):
    """Checks if the area is within Lat (45 to 48) and Lon (6 to 16)
    
    Parameters
    ----------
    area : list of float
        A list of four coordinates in the order: [North, West, South, East].

    Raises
    ------
    ValueError
        If any coordinate is outside the bounds
    """
    
    n, w, s, e = area
    ALP_N, ALP_S, ALP_W, ALP_E = 48.0, 45.0, 6.0, 16.0

    if n > ALP_N or s < ALP_S or w < ALP_W or e > ALP_E:
        raise ValueError(
            f"Area outside terrain bounds. Bounds: Lat [{ALP_S}, {ALP_N}], Lon [{ALP_W}, {ALP_E}]"
        )

def validate_inputs(area, year, month, day, time):
    """Performs date, time, and coordinate validation.

    Parameters
    ----------
    area : list of float
        Coordinates [North, West, South, East]
    year : str or int
        Year of the data request (YYYY)
    month : str or int
        Month of the data request (MM)
    day : str or int
        Day of the data request (DD)
    time : str
        Time in HH:MM format

    Returns
    -------
    bool
        True if all validations pass.
    """
    try:
        datetime.date(int(year), int(month), int(day))
        datetime.datetime.strptime(time, "%H:%M")
    except ValueError as e:
        raise ValueError(f"Input Format Error: {e}")
        
    check_alpine_bounds(area)
    
    if not os.path.exists(os.path.expanduser("~/.cdsapirc")):
        raise FileNotFoundError("Authentication Error: ~/.cdsapirc file not found.")
    return True

def load_era5_data(output_filename, year, month, day, time, area):
    """Download data from CDS API."""
    if os.path.exists(output_filename):
        print(f"File '{output_filename}' exists. Skipping.")
        return

    validate_inputs(area, year, month, day, time)

    levels = [
        '450', '500', '550', '600', 
        '650', '700', '750', '775', '800', '825', 
        '850', '875', '900', '925', '950', '975', '1000'
    ]
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ['temperature', 'u_component_of_wind', 
                         'v_component_of_wind', 'geopotential'],
            'pressure_level': levels,
            'year': year, 
            'month': month,
            'day': day,
            'time': time,
            'area': area,
        },
        output_filename)

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
    check_data_availability(param, level=lvl, time=time)
    
    with xr.open_dataset(cfg.datafile).load() as ds:
        if isinstance(time, str):
            da = ds[param].sel(pressure_level=lvl).sel(valid_time=time)
        else:
            da = ds[param].sel(pressure_level=lvl).isel(valid_time=time)
    return da

if __name__ == "__main__": 
    if len(sys.argv) < 10:
        print("Usage: python era5.py <filename> <year> <month> <day> <time> <N> <W> <S> <E>")
    else:
        try:
            output_filename = sys.argv[1]
            year, month, day, time = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
            coords = [float(sys.argv[6]), float(sys.argv[7]), 
                      float(sys.argv[8]), float(sys.argv[9])]
            
            load_era5_data(output_filename , year, month, day, time, coords)
        except Exception as e:
            print(f"\nERROR: {e}")
