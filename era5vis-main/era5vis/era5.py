"""Functions interacting with the ERA5 dataset."""

"""
This module provides functions for:
- Validating geographic bounds for the Alpine region.
- Authenticating and downloading pressure-level data from the CDS API.
- Checking data availability within local NetCDF datasets.
- Extracting horizontal cross-sections of atmospheric parameters.

Author: Lawrence Ansu Mensah, Andreas Friesinger where stated
Date: 2026-01-14
"""

import os
import sys
import cdsapi
import datetime
import numpy as np
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
        if param not in ds.data_vars:
            raise ValueError(f"Variable '{param}' not found in dataset. Available: {list(ds.data_vars)}")

        if level is not None:
            if level not in ds.pressure_level.values:
                raise ValueError(f"Level {level} hPa not found. Available: {ds.pressure_level.values}")

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

def validate_inputs(area):
   """
    Performs coordinate and credential validation for ERA5 downloads.

    This function checks if the requested geographic area is within the 
    predefined Alpine bounds and verifies the existence of the CDS API 
    configuration file required for authentication.

    Parameters
    ----------
    area : list of float
        A list of four coordinates in the order: [North, West, South, East].

    Returns
    -------
    bool
        True if the area is within bounds and credentials exist.

    Raises
    ------
    ValueError
        If the coordinates are outside the allowed Alpine range.
    FileNotFoundError
        If the ~/.cdsapirc file is missing, providing instructions for setup.
        
    """

   #Coordinate check    
   check_alpine_bounds(area)
    
    #Credential check
   if not os.path.exists(os.path.expanduser("~/.cdsapirc")):
        raise FileNotFoundError(
        "\n--- CDS API Authentication Missing ---\n"
        "To download ERA5 data, you need a credentials file.\n"
        "1. Create a file named '.cdsapirc' in your home directory.\n"
        "2. Add the following lines (get your key from https://cds.climate.copernicus.eu):\n\n"
        "url: https://cds.climate.copernicus.eu/api/v2\n"
        "key: YOUR_UID:YOUR_API_KEY\n"
        )
   return True

def load_era5_data(output_filename, start_date, area, end_date=None):
    """
    Download ERA5 pressure level data from the CDS API.

    The function requests temperature, wind components (u, v), and geopotential
    data for a specified Alpine area and time.

    Parameters
    ----------
    output_filename : str
        The path and name of the NetCDF file to be created.
    start_date : datetime.datetime
        The starting date and hour for the data request.
    area : list of float
        Coordinates for the download area [North, West, South, East].
    end_date : datetime.datetime, optional
        The ending date and hour for a time-series request. If None, only
        the `start_date` is downloaded.

    Returns
    -------
    xarray.Dataset
       The downloaded ERA5 data loaded into an xarray object."""
        
    if os.path.exists(output_filename):
        print(f"File '{output_filename}' exists. Skipping.")
        return xr.open_dataset(output_filename)

    validate_inputs(area)
    
    years = [str(start_date.year)]
    months = [f"{start_date.month:02d}"]
    days = [f"{start_date.day:02d}"]
    times = [f"{start_date.hour:02d}:00"]

    #if end_date is provided and different from start_date
    if end_date and end_date != start_date:
        if str(end_date.year) not in years: 
            years.append(str(end_date.year))
        if f"{end_date.month:02d}" not in months: 
            months.append(f"{end_date.month:02d}")
        if f"{end_date.day:02d}" not in days: 
            days.append(f"{end_date.day:02d}")
        if f"{end_date.hour:02d}:00" not in times: 
            times.append(f"{end_date.hour:02d}:00")
    

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
            'year': years, 
            'month': months,
            'day': days,
            'time': times,
            'area': area,
        },
        output_filename)

    return xr.open_dataset(output_filename)

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
    if len(sys.argv) < 7:
        print("Usage: python era5.py <filename> <YYYY-MM-DD-HH> <N> <W> <S> <E>" )
    else:
        try:
            output_filename = sys.argv[1]
            date = datetime.datetime.strptime(sys.argv[2], "%Y-%m-%d-%H")
            coords = [float(sys.argv[3]), float(sys.argv[4]), 
                      float(sys.argv[5]), float(sys.argv[6])]
            
            load_era5_data(output_filename , date, coords)
        except Exception as e:
            print(f"\nERROR: {e}")

def compress_era(dataset):
   
    """
    Compress ERA5 dataset using numpy dtypes.

    Parameters
    ----------
    dataset: xarray.Dataset
        ERA5 dataset to be compressed

    Returns
    -------
    dataset: xarray.Dataset
        Compressed ERA5 dataset
    
    Author: Andreas Friesinger
    """
    dtype_map = {
        # coordinates / dimensions
        "latitude": np.float32,
        "longitude": np.float32,
        "pressure_level": np.int16,

        # data variables
        "t": np.float16,
        "u": np.float16,
        "v": np.float16,
        "z": np.uint16,
        "gph": np.uint16, 
        #'elevation': np.int16,
        #'aspect_deg': np.int16,
        #'slope': np.int16
    }

    for var, dtype in dtype_map.items():
        if var in dataset:
            dataset[var] = dataset[var].astype(dtype)
    return dataset
