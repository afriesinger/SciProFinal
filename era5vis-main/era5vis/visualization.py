"""
Module for the visualization of the non dimensional mountain height calculated from ERA5 data.

This module provides functions for:
    - calculating pressure depending on height
    - selection of data for a south to north vertical crosssection
    - applying a terrain mask to data of H
    - plotting a vertical crosssection of H including terrain

Requires: Netcdf file which includes H, U, V, terrain elevation

Author: Anna Buchhauser
Date: 2026-01-19
"""
import sys
import os.path

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch



# needed functions
def height_to_pressure(z):
    """
    Convert height (m) to pressure (hPa) using standard atmosphere 
    conditions and the barometric formula.

    Parameters
    ----------
    z: float
        terrain elevation in m

    Returns
    -------
    p: float
        pressure at terrain height in hPa
    """
    
    p0 = 1013.25  # standard pressure (hPa)
    T0 = 288.15   # standard temperature (K)
    g = 9.81   # gravitational constant
    R = 287.05   # specific gas constant (J kg^-1 K^-1)
    L = 0.0065    # temperature lapse rate (K/m)

    p = p0 * (1 - L * z / T0) ** (g / (R * L))   # barometric formula

    return p

def add_color_code(ds):
    """
    Add a variable to the dataset that indicates the flow situation 
    based on H with colors. Red is blocked flow, green is flow over 
    and blue is uneffected flow.

    Parameters
    ----------
    ds: dataset
        dataset containing all variables
    """
    
    ds['H_color'] = xr.where(ds['H'] >= 1, 'red', 'green') #  red = H >= 1 blocked, green = H <1 flow over
    ds['H_color'] = xr.where(np.isnan(ds['H']), 'white', ds['H_color'])  # filter NaNs
    ds['H_color'] = xr.where(ds['perpendicular_wind_speed'] == 0, 'blue', ds['H_color']) # unaffected flow

def select_data(ds, lon, start_lat=45.5, end_lat=47.8):
    """
    Select wanted area for a vertical cross section 
    from the dataset.

    Parameters
    ----------
    ds: dataset
        dataset containing all variables
    lon: float 
        longitude of the N-S oriented crosssection
    start_lat: float
        start latitude of the vertical crosssection
    end_lat: float
        end latitude of the vertical crosssection

    Returns
    -------
    ds_crosssec: dataset
        dataset containing all variables along the given longitude
    timestamp: Timestamp
        date and time of the downloaded data
    """ 
    timestamp = pd.to_datetime(ds.valid_time.values[0])
    
    ds_crosssec = ds.sel(longitude=lon, method='nearest').sel(valid_time=timestamp) 
    ds_crosssec = ds_crosssec.sel(latitude=slice(end_lat, start_lat))

    return ds_crosssec, timestamp

def apply_terrain_mask(ds_crosssec, terrain_p):
    """
    Apply a terrain mask to H, 
    so gridpoints beneath the terrain are not plotted.

    Parameters
    ----------
    ds_crosssec: dataset
        dataset containing all variables along the given longitude
    terrain_p: dataarray
        terrain elevation converted to pressure levels
    """
    
    p = ds_crosssec.pressure_level
    lat = ds_crosssec.latitude
    
    p2d, _ = xr.broadcast(p, lat)
    terrain_p2d, _ = xr.broadcast(terrain_p, p)
    
    ds_crosssec['H_masked'] = ds_crosssec['H'].where(p2d < terrain_p2d)
    ds_crosssec['H_color_masked'] = ds_crosssec['H_color'].where(p2d < terrain_p2d)


def vertical_crosssection(ds_crosssec, terrain_p, timestamp, lon, filepath=None):
    """
    Create a vertical crosssection of H including the terrain. 
    H is colorcoded for the different flow possibilites.

    Parameters
    ----------
    ds_crosssec: dataset
        dataset containing all variables along the given longitude
    terrain_p: dataarray
        terrain elevation in pressure levels
    timestamp: datetime
        date and time of the downloaded data
    filepath: string
        path specifying where the figure is saved
    """
    H_masked = ds_crosssec['H_masked']
    H_color_masked = ds_crosssec['H_color_masked']
    U = ds_crosssec['u']
    V = ds_crosssec['v']
    
    # create color map
    color_map = {
        'red': 'red',
        'green': 'green',
        'blue': 'blue',
        'white': 'white'
    }

    mapper = np.vectorize(lambda x: color_map.get(x, 'white')) 
    colors = xr.apply_ufunc(mapper, H_color_masked.fillna('white'), 
                            dask='allowed' )

    # meshgrid
    lat, p = np.meshgrid(H_masked.latitude, H_masked.pressure_level)

    # create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.invert_yaxis() # pressure decreases with height

    # scatter plot H
    ax.scatter(lat.ravel(), p.ravel(), c=colors.values.ravel(), s=20)

    # plot wind barbs
    stepsize = 8
    ax.barbs(lat[::, ::stepsize], 
             p[::, ::stepsize], 
             V.values[::, ::stepsize], 
             U.values[::, ::stepsize],
             length=5, barbcolor='darkgrey')

    # plot terrain
    ax.plot(terrain_p.latitude, terrain_p, color='black')
    ax.fill_between(terrain_p.latitude, terrain_p, y2=terrain_p.values.max(), color='lightgrey')

    # add lables, title and grid
    ax.set_xlabel('latitude[°]')
    ax.set_ylabel('pressure [hpa]')
    ax.set_title(f'Non-dimensional mountain height ({lon}° E, {timestamp})')
    ax.grid(alpha=0.4)

    # add legend
    legend_elements = [
    Patch(facecolor='red',   label='H ≥ 1: flow around'),
    Patch(facecolor='green', label='H < 1: flow over'),
    Patch(facecolor='blue',  label='zero perpendicular wind'),
    ]

    ax.legend(handles=legend_elements, loc='lower right')

    # save plot if requested
    if filepath is not None:
        fig.savefig(filepath)
        plt.close()

    return fig 

# main function
def create_plot(ds, lon):
    """Create vertical cross-section of H

    Parameters
    ----------
    ds: dataset
        dataset containing all variables
    lon: float
        longitude of the vertical cross-section
    """

    # add variable with color code for H
    add_color_code(ds)
    
    # select data for cross-section
    ds_crosssec, timestamp = select_data(ds, 11.2)
    terrain_p = height_to_pressure(ds_crosssec['terrain_elevation'])
    apply_terrain_mask(ds_crosssec, terrain_p)
    
    # plot vertical crosssection
    vertical_crosssection(ds_crosssec, terrain_p, timestamp, lon)

