"""
Module for the visualization of the non dimensional mountain height calculated from ERA5 data.

This module provides functions for:

Requires: 

Author: Anna Buchhauser
Date: 2026-01-04
"""
import sys
import os.path

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# needed functions

# data selection
def select_terrain_data(lon, start_lat, end_lat):
    '''
    This function selects wanted area for a vertical cross section 
    from the terrain dataset. 
    '''
    # select terrain data
    # select terrain mask for given area
    oned_mask = ds_terrain['terrain_mask'].sel(longitude=lon, method='nearest') 
    crossec_mask = oned_mask.sel(latitude=slice(end_lat, start_lat))
    # select elevation for given area
    oned_elevation = ds_terrain['elevation'].sel(longitude=lon, method='nearest') 
    crossec_elevation = oned_elevation.sel(latitude=slice(end_lat, start_lat))
    latitudes = crossec_elevation.latitude
    # apply terrain mask
    masked_elevation = np.where(crossec_mask, crossec_elevation, np.nan)

    return masked_elevation, latitudes

def select_h_data(lon, start_lat, end_lat):
    
    # select h data
    # select h for given area
    #oned_h = ds_h['h_nd'].sel(longitude=lon, method='nearest') 
    oned_h = ds_h['H'].sel(longitude=lon, method='nearest')
    crossec_h = oned_h.sel(latitude=slice(end_lat, start_lat))
    h_latitudes = crossec_h.latitude
    #h_elevation = crossec_h.elevation
    print(crossec_h.data_vars)
    h_elevation = crossec_h.terrain_elevation

    return crossec_h, h_latitudes, h_elevation

# save produced figuer
def save_fig():
    '''Save the produced plot as png to a specified directory.'''

    # check wether directory was specified
    if '--output-dir' in sys.argv:
        index = sys.argv.index('--output-dir')
        output_dir = sys.argv[index + 1]
            
        # check if directory exists
        if os.path.isdir(output_dir):
            # save in specified directory
            path = os.path.join(output_dir, f'h_nd_vertical_crossection.png')
            plt.savefig(path)
        else:
            raise NameError('This directory does not exist.')
            sys.exit()
                
    else:            
        # save in working directory
        plt.savefig(f'h_nd_vertical_crosssection.png')

        
# create plots
def create_vertical_crosssec(elevation, latitudes, h, h_elevation, h_latitudes):
    '''
    This function plots a vertical cross-section of the selected area, 
    displaying the terrain as well as the non-dimesnional mountain height.
    '''

    # convert h to numpy array
    h_nd = h.values
    
    # meshgrid for plotting h
    LAT, Z = np.meshgrid(h_latitudes, h_elevation)
    mask = np.isfinite(h_nd) # mask to only plot h where no terrain

    # categorize h
    blocked = h_nd > 1
    flow_over = h_nd <= 1

    # create plot
    fig, ax = plt.subplots(figsize=(8,5))
    # plot terrain
    ax.plot(latitudes, elevation, color='black')
    ax.fill_between(latitudes, elevation, color='lightgrey')
    # plot h
    ax.scatter(LAT[mask & blocked], Z[mask & blocked], color='red', label='blocked/flow around')
    ax.scatter(LAT[mask & flow_over], Z[mask & flow_over], color='green', label='flow over')
    
    # titles and labels
    ax.set_title('Non-dimensional mountain height')
    ax.set_ylabel('elevation [m]')
    ax.set_xlabel('latitude [m]')
    ax.legend(loc='lower right')

    # save plot
    save_fig()
    plt.show()




# main block
if __name__ == "__main__":
    # open terrain data
    ds_terrain = xr.open_dataset('/media/afriesinger/Volume/Projekte/Gleitschirmfliegen/Studium/Programming/SciProFinal/era5vis-main/era5vis/terrain_cache/terrain_aspect_1km.nc')
    # open h_nd data
    #ds_h = xr.open_dataset('hnd_mockup_data.nc') 
    ds_h = xr.open_dataset('/media/afriesinger/Volume/Projekte/Gleitschirmfliegen/Studium/Programming/SciProFinal/era5vis-main/era5vis/data/era5_test_with_NH.nc') 


    # get lon,lat from arg
    lon = sys.argv[1]
    start_lat = sys.argv[2]
    end_lat = sys.argv[3]

    # select data
    elevation, latitudes = select_terrain_data(lon, start_lat, end_lat)
    h, h_latitudes, h_elevation = select_h_data(lon, start_lat, end_lat)

    # plot vertical crosssection
    create_vertical_crosssec(elevation, latitudes, h, h_elevation, h_latitudes)
