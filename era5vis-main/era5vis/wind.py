"""
Wind-terrain interaction module for ERA5 visualization.

This module provides functions for:
- Computing wind-terrain interaction metrics
- Enhancing ERA5 data with terrain and wind analysis

Author: Andreas Friesinger
Date: 2025-12-31
"""

import numpy as np
import xarray as xr


# Constants
G = 9.80665  # Standard gravity (m/s²)





def compute_wind_terrain_interaction(
    era5_data: xr.Dataset,
    terrain_ds: xr.Dataset
) -> xr.Dataset:
    """
    Compute wind-terrain interaction metrics.
    
    Adds variables quantifying how wind interacts with terrain slopes:
    - wind_terrain_angle: angle between wind and slope aspect
    - perpendicular_wind: wind component perpendicular to slope (positive = upslope)
    - parallel_wind: wind component parallel to slope
    
    Parameters
    ----------
    era5_data : xr.Dataset
        ERA5 dataset with u, v wind components
    terrain_ds : xr.Dataset
        Terrain aspect dataset
        
    Returns
    -------
    xr.Dataset
        ERA5 dataset with wind-terrain interaction variables
    """
    from scipy.interpolate import RegularGridInterpolator
    
    # Get terrain aspect interpolated to ERA5 grid
    terrain_aspect = terrain_ds['aspect'].values
    terrain_lats = terrain_ds['latitude'].values
    terrain_lons = terrain_ds['longitude'].values
    
    # Interpolate aspect to ERA5 grid
    interp_func = RegularGridInterpolator(
        (terrain_lats[::-1], terrain_lons),
        terrain_aspect[::-1, :],
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    
    era5_lats = era5_data.latitude.values
    era5_lons = era5_data.longitude.values
    lon_grid, lat_grid = np.meshgrid(era5_lons, era5_lats)
    points = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
    aspect_on_era5 = interp_func(points).reshape(lat_grid.shape)
    
    aspect_da = xr.DataArray(
        aspect_on_era5,
        dims=['latitude', 'longitude'],
        coords={'latitude': era5_data.latitude, 'longitude': era5_data.longitude}
    )
    
    # Get wind components
    if 'u' not in era5_data or 'v' not in era5_data:
        raise ValueError("ERA5 dataset must contain 'u' and 'v' wind components")
    
    u = era5_data['u']  # East-West component (positive = eastward)
    v = era5_data['v']  # North-South component (positive = northward)
    
    # Wind direction (where wind is coming FROM, meteorological convention)
    wind_dir = np.arctan2(-u, -v)  # radians, 0=N, π/2=E
    
    # Wind speed
    wind_speed = np.sqrt(u**2 + v**2)
    
    # Angle between wind direction and slope aspect
    # aspect = downhill direction, wind_dir = where wind comes from
    # For upslope flow: wind coming from opposite of downhill = wind_dir ≈ aspect + π
    angle_diff = wind_dir - aspect_da
    
    # Perpendicular wind component (positive = upslope flow, negative = downslope)
    # cos(angle_diff - π) = -cos(angle_diff) gives upslope as positive
    perpendicular = -wind_speed * np.cos(angle_diff)
    
    # Parallel wind component
    parallel = wind_speed * np.sin(angle_diff)
    
    # Add to dataset
    result = era5_data.copy()
    result['slope_aspect'] = aspect_da
    result['slope_aspect'].attrs = {'units': 'radians', 'long_name': 'Terrain slope aspect on ERA5 grid'}
    result['wind_speed'] = wind_speed
    result['wind_speed'].attrs = {'units': 'm/s', 'long_name': 'Wind speed'}
    result['perpendicular_wind'] = perpendicular
    result['perpendicular_wind'].attrs = {'units': 'm/s', 'long_name': 'Wind perpendicular to slope (positive=upslope)'}
    result['parallel_wind'] = parallel
    result['parallel_wind'].attrs = {'units': 'm/s', 'long_name': 'Wind parallel to slope'}
    
    return result


# """ def enhance_era5_with_terrain(
#     era5_data: xr.Dataset,
#     terrain_ds: xr.Dataset
# ) -> xr.Dataset:
#     """
#     Enhance ERA5 dataset with complete terrain information.
    
#     This is the main entry point for adding terrain data to ERA5.
#     Adds:
#     - terrain_elevation: SRTM elevation on ERA5 grid
#     - terrain: boolean mask for below-terrain points
#     - geopotential_height: converted from geopotential
#     - slope_aspect: terrain aspect on ERA5 grid
#     - perpendicular_wind: wind component perpendicular to slope
#     - parallel_wind: wind component parallel to slope
    
#     Parameters
#     ----------
#     era5_data : xr.Dataset
#         ERA5 dataset with z, u, v variables
#     terrain_ds : xr.Dataset
#         Terrain aspect dataset from terrain.load_terrain_aspect_dataset()
        
#     Returns
#     -------
#     xr.Dataset
#         Enhanced ERA5 dataset with terrain variables
#     """
#     # Add terrain intersection
#     result = compute_terrain_intersection(
#         era5_data=era5_data,
#         terrain_ds=terrain_ds
#     )
    
#     # Add wind-terrain interaction
#     result = compute_wind_terrain_interaction(
#         era5_data=result,
#         terrain_ds=terrain_ds
#     )
    
#     return result """
