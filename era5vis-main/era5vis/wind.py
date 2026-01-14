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


def _get_downwind_points(era_lat, era_lon, wind_dir, range_km=10):
    """
    Generate points in a line downwind from ERA gridpoint.
    
    Parameters
    ----------
    era_lat : float
        Latitude of ERA gridpoint (degrees)
    era_lon : float
        Longitude of ERA gridpoint (degrees)
    wind_dir : float
        Wind direction in degrees
    range_km : float
        Search range in kilometers
        
    Returns
    -------
    downwind_lats : np.ndarray
        1D array of latitudes
    downwind_lons : np.ndarray
        1D array of longitudes
    """
    # Generate points along the downwind direction
    downwind_dir = int(wind_dir) + 180  # Opposite direction
    
    n_points = max(2, int(range_km * 2))  # Number of sample points (at least 2)
    max_dist_m = range_km * 1000 
    distances = np.linspace(0, max_dist_m, n_points)
    
    # Use simple lat/lon offset (good approx. for small distances)
    # 1 degree latitude ≈ 111 km
    # 1 degree longitude ≈ 111 * cos(lat) km
    lat_offset = (distances / 111000) * np.sin(np.radians(downwind_dir))
    lon_offset = (distances / (111000 * np.cos(np.radians(era_lat)))) * np.cos(np.radians(downwind_dir))
    
    downwind_lats = era_lat + lat_offset
    downwind_lons = era_lon + lon_offset
    
    return downwind_lats, downwind_lons


def _find_highest_terrain_downwind(downwind_lats, downwind_lons, interp_elev, interp_aspect, pressure_level_height):
    """
    Find first terrain point downwind that's higher than pressure level.
    
    Parameters
    ----------
    downwind_lats : np.ndarray
        Latitudes along downwind direction
    downwind_lons : np.ndarray
        Longitudes along downwind direction
    interp_elev : RegularGridInterpolator
        Terrain elevation interpolator
    interp_aspect : RegularGridInterpolator
        Terrain aspect interpolator
    pressure_level_height : float
        Height at pressure level (meters)
        
    Returns
    -------
    terrain_height : float or np.nan
        Height of first terrain above pressure level
    terrain_aspect : float or np.nan
        Aspect of the first terrain point above pressure level
    """
    
    points = np.stack([downwind_lats, downwind_lons], axis=-1)
    elevations = interp_elev(points)
    aspects = interp_aspect(points)
    
    # Find first point where terrain is higher than pressure level
    above_level = elevations > pressure_level_height
    
    if np.any(above_level):
        first_hit_idx = np.where(above_level)[0][0]
        terrain_height = float(elevations[first_hit_idx])
        terrain_aspect = float(aspects[first_hit_idx])
    else:
        terrain_height = np.nan
        terrain_aspect = np.nan
    
    return terrain_height, terrain_aspect

def calc_wind(u, v):
    """
    Compute wind direction in degrees from u and v components.
    
    Parameters:
    u : array-like
        U-component of wind (east-west).
    v : array-like
        V-component of wind (north-south).
        
    Returns:
    wind_dir : array-like
        Wind direction in degrees (uint16, 0-360).
    wind_speed : array-like
        Wind speed in m/s.
    """

    wind_dir = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
    wind_dir = wind_dir.astype(np.uint16)
    wind_speed = np.round(np.sqrt(u**2 + v**2), 2)  
    return wind_dir, wind_speed


def perpendicular_wind_component(wind_dir, wind_speed, slope_aspect):
    """
    Compute the perpendicular component of wind relative to a slope aspect.
    
    The perpendicular component is the projection of wind speed onto the direction
    perpendicular to the slope (i.e., wind blowing INTO the slope).
    
    Parameters:
    wind_dir : float
        Wind direction in degrees (0-360).
    wind_speed : float
        Wind speed in m/s.
    slope_aspect : float
        Slope aspect direction in degrees (0-360, direction the slope faces).
        
    Returns:
    perpendicular_wind_speed : float
        Component of wind speed perpendicular to slope. Positive = wind into slope.
    """
    if wind_dir < 0 or wind_dir > 360:
        raise ValueError("Wind direction must be between 0 and 360 degrees.")
    if slope_aspect < 0 or slope_aspect > 360:
        raise ValueError("Slope aspect must be between 0 and 360 degrees.")
    
    angle_diff = (wind_dir - slope_aspect) % 360
    
    # Normalize
    if angle_diff > 180:
        angle_diff -= 360
    
    # Check if wind is within ±90° of slope aspect (wind blowing toward the slope)
    if abs(angle_diff) <= 90:
        perpendicular_wind_speed = wind_speed * np.cos(np.radians(angle_diff))
    else:
        # Wind is blowing away from the slope
        perpendicular_wind_speed = 0

    # final check
    if perpendicular_wind_speed < 0 or perpendicular_wind_speed > wind_speed:
        raise ValueError("Computed perpendicular wind speed is out of valid range.")
    
    return np.round(perpendicular_wind_speed,2)


def compute_downwind_terrain_height(
    era5_data: xr.Dataset,
    terrain_ds: xr.Dataset,
    range_km: float = 1.0
) -> xr.DataArray:
    """
    Compute downwind terrain height for each ERA5 gridpoint and pressure level.
    
    For each ERA5 gridpoint at each pressure level:
    1. Extract wind direction at that gridpoint
    2. Search downwind within the specified range
    3. Find the first terrain point higher than the pressure level
    4. Return the height of that terrain point
    
    Parameters
    ----------
    era5_data : xr.Dataset
        ERA5 dataset with u, v wind components and z (geopotential)
    terrain_ds : xr.Dataset
        Terrain aspect dataset with elevation, aspect, latitude, longitude
    range_km : float
        Search range downwind in kilometers (default: 1.0 km)
        
    Returns
    -------
    xr.DataArray
        Downwind terrain height with same shape as wind field (time, pressure_level, latitude, longitude)
    
    """

    if 'u' not in era5_data or 'v' not in era5_data:
        raise ValueError("ERA5 dataset must contain 'u' and 'v' wind components")
    if 'z' not in era5_data:
        raise ValueError("ERA5 dataset must contain 'z' (geopotential)")
    
    wind_dir, wind_speed = calc_wind(era5_data['u'], era5_data['v'])
    height_at_level = era5_data['gph']
    
    downwind_terrain_heights = xr.full_like(wind_speed, np.nan, dtype=float)
    lats = era5_data.latitude.values
    lons = era5_data.longitude.values
    
    # VECTORIZED: Generate downwind points for all spatial locations at once
    n_points_downwind = max(2, int(range_km * 2))
    max_dist_m = range_km * 1000
    distances = np.linspace(0, max_dist_m, n_points_downwind)
    
    nt, np_level, nlat_era5, nlon_era5 = wind_dir.shape
    
    lat_grid_2d, lon_grid_2d = np.meshgrid(lats, lons, indexing='ij')  # shape: (nlat_era5, nlon_era5)
    
    # Get terrain data for direct indexing (no interpolation)
    terrain_elevation = terrain_ds['elevation'].values  
    terrain_aspect_vals = terrain_ds['aspect_deg'].values  
    terrain_lats_arr = terrain_ds['latitude'].values 
    terrain_lons_arr = terrain_ds['longitude'].values  
    
    # Initialize arrays 
    downwind_terrain_elev_all = np.full(
        (n_points_downwind, nlat_era5, nlon_era5), np.nan
    )
    downwind_terrain_aspect_all = np.full(
        (n_points_downwind, nlat_era5, nlon_era5), np.nan
    )
    
    # For each downwind distance, compute downwind positions and get terrain values
    wind_dir_avg = np.nanmean(wind_dir.values, axis=(0, 1))  # shape: (nlat_era5, nlon_era5)
    downwind_dir_avg = wind_dir_avg + 180  # Opposite direction
    
    for pt_idx in range(n_points_downwind):
        dist = distances[pt_idx]
        
        # Compute downwind offsets for ALL points at once (vectorized)
        lat_offset = (dist / 111000) * np.sin(np.radians(downwind_dir_avg))
        lon_offset = (dist / (111000 * np.cos(np.radians(lat_grid_2d)))) * np.cos(np.radians(downwind_dir_avg))
        
        # Downwind positions for all points
        downwind_lat_all = lat_grid_2d + lat_offset
        downwind_lon_all = lon_grid_2d + lon_offset
        
        # Find nearest terrain grid indices for all downwind positions
        # NOTE: terrain_lats_arr is DESCENDING, terrain_lons_arr is ASCENDING
        lat_indices_down = np.searchsorted(-terrain_lats_arr, -downwind_lat_all.ravel(), side='left')
        lon_indices_down = np.searchsorted(terrain_lons_arr, downwind_lon_all.ravel(), side='left')
        
        # Clip to valid range
        lat_indices_down = np.clip(lat_indices_down, 0, len(terrain_lats_arr) - 1)
        lon_indices_down = np.clip(lon_indices_down, 0, len(terrain_lons_arr) - 1)
        
        # Extract terrain values at downwind locations
        downwind_terrain_elev_all[pt_idx] = terrain_elevation[lat_indices_down, lon_indices_down].reshape(nlat_era5, nlon_era5)
        downwind_terrain_aspect_all[pt_idx] = terrain_aspect_vals[lat_indices_down, lon_indices_down].reshape(nlat_era5, nlon_era5)
    
    # Find first terrain point above pressure level (vectorized across spatial dimensions)
    height_threshold = height_at_level.values  # shape: (nt, np_level, nlat_era5, nlon_era5)
    
    # For each spatial location, find the first downwind point with terrain > pressure height
    for lat_idx in range(nlat_era5):
        for lon_idx in range(nlon_era5):
            # For this spatial location, check all time and pressure levels
            for t_idx in range(nt):
                for p_idx in range(np_level):                    
                    # Skip if no valid wind data
                    if np.isnan(wind_dir.values[t_idx, p_idx, lat_idx, lon_idx]) or \
                       np.isnan(wind_speed.values[t_idx, p_idx, lat_idx, lon_idx]):
                        continue
                    
                    threshold = height_threshold[t_idx, p_idx, lat_idx, lon_idx]
                    
                    # Search through downwind points
                    for pt_idx in range(n_points_downwind):
                        terrain_height = downwind_terrain_elev_all[pt_idx, lat_idx, lon_idx]
                        
                        # Check if this point is above threshold
                        if not np.isnan(terrain_height) and terrain_height > threshold:
                            downwind_terrain_heights.values[t_idx, p_idx, lat_idx, lon_idx] = terrain_height
                            break  
    
    return downwind_terrain_heights


def compute_wind_terrain_interaction(
    era5_data: xr.Dataset,
    terrain_ds: xr.Dataset,
    range_km: float = 1.0
) -> xr.Dataset:
    """
    Compute complete wind-terrain interaction metrics (vectorized).
    
    For each ERA5 gridpoint at each pressure level:
    1. Compute wind speed and direction
    2. Find downwind terrain height (first terrain above pressure level)
    3. Calculate perpendicular wind component relative to terrain aspect
    
    Parameters
    ----------
    era5_data : xr.Dataset
        ERA5 dataset with u, v wind components and z (geopotential)
    terrain_ds : xr.Dataset
        Terrain dataset with elevation, aspect_deg (latitude, longitude)
    range_km : float
        Search range downwind in kilometers (default: 1.0 km)
        
    Returns
    -------
    xr.Dataset
        Input ERA5 dataset with added variables:
        - wind_speed: Wind speed at each point (m/s)
        - wind_direction: Wind direction where wind comes FROM (degrees)
        - downwind_terrain_height: Height of first terrain above pressure level (m)
        - perpendicular_wind_speed: Wind component perpendicular to terrain aspect (m/s)
    """
    # Input validation
    if 'u' not in era5_data or 'v' not in era5_data:
        raise ValueError("ERA5 dataset must contain 'u' and 'v' wind components")
    if 'z' not in era5_data:
        raise ValueError("ERA5 dataset must contain 'z' (geopotential)")
    
    wind_dir, wind_speed = calc_wind(era5_data['u'], era5_data['v'])
    
    downwind_terrain_heights = compute_downwind_terrain_height(
        era5_data, terrain_ds, range_km=range_km
    )
    
    terrain_lats_arr = terrain_ds['latitude'].values
    terrain_lons_arr = terrain_ds['longitude'].values
    terrain_aspect_vals = terrain_ds['aspect_deg'].values
    
    era5_lats = era5_data.latitude.values
    era5_lons = era5_data.longitude.values
    nt, np_level, nlat_era5, nlon_era5 = wind_dir.shape
    
    # Create mapping from ERA5 to terrain grid
    if nlat_era5 == len(terrain_lats_arr) and nlon_era5 == len(terrain_lons_arr):
        # Perfect alignment - use direct indexing
        terrain_aspect_at_era5 = terrain_aspect_vals
    else:
        # Find nearest terrain indices for each ERA5 point
        lat_indices = np.searchsorted(terrain_lats_arr, era5_lats, side='left')
        lon_indices = np.searchsorted(terrain_lons_arr, era5_lons, side='left')
        lat_indices = np.clip(lat_indices, 0, len(terrain_lats_arr) - 1)
        lon_indices = np.clip(lon_indices, 0, len(terrain_lons_arr) - 1)
        terrain_aspect_at_era5 = terrain_aspect_vals[np.ix_(lat_indices, lon_indices)]
    

    perpendicular_winds = xr.full_like(wind_speed, np.nan, dtype=float)
    
    # Vectorized perpendicular wind calculation
    valid_terrain = ~np.isnan(downwind_terrain_heights.values)
    
    for t_idx in range(nt):
        for p_idx in range(np_level):
            for lat_idx in range(nlat_era5):
                for lon_idx in range(nlon_era5):
                    if valid_terrain[t_idx, p_idx, lat_idx, lon_idx]:
                        wd = wind_dir.values[t_idx, p_idx, lat_idx, lon_idx]
                        ws = wind_speed.values[t_idx, p_idx, lat_idx, lon_idx]
                        ta = terrain_aspect_at_era5[lat_idx, lon_idx]
                        
                        if not np.isnan(wd) and not np.isnan(ws) and not np.isnan(ta) and ws > 0.1:
                            perp_wind = perpendicular_wind_component(wd, ws, ta)
                            perpendicular_winds.values[t_idx, p_idx, lat_idx, lon_idx] = perp_wind
    
    result = era5_data.copy()
    result['wind_speed'] = wind_speed
    result['wind_speed'].attrs = {'units': 'm/s', 'long_name': 'Wind speed'}
    result['wind_direction'] = wind_dir.astype(np.uint16)
    result['wind_direction'].attrs = {'units': 'degrees', 'long_name': 'Wind direction (from)'}

    downwind_heights_filled = downwind_terrain_heights.fillna(0) #handle NaN in Uint16
    result['downwind_terrain_height'] = downwind_heights_filled.astype(np.uint16)
    result['downwind_terrain_height'].attrs = {
        'units': 'm',
        'long_name': f'First terrain elevation above pressure level (within {range_km}km downwind)'
    }
    result['perpendicular_wind_speed'] = perpendicular_winds.astype(np.float32)
    result['perpendicular_wind_speed'].attrs = {
        'units': 'm/s',
        'long_name': 'Wind component perpendicular to terrain aspect'
    }
    
    return result
