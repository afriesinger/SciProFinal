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
    # Wind direction is where it comes FROM, so downwind direction is opposite
    downwind_dir = int(wind_dir) + 180  # Opposite direction
    
    n_points = int(range_km/2)  # Number of sample points
    max_dist_m = range_km * 1000  # Convert to meters
    distances = np.linspace(0, max_dist_m, n_points)
    
    # Use simple lat/lon offset (good approximation for small distances)
    # 1 degree latitude ≈ 111 km
    # 1 degree longitude ≈ 111 * cos(lat) km
    lat_offset = (distances / 111000) * np.cos(np.radians(downwind_dir))
    lon_offset = (distances / (111000 * np.cos(np.radians(era_lat)))) * np.sin(np.radians(downwind_dir))
    
    downwind_lats = era_lat + lat_offset
    downwind_lons = era_lon + lon_offset
    
    return downwind_lats, downwind_lons


def _find_highest_terrain_downwind(downwind_lats, downwind_lons, terrain_ds, pressure_level_height):
    """
    Find first terrain point downwind that's higher than pressure level.
    
    Parameters
    ----------
    downwind_lats : np.ndarray
        Latitudes along downwind direction
    downwind_lons : np.ndarray
        Longitudes along downwind direction
    terrain_ds : xr.Dataset
        Terrain dataset with 'elevation' and 'aspect' variables
    pressure_level_height : float
        Height at pressure level (meters)
        
    Returns
    -------
    terrain_height : float or np.nan
        Height of first terrain above pressure level
    terrain_aspect : float or np.nan
        Aspect of the first terrain point above pressure level
    """
    from scipy.interpolate import RegularGridInterpolator
    
    terrain_elevation = terrain_ds['elevation'].values
    terrain_lats = terrain_ds['latitude'].values
    terrain_lons = terrain_ds['longitude'].values
    
    # Create interpolator for terrain elevation
    interp_elev = RegularGridInterpolator(
        (terrain_lats[::-1], terrain_lons),
        terrain_elevation[::-1, :],
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    
    # Create interpolator for aspect
    terrain_aspect_vals = terrain_ds['aspect_deg'].values
    interp_aspect = RegularGridInterpolator(
        (terrain_lats[::-1], terrain_lons),
        terrain_aspect_vals[::-1, :],
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    
    # Stack points for interpolation
    points = np.stack([downwind_lats, downwind_lons], axis=-1)
    
    # Interpolate elevations along the downwind line
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
        Wind direction in degrees.
    """

    wind_dir = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
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


def compute_wind_terrain_interaction(
    era5_data: xr.Dataset,
    terrain_ds: xr.Dataset,
    range_km: float = 1.0
) -> xr.Dataset:
    """
    Compute wind-terrain interaction metrics using terrain search in downwind direction.
    
    For each ERA5 gridpoint at each pressure level:
    1. Extract wind speed and direction at that gridpoint
    2. Search downwind within the specified range
    3. Find the first terrain point higher than the pressure level
    4. Calculate the angle between wind direction and terrain slope aspect
    
    Points where the pressure level geopotential is below terrain elevation are skipped.
    
    Parameters
    ----------
    era5_data : xr.Dataset
        ERA5 dataset with u, v wind components and z (geopotential)
    terrain_ds : xr.Dataset
        Terrain aspect dataset with elevation, aspect, latitude, longitude
    range_km : float
        Search range downwind in kilometers (default: 10 km)
        
    Returns
    -------
    xr.Dataset
        ERA5 dataset with added xarray variables:
        - wind_speed: Wind speed at gridpoint
        - wind_direction: Wind direction (where wind comes FROM)
        - downwind_terrain_height: Height of first terrain above pressure level
    
    """
    # Input validation
    if 'u' not in era5_data or 'v' not in era5_data:
        raise ValueError("ERA5 dataset must contain 'u' and 'v' wind components")
    if 'z' not in era5_data:
        raise ValueError("ERA5 dataset must contain 'z' (geopotential)")
    
    # Compute wind metrics (vectorized across all dimensions at once)
    wind_dir, wind_speed = calc_wind(era5_data['u'], era5_data['v'])
    height_at_level = era5_data['z'] / G
    
    # Initialize output arrays with same shape as wind_speed
    downwind_terrain_heights = xr.full_like(wind_speed, np.nan, dtype=float)
    perpendicular_winds = xr.full_like(wind_speed, np.nan, dtype=float)
    
    # Get spatial coordinate arrays
    lats = era5_data.latitude.values
    lons = era5_data.longitude.values
    
    # Get terrain intersection mask if available
    # (True = gridpoint stuck below terrain, should skip)
    terrain_mask = None
    if 'terrain' in era5_data:
        terrain_mask = era5_data['terrain'].values
    
    # Define computation function for a single spatial location
    def compute_terrain_at_location(lat, lon):
        """Compute terrain metrics for all time-pressure points at a spatial location."""
        # Extract 1D time-pressure slice at this location
        wd_loc = wind_dir.sel(latitude=lat, longitude=lon, method='nearest')
        ws_loc = wind_speed.sel(latitude=lat, longitude=lon, method='nearest')
        h_loc = height_at_level.sel(latitude=lat, longitude=lon, method='nearest')
        
        # Get terrain intersection mask at this location (if available)
        terrain_mask_loc = None
        if terrain_mask is not None:
            lat_idx = np.argmin(np.abs(lats - lat))
            lon_idx = np.argmin(np.abs(lons - lon))
            # Extract 1D time-pressure mask at this location
            # terrain_mask shape: (valid_time, pressure_level, latitude, longitude)
            terrain_mask_loc = terrain_mask[:, :, lat_idx, lon_idx]
        
        # Convert to numpy for iteration (handles any dimension structure)
        wd_vals = wd_loc.values.ravel()
        ws_vals = ws_loc.values.ravel()
        h_vals = h_loc.values.ravel()
        
        # If terrain_mask exists, get corresponding 1D mask values
        terrain_stuck_vals = None
        if terrain_mask_loc is not None:
            # terrain_mask_loc is already a numpy array (extracted from terrain_mask)
            terrain_stuck_vals = terrain_mask_loc.ravel()
        
        # Process each time-pressure point with lambda
        compute_point = lambda wd, ws, h: (
            _find_highest_terrain_downwind(
                *_get_downwind_points(lat, lon, wd, range_km=range_km),
                terrain_ds, h
            ) if not (np.isnan(ws) or np.isnan(wd) or np.isnan(h) or ws < 0.1) 
            else (np.nan, np.nan)
        )
        
        # Apply computation to all points at this location
        for idx, (wd_val, ws_val, h_val) in enumerate(zip(wd_vals, ws_vals, h_vals)):
            # Check if this specific point is stuck in terrain
            stuck = terrain_stuck_vals[idx] if terrain_stuck_vals is not None else False
            
            # Skip computation if stuck in terrain
            if stuck:
                continue
            
            terrain_height, terrain_aspect = compute_point(wd_val, ws_val, h_val)
            
            # Calculate perpendicular wind component if terrain found
            if not np.isnan(terrain_aspect):
                perp_wind = perpendicular_wind_component(wd_val, ws_val, terrain_aspect)
                
                # Assign to output arrays (reconstruct multi-dimensional index)
                multi_idx = np.unravel_index(idx, wd_loc.values.shape)
                downwind_terrain_heights.values[multi_idx] = terrain_height
                perpendicular_winds.values[multi_idx] = perp_wind
    
    # Iterate over all spatial locations on ERA5 grid
    # Need the looping because of different shapes of datasets
    for lat in lats:
        for lon in lons:
            compute_terrain_at_location(lat, lon)
    
    # Assemble result dataset
    result = era5_data.copy()
    result['wind_speed'] = wind_speed
    result['wind_speed'].attrs = {'units': 'm/s', 'long_name': 'Wind speed'}
    result['wind_direction'] = wind_dir
    result['wind_direction'].attrs = {'units': 'degree', 'long_name': 'Wind direction (where wind comes FROM)'}
    result['downwind_terrain_height'] = downwind_terrain_heights
    result['downwind_terrain_height'].attrs = {
        'units': 'm',
        'long_name': f'Height of first terrain above pressure level within {range_km}km downwind'
    }
    result['perpendicular_wind_speed'] = perpendicular_winds
    result['perpendicular_wind_speed'].attrs = {'units': 'm/s', 'long_name': 'perpendicular wind speed'}
    
    return result
