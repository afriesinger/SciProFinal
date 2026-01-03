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
        Wind direction in radians (where wind is coming FROM, 0=N, π/2=E)
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
    downwind_dir = wind_dir + np.pi  # Opposite direction
    
    n_points = 50  # Number of sample points
    max_dist_m = range_km * 1000  # Convert to meters
    distances = np.linspace(0, max_dist_m, n_points)
    
    # Use simple lat/lon offset (good approximation for small distances)
    # 1 degree latitude ≈ 111 km
    # 1 degree longitude ≈ 111 * cos(lat) km
    lat_offset = (distances / 111000) * np.cos(downwind_dir)
    lon_offset = (distances / (111000 * np.cos(np.radians(era_lat)))) * np.sin(downwind_dir)
    
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
    Meteorological convention: 0° = from North, 90° = from East, 180° = from South, 270° = from West.
    
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
    wind_speed = np.sqrt(u**2 + v**2)
    return wind_dir, wind_speed



def perpendicular_wind_component(wind_dir, wind_speed, slope_aspect):
    """
    Compute the perpendicular component of wind relative to a slope aspect.
    
    The perpendicular component is the projection of wind speed onto the direction
    perpendicular to the slope (i.e., wind blowing INTO the slope).
    
    Parameters:
    wind_dir : float
        Wind direction in degrees (0-360, meteorological convention: 0=N, 90=E, 180=S, 270=W).
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
    
    # Compute the angle difference between wind direction and slope aspect
    # This gives us how aligned the wind is with the slope
    angle_diff = (wind_dir - slope_aspect) % 360
    
    # Normalize to [-180, 180] for easier interpretation
    if angle_diff > 180:
        angle_diff -= 360
    
    # Check if wind is within ±90° of slope aspect (wind blowing toward the slope)
    # If |angle_diff| <= 90°, the wind has a component toward the slope
    if abs(angle_diff) <= 90:
        perpendicular_wind_speed = wind_speed * np.cos(np.radians(angle_diff))
    else:
        # Wind is blowing away from the slope
        perpendicular_wind_speed = 0

    ##final check
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
    
    This approach does NOT interpolate the high-resolution terrain grid to the ERA5 grid.
    Instead, it performs targeted lookups along the downwind direction.
    
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
    
    # Get wind components - keep as xarray DataArrays
    if 'u' not in era5_data or 'v' not in era5_data:
        raise ValueError("ERA5 dataset must contain 'u' and 'v' wind components")
    
    u = era5_data['u']
    v = era5_data['v']
    
    # Get geopotential height - keep as xarray DataArray
    if 'z' not in era5_data:
        raise ValueError("ERA5 dataset must contain 'z' (geopotential)")
    
    z = era5_data['z']
    
    # Compute wind speed and direction with xarray operations
    wind_dir, wind_speed = calc_wind(u, v)
    
    # Convert geopotential to geometric height in meters
    height_at_level = z / G
    
    # Get coordinate dimensions
    era_lats = era5_data.latitude.values
    era_lons = era5_data.longitude.values
    has_pressure = 'pressure_level' in era5_data.dims
    has_time = 'valid_time' in era5_data.dims
    
    pressure_levels = era5_data.pressure_level.values if has_pressure else None
    valid_times = era5_data.valid_time.values if has_time else None
    
    # Initialize output DataArrays (copy structure from wind_speed)
    downwind_terrain_heights = wind_speed.copy(deep=True)
    downwind_terrain_heights[:] = np.nan
    
    perpendicular_winds = wind_speed.copy(deep=True)
    perpendicular_winds[:] = np.nan
    
    # Define computation for a single point
    def compute_point(u_val, v_val, z_val, lat, lon):
        """Compute metrics for a single gridpoint."""
        # Extract scalar values
        wind_dir_pt, wind_speed_pt  = calc_wind(u_val, v_val)
      
        pressure_height_pt = float(z_val / G)
        
        # Skip if wind speed is too small or invalid
        if np.isnan(wind_speed_pt) or wind_speed_pt < 0.1:
            return np.nan, np.nan
        if np.isnan(wind_dir_pt) or np.isnan(pressure_height_pt):
            return np.nan, np.nan
        
        # Get downwind points
        downwind_lats, downwind_lons = _get_downwind_points(
            lat, lon, wind_dir_pt, range_km=range_km
        )
        
        # Find terrain downwind
        terrain_height, terrain_aspect = _find_highest_terrain_downwind(
            downwind_lats, downwind_lons, terrain_ds, pressure_height_pt
        )
        
        # Calculate angle if terrain found
        if not np.isnan(terrain_aspect):
            #angle = _compute_wind_slope_angle(wind_dir_pt, terrain_aspect)
            perpendicular_wind = perpendicular_wind_component(wind_dir_pt, wind_speed_pt, terrain_aspect)

        else:
            #angle = np.nan
            perpendicular_wind = np.nan
        
        return terrain_height, perpendicular_wind
    
    # Convert to numpy for faster indexed access
    u_np = u.values
    v_np = v.values
    z_np = z.values
    
    # Process each gridpoint with all dimensions
    if has_time and has_pressure:
        # 4D case: (valid_time, pressure_level, latitude, longitude)
        for t_idx in range(len(valid_times)):
            for p_idx in range(len(pressure_levels)):
                for lat_idx, lat in enumerate(era_lats):
                    for lon_idx, lon in enumerate(era_lons):
                        u_val = float(u_np[t_idx, p_idx, lat_idx, lon_idx])
                        v_val = float(v_np[t_idx, p_idx, lat_idx, lon_idx])
                        z_val = float(z_np[t_idx, p_idx, lat_idx, lon_idx])
                        
                        terrain_h, perpendicular_wind = compute_point(u_val, v_val, z_val, lat, lon)
                        
                        downwind_terrain_heights.values[t_idx, p_idx, lat_idx, lon_idx] = terrain_h
                        perpendicular_winds.values[t_idx, p_idx, lat_idx, lon_idx] = perpendicular_wind
    
    
    # Add all variables to result dataset
    result = era5_data.copy()

    result['perpendicular_wind_speed'] = perpendicular_winds
    result['perpendicular_wind_speed'].attrs = {'units': 'm/s', 'long_name': 'perpendicular wind speed'}
    
    result['wind_speed'] = wind_speed
    result['wind_speed'].attrs = {'units': 'm/s', 'long_name': 'Wind speed'}
    
    result['wind_direction'] = wind_dir
    result['wind_direction'].attrs = {
        'units': 'degree',
        'long_name': 'Wind direction (where wind comes FROM)'
    }
    
    result['downwind_terrain_height'] = downwind_terrain_heights
    result['downwind_terrain_height'].attrs = {
        'units': 'm',
        'long_name': f'Height of first terrain above pressure level within {range_km}km downwind'
    }
    

    
    return result
