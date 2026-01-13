"""
Terrain processing module for ERA5 visualization.

This module provides functions for:
- Loading SRTM terrain data from cached GeoTiff
- Computing slope aspect and terrain properties
- Creating xarray datasets with terrain information

Author: Andreas Friesinger
Date: 2025-12-31
"""

import numpy as np
import xarray as xr
import os
from typing import Tuple

G = 9.80665  # Gravity (m/s²)


def load_terrain_from_tif(
    tif_path: str = "./terrain_cache/srtm_alps_30m.tif"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Load terrain data from GeoTiff file.
    
    Parameters
    ----------
    tif_path : str
        Path to the GeoTiff file
        
    Returns
    -------
    terrain : np.ndarray
        2D elevation array (int32) with no-data values filled using neighbor mean
    lats : np.ndarray
        1D latitude array (decreasing, N to S)
    lons : np.ndarray
        1D longitude array (increasing, W to E)
    resolution_m : float
        Approximate resolution in meters
    """
    import rasterio
    from scipy.ndimage import uniform_filter
    
    if not os.path.exists(tif_path):
        raise FileNotFoundError(
            f"Terrain file not found: {tif_path}\n"
            "Please download srtm_alps_30m.tif and place it in terrain_cache/"
        )
    
    with rasterio.open(tif_path) as ds:
        terrain = ds.read(1)  # Keep as int16 first
        bounds = ds.bounds
        
    lat_min, lat_max = bounds.bottom, bounds.top
    lon_min, lon_max = bounds.left, bounds.right
    
    lats = np.linspace(lat_max, lat_min, terrain.shape[0])
    lons = np.linspace(lon_min, lon_max, terrain.shape[1])
    
    # Identify no-data values: -32768 is standard sentinel for int16 GeoTIFFs
    # Also mark any elevation < 0m as no-data (water, artifacts, errors)
    no_data_mask = (terrain == -32768) | (terrain < 0)
    no_data_count_initial = no_data_mask.sum()
    
    terrain = terrain.astype(np.float32)
    
    # Fill no-data values iteratively using 3x3 neighborhood mean
    if no_data_mask.any():
        max_iterations = 10
        for iteration in range(max_iterations):         
            weighted_sum = uniform_filter(np.where(no_data_mask, 0, terrain), 
                                         size=3, mode='constant', cval=0)
            weight_count = uniform_filter(np.where(no_data_mask, 0, 1).astype(float), 
                                         size=3, mode='constant', cval=0)
            # Avoid division by zero
            weight_count[weight_count == 0] = 1
            neighborhood_mean = weighted_sum / weight_count
            
            can_fill = (weight_count > 0) & no_data_mask
            terrain[can_fill] = neighborhood_mean[can_fill]
            no_data_mask = no_data_mask & ~can_fill
            
            filled_this_iteration = can_fill.sum()

    terrain = np.round(terrain).astype(np.int32)
    
    # Estimate resolution in meters
    mean_lat = 0.5 * (lat_min + lat_max)
    resolution_m = int(round((lons[1] - lons[0]) * 111000.0 * np.cos(np.radians(mean_lat))))
    
    return terrain, lats, lons, resolution_m


def downsample_terrain(
    terrain: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    source_resolution_m: float,
    target_resolution_m: float = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:

    """
    Downsample terrain to a coarser resolution.

    Parameters
    ----------
    terrain : np.ndarray
        2D elevation array (may contain NaN for no-data)
    lats : np.ndarray
        1D latitude array (decreasing, N to S)
    lons : np.ndarray
        1D longitude array (increasing, W to E)
    source_resolution_m : float
        Approximate resolution of the input terrain
    target_resolution_m : float, optional
        Desired resolution of the output terrain, default is 1000m

    Returns
    -------
    terrain_ds : np.ndarray
        Downsampled elevation array
    lats_ds : np.ndarray
        Downsampled latitude array
    lons_ds : np.ndarray
        Downsampled longitude array
    actual_resolution_m : float
        Actual resolution of the output terrain
    """
    from scipy.ndimage import zoom as scipy_zoom
    from scipy.ndimage import gaussian_filter
    
    downsample_factor = int(round(target_resolution_m / source_resolution_m))
    
    if downsample_factor <= 1:
        return terrain, lats, lons, source_resolution_m
    
    # Downsample using bilinear interpolation, then convert back to int32
    terrain_ds = scipy_zoom(terrain.astype(float), 1 / downsample_factor, order=1)
    terrain_ds = np.round(terrain_ds).astype(np.int32)
    
    rows_ds, cols_ds = terrain_ds.shape
    lats_ds = np.linspace(lats[0], lats[-1], rows_ds)
    lons_ds = np.linspace(lons[0], lons[-1], cols_ds)
    actual_resolution_m = downsample_factor * source_resolution_m
    
    print(f"Downsampled: {terrain.shape} -> {terrain_ds.shape}, resolution ~{actual_resolution_m}m")
    print(f"  Elevation range: {terrain_ds.min()}m to {terrain_ds.max()}m")
    
    return terrain_ds, lats_ds, lons_ds, actual_resolution_m


def compute_terrain_aspect_dataset(
    terrain: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    resolution_m: float,
    min_elevation: float = 50.0,
    min_slope: float = 3.0,
    smooth_sigma: float = 1.5
) -> xr.Dataset:
    """
    Compute slope aspect and terrain properties, return as xarray Dataset.

    Parameters
    ----------
    terrain : np.ndarray
        2D elevation array
    lats : np.ndarray
        1D latitude array (decreasing, N to S)
    lons : np.ndarray
        1D longitude array (increasing, W to E)
    resolution_m : float
        Approximate resolution in meters
    min_elevation : float, optional
        Minimum elevation for terrain mask, default 500.0
    min_slope : float, optional
        Minimum slope for terrain mask, default 1.0
    smooth_sigma : float, optional
        Sigma for Gaussian smoothing before gradient calculation, default 1.5

    Returns
    -------
    ds : xr.Dataset
        xarray Dataset containing terrain elevation, aspect, slope, and mask
    """
    from scipy.ndimage import gaussian_filter
    
    # Smooth terrain for gradient calculation
    terrain_smooth = gaussian_filter(terrain.astype(float), sigma=smooth_sigma)
    
    dz_dy, dz_dx = np.gradient(terrain_smooth, resolution_m)
    
    aspect_deg = np.degrees(np.arctan2(-dz_dx, -dz_dy)) % 360
    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    
    aspect_deg_int = np.round(aspect_deg).astype(np.int32) % 360
    slope_deg_int = np.round(slope_deg).astype(np.int32)
    
    terrain_mask = (terrain >= min_elevation) & (slope_deg >= min_slope)
    
    ds = xr.Dataset(
        {
            'elevation': (['latitude', 'longitude'], terrain),
            'aspect_deg': (['latitude', 'longitude'], aspect_deg_int),
            'slope': (['latitude', 'longitude'], slope_deg_int),
            'terrain_mask': (['latitude', 'longitude'], terrain_mask),
        },
        coords={
            'latitude': lats,
            'longitude': lons,
        },
        attrs={
            'resolution_m': resolution_m,
            'min_elevation': min_elevation,
            'min_slope': min_slope,
            'description': 'Terrain slope aspect dataset computed from SRTM data',
        }
    )
    
    # Add variable attributes
    ds['elevation'].attrs = {'units': 'm', 'long_name': 'Terrain elevation'}
    ds['aspect_deg'].attrs = {'units': 'degrees', 'long_name': 'Slope aspect (0-360)'}
    ds['slope'].attrs = {'units': 'degrees', 'long_name': 'Slope steepness'}
    ds['terrain_mask'].attrs = {'long_name': 'Valid terrain mask'}
    
    return ds


def load_terrain_aspect_dataset(
    cache_path: str = "./terrain_cache/terrain_aspect_1km.nc"
) -> xr.Dataset:

    """
    Load pre-computed terrain aspect dataset from NetCDF file.

    Parameters
    ----------
    cache_path : str
        Path to the terrain aspect cache file (default: ./terrain_cache/terrain_aspect_1km.nc)

    Returns
    -------
    xr.Dataset
        xarray Dataset containing terrain elevation, aspect, slope, and mask
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Terrain aspect cache not found: {cache_path}")
    
    ds = xr.open_dataset(cache_path)
    print(f"Loaded terrain aspect dataset: {ds.sizes['latitude']} x {ds.sizes['longitude']}")
    return ds


def compute_terrain_intersection(
    era5_data: xr.Dataset,
    terrain_ds: xr.Dataset
) -> xr.Dataset:
    """
    Add terrain intersection information to ERA5 dataset.
    
    The intersection is determined by comparing the geopotential height at each 
    pressure level with the terrain elevation at each lat/lon grid point.
    
    Parameters
    ----------
    era5_data : xr.Dataset
        ERA5 dataset containing 'gph' (geopotential-height) variable
    terrain_ds : xr.Dataset
        Terrain aspect dataset from terrain.load_terrain_aspect_dataset()
    
    Returns
    -------
    xr.Dataset
        ERA5 dataset with added terrain information:
        - terrain: boolean mask where pressure level is below terrain
        - terrain_elevation: SRTM elevation interpolated to ERA5 grid
    """
    from scipy.interpolate import RegularGridInterpolator
    
    
    terrain_elev = terrain_ds['elevation'].values
    terrain_lats = terrain_ds['latitude'].values
    terrain_lons = terrain_ds['longitude'].values
    
    # RegularGridInterpolator expects increasing coordinates
    interp_func = RegularGridInterpolator(
        (terrain_lats[::-1], terrain_lons),  # Flip lats to be increasing
        terrain_elev[::-1, :],                # Flip data to match
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    
    # Create grid of ERA5 coordinates
    era5_lats = era5_data.latitude.values
    era5_lons = era5_data.longitude.values
    lon_grid, lat_grid = np.meshgrid(era5_lons, era5_lats)
    
    # Interpolate terrain to ERA5 grid
    points = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
    terrain_on_era5_values = interp_func(points).reshape(lat_grid.shape)
    
    # Convert back to int32 to match terrain data type
    terrain_on_era5_values = np.round(np.nan_to_num(terrain_on_era5_values, nan=0)).astype(np.int32)
    
    terrain_on_era5 = xr.DataArray(
        terrain_on_era5_values,
        dims=['latitude', 'longitude'],
        coords={'latitude': era5_data.latitude, 'longitude': era5_data.longitude}
    )
       
    # Convert geopotential to geopotential height
    if 'gph' in era5_data:
        geopotential_height = era5_data['gph']
    else:
        raise ValueError("ERA5 dataset must contain 'gph' (geopotential-height) variable")
    
    # Create terrain intersection mask
    terrain_broadcast = terrain_on_era5.broadcast_like(geopotential_height)
    terrain_mask = geopotential_height < terrain_broadcast
    terrain_mask.attrs = {
        'long_name': 'Terrain intersection mask',
        'description': 'True where geopotential height is below terrain elevation'
    }
    
    # Create output dataset
    result = era5_data.copy()
    result['terrain'] = terrain_mask
    result['terrain_elevation'] = terrain_on_era5.astype(np.int32)
    result['terrain_elevation'].attrs = {
        'units': 'm',
        'long_name': 'SRTM terrain elevation',
        'source': 'SRTM'
    }
    
    
    return result

def interpolate_to_grid(source_dataset, target_dataset):
    """
    Interpolate ERA5 data onto a target grid using xarray's interpolation methods.
    
    Parameters
    ----------
    source_dataset : xr.Dataset
        Source dataset with variables on (time, pressure_level, latitude, longitude)
    target_dataset : xr.Dataset
        Target dataset defining the grid (uses latitude and longitude coordinates)
    
    Returns
    -------
    xr.Dataset
        Interpolated dataset on target grid with same variables and coordinates
    """
    target_lats = target_dataset.latitude.values
    target_lons = target_dataset.longitude.values
    
    # Use xarray's interp method with vectorized operations
    # Create new coordinates for target grid
    new_coords = {
        'latitude': target_lats,
        'longitude': target_lons,
    }
    
    # Interpolate using xarray's built-in method (handles all dims at once)
    result = source_dataset.interp(
        coords=new_coords,
        method='linear',
        kwargs={'fill_value': np.nan}
    )
    
    # Preserve attributes from original dataset
    for var_name in result.data_vars:
        if var_name in source_dataset.data_vars:
            result[var_name].attrs.update(source_dataset[var_name].attrs)
    
    return result


