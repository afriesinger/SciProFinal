"""
Terrain processing module for ERA5 visualization.

This module provides functions for:
- Loading SRTM terrain data from cached GeoTiff
- Computing slope aspect and terrain properties
- Creating xarray datasets with terrain information

Requires: srtm_alps_30m.tif in terrain_cache/ directory

Author: Andreas Friesinger
Date: 2025-12-31
"""

import numpy as np
import xarray as xr
import os
from typing import Tuple

G = 9.80665  # Standard gravity (m/s²)


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
        2D elevation array
    lats : np.ndarray
        1D latitude array (decreasing, N to S)
    lons : np.ndarray
        1D longitude array (increasing, W to E)
    resolution_m : float
        Approximate resolution in meters
    """
    import rasterio
    
    if not os.path.exists(tif_path):
        raise FileNotFoundError(
            f"Terrain file not found: {tif_path}\n"
            "Please download srtm_alps_30m.tif and place it in terrain_cache/"
        )
    
    with rasterio.open(tif_path) as ds:
        terrain = ds.read(1)
        bounds = ds.bounds
        
    lat_min, lat_max = bounds.bottom, bounds.top
    lon_min, lon_max = bounds.left, bounds.right
    
    lats = np.linspace(lat_max, lat_min, terrain.shape[0])
    lons = np.linspace(lon_min, lon_max, terrain.shape[1])
    
    # Estimate resolution in meters
    mean_lat = 0.5 * (lat_min + lat_max)
    resolution_m = int(round((lons[1] - lons[0]) * 111000.0 * np.cos(np.radians(mean_lat))))
    
    print(f"Loaded terrain: {terrain.shape}, resolution ~{resolution_m}m")
    print(f"  Bounds: lat({lat_min:.2f}, {lat_max:.2f}), lon({lon_min:.2f}, {lon_max:.2f})")
    
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
    """
    from scipy.ndimage import zoom as scipy_zoom
    from scipy.ndimage import gaussian_filter
    
    downsample_factor = int(round(target_resolution_m / source_resolution_m))
    
    if downsample_factor <= 1:
        return terrain, lats, lons, source_resolution_m
    
    # Smooth before downsampling (anti-aliasing)
    terrain_smooth = gaussian_filter(terrain.astype(float), sigma=downsample_factor / 2)
    terrain_ds = scipy_zoom(terrain_smooth, 1 / downsample_factor, order=1)
    
    rows_ds, cols_ds = terrain_ds.shape
    lats_ds = np.linspace(lats[0], lats[-1], rows_ds)
    lons_ds = np.linspace(lons[0], lons[-1], cols_ds)
    actual_resolution_m = downsample_factor * source_resolution_m
    
    print(f"Downsampled: {terrain.shape} -> {terrain_ds.shape}, resolution ~{actual_resolution_m}m")
    
    return terrain_ds, lats_ds, lons_ds, actual_resolution_m


def compute_terrain_aspect_dataset(
    terrain: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    resolution_m: float,
    min_elevation: float = 500.0,
    min_slope: float = 1.0,
    smooth_sigma: float = 1.5
) -> xr.Dataset:
    """
    Compute slope aspect and related terrain properties, return as xarray Dataset.
    """
    from scipy.ndimage import gaussian_filter
    
    # Smooth terrain for gradient calculation
    terrain_smooth = gaussian_filter(terrain.astype(float), sigma=smooth_sigma)
    
    # Compute gradients (dz/dy, dz/dx)
    dz_dy, dz_dx = np.gradient(terrain_smooth, resolution_m)
    
    # Aspect: direction the slope faces (downhill direction)
    aspect = np.arctan2(-dz_dx, -dz_dy)  # radians
    
    # Convert to 0-360 degrees
    aspect_deg = np.degrees(aspect) % 360
    
    # Slope magnitude and angle
    slope_mag = np.sqrt(dz_dx**2 + dz_dy**2)
    slope_deg = np.degrees(np.arctan(slope_mag))
    
    # Terrain mask
    terrain_mask = (terrain >= min_elevation) & (slope_deg >= min_slope)
    
    # Create xarray Dataset
    ds = xr.Dataset(
        {
            'elevation': (['latitude', 'longitude'], terrain),
            'aspect': (['latitude', 'longitude'], aspect),
            'aspect_deg': (['latitude', 'longitude'], aspect_deg),
            'slope': (['latitude', 'longitude'], slope_deg),
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
    ds['aspect'].attrs = {'units': 'radians', 'long_name': 'Slope aspect (downhill direction)'}
    ds['aspect_deg'].attrs = {'units': 'degrees', 'long_name': 'Slope aspect (0-360)'}
    ds['slope'].attrs = {'units': 'degrees', 'long_name': 'Slope steepness'}
    ds['terrain_mask'].attrs = {'long_name': 'Valid terrain mask'}
    
    return ds


def load_terrain_aspect_dataset(
    cache_path: str = "./terrain_cache/terrain_aspect_1km.nc"
) -> xr.Dataset:
    """
    Load pre-computed terrain aspect dataset from NetCDF file.
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
        ERA5 dataset containing 'z' (geopotential) variable
    terrain_ds : xr.Dataset
        Terrain aspect dataset from terrain.load_terrain_aspect_dataset()
    
    Returns
    -------
    xr.Dataset
        ERA5 dataset with added terrain information:
        - terrain: boolean mask where pressure level is below terrain
        - terrain_elevation: SRTM elevation interpolated to ERA5 grid
        - geopotential_height: converted from geopotential
    """
    from scipy.interpolate import RegularGridInterpolator
    
    # Get ERA5 coordinate bounds
    lat_min = float(era5_data.latitude.min())
    lat_max = float(era5_data.latitude.max())
    lon_min = float(era5_data.longitude.min())
    lon_max = float(era5_data.longitude.max())
    
    print(f"ERA5 domain: lat({lat_min:.1f}, {lat_max:.1f}), lon({lon_min:.1f}, {lon_max:.1f})")
    
    terrain_elev = terrain_ds['elevation'].values
    terrain_lats = terrain_ds['latitude'].values
    terrain_lons = terrain_ds['longitude'].values
    
    # Interpolate terrain to ERA5 grid
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
    
    terrain_on_era5 = xr.DataArray(
        terrain_on_era5_values,
        dims=['latitude', 'longitude'],
        coords={'latitude': era5_data.latitude, 'longitude': era5_data.longitude}
    )
    
    print(f"Terrain elevation range on ERA5 grid: {float(np.nanmin(terrain_on_era5_values)):.0f}m to {float(np.nanmax(terrain_on_era5_values)):.0f}m")
    
    # Convert geopotential to geopotential height
    if 'z' in era5_data:
        geopotential_height = era5_data['z'] / G
        geopotential_height.attrs = {'units': 'm', 'long_name': 'Geopotential height'}
    else:
        raise ValueError("ERA5 dataset must contain 'z' (geopotential) variable")
    
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
    result['terrain_elevation'] = terrain_on_era5
    result['terrain_elevation'].attrs = {
        'units': 'm',
        'long_name': 'SRTM terrain elevation',
        'source': 'SRTM'
    }
    result['geopotential_height'] = geopotential_height
    
    # Print summary
    print(f"\n Terrain intersection summary:")
    for plev in era5_data.pressure_level.values:
        mask_at_level = terrain_mask.isel(valid_time=0).sel(pressure_level=plev)
        n_intersect = int(mask_at_level.sum())
        n_total = mask_at_level.size
        pct = 100 * n_intersect / n_total
        height_at_level = geopotential_height.isel(valid_time=0).sel(pressure_level=plev)
        mean_height = float(height_at_level.mean())
        print(f"  {plev:4.0f} hPa: mean height ~{mean_height:5.0f}m, {n_intersect:5d}/{n_total} ({pct:5.1f}%) below terrain")
    
    return result

def interpolate_to_grid(source_dataset, target_dataset):
    """
    Interpolate ERA5 data onto a target grid.
    
    Parameters
    ----------
    source_dataset : xr.Dataset
        Source dataset with 4D variables (time, pressure_level, latitude, longitude)
    target_dataset : xr.Dataset
        Target dataset defining the grid (uses latitude and longitude coordinates)
    
    Returns
    -------
    xr.Dataset
        Interpolated dataset on target grid with same variables and coordinates
    """
    from scipy.interpolate import RegularGridInterpolator
    
    source_lats = source_dataset.latitude.values
    source_lons = source_dataset.longitude.values
    target_lats = target_dataset.latitude.values
    target_lons = target_dataset.longitude.values
    
    # Create output dataset structure
    result = xr.Dataset(
        coords={
            'latitude': target_lats,
            'longitude': target_lons,
            'valid_time': source_dataset.valid_time,
            'pressure_level': source_dataset.pressure_level,
        }
    )
    
    # Points to interpolate to
    points = np.array(np.meshgrid(target_lats, target_lons, indexing='ij')).T.reshape(-1, 2)
    
    # Filter 4D variables
    four_d_vars = {name: var for name, var in source_dataset.data_vars.items() 
                   if len(var.dims) == 4}
    
    for var_name, var_data in four_d_vars.items():
        data_stacked = var_data.stack(tp=('valid_time', 'pressure_level'))
        output_shape = (len(source_dataset.valid_time), len(source_dataset.pressure_level),
                        len(target_lats), len(target_lons))
        interp_array = np.zeros(output_shape)
        
        for idx, (tp, data_slice) in enumerate(data_stacked.groupby('tp')):
            interp_func = RegularGridInterpolator(
                (source_lats, source_lons),
                data_slice.values,
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )
            t_idx = idx // len(source_dataset.pressure_level)
            p_idx = idx % len(source_dataset.pressure_level)
            interp_array[t_idx, p_idx] = interp_func(points).reshape(
                len(target_lats), len(target_lons)
            )
        
        result[var_name] = (('valid_time', 'pressure_level', 'latitude', 'longitude'), 
                            interp_array)
        if var_data.attrs:
            result[var_name].attrs.update(var_data.attrs)
    
    return result


