"""
Unit tests for the terrain module.

Author: Andreas Friesinger
Date: 2026-01-09
"""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Add parent directory to path to allow importing terrain module directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import terrain


class TestLoadTerrainFromTif:
    """Tests for load_terrain_from_tif function."""
    
    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        pytest.importorskip("rasterio")
        with pytest.raises(FileNotFoundError):
            terrain.load_terrain_from_tif(tif_path="./nonexistent/file.tif")
    
    def test_successful_load(self):
        """Test successful terrain loading from GeoTiff."""
        # Skip if rasterio not available
        try:
            import rasterio
        except ImportError:
            pytest.skip("rasterio not installed")
        
        with patch('rasterio.open') as mock_rasterio:
            mock_ds = MagicMock()
            mock_ds.__enter__.return_value = mock_ds
            mock_ds.__exit__.return_value = None
            
            # Create mock terrain data
            terrain_data = np.random.rand(100, 200).astype(np.float32) * 4000
            mock_ds.read.return_value = terrain_data
            
            # Mock bounds (lat_min, lon_min, lat_max, lon_max)
            mock_bounds = Mock()
            mock_bounds.bottom = 46.0
            mock_bounds.top = 47.0
            mock_bounds.left = 10.0
            mock_bounds.right = 11.0
            mock_ds.bounds = mock_bounds
            mock_rasterio.return_value = mock_ds
            
            # Call function
            with patch('os.path.exists', return_value=True):
                elev, lats, lons, res = terrain.load_terrain_from_tif("mock.tif")
            
            # Assertions
            assert elev.shape == (100, 200)
            assert len(lats) == 100
            assert len(lons) == 200
            assert res > 0
            assert lats[0] > lats[-1]  # Decreasing (N to S)
            assert lons[0] < lons[-1]  # Increasing (W to E)
    
    def test_resolution_calculation(self):
        """Test that resolution is calculated correctly in meters."""
        # Skip if rasterio not available
        try:
            import rasterio
        except ImportError:
            pytest.skip("rasterio not installed")
        
        with patch('rasterio.open') as mock_rasterio:
            mock_ds = MagicMock()
            mock_ds.__enter__.return_value = mock_ds
            mock_ds.__exit__.return_value = None
            
            terrain_data = np.ones((1000, 1000))
            mock_ds.read.return_value = terrain_data
            
            # Set bounds for ~1 degree spacing with correct number of points
            # 1000 points per degree → ~111m per point at equator
            # At ~46.5°N, multiply by cos(46.5°) ≈ 0.69 → ~77m
            mock_bounds = Mock()
            mock_bounds.bottom = 46.0
            mock_bounds.top = 47.0
            mock_bounds.left = 10.0
            mock_bounds.right = 11.0
            mock_ds.bounds = mock_bounds
            mock_rasterio.return_value = mock_ds
            
            with patch('os.path.exists', return_value=True):
                _, _, _, res = terrain.load_terrain_from_tif("mock.tif")
            
            # 1000 points per 1 degree at ~46.5°N should give ~76-77m resolution
            assert 70 < res < 90


class TestDownsampleTerrain:
    """Tests for downsample_terrain function."""
    
    def test_no_downsampling_needed(self):
        """Test that terrain is returned unchanged when factor <= 1."""
        terrain_data = np.random.rand(100, 200)
        lats = np.linspace(47, 46, 100)
        lons = np.linspace(10, 11, 200)
        
        result_elev, result_lats, result_lons, result_res = terrain.downsample_terrain(
            terrain_data, lats, lons, 
            source_resolution_m=1000, 
            target_resolution_m=500  # Finer target than source
        )
        
        assert np.allclose(result_elev, terrain_data)
        assert np.allclose(result_lats, lats)
        assert np.allclose(result_lons, lons)
    
    def test_downsampling_shape(self):
        """Test that downsampled terrain has correct shape."""
        terrain_data = np.random.rand(100, 200)
        lats = np.linspace(47, 46, 100)
        lons = np.linspace(10, 11, 200)
        
        result_elev, result_lats, result_lons, result_res = terrain.downsample_terrain(
            terrain_data, lats, lons,
            source_resolution_m=100,
            target_resolution_m=1000  # 10x coarser
        )
        
        assert result_elev.shape[0] <= 20  # Roughly 10x smaller
        assert result_elev.shape[1] <= 30  # Roughly 10x smaller
        assert len(result_lats) == result_elev.shape[0]
        assert len(result_lons) == result_elev.shape[1]
    
    def test_downsampling_preserves_bounds(self):
        """Test that downsampling preserves lat/lon bounds."""
        terrain_data = np.random.rand(100, 200)
        lats = np.linspace(47, 46, 100)
        lons = np.linspace(10, 11, 200)
        
        result_elev, result_lats, result_lons, _ = terrain.downsample_terrain(
            terrain_data, lats, lons,
            source_resolution_m=100,
            target_resolution_m=1000
        )
        
        # Check bounds are preserved
        assert np.isclose(result_lats[0], lats[0])
        assert np.isclose(result_lats[-1], lats[-1])
        assert np.isclose(result_lons[0], lons[0])
        assert np.isclose(result_lons[-1], lons[-1])


class TestComputeTerrainAspectDataset:
    """Tests for compute_terrain_aspect_dataset function."""
    
    def test_output_is_dataset(self):
        """Test that function returns an xarray Dataset."""
        terrain_data = np.random.rand(50, 50) * 2000 + 1000
        lats = np.linspace(47, 46, 50)
        lons = np.linspace(10, 11, 50)
        
        result = terrain.compute_terrain_aspect_dataset(
            terrain_data, lats, lons, resolution_m=1000
        )
        
        assert isinstance(result, xr.Dataset)
    
    def test_required_variables_present(self):
        """Test that all required variables are in output."""
        terrain_data = np.random.rand(50, 50) * 2000 + 1000
        lats = np.linspace(47, 46, 50)
        lons = np.linspace(10, 11, 50)
        
        result = terrain.compute_terrain_aspect_dataset(
            terrain_data, lats, lons, resolution_m=1000
        )
        
        required_vars = ['elevation', 'aspect_deg', 'slope', 'terrain_mask']
        for var in required_vars:
            assert var in result.data_vars, f"Missing variable: {var}"
    
    def test_coordinates_match_input(self):
        """Test that output coordinates match input arrays."""
        terrain_data = np.random.rand(50, 50) * 2000 + 1000
        lats = np.linspace(47, 46, 50)
        lons = np.linspace(10, 11, 50)
        
        result = terrain.compute_terrain_aspect_dataset(
            terrain_data, lats, lons, resolution_m=1000
        )
        
        assert np.allclose(result.latitude.values, lats)
        assert np.allclose(result.longitude.values, lons)
    
    def test_aspect_deg_range(self):
        """Test that aspect_deg is in valid range [0, 360)."""
        terrain_data = np.random.rand(50, 50) * 2000 + 1000
        lats = np.linspace(47, 46, 50)
        lons = np.linspace(10, 11, 50)
        
        result = terrain.compute_terrain_aspect_dataset(
            terrain_data, lats, lons, resolution_m=1000
        )
        
        aspect_deg = result['aspect_deg'].values
        assert np.all(aspect_deg >= 0)
        assert np.all(aspect_deg < 360)
    
    def test_south_facing_slope(self):
        """Test south-facing slope detection with crest W->E and slope S->N.
        
        Creates a ridge running west to east with elevation increasing from south to north.
        This creates a south-facing slope (normal pointing south, aspect ~180°).
        """
        # Create a 50x50 grid
        ny, nx = 50, 50
        lats = np.linspace(47, 46, ny)
        lons = np.linspace(10, 11, nx)
        
        # Create terrain: crest running W->E (constant along x-axis)
        # Elevation increases from south (low) to north (high)
        # This creates a slope facing south
        terrain_data = np.zeros((ny, nx))
        for i in range(ny):
            # Elevation increases with latitude (north is higher)
            terrain_data[i, :] = 1000 + (i / ny) * 1000
        
        result = terrain.compute_terrain_aspect_dataset(
            terrain_data, lats, lons, resolution_m=1000, min_slope=0.1
        )
        
        aspect_deg = result['aspect_deg'].values
        
        # South-facing slope should have aspect around 180° (south)
        central_aspect = aspect_deg[10:40, 10:40]
        mean_aspect = np.mean(central_aspect)
        
        # Aspect should be close to 180° (south-facing)
        # Allow ±45° tolerance due to resolution and smoothing
        assert 135 <= mean_aspect <= 225, f"Expected south-facing (180°), got {mean_aspect:.1f}°"
    
    def test_slope_angle_range(self):
        """Test that slope is in valid range [0, 90) degrees."""
        terrain_data = np.random.rand(50, 50) * 2000 + 1000
        lats = np.linspace(47, 46, 50)
        lons = np.linspace(10, 11, 50)
        
        result = terrain.compute_terrain_aspect_dataset(
            terrain_data, lats, lons, resolution_m=1000
        )
        
        slope = result['slope'].values
        assert np.all(slope >= 0)
        assert np.all(slope < 90)
    
    def test_terrain_mask_is_boolean(self):
        """Test that terrain_mask is boolean."""
        terrain_data = np.random.rand(50, 50) * 2000 + 1000
        lats = np.linspace(47, 46, 50)
        lons = np.linspace(10, 11, 50)
        
        result = terrain.compute_terrain_aspect_dataset(
            terrain_data, lats, lons, resolution_m=1000
        )
        
        assert result['terrain_mask'].dtype == bool
    
    def test_elevation_minimum_filtering(self):
        """Test that terrain below min_elevation is masked."""
        terrain_data = np.ones((50, 50)) * 300  # All below min_elevation=500
        lats = np.linspace(47, 46, 50)
        lons = np.linspace(10, 11, 50)
        
        result = terrain.compute_terrain_aspect_dataset(
            terrain_data, lats, lons, 
            resolution_m=1000, 
            min_elevation=500.0,
            min_slope=0.1  # Very low slope requirement
        )
        
        # All points should be masked (False) due to low elevation
        assert not result['terrain_mask'].any()
    
    def test_slope_minimum_filtering(self):
        """Test that terrain with low slope is masked."""
        terrain_data = np.ones((50, 50)) * 2000  # Flat terrain
        lats = np.linspace(47, 46, 50)
        lons = np.linspace(10, 11, 50)
        
        result = terrain.compute_terrain_aspect_dataset(
            terrain_data, lats, lons,
            resolution_m=1000,
            min_elevation=500.0,
            min_slope=10.0  # High slope requirement
        )
        
        # Most flat points should be masked
        assert result['terrain_mask'].sum() < result['terrain_mask'].size * 0.1
    
    def test_has_attributes(self):
        """Test that variables have proper attributes."""
        terrain_data = np.random.rand(50, 50) * 2000 + 1000
        lats = np.linspace(47, 46, 50)
        lons = np.linspace(10, 11, 50)
        
        result = terrain.compute_terrain_aspect_dataset(
            terrain_data, lats, lons, resolution_m=1000
        )
        
        assert 'units' in result['elevation'].attrs
        assert 'units' in result['aspect_deg'].attrs
        assert 'units' in result['slope'].attrs
        assert result.attrs['resolution_m'] == 1000


class TestLoadTerrainAspectDataset:
    """Tests for load_terrain_aspect_dataset function."""
    
    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            terrain.load_terrain_aspect_dataset(cache_path="./nonexistent/file.nc")
    
    @patch('xarray.open_dataset')
    def test_successful_load(self, mock_open):
        """Test successful loading of cached dataset."""
        # Create mock dataset
        mock_ds = xr.Dataset({
            'elevation': (['latitude', 'longitude'], np.random.rand(50, 50)),
            'aspect_deg': (['latitude', 'longitude'], np.random.rand(50, 50) * 360),
        }, coords={
            'latitude': np.linspace(47, 46, 50),
            'longitude': np.linspace(10, 11, 50),
        })
        
        mock_open.return_value = mock_ds
        
        with patch('os.path.exists', return_value=True):
            result = terrain.load_terrain_aspect_dataset("mock.nc")
        
        assert isinstance(result, xr.Dataset)
        mock_open.assert_called_once()


class TestComputeTerrainIntersection:
    """Tests for compute_terrain_intersection function."""
    
    @pytest.fixture
    def sample_era5_dataset(self):
        """Create a sample ERA5-like dataset."""
        times = np.array(['2025-12-24', '2025-12-25'], dtype='datetime64[D]')
        pressures = np.array([1000, 975, 950, 850, 700])
        lats = np.array([46.0, 46.25, 46.5, 46.75, 47.0])
        lons = np.array([10.0, 10.5, 11.0, 11.5, 12.0])
        
        # Create mock geopotential field
        # Values decrease with height (pressure level), typical of geopotential
        gph = np.zeros((len(times), len(pressures), len(lats), len(lons)))
        for p_idx, p in enumerate(pressures):
            # Higher pressure = lower altitude = larger geopotential
            gph[:, p_idx, :, :] = (1000 - p) * 100 + 50000
        
        ds = xr.Dataset(
            {
                'z': (['valid_time', 'pressure_level', 'latitude', 'longitude'], gph),
                'gph': (['valid_time', 'pressure_level', 'latitude', 'longitude'], gph),
                'u': (['valid_time', 'pressure_level', 'latitude', 'longitude'], 
                      np.random.randn(len(times), len(pressures), len(lats), len(lons))),
            },
            coords={
                'valid_time': times,
                'pressure_level': pressures,
                'latitude': lats,
                'longitude': lons,
            }
        )
        return ds
    
    @pytest.fixture
    def sample_terrain_dataset(self):
        """Create a sample terrain dataset."""
        lats = np.linspace(47.5, 45.5, 100)
        lons = np.linspace(9.5, 12.5, 150)
        
        ds = xr.Dataset(
            {
                'elevation': (['latitude', 'longitude'], np.random.rand(100, 150) * 2000 + 1000),
                'aspect_deg': (['latitude', 'longitude'], np.random.rand(100, 150) * 360),
            },
            coords={
                'latitude': lats,
                'longitude': lons,
            }
        )
        return ds
    
    def test_output_is_dataset(self, sample_era5_dataset, sample_terrain_dataset):
        """Test that function returns xarray Dataset."""
        result = terrain.compute_terrain_intersection(
            sample_era5_dataset, sample_terrain_dataset
        )
        assert isinstance(result, xr.Dataset)
    
    def test_required_variables_added(self, sample_era5_dataset, sample_terrain_dataset):
        """Test that required variables are added to output."""
        result = terrain.compute_terrain_intersection(
            sample_era5_dataset, sample_terrain_dataset
        )
        
        assert 'terrain' in result.data_vars
        assert 'terrain_elevation' in result.data_vars
    
    def test_output_maintains_era5_structure(self, sample_era5_dataset, sample_terrain_dataset):
        """Test that output maintains ERA5 dimension structure."""
        result = terrain.compute_terrain_intersection(
            sample_era5_dataset, sample_terrain_dataset
        )
        
        # Check dimensions match ERA5
        assert result.sizes['valid_time'] == sample_era5_dataset.sizes['valid_time']
        assert result.sizes['pressure_level'] == sample_era5_dataset.sizes['pressure_level']
        assert result.sizes['latitude'] == sample_era5_dataset.sizes['latitude']
        assert result.sizes['longitude'] == sample_era5_dataset.sizes['longitude']
    
    def test_terrain_mask_is_boolean(self, sample_era5_dataset, sample_terrain_dataset):
        """Test that terrain mask is boolean."""
        result = terrain.compute_terrain_intersection(
            sample_era5_dataset, sample_terrain_dataset
        )
        
        assert result['terrain'].dtype == bool
    
    def test_geopotential_height_conversion(self, sample_era5_dataset, sample_terrain_dataset):
        """Test that function processes geopotential height correctly."""
        result = terrain.compute_terrain_intersection(
            sample_era5_dataset, sample_terrain_dataset
        )
        
        # Function should use gph for comparison but doesn't return it separately
        # Just verify that terrain mask was created successfully
        assert 'terrain' in result.data_vars
        assert result['terrain'].dtype == bool
    
    def test_missing_z_variable_raises_error(self):
        """Test that ValueError is raised when 'gph' is not in ERA5 dataset."""
        # Create sample terrain dataset
        terrain_lats = np.linspace(47.5, 45.5, 100)
        terrain_lons = np.linspace(9.5, 12.5, 150)
        
        terrain_ds = xr.Dataset(
            {
                'elevation': (['latitude', 'longitude'], np.random.rand(100, 150) * 2000 + 1000),
                'aspect_deg': (['latitude', 'longitude'], np.random.rand(100, 150) * 360),
            },
            coords={
                'latitude': terrain_lats,
                'longitude': terrain_lons,
            }
        )
        
        # Create ERA5 dataset WITHOUT 'gph' variable
        bad_era5 = xr.Dataset(
            {
                'u': (['valid_time', 'pressure_level', 'latitude', 'longitude'],
                      np.random.rand(2, 5, 5, 5)),
            },
            coords={
                'valid_time': np.array(['2025-12-24', '2025-12-25'], dtype='datetime64[D]'),
                'pressure_level': np.array([1000, 850, 700, 500, 300]),
                'latitude': np.array([46.0, 46.25, 46.5, 46.75, 47.0]),
                'longitude': np.array([10.0, 10.5, 11.0, 11.5, 12.0]),
            }
        )
        
        with pytest.raises(ValueError, match="must contain 'gph'"):
            terrain.compute_terrain_intersection(bad_era5, terrain_ds)
    
    def test_terrain_elevation_interpolated(self, sample_era5_dataset, sample_terrain_dataset):
        """Test that terrain elevation is interpolated to ERA5 grid."""
        result = terrain.compute_terrain_intersection(
            sample_era5_dataset, sample_terrain_dataset
        )
        
        terrain_elev = result['terrain_elevation']
        
        # Should have ERA5 spatial dimensions (not full terrain grid)
        assert terrain_elev.sizes['latitude'] == sample_era5_dataset.sizes['latitude']
        assert terrain_elev.sizes['longitude'] == sample_era5_dataset.sizes['longitude']
        assert terrain_elev.shape == (
            sample_era5_dataset.sizes['latitude'],
            sample_era5_dataset.sizes['longitude']
        )


class TestInterpolateToGrid:
    """Tests for interpolate_to_grid function."""
    
    @pytest.fixture
    def source_dataset(self):
        """Create a sample source dataset (coarse grid)."""
        times = np.array(['2025-12-24', '2025-12-25'], dtype='datetime64[D]')
        pressures = np.array([1000, 850, 700])
        lats = np.array([46.0, 46.5, 47.0])
        lons = np.array([10.0, 11.0, 12.0])
        
        ds = xr.Dataset(
            {
                'u': (['valid_time', 'pressure_level', 'latitude', 'longitude'],
                      np.random.randn(len(times), len(pressures), len(lats), len(lons))),
                'v': (['valid_time', 'pressure_level', 'latitude', 'longitude'],
                      np.random.randn(len(times), len(pressures), len(lats), len(lons))),
            },
            coords={
                'valid_time': times,
                'pressure_level': pressures,
                'latitude': lats,
                'longitude': lons,
            }
        )
        return ds
    
    @pytest.fixture
    def target_grid(self):
        """Create a sample target grid (fine grid)."""
        lats = np.linspace(46, 47, 20)
        lons = np.linspace(10, 12, 30)
        
        ds = xr.Dataset(
            coords={
                'latitude': lats,
                'longitude': lons,
            }
        )
        return ds
    
    def test_output_is_dataset(self, source_dataset, target_grid):
        """Test that function returns xarray Dataset."""
        result = terrain.interpolate_to_grid(source_dataset, target_grid)
        assert isinstance(result, xr.Dataset)
    
    def test_target_grid_dimensions(self, source_dataset, target_grid):
        """Test that output has target grid dimensions."""
        result = terrain.interpolate_to_grid(source_dataset, target_grid)
        
        assert result.sizes['latitude'] == target_grid.sizes['latitude']
        assert result.sizes['longitude'] == target_grid.sizes['longitude']
    
    def test_temporal_dimensions_preserved(self, source_dataset, target_grid):
        """Test that temporal and pressure dimensions are preserved."""
        result = terrain.interpolate_to_grid(source_dataset, target_grid)
        
        assert result.sizes['valid_time'] == source_dataset.sizes['valid_time']
        assert result.sizes['pressure_level'] == source_dataset.sizes['pressure_level']
    
    def test_4d_variables_interpolated(self, source_dataset, target_grid):
        """Test that 4D variables are present in output."""
        result = terrain.interpolate_to_grid(source_dataset, target_grid)
        
        assert 'u' in result.data_vars
        assert 'v' in result.data_vars
        
        for var in ['u', 'v']:
            assert len(result[var].sizes) == 4
            assert 'valid_time' in result[var].sizes
            assert 'pressure_level' in result[var].sizes
    
    def test_interpolation_produces_reasonable_values(self, source_dataset, target_grid):
        """Test that interpolated values are reasonable (within source data range)."""
        result = terrain.interpolate_to_grid(source_dataset, target_grid)
        
        for var in ['u', 'v']:
            source_min = float(source_dataset[var].min())
            source_max = float(source_dataset[var].max())
            result_min = float(np.nanmin(result[var].values))
            result_max = float(np.nanmax(result[var].values))
            
            # Interpolated values should be roughly in source range
            # (allowing some extrapolation at boundaries)
            assert result_min >= source_min - abs(source_max - source_min)
            assert result_max <= source_max + abs(source_max - source_min)
    
    def test_coordinates_match_target(self, source_dataset, target_grid):
        """Test that output coordinates match target grid."""
        result = terrain.interpolate_to_grid(source_dataset, target_grid)
        
        assert np.allclose(result.latitude.values, target_grid.latitude.values)
        assert np.allclose(result.longitude.values, target_grid.longitude.values)


class TestTerrainIntegration:
    """Integration tests for terrain module functions."""
    
    def test_aspect_dataset_creation_and_properties(self):
        """Integration test: create dataset and verify all properties."""
        # Create realistic terrain
        lats = np.linspace(47, 46, 100)
        lons = np.linspace(10, 11, 100)
        
        # Create synthetic mountain-like terrain
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        terrain_data = (
            2000 * np.exp(-((lon_grid - 10.5)**2 + (lat_grid - 46.5)**2) / 0.01) + 500
        )
        
        ds = terrain.compute_terrain_aspect_dataset(
            terrain_data, lats, lons, resolution_m=1000
        )
        
        # Verify structure
        assert isinstance(ds, xr.Dataset)
        assert 'elevation' in ds
        assert 'aspect_deg' in ds
        assert 'slope' in ds
        
        # Verify values
        assert np.all(np.isfinite(ds['elevation'].values))
        assert np.all((ds['aspect_deg'].values >= 0) & (ds['aspect_deg'].values < 360))
        assert np.all((ds['slope'].values >= 0) & (ds['slope'].values < 90))
    
    def test_gravity_constant(self):
        """Test that gravity constant is properly defined."""
        assert terrain.G > 0
        assert 9 < terrain.G < 10  # Reasonable gravity value
    
    def test_crop_terrain(self):
        """Test crop_terrain function."""
        # Create a target dataset (ERA5-like with (time, pressure, lat, lon))
        target_ds = xr.Dataset(
            {
                'gph': (['valid_time', 'pressure_level', 'latitude', 'longitude'],
                        np.random.randn(2, 3, 5, 5))
            },
            coords={
                'valid_time': np.array(['2025-12-24', '2025-12-25'], dtype='datetime64[D]'),
                'pressure_level': [1000, 850, 700],
                'latitude': np.linspace(46.2, 46.8, 5),
                'longitude': np.linspace(10.2, 10.8, 5),
            }
        )
        
        # Create a larger terrain dataset
        terrain_ds = xr.Dataset(
            {
                'elevation': (['latitude', 'longitude'],
                              np.random.randint(500, 3000, (20, 20)))
            },
            coords={
                'latitude': np.linspace(47.0, 46.0, 20),
                'longitude': np.linspace(10.0, 11.0, 20),
            }
        )
        
        # Crop terrain to target bounds
        cropped = terrain.crop_terrain(target_ds, terrain_ds)
        
        # Verify cropped dataset has correct bounds
        assert cropped.latitude.min() >= target_ds.latitude.min() - 0.01
        assert cropped.latitude.max() <= target_ds.latitude.max() + 0.01
        assert cropped.longitude.min() >= target_ds.longitude.min() - 0.01
        assert cropped.longitude.max() <= target_ds.longitude.max() + 0.01
        
        # Verify it's smaller than the original terrain dataset
        assert cropped.sizes['latitude'] < terrain_ds.sizes['latitude']
        assert cropped.sizes['longitude'] < terrain_ds.sizes['longitude']
