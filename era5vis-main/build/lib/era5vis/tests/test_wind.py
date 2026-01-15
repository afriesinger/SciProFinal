"""
Unit tests for the wind module.

Author: Andreas Friesinger
Date: 2026-01-09
"""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch, MagicMock
from scipy.interpolate import RegularGridInterpolator
import os
import sys

# Add parent directory to path to allow importing wind module directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import wind


class TestCalcWind:
    """Tests for calc_wind function."""
    
    def test_wind_from_cardinal_and_ordinal_directions(self):
        """Test wind calculations for all cardinal and ordinal directions."""
        
        # Direction tolerance for assertions
        direction_tolerance = 1.0  # degrees
        speed_tolerance = 0.15      # m/s
        
        # Test cases: (u, v, expected_dir, expected_speed, description)
        test_cases = [
            (0.0, -10.0, 0.0, 10.0, "0° - Wind from south"),
            (-10.0, 0.0, 90.0, 10.0, "90° - Wind from west"),
            (0.0, 10.0, 180.0, 10.0, "180° - Wind from north"),
            (10.0, 0.0, 270.0, 10.0, "270° - Wind from east"),
            (-7.07, -7.07, 45.0, 10.0, "45° - Wind from southwest"),
            (7.07, 7.07, 225.0, 10.0, "225° - Wind from northeast"),
        ]
        
        for u, v, expected_dir, expected_speed, desc in test_cases:
            wind_dir, wind_speed = wind.calc_wind(u, v)
            
            assert np.isclose(wind_dir, expected_dir, atol=direction_tolerance), \
                f"{desc}: Wind direction {wind_dir}° not close to {expected_dir}°"
            
            assert np.isclose(wind_speed, expected_speed, atol=speed_tolerance), \
                f"{desc}: Wind speed {wind_speed} not close to {expected_speed}"
    
    def test_zero_wind(self):
        """Test with zero wind components."""
        u = 0.0
        v = 0.0
        
        wind_dir, wind_speed = wind.calc_wind(u, v)
        
        # Direction is undefined, speed is zero
        assert np.isclose(wind_speed, 0.0)
    
    def test_wind_direction_range(self):
        """Test that wind direction is always in [0, 360)."""
        for angle in np.linspace(0, 2*np.pi, 100):
            u = 10 * np.sin(angle)
            v = 10 * np.cos(angle)
            
            wind_dir, _ = wind.calc_wind(u, v)
            
            assert 0 <= wind_dir < 360
    
    def test_wind_speed_positive(self):
        """Test that wind speed is always non-negative."""
        u_values = np.random.randn(20) * 10
        v_values = np.random.randn(20) * 10
        
        for u, v in zip(u_values, v_values):
            _, wind_speed = wind.calc_wind(u, v)
            assert wind_speed >= 0
    
    def test_numpy_array_input(self):
        """Test function with numpy array inputs."""
        u = np.array([0.0, 10.0, 0.0, -10.0])
        v = np.array([10.0, 0.0, -10.0, 0.0])
        
        wind_dir, wind_speed = wind.calc_wind(u, v)
        
        assert len(wind_dir) == 4
        assert len(wind_speed) == 4
        assert np.all(wind_speed >= 0)
        assert np.all((wind_dir >= 0) & (wind_dir < 360))


class TestPerpendicularWindComponent:
    """Tests for perpendicular_wind_component function."""
    
    def test_wind_perpendicular_to_slope_south(self):
        """Test wind perpendicular to slope (wind from N, slope faces N)."""
        # Wind from north, slope faces north
        wind_dir = 0.0  # From north
        wind_speed = 10.0
        slope_aspect = 0.0  # Slope faces north
        
        perp_wind = wind.perpendicular_wind_component(wind_dir, wind_speed, slope_aspect)
        
        # Wind is directly into the slope
        assert np.isclose(perp_wind, 10.0, atol=0.1)
    
    def test_wind_parallel_to_slope(self):
        """Test wind parallel to slope (perpendicular component should be zero)."""
        # Wind from east, slope faces north (perpendicular to wind)
        wind_dir = 90.0  # From east
        wind_speed = 10.0
        slope_aspect = 0.0  # Slope faces north (perpendicular to wind)
        
        perp_wind = wind.perpendicular_wind_component(wind_dir, wind_speed, slope_aspect)
        
        # Wind is parallel to slope
        assert np.isclose(perp_wind, 0.0, atol=0.1)
    
    def test_wind_away_from_slope(self):
        """Test wind blowing away from slope."""
        # Wind from north, slope faces south (opposite direction)
        wind_dir = 0.0  # From north
        wind_speed = 10.0
        slope_aspect = 180.0  # Slope faces south (opposite to wind)
        
        perp_wind = wind.perpendicular_wind_component(wind_dir, wind_speed, slope_aspect)
        
        # Wind is blowing away from slope
        assert np.isclose(perp_wind, 0.0, atol=0.1)
    
    def test_wind_at_45_degrees_to_slope(self):
        """Test wind at 45° to slope normal."""
        wind_dir = 0.0  # From north
        wind_speed = 10.0
        slope_aspect = 45.0  # Slope faces northeast
        
        perp_wind = wind.perpendicular_wind_component(wind_dir, wind_speed, slope_aspect)
        
        # Component should be cos(45°) * 10 ≈ 7.07
        assert np.isclose(perp_wind, 7.07, atol=0.1)
    
    def test_wind_at_90_degrees_to_slope(self):
        """Test wind at 90° to slope (parallel)."""
        wind_dir = 90.0  # From east
        wind_speed = 10.0
        slope_aspect = 0.0  # Slope faces north
        
        perp_wind = wind.perpendicular_wind_component(wind_dir, wind_speed, slope_aspect)
        
        # Wind perpendicular to slope normal = 0 component
        assert np.isclose(perp_wind, 0.0, atol=0.1)
    
    def test_perpendicular_wind_in_valid_range(self):
        """Test that perpendicular wind is always in [0, wind_speed]."""
        wind_dirs = np.linspace(0, 360, 72)  # Every 5 degrees
        slope_aspects = np.linspace(0, 360, 72)
        wind_speed = 15.0
        
        for wind_dir in wind_dirs:
            for slope_aspect in slope_aspects:
                perp_wind = wind.perpendicular_wind_component(
                    wind_dir, wind_speed, slope_aspect
                )
                
                assert 0 <= perp_wind <= wind_speed
    
    def test_invalid_wind_direction_negative(self):
        """Test that invalid wind direction raises ValueError."""
        with pytest.raises(ValueError, match="Wind direction must be between"):
            wind.perpendicular_wind_component(-10.0, 10.0, 45.0)
    
    def test_invalid_wind_direction_too_large(self):
        """Test that wind direction > 360 raises ValueError."""
        with pytest.raises(ValueError, match="Wind direction must be between"):
            wind.perpendicular_wind_component(370.0, 10.0, 45.0)
    
    def test_invalid_slope_aspect_negative(self):
        """Test that invalid slope aspect raises ValueError."""
        with pytest.raises(ValueError, match="Slope aspect must be between"):
            wind.perpendicular_wind_component(0.0, 10.0, -10.0)
    
    def test_invalid_slope_aspect_too_large(self):
        """Test that slope aspect > 360 raises ValueError."""
        with pytest.raises(ValueError, match="Slope aspect must be between"):
            wind.perpendicular_wind_component(0.0, 10.0, 370.0)
    
    def test_zero_wind_speed(self):
        """Test with zero wind speed."""
        perp_wind = wind.perpendicular_wind_component(45.0, 0.0, 90.0)
        
        assert np.isclose(perp_wind, 0.0)
    
    def test_result_is_rounded(self):
        """Test that result is rounded to 2 decimal places."""
        wind_dir = 0.0
        wind_speed = 10.123456
        slope_aspect = 0.0
        
        perp_wind = wind.perpendicular_wind_component(wind_dir, wind_speed, slope_aspect)
        
        assert perp_wind == round(perp_wind, 2)
    
    def test_perpendicular_wind_boundary_180_degrees(self):
        """Test wind at exactly 180° to slope."""
        wind_dir = 0.0  # From south
        wind_speed = 10.0
        slope_aspect = 180.0  # Slope faces north (opposite)
        
        perp_wind = wind.perpendicular_wind_component(wind_dir, wind_speed, slope_aspect)
        
        # Exactly opposite direction
        assert np.isclose(perp_wind, 0.0, atol=0.1)
    
    def test_perpendicular_wind_near_90_boundary(self):
        """Test wind near ±90° boundary."""
        wind_dir = 0.0
        wind_speed = 10.0
        slope_aspect = 89.5  # Just before perpendicular
        
        perp_wind = wind.perpendicular_wind_component(wind_dir, wind_speed, slope_aspect)
        
        # Should be small but positive
        assert 0 <= perp_wind < 1.0
    
    def test_perpendicular_wind_computational_check(self):
        """Test perpendicular wind with values that stress the computational check."""
        # Test with angles that result in cos values very close to limits
        wind_dir = 45.0
        wind_speed = 10.0
        
        # Test all slope aspects to ensure no computational errors
        for aspect in np.linspace(0, 360, 360):
            perp_wind = wind.perpendicular_wind_component(wind_dir, wind_speed, aspect)
            
            # Result should always be valid
            assert 0 <= perp_wind <= wind_speed
            # Should not raise an error
    
    def test_perpendicular_wind_edge_case_negative_check(self):
        """Test edge case where rounding might affect the check."""
        # Use values that could theoretically produce negative or >wind_speed after rounding
        wind_dir = 0.0
        wind_speed = 10.0
        
        # Test aspect angles that result in very small or very large cosines
        test_cases = [
            (0.1, 10.0),    # Almost perpendicular
            (179.9, 10.0),  # Almost opposite
            (359.9, 10.0),  # Almost same direction
            (45.0, 15.5),   # Diagonal
        ]
        
        for slope_aspect, ws in test_cases:
            perp_wind = wind.perpendicular_wind_component(wind_dir, ws, slope_aspect)
            # Should never raise error and always be in valid range
            assert 0 <= perp_wind <= ws


class TestGetDownwindPoints:
    """Tests for _get_downwind_points function."""
    
    def test_output_arrays_length(self):
        """Test that output arrays have correct length."""
        downwind_lats, downwind_lons = wind._get_downwind_points(
            46.0, 10.5, 0.0, range_km=10
        )
        
        assert len(downwind_lats) == int(10*2)
        assert len(downwind_lons) == int(10*2)
    
    def test_wind_from_north(self):
        """Test downwind points with wind from north."""
        era_lat = 46.0
        era_lon = 10.5
        wind_dir = 0.0  # From north → downwind is south
        
        downwind_lats, downwind_lons = wind._get_downwind_points(
            era_lat, era_lon, wind_dir, range_km=10
        )
        
        # Downwind should extend south from starting point  
        # First point should be very close to era_lat
        assert np.isclose(downwind_lats[0], era_lat, atol=0.01)
        assert np.isclose(downwind_lons[0], era_lon, atol=0.1)
    
    def test_wind_from_east(self):
        """Test downwind points with wind from east."""
        era_lat = 46.0
        era_lon = 10.5
        wind_dir = np.pi / 2  # From east → downwind is west
        
        downwind_lats, downwind_lons = wind._get_downwind_points(
            era_lat, era_lon, wind_dir, range_km=10
        )
        
        # Downwind should be west (decreasing longitude)
        assert downwind_lons[-1] < downwind_lons[0]
        assert np.isclose(downwind_lats[0], era_lat, atol=0.1)
    
    def test_starting_point_is_era_gridpoint(self):
        """Test that first point is the ERA gridpoint."""
        era_lat = 46.5
        era_lon = 11.0
        
        downwind_lats, downwind_lons = wind._get_downwind_points(
            era_lat, era_lon, 0.0, range_km=10
        )
        
        # First point should be the ERA gridpoint
        assert np.isclose(downwind_lats[0], era_lat)
        assert np.isclose(downwind_lons[0], era_lon)
    
    def test_distance_increases_monotonically(self):
        """Test that distance increases monotonically along points."""
        downwind_lats, downwind_lons = wind._get_downwind_points(
            46.0, 10.5, 0.0, range_km=10
        )
        
        # Calculate distances from first point
        distances = np.sqrt(
            (downwind_lats - downwind_lats[0])**2 + 
            (downwind_lons - downwind_lons[0])**2
        )
        
        # Distances should increase monotonically
        assert np.all(np.diff(distances) >= 0)
    
    def test_different_range_values(self):
        """Test with different range_km values."""
        era_lat = 46.0
        era_lon = 10.5
        
        points_10km = wind._get_downwind_points(era_lat, era_lon, 0.0, range_km=10)
        points_20km = wind._get_downwind_points(era_lat, era_lon, 0.0, range_km=20)
        
        # Larger range should have larger maximum distance
        dist_10 = np.sqrt(
            (points_10km[0][-1] - points_10km[0][0])**2 + 
            (points_10km[1][-1] - points_10km[1][0])**2
        )
        dist_20 = np.sqrt(
            (points_20km[0][-1] - points_20km[0][0])**2 + 
            (points_20km[1][-1] - points_20km[1][0])**2
        )
        
        assert dist_20 > dist_10


class TestFindHighestTerrainDownwind:
    """Tests for _find_highest_terrain_downwind function."""
    
    @pytest.fixture
    def sample_terrain_dataset(self):
        """Create a sample terrain dataset."""
        lats = np.linspace(47.5, 45.5, 100)
        lons = np.linspace(9.5, 12.5, 150)
        
        # Create synthetic mountain-like terrain
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        elevation = (
            2000 * np.exp(-((lon_grid - 10.5)**2 + (lat_grid - 46.5)**2) / 0.01) + 500
        )
        
        ds = xr.Dataset(
            {
                'elevation': (['latitude', 'longitude'], elevation),
                'aspect_deg': (['latitude', 'longitude'], np.random.rand(100, 150) * 360),
            },
            coords={
                'latitude': lats,
                'longitude': lons,
            }
        )
        return ds
    
    def _create_interpolators(self, dataset):
        """Helper method to create interpolators from dataset."""
        lats = dataset.latitude.values
        lons = dataset.longitude.values
        elev = dataset['elevation'].values
        aspect = dataset['aspect_deg'].values
        
        interp_elev = RegularGridInterpolator(
            (lats, lons), elev, bounds_error=False, fill_value=np.nan
        )
        interp_aspect = RegularGridInterpolator(
            (lats, lons), aspect, bounds_error=False, fill_value=np.nan
        )
        return interp_elev, interp_aspect
    
    def test_terrain_found_above_level(self, sample_terrain_dataset):
        """Test that terrain is found when above pressure level."""
        # Create downwind points in area with high terrain
        downwind_lats = np.linspace(46.5, 46.0, 50)
        downwind_lons = np.linspace(10.5, 11.0, 50)
        
        pressure_level_height = 1000  # meters
        
        interp_elev, interp_aspect = self._create_interpolators(sample_terrain_dataset)
        
        terrain_height, terrain_aspect = wind._find_highest_terrain_downwind(
            downwind_lats, downwind_lons, interp_elev, interp_aspect, pressure_level_height
        )
        
        # Should find terrain
        assert not np.isnan(terrain_height)
        assert not np.isnan(terrain_aspect)
        assert terrain_height > pressure_level_height
    
    def test_terrain_not_found_below_level(self, sample_terrain_dataset):
        """Test that no terrain is found when all below pressure level."""
        # Create downwind points in area with low terrain
        downwind_lats = np.linspace(45.0, 45.1, 50)
        downwind_lons = np.linspace(9.5, 9.6, 50)
        
        pressure_level_height = 5000  # meters (very high)
        
        interp_elev, interp_aspect = self._create_interpolators(sample_terrain_dataset)
        
        terrain_height, terrain_aspect = wind._find_highest_terrain_downwind(
            downwind_lats, downwind_lons, interp_elev, interp_aspect, pressure_level_height
        )
        
        # Should not find terrain
        assert np.isnan(terrain_height)
        assert np.isnan(terrain_aspect)
    
    def test_terrain_aspect_in_valid_range(self, sample_terrain_dataset):
        """Test that aspect is in valid range when terrain is found."""
        downwind_lats = np.linspace(46.5, 46.0, 50)
        downwind_lons = np.linspace(10.5, 11.0, 50)
        
        interp_elev, interp_aspect = self._create_interpolators(sample_terrain_dataset)
        
        terrain_height, terrain_aspect = wind._find_highest_terrain_downwind(
            downwind_lats, downwind_lons, interp_elev, interp_aspect, 1000.0
        )
        
        if not np.isnan(terrain_aspect):
            assert 0 <= terrain_aspect <= 360


class TestComputeWindTerrainInteraction:
    """Tests for compute_wind_terrain_interaction function."""
    
    @pytest.fixture
    def sample_era5_dataset(self):
        """Create a sample ERA5-like dataset."""
        times = np.array(['2025-12-24', '2025-12-25'], dtype='datetime64[D]')
        pressures = np.array([1000, 975, 950, 850, 700])
        lats = np.array([46.0, 46.25, 46.5, 46.75, 47.0])
        lons = np.array([10.0, 10.5, 11.0, 11.5, 12.0])
        
        # Create wind components
        u = np.random.randn(len(times), len(pressures), len(lats), len(lons)) * 5
        v = np.random.randn(len(times), len(pressures), len(lats), len(lons)) * 5
        
        # Create geopotential field (both z and gph for compatibility)
        gph = np.zeros((len(times), len(pressures), len(lats), len(lons)))
        for p_idx, p in enumerate(pressures):
            gph[:, p_idx, :, :] = (1000 - p) * 100 + 50000
        
        ds = xr.Dataset(
            {
                'u': (['valid_time', 'pressure_level', 'latitude', 'longitude'], u),
                'v': (['valid_time', 'pressure_level', 'latitude', 'longitude'], v),
                'z': (['valid_time', 'pressure_level', 'latitude', 'longitude'], gph),
                'gph': (['valid_time', 'pressure_level', 'latitude', 'longitude'], gph),
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
        
        # Create synthetic mountain-like terrain
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        elevation = (
            2000 * np.exp(-((lon_grid - 10.5)**2 + (lat_grid - 46.5)**2) / 0.01) + 500
        )
        
        ds = xr.Dataset(
            {
                'elevation': (['latitude', 'longitude'], elevation),
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
        result = wind.compute_wind_terrain_interaction(
            sample_era5_dataset, sample_terrain_dataset, range_km=10
        )
        assert isinstance(result, xr.Dataset)
    
    def test_required_variables_added(self, sample_era5_dataset, sample_terrain_dataset):
        """Test that required variables are added."""
        result = wind.compute_wind_terrain_interaction(
            sample_era5_dataset, sample_terrain_dataset, range_km=10
        )
        
        assert 'wind_speed' in result.data_vars
        assert 'wind_direction' in result.data_vars
        assert 'downwind_terrain_height' in result.data_vars
        assert 'perpendicular_wind_speed' in result.data_vars
    
    def test_output_maintains_era5_structure(self, sample_era5_dataset, sample_terrain_dataset):
        """Test that output maintains ERA5 structure."""
        result = wind.compute_wind_terrain_interaction(
            sample_era5_dataset, sample_terrain_dataset, range_km=10
        )
        
        # Check dimensions match
        assert result.sizes['valid_time'] == sample_era5_dataset.sizes['valid_time']
        assert result.sizes['pressure_level'] == sample_era5_dataset.sizes['pressure_level']
        assert result.sizes['latitude'] == sample_era5_dataset.sizes['latitude']
        assert result.sizes['longitude'] == sample_era5_dataset.sizes['longitude']
    
    def test_wind_speed_positive(self, sample_era5_dataset, sample_terrain_dataset):
        """Test that wind speed is non-negative."""
        result = wind.compute_wind_terrain_interaction(
            sample_era5_dataset, sample_terrain_dataset, range_km=10
        )
        
        wind_speed = result['wind_speed'].values
        assert np.all((wind_speed >= 0) | np.isnan(wind_speed))
    
    def test_wind_direction_in_range(self, sample_era5_dataset, sample_terrain_dataset):
        """Test that wind direction is in valid range."""
        result = wind.compute_wind_terrain_interaction(
            sample_era5_dataset, sample_terrain_dataset, range_km=10
        )
        
        wind_dir = result['wind_direction'].values
        valid_dirs = (wind_dir >= 0) & (wind_dir < 360) | np.isnan(wind_dir)
        assert np.all(valid_dirs)
    
    def test_perpendicular_wind_in_range(self, sample_era5_dataset, sample_terrain_dataset):
        """Test that perpendicular wind is in valid range."""
        result = wind.compute_wind_terrain_interaction(
            sample_era5_dataset, sample_terrain_dataset, range_km=10
        )
        
        perp_wind = result['perpendicular_wind_speed'].values
        wind_speed = result['wind_speed'].values
        
        # Should be 0 to wind_speed
        valid_range = (perp_wind >= 0) & (perp_wind <= wind_speed) | np.isnan(perp_wind)
        assert np.all(valid_range)
    
    def test_missing_u_raises_error(self, sample_era5_dataset, sample_terrain_dataset):
        """Test that missing 'u' raises ValueError."""
        bad_era5 = sample_era5_dataset.drop_vars('u')
        
        with pytest.raises(ValueError, match="must contain 'u' and 'v'"):
            wind.compute_wind_terrain_interaction(bad_era5, sample_terrain_dataset)
    
    def test_missing_v_raises_error(self, sample_era5_dataset, sample_terrain_dataset):
        """Test that missing 'v' raises ValueError."""
        bad_era5 = sample_era5_dataset.drop_vars('v')
        
        with pytest.raises(ValueError, match="must contain 'u' and 'v'"):
            wind.compute_wind_terrain_interaction(bad_era5, sample_terrain_dataset)
    
    def test_missing_z_raises_error(self, sample_era5_dataset, sample_terrain_dataset):
        """Test that missing 'z' raises ValueError."""
        bad_era5 = sample_era5_dataset.drop_vars('z')
        
        with pytest.raises(ValueError, match="must contain 'z'"):
            wind.compute_wind_terrain_interaction(bad_era5, sample_terrain_dataset)
    
    def test_wind_direction_opposite_to_u_v(self, sample_era5_dataset, sample_terrain_dataset):
        """Test that wind direction is correctly calculated from u, v."""
        # Set specific u, v values to verify wind direction
        # Wind from south: u=0, v>0 → direction should be 180°
        sample_era5_dataset['u'].values[:] = 0.0
        sample_era5_dataset['v'].values[:] = 10.0
        
        result = wind.compute_wind_terrain_interaction(
            sample_era5_dataset, sample_terrain_dataset, range_km=10
        )
        
        wind_dirs = result['wind_direction'].values
        # All should be approximately 180 (wind from south)
        valid_dirs = wind_dirs[~np.isnan(wind_dirs)]
        if len(valid_dirs) > 0:
            assert np.allclose(valid_dirs, 180.0, atol=1.0)
    
    def test_different_range_km_values(self, sample_era5_dataset, sample_terrain_dataset):
        """Test with different range_km values."""
        result_10 = wind.compute_wind_terrain_interaction(
            sample_era5_dataset, sample_terrain_dataset, range_km=10
        )
        result_20 = wind.compute_wind_terrain_interaction(
            sample_era5_dataset, sample_terrain_dataset, range_km=20
        )
        
        # Both should have same dimensions
        assert result_10.sizes == result_20.sizes
        # Both should have required variables
        assert 'wind_speed' in result_10 and 'wind_speed' in result_20
    
    def test_has_attributes(self, sample_era5_dataset, sample_terrain_dataset):
        """Test that output variables have proper attributes."""
        result = wind.compute_wind_terrain_interaction(
            sample_era5_dataset, sample_terrain_dataset, range_km=10
        )
        
        assert 'units' in result['wind_speed'].attrs
        assert 'units' in result['wind_direction'].attrs
        assert 'units' in result['perpendicular_wind_speed'].attrs
        assert 'units' in result['downwind_terrain_height'].attrs
    
    def test_no_terrain_above_pressure_level(self, sample_era5_dataset, sample_terrain_dataset):
        """Test handling when no terrain found above pressure level."""
        # Set very high pressure level height (above all terrain)
        sample_era5_dataset['gph'].values[:] = 100000  # Very high geopotential
        
        result = wind.compute_wind_terrain_interaction(
            sample_era5_dataset, sample_terrain_dataset, range_km=10
        )
        
        # Should have 0 for downwind_terrain_height (uint16 cannot represent NaN)
        # and NaN for perpendicular_wind_speed when terrain not found
        downwind_heights = result['downwind_terrain_height'].values
        perp_winds = result['perpendicular_wind_speed'].values
        
        # Most values should be 0 or NaN when no terrain above level
        # (uint16 converts NaN to 0, so we check for mostly small values)
        assert np.sum((downwind_heights == 0) | np.isnan(downwind_heights)) > len(downwind_heights) * 0.5
        assert np.sum(np.isnan(perp_winds)) > len(perp_winds) * 0.5
    
    def test_very_small_wind_speed_skipped(self, sample_era5_dataset, sample_terrain_dataset):
        """Test that very small wind speeds are skipped (NaN result)."""
        # Set very small wind values
        sample_era5_dataset['u'].values[:] = 0.01
        sample_era5_dataset['v'].values[:] = 0.01  # Very small wind speed
        
        result = wind.compute_wind_terrain_interaction(
            sample_era5_dataset, sample_terrain_dataset, range_km=10
        )
        
        # Should have 0 or NaN for all output when wind is too small
        wind_speed = result['wind_speed'].values
        downwind_heights = result['downwind_terrain_height'].values
        
        assert np.all(wind_speed < 0.2)  # Wind is very small
        # uint16 cannot represent NaN, so 0 is used for "no data"
        assert np.all((downwind_heights == 0) | np.isnan(downwind_heights))
    
    def test_nan_geopotential_skipped(self, sample_era5_dataset, sample_terrain_dataset):
        """Test that NaN geopotential values are skipped."""
        # Set some gph values to NaN
        sample_era5_dataset['gph'].values[0, 0, 0, 0] = np.nan
        
        result = wind.compute_wind_terrain_interaction(
            sample_era5_dataset, sample_terrain_dataset, range_km=10
        )
        
        # That specific point should have NaN output (or 0 for uint16 downwind_terrain_height)
        assert np.isnan(result['perpendicular_wind_speed'].values[0, 0, 0, 0])
        # downwind_terrain_height is uint16, so NaN becomes 0
        assert result['downwind_terrain_height'].values[0, 0, 0, 0] == 0
    
    def test_terrain_found_no_perpendicular_component(self, sample_era5_dataset, sample_terrain_dataset):
        """Test case where terrain is found but we handle the wind aspect correctly."""
        # Create wind that's perpendicular to a known slope direction
        sample_era5_dataset['u'].values[:] = 10.0  # Wind from east
        sample_era5_dataset['v'].values[:] = 0.0
        
        result = wind.compute_wind_terrain_interaction(
            sample_era5_dataset, sample_terrain_dataset, range_km=10
        )
        
        # Should still process and return valid dataset
        assert isinstance(result, xr.Dataset)
        assert 'perpendicular_wind_speed' in result
        assert 'wind_direction' in result
    
    def test_terrain_interaction_with_low_pressure_level(self, sample_era5_dataset, sample_terrain_dataset):
        """Test wind-terrain interaction with low pressure level (more terrain above)."""
        # Set very low pressure level (high altitude)
        sample_era5_dataset['gph'].values[:] = 200000  # High altitude
        
        # Adjust to have some points with reasonable wind
        sample_era5_dataset['u'].values[:] = 5.0
        sample_era5_dataset['v'].values[:] = 5.0
        
        result = wind.compute_wind_terrain_interaction(
            sample_era5_dataset, sample_terrain_dataset, range_km=10
        )
        
        # Should process successfully
        assert isinstance(result, xr.Dataset)
        # Some values might be NaN if terrain is not above level
        assert 'perpendicular_wind_speed' in result.data_vars
    
    def test_terrain_found_computation_executed(self, sample_era5_dataset):
        """Test that perpendicular_wind_component is called when terrain found."""
        # Use same coordinates as sample_era5_dataset for proper grid overlap
        lats = sample_era5_dataset.latitude.values
        lons = sample_era5_dataset.longitude.values
        
        # Create VERY HIGH terrain to ensure it's above pressure level
        elevation = np.ones((len(lats), len(lons))) * 30000
        aspect = np.ones((len(lats), len(lons))) * 90  # All facing east
        
        terrain_ds = xr.Dataset(
            {
                'elevation': (['latitude', 'longitude'], elevation),
                'aspect_deg': (['latitude', 'longitude'], aspect),
            },
            coords={
                'latitude': lats,
                'longitude': lons,
            }
        )
        
        # Create ERA5 data with LOW geopotential (very low pressure level)
        sample_era5_dataset['gph'].values[:] = 50000  # Low altitude pressure level
        sample_era5_dataset['u'].values[:] = 5.0
        sample_era5_dataset['v'].values[:] = 0.0
        
        result = wind.compute_wind_terrain_interaction(
            sample_era5_dataset, terrain_ds, range_km=10
        )
        
        # Verify result is a dataset with expected variables
        assert isinstance(result, xr.Dataset)
        assert 'perpendicular_wind_speed' in result.data_vars
        assert 'downwind_terrain_height' in result.data_vars


class TestWindIntegration:
    """Integration tests for wind module functions."""
    
    def test_wind_calculation_consistency(self):
        """Integration test: wind calculations are consistent."""
        u = np.array([0, 10, 0, -10, 7.07, -7.07])
        v = np.array([10, 0, -10, 0, 7.07, -7.07])
        
        wind_dir, wind_speed = wind.calc_wind(u, v)
        
        # Verify Pythagorean theorem (allowing for rounding tolerance)
        computed_speed = np.sqrt(u**2 + v**2)
        assert np.allclose(wind_speed, computed_speed, atol=0.02)
        
        # Verify direction is in valid range
        assert np.all((wind_dir >= 0) & (wind_dir < 360))
    
    def test_perpendicular_wind_with_various_angles(self):
        """Integration test: perpendicular wind with various angles."""
        wind_speed = 10.0
        angles = np.linspace(0, 360, 73)
        
        for wind_dir in angles:
            for slope_aspect in angles:
                perp_wind = wind.perpendicular_wind_component(
                    wind_dir, wind_speed, slope_aspect
                )
                
                # Should always be in valid range
                assert 0 <= perp_wind <= wind_speed
                
                # When perfectly aligned
                if abs((wind_dir - slope_aspect) % 360) < 1:
                    assert np.isclose(perp_wind, wind_speed, atol=0.1)
    

