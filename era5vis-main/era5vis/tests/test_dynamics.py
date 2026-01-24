"""
Tests for the dynamics module.

Author: Anna Buchhauser, Lawrence Ansu Mensah
Date: 2026-01-18
"""

import numpy as np
import xarray as xr
import pytest

from era5vis import dynamics

# tests for geopotential_height()
def test_geopotential_height_single_value():
    """Test the conversion from geopotential to height for a single point"""
    
    z = np.array(9.81)
    result = dynamics.geopotential_height(z)
    assert result == 1


# test with a numpy array
def test_geopotential_height_array():
    """Test the conversion across an entire data array"""

    z = np.array([0, 9810, 19620])
    expected = np.array([0, 1000, 2000], dtype=np.uint16)
    result = dynamics.geopotential_height(z)
    np.testing.assert_array_equal(result, expected)
    

# tests for potential_temperature()
def test_potential_temperature_known_value():
    """Test for calculating potential temperature of a known value"""

    p = np.array(1000.0)
    T = np.array(300.0)
    theta = dynamics.potential_temperature(T, p)
    assert np.allclose(theta, 300.0) 

def test_potential_temperature_array():
    """Test shape of array"""

    T = np.array([290.0, 300.0])
    p = np.array([900.0, 1000.0])
    theta = dynamics.potential_temperature(T, p)
    assert T.shape == theta.shape

def test_potential_temperature_invalid_T():
    """Test if invalid temperature value raises error"""

    T = np.array(-300.0)
    p = np.array(1000.0)
    with pytest.raises(ValueError, match='T needs to be above -273 °C'):
        dynamics.potential_temperature(T, p)

def test_potential_temperature_invalid_p():
    """Test if invalid pressure value raises error"""

    T = np.array(300.0)
    p = np.array(3000.0)
    with pytest.raises(ValueError, match='p need to be between'):
        dynamics.potential_temperature(T, p)


# tests for brunt_vaeisaelae_freq()
def test_brunt_vaeisaelae_freq_mean():
    """Test calculating the mean frequency"""

    theta_up = np.array(310.0)
    theta_down = np.array(300.0)
    z_up = np.array(2000.0)
    z_down = np.array(1000.0)

    N = dynamics.brunt_vaeisaelae_freq(theta_up, theta_down, z_up, z_down)
    expected = np.sqrt((9.81 / 305.0) * (10.0 / 1000.0))
    
    assert np.allclose(N, expected)

def test_brunt_vaeisaelae_freq_min():
    """Test calculating the minimum frequency"""
    
    theta_up = np.array(310.0)
    theta_down = np.array(300.0)
    z_up = np.array(2000.0)
    z_down = np.array(1000.0)

    N = dynamics.brunt_vaeisaelae_freq(theta_up, theta_down, z_up, z_down, method='min')
    expected = np.sqrt((9.81 / 300.0) * (10.0 / 1000.0))
    
    assert np.allclose(N, expected)

def test_brunt_vaeisaelae_level_heights():
    """Test if invalid order of level heights raises correct error"""

    theta_up = np.array(310.0)
    theta_down = np.array(300.0)
    z_up = np.array(200.0)
    z_down = np.array(1000.0)

    with pytest.raises(ValueError, match='z_down need to be smaller'):
        dynamics.brunt_vaeisaelae_freq(theta_up, theta_down, z_up, z_down)

def test_brunt_vaeisaelae_invalid_method():
    """Test if invalid passing invalid method raises correct error"""

    theta_up = np.array(310.0)
    theta_down = np.array(300.0)
    z_up = np.array(2000.0)
    z_down = np.array(1000.0)
    
    with pytest.raises(ValueError, 
                       match=r'Valid Arguments for method'):
        dynamics.brunt_vaeisaelae_freq(theta_up, theta_down, z_up, z_down, method='max')


# tests for nondim_mtn_height()
def test_nondim_mtn_height_known_value():
    """Test if non dimensional mountain height is calculated correctly."""

    N = np.array(0.01)
    h = np.array(1000)
    U_up = np.array(10.0)
    U_down = np.array(10.0)

    H = dynamics.nondim_mtn_height(N, h, U_up, U_down)

    assert np.allclose(H, 1.0)

def test_nondim_mtn_height_negative_N():
    """Test if negative N raises correct error"""
    N = np.array(-0.01)
    h = np.array(1000)
    U_up = np.array(10.0)
    U_down = np.array(10.0)
    
    with pytest.raises(ValueError, 
                       match='N need to be positive and smaller than 1'):
        dynamics.nondim_mtn_height(N, h, U_up, U_down)

def test_nondim_mtn_height_negative_wind():
    """Test if negative wind raises correct error"""
    
    N = np.array(0.01)
    h = np.array(1000)
    U_up = np.array(-10.0)
    U_down = np.array(-10.0)
    
    with pytest.raises(ValueError, match='U need to be positive'):
        dynamics.nondim_mtn_height(N, h, U_up, U_down)

def test_nondim_mtn_height_negative_h():
    """Test if negative mountain height raises correct error"""
    
    N = np.array(0.01)
    h = np.array(-1000)
    U_up = np.array(10.0)
    U_down = np.array(10.0)
    
    with pytest.raises(ValueError, match='h need to be positive'):
        dynamics.nondim_mtn_height(N, h, U_up, U_down)


# integration tests for compute_N_H()
def test_compute_N_H_basic():
    """Test if H and N are added to dataset and if shape is correct."""

    pressure = [1000, 900, 800]

    data = xr.Dataset(
        data_vars={
            'theta': (('pressure_level',), [300, 305, 310]),
            'gph': (('pressure_level',), [0, 1000, 2000]),
            'perpendicular_wind_speed': (('pressure_level',), [10, 10, 10]),
            'downwind_terrain_height': ((), 1000),
        },
        coords={'pressure_level': pressure},
    )

    result = dynamics.compute_N_H(data)

    assert 'N' in result
    assert 'H' in result
    # N and H have same shape as input pressure levels (last value is NaN from calculation)
    assert result['N'].shape[0] == len(pressure)

def test_compute_N_H_masking():
    """Test if masking works correctly"""
    data = xr.Dataset(
        data_vars={
            'theta': (('pressure_level',), [300, 305]),
            'gph': (('pressure_level',), [0, 1000]),
            'perpendicular_wind_speed': (('pressure_level',), [0, 10]),
            'downwind_terrain_height': ((), 1000),
        },
        coords={'pressure_level': [1000, 900]},
    )

    result = dynamics.compute_N_H(data)
    assert np.isnan(result['N'].values).all()
    assert np.isnan(result['H'].values).all()
