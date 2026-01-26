"""
Tests for the visualization module.

Author: Anna Buchhauser
Date: 2026-01-21
"""
import os
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import pytest

import visualization

# tests for height_to_pressure()
def test_height_to_pressure_known_value():
    """Test for calculating pressure of a known value at a certain height"""
    
    p = visualization.height_to_pressure(0.0)
    expected = 1013.25
    
    assert np.isclose(p, expected) 

def test_height_to_pressure_array():
    """Test shape of array"""

    z = np.array([900.0, 1000.0])
    p = visualization.height_to_pressure(z)
    assert z.shape == p.shape


# tests for add_color_code()
def test_add_color_code_basic():
    """Tests if the color code is applied correctly"""

    # create mockup data
    H = xr.DataArray([1.2, 0.8, np.nan, 2.0], dims='x')
    perpendicular_wind = xr.DataArray([1.0, 1.0, 1.0, 0.0], dims='x')
    color_ds = xr.Dataset({'H': H, 
                     'perpendicular_wind_speed': perpendicular_wind})

    visualization.add_color_code(color_ds)
    expected = [
        'red',    # H >= 1
        'green',  # H < 1
        'white',  # NaN
        'blue',   # perpendicular wind = 0
    ]
    assert list(color_ds['H_color'].values) == expected


    
# tests for select_data()    
def test_select_data_correct_range():
    """Tests if correct range is selected and time dimension removed"""

    # create mockup data
    lat = np.linspace(48, 45, 7)
    lon = [10.0, 11.0]
    time = [pd.to_datetime('2025-09-02T12:00:00')]

    data = xr.DataArray(
        np.random.rand(1, 2, 7),
        dims=('valid_time', 'longitude', 'latitude'),
        coords={
            'valid_time': time,
            'longitude': lon,
            'latitude': lat,
        },
    )

    ds_mockup = xr.Dataset({'H': data})
    ds_crosssec, _ = visualization.select_data(ds_mockup, 10, start_lat=46, end_lat=47.5)

    assert 'valid_time' not in ds_crosssec.dims
    assert ds_crosssec.latitude.min() >= 46
    assert ds_crosssec.latitude.max() <= 47.5

    
# tests for apply_terrain_mask()
@pytest.fixture
def mockup_data_crosssec():
    """Creates mockup cross-section dataset"""

    lat = [46, 47]
    p = [800, 900, 1000]

    H = xr.DataArray(
        np.ones((3, 2)),
        dims=('pressure_level', 'latitude'),
        coords={'pressure_level': p, 'latitude': lat},
    )

    ds_crosssec = xr.Dataset(
        {
            'H': H,
            'H_color': xr.full_like(H, 'red', dtype=object),
            'U': xr.zeros_like(H),
            'V': xr.zeros_like(H),
        }
    )


    return ds_crosssec
    
@pytest.fixture
def mockup_terrain_p():
    """Creates array with mockup pressure levels"""
    
    lat = [46, 47]
    terrain_p = xr.DataArray([900, 850], dims='latitude', coords={'latitude': lat})

    return terrain_p
    
    
def test_apply_terrain_mask(mockup_terrain_p, mockup_data_crosssec):
    """Test if datapoints beneath terrain height are masked correctly"""

    visualization.apply_terrain_mask(mockup_data_crosssec, mockup_terrain_p)
    H_masked = mockup_data_crosssec['H_masked']

    assert H_masked.sel(pressure_level=1000).isnull().all()
    assert H_masked.sel(pressure_level=800).notnull().all()


# tests for vertical_crosssec()
@pytest.fixture
def mockup_masked_data_crosssec():
    """Creates masked dataset for crosssection"""

    # create mockup data
    lat = [46, 47]
    p = [800, 900, 1000]

    H = xr.DataArray(
        np.ones((3, 2)),
        dims=('pressure_level', 'latitude'),
        coords={'pressure_level': p, 'latitude': lat},
    )

    ds_crosssec = xr.Dataset(
        {
            'H_masked': H,
            'H_color_masked': xr.full_like(H, 'red', dtype=object),
            'u': xr.zeros_like(H),
            'v': xr.zeros_like(H),
        }
    )


    return ds_crosssec
    
def test_vertical_crosssection(mockup_terrain_p, mockup_masked_data_crosssec, monkeypatch):
    """Tests if figure is not empty and saved"""
    #replace plt.show to avoid displaying the plot during testing
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    
    filepath =  Path('test.png')
    
    fig = visualization.vertical_crosssection(
        mockup_masked_data_crosssec, mockup_terrain_p, 
        pd.Timestamp('2025-09-02T12:00:00'), lon=11.2,
        filepath=filepath
    )

    assert fig is not None
    assert Path.is_file(filepath)

    plt.close('all')