''' Test functions for era5.py '''
""" 
Author: Lawrence Ansu Mensah, Andreas Friesinger
Date: 2026-01-23
"""
import os
import datetime
import numpy as np
import xarray as xr
import pytest
from unittest.mock import MagicMock
from era5vis import era5, cfg
from era5vis.era5 import check_alpine_bounds, validate_inputs, check_data_availability


def test_horiz_cross_section(retrieve_param_level_from_ds):

    # extract the horizontal cross section
    param, level = retrieve_param_level_from_ds
    da = era5.horiz_cross_section(param, level, 0)

    # check that the correct parameter is extracted
    assert da.GRIB_shortName == param

    # check that the DataArray has the correct type and dimensions
    assert isinstance(da, xr.DataArray)
    assert da.dims == ('latitude', 'longitude')

    # check that pressure_level and valid_time are indeed scalars
    da.pressure_level.item()
    da.valid_time.item()

def test_check_alpine_bounds_valid():
    """Test that valid coordinates do not raise an error"""
    #Inside the Alps
    area = [47.5, 11.0, 47.0, 11.5]
    assert check_alpine_bounds(area) is None

def test_check_alpine_bounds_invalid():
    """Test that coordinates outside the Alps raise a ValueError"""
    # Outside the Alps 
    area = [52.5, 13.0, 52.0, 13.5]
    with pytest.raises(ValueError, match="Area outside terrain bounds"):
        check_alpine_bounds(area)


def test_validate_inputs_no_credentials(monkeypatch):
    """Test that missing CDS API credentials raises FileNotFoundError"""
    #a mock  to simulate a missing file without deleting real data.
    mock_exists = MagicMock(return_value = False)
    
    # Swap the real os.path.exists with our mock
    monkeypatch.setattr(os.path, "exists", mock_exists)
    
    # confirm the error handling works as expected
    with pytest.raises(FileNotFoundError, match = "CDS API Authentication Missing"):
        validate_inputs([47, 10, 46, 11])



def test_check_availability_missing_file():
    """Verify system handles a missing NetCDF data file"""
    # store original path
    original_file = cfg.datafile

    # point config. to a non-existent file
    cfg.datafile = "non_existent_file.nc"

    # confirm that check_data_availability catches the missing file
    with pytest.raises(FileNotFoundError):
        check_data_availability("t")
    
    # restore original configuration
    cfg.datafile = original_file


def test_datetime_parsing():
    """Verify that the CLI date string is correctly converted to a datetime object"""
    date_str = "2025-01-14-12"

    # Parse the string using the module's formatting logic
    parsed_date = datetime.datetime.strptime(date_str, "%Y-%m-%d-%H")

    # Check each component to ensure formatting is right 
    assert parsed_date.year == 2025
    assert parsed_date.month == 1
    assert parsed_date.hour == 12



def test_compress_era(retrieve_param_level_from_ds):
    """Test that compress_era() converts data to the correct dtypes"""
    
    # Load the test dataset
    with xr.open_dataset(cfg.datafile) as ds:
        ds_original = ds.copy()
    
    # Compress the dataset
    ds_compressed = era5.compress_era(ds_original)
    
    # Check that the expected variables have been converted to the correct dtypes
    dtype_map = {
        "latitude": np.float32,
        "longitude": np.float32,
        "pressure_level": np.int16,
        "t": np.float16,
        "u": np.float16,
        "v": np.float16,
        "z": np.uint16,
        "gph": np.uint16,
    }
    
    for var, expected_dtype in dtype_map.items():
        if var in ds_compressed:
            assert ds_compressed[var].dtype == expected_dtype, \
                f"Variable {var} has dtype {ds_compressed[var].dtype}, expected {expected_dtype}"
    
    # Check that the dataset still has the same structure
    assert set(ds_compressed.dims) == set(ds_original.dims)
    assert set(ds_compressed.coords) == set(ds_original.coords)
    assert len(ds_compressed.data_vars) == len(ds_original.data_vars)


def test_safe_to_netcdf(tmp_path):
    """Test that safe_to_netcdf() correctly saves a dataset with proper dtypes"""
    
    # Load the test dataset
    with xr.open_dataset(cfg.datafile) as ds:
        ds_test = ds.copy()
    
    # Create output filename in temporary directory
    output_file = tmp_path / "test_output.nc"
    
    # Save the dataset using safe_to_netcdf
    era5.safe_to_netcdf(ds_test, str(output_file))
    
    # Verify the file was created
    assert output_file.exists(), "Output NetCDF file was not created"
    
    # Load the saved file and verify its contents
    with xr.open_dataset(str(output_file)) as ds_loaded:
        # Check that expected variables have been converted to float32
        dtype_map = {
            "t": np.float32,
            "u": np.float32,
            "v": np.float32,
            "wind_speed": np.float32,
        }
        
        for var, expected_dtype in dtype_map.items():
            if var in ds_loaded:
                assert ds_loaded[var].dtype == expected_dtype, \
                    f"Variable {var} has dtype {ds_loaded[var].dtype}, expected {expected_dtype}"
        
        # Verify that dimensions and coordinates are preserved
        assert set(ds_loaded.dims) == set(ds_test.dims), \
            "Dataset dimensions were not preserved"
        assert set(ds_loaded.coords) == set(ds_test.coords), \
            "Dataset coordinates were not preserved"


def test_safe_to_netcdf_data_integrity(tmp_path):
    """Test that safe_to_netcdf() preserves data values correctly"""
    
    # Create a minimal test dataset with known values
    test_data = xr.Dataset({
        "t": (("x", "y"), np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)),
        "u": (("x", "y"), np.array([[10.1, 20.2], [30.3, 40.4]], dtype=np.float64)),
    })
    
    # Create output filename in temporary directory
    output_file = tmp_path / "test_data_integrity.nc"
    
    # Save the dataset using safe_to_netcdf
    era5.safe_to_netcdf(test_data, str(output_file))
    
    # Load the saved file
    with xr.open_dataset(str(output_file)) as ds_loaded:
        # Check that data values are preserved (within float32 precision)
        np.testing.assert_array_almost_equal(
            ds_loaded["t"].values,
            np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32),
            decimal=5,
            err_msg="Temperature data was not preserved correctly"
        )
        
        np.testing.assert_array_almost_equal(
            ds_loaded["u"].values,
            np.array([[10.1, 20.2], [30.3, 40.4]], dtype=np.float32),
            decimal=5,
            err_msg="Wind u component data was not preserved correctly"
        )


def test_safe_to_netcdf_does_not_modify_original(tmp_path):
    """Test that safe_to_netcdf() does not modify the original dataset"""
    
    # Load the test dataset
    with xr.open_dataset(cfg.datafile) as ds:
        ds_original = ds.copy()
    
    # Store original dtypes
    original_dtypes = {var: ds_original[var].dtype for var in ds_original.data_vars}
    
    # Create output filename in temporary directory
    output_file = tmp_path / "test_no_modify.nc"
    
    # Save the dataset using safe_to_netcdf
    era5.safe_to_netcdf(ds_original, str(output_file))
    
    # Verify original dataset was not modified
    for var, original_dtype in original_dtypes.items():
        assert ds_original[var].dtype == original_dtype, \
            f"Original dataset variable {var} dtype was modified"
