""" Test functions for cli """

import era5vis
from era5vis.cli import modellevel, terrain
from unittest.mock import patch, MagicMock
import pytest


def test_help(capsys):

    # check that empty arguments return the help
    modellevel([])
    captured = capsys.readouterr()
    assert 'Usage:' in captured.out
    print(captured.out)

    # check that -h and --help return the help
    modellevel(['-h'])
    captured = capsys.readouterr()
    assert 'Usage:' in captured.out

    modellevel(['--help'])
    captured = capsys.readouterr()
    assert 'Usage:' in captured.out


def test_version(capsys):

    # check that -v and --version return version information
    modellevel(['-v'])
    captured = capsys.readouterr()
    assert era5vis.__version__ in captured.out

    modellevel(['--version'])
    captured = capsys.readouterr()
    assert era5vis.__version__ in captured.out


def test_print_html(capsys, retrieve_param_level_time_from_ds):

    param, level, time = retrieve_param_level_time_from_ds

    # check that correctly formatted calls run successfully
    modellevel(['-p', param, '-lvl', level, '--no-browser'])
    captured = capsys.readouterr()
    assert 'File successfully generated at:' in captured.out

    modellevel(['-p', param, '-lvl', level, '-ti', '0', '--no-browser'])
    captured = capsys.readouterr()
    assert 'File successfully generated at:' in captured.out

    modellevel(['-p', param, '-lvl', level, '-t', '202510010000', '--no-browser'])
    captured = capsys.readouterr()
    assert 'File successfully generated at:' in captured.out


def test_error(capsys):

    # check that incorrectly formatted calls raise an error
    modellevel(['-p', 'z'])
    captured = capsys.readouterr()
    assert 'command not understood' in captured.out


class TestTerrainCLI:
    """Tests for era5vis_terrain CLI function"""
    
    def test_terrain_version(self, capsys):
        """Test that -v and --version return version information"""
        with pytest.raises(SystemExit):
            terrain(['-v'])
        captured = capsys.readouterr()
        assert era5vis.__version__ in captured.out
    
    def test_terrain_no_arguments_error(self, capsys):
        """Test that no arguments raises an error"""
        with pytest.raises(SystemExit):
            terrain([])
        captured = capsys.readouterr()
        assert 'Either -d/--download or -recreate/--recreate must be specified' in captured.err
    
    def test_terrain_missing_path_error(self, capsys):
        """Test that missing path for -recreate raises an error"""
        with pytest.raises(SystemExit):
            terrain(['-recreate'])
        captured = capsys.readouterr()
        # argparse will complain about argument -recreate needing a value
        assert 'argument' in captured.err or 'expected one argument' in captured.err
    
    def test_terrain_help(self, capsys):
        """Test that --help shows help message"""
        with pytest.raises(SystemExit):
            terrain(['--help'])
        captured = capsys.readouterr()
        assert 'Recreate terrain datasets' in captured.out
        assert 'Download Alps TIF file' in captured.out
        assert 'Target resolution' in captured.out
    
    @patch('era5vis.terrain.load_terrain_from_tif')
    @patch('era5vis.terrain.compute_terrain_aspect_dataset')
    def test_terrain_recreate_local_file(self, mock_aspect, mock_load, capsys, tmp_path):
        """Test terrain recreation from a local file"""
        import numpy as np
        import xarray as xr
        
        # Create mock return values
        mock_terrain = np.ones((100, 100), dtype=np.int32)
        mock_lats = np.linspace(47, 46, 100)
        mock_lons = np.linspace(10, 11, 100)
        mock_load.return_value = (mock_terrain, mock_lats, mock_lons, 30)
        
        # Create a mock dataset for aspect
        mock_dataset = xr.Dataset({
            'elevation': (['latitude', 'longitude'], mock_terrain),
            'aspect_deg': (['latitude', 'longitude'], np.ones((100, 100), dtype=np.int32) * 90),
            'slope': (['latitude', 'longitude'], np.ones((100, 100), dtype=np.int32) * 20),
            'terrain_mask': (['latitude', 'longitude'], np.ones((100, 100), dtype=bool)),
        }, coords={
            'latitude': mock_lats,
            'longitude': mock_lons,
        })
        mock_aspect.return_value = mock_dataset
        
        # Create a temporary file path
        test_file = str(tmp_path / "test.tif")
        
        # Test with quiet mode
        terrain(['-recreate', test_file, '-q'])
        
        # Verify load_terrain_from_tif was called with correct path
        mock_load.assert_called_once_with(test_file)
        mock_aspect.assert_called_once()
    
    @patch('era5vis.terrain.load_terrain_from_tif')
    @patch('era5vis.terrain.downsample_terrain')
    @patch('era5vis.terrain.compute_terrain_aspect_dataset')
    def test_terrain_recreate_with_resolution(self, mock_aspect, mock_downsample, mock_load, capsys, tmp_path):
        """Test terrain recreation with custom resolution"""
        import numpy as np
        import xarray as xr
        
        # Create mock return values
        mock_terrain = np.ones((100, 100), dtype=np.int32)
        mock_lats = np.linspace(47, 46, 100)
        mock_lons = np.linspace(10, 11, 100)
        # Return 30m resolution
        mock_load.return_value = (mock_terrain, mock_lats, mock_lons, 30)
        
        # Mock downsample to return coarser resolution
        downsampled_terrain = np.ones((50, 50), dtype=np.int32)
        downsampled_lats = np.linspace(47, 46, 50)
        downsampled_lons = np.linspace(10, 11, 50)
        mock_downsample.return_value = (downsampled_terrain, downsampled_lats, downsampled_lons, 2000)
        
        # Create a mock dataset
        mock_dataset = xr.Dataset({
            'elevation': (['latitude', 'longitude'], downsampled_terrain),
            'aspect_deg': (['latitude', 'longitude'], np.ones((50, 50), dtype=np.int32) * 90),
            'slope': (['latitude', 'longitude'], np.ones((50, 50), dtype=np.int32) * 20),
            'terrain_mask': (['latitude', 'longitude'], np.ones((50, 50), dtype=bool)),
        }, coords={
            'latitude': downsampled_lats,
            'longitude': downsampled_lons,
        })
        mock_aspect.return_value = mock_dataset
        
        test_file = str(tmp_path / "test.tif")
        
        # Test with 2km resolution (should trigger downsampling)
        terrain(['-recreate', test_file, '-res', '2.0', '-q'])
        
        # Verify downsample was called
        mock_downsample.assert_called_once()
        # Check that resolution argument was passed correctly
        call_args = mock_downsample.call_args
        assert call_args[0][4] == 2000  # target_res_m = 2.0 * 1000
    
    @patch('urllib.request.urlretrieve')
    @patch('era5vis.terrain.load_terrain_from_tif')
    @patch('era5vis.terrain.compute_terrain_aspect_dataset')
    def test_terrain_download_with_force(self, mock_aspect, mock_load, mock_download, capsys, tmp_path):
        """Test terrain download with --force flag (no confirmation)"""
        import numpy as np
        import xarray as xr
        
        # Create mock return values
        mock_terrain = np.ones((100, 100), dtype=np.int32)
        mock_lats = np.linspace(47, 46, 100)
        mock_lons = np.linspace(10, 11, 100)
        mock_load.return_value = (mock_terrain, mock_lats, mock_lons, 30)
        
        mock_dataset = xr.Dataset({
            'elevation': (['latitude', 'longitude'], mock_terrain),
            'aspect_deg': (['latitude', 'longitude'], np.ones((100, 100), dtype=np.int32) * 90),
            'slope': (['latitude', 'longitude'], np.ones((100, 100), dtype=np.int32) * 20),
            'terrain_mask': (['latitude', 'longitude'], np.ones((100, 100), dtype=bool)),
        }, coords={
            'latitude': mock_lats,
            'longitude': mock_lons,
        })
        mock_aspect.return_value = mock_dataset
        
        # Mock the download function to do nothing (just pretend file was downloaded)
        mock_download.return_value = None
        
        # Test with --force and --quiet
        terrain(['-d', '-f', '-q'])
        
        # Verify download was called
        mock_download.assert_called_once()
        # Verify load_terrain_from_tif was called with the downloaded file
        mock_load.assert_called_once()
    
    @patch('builtins.input', return_value='no')
    @patch('urllib.request.urlretrieve')
    def test_terrain_download_user_declines(self, mock_download, mock_input, capsys):
        """Test terrain download when user declines confirmation"""
        # Test with user declining the download
        terrain(['-d', '-q'])
        
        # Verify input was called (asking for confirmation)
        mock_input.assert_called_once()
        # Verify download was NOT called
        mock_download.assert_not_called()
        
    def test_terrain_quiet_suppresses_output(self, capsys, tmp_path):
        """Test that --quiet flag suppresses normal output"""
        import numpy as np
        import xarray as xr
        
        with patch('era5vis.terrain.load_terrain_from_tif') as mock_load, \
             patch('era5vis.terrain.compute_terrain_aspect_dataset') as mock_aspect, \
             patch('era5vis.terrain.downsample_terrain') as mock_downsample:
            
            # Create mock return values
            mock_terrain = np.ones((50, 50), dtype=np.int32)
            mock_lats = np.linspace(47, 46, 50)
            mock_lons = np.linspace(10, 11, 50)
            mock_load.return_value = (mock_terrain, mock_lats, mock_lons, 30)
            
            # Mock downsample to return downsampled data (90 m resolution is much less than 1000m default)
            downsampled_terrain = np.ones((40, 40), dtype=np.int32)
            downsampled_lats = np.linspace(47, 46, 40)
            downsampled_lons = np.linspace(10, 11, 40)
            mock_downsample.return_value = (downsampled_terrain, downsampled_lats, downsampled_lons, 750)
            
            mock_dataset = xr.Dataset({
                'elevation': (['latitude', 'longitude'], downsampled_terrain),
                'aspect_deg': (['latitude', 'longitude'], np.ones((40, 40), dtype=np.int32) * 90),
                'slope': (['latitude', 'longitude'], np.ones((40, 40), dtype=np.int32) * 20),
                'terrain_mask': (['latitude', 'longitude'], np.ones((40, 40), dtype=bool)),
            }, coords={
                'latitude': downsampled_lats,
                'longitude': downsampled_lons,
            })
            mock_aspect.return_value = mock_dataset
            
            test_file = str(tmp_path / "test.tif")
            
            # Test with quiet mode
            terrain(['-recreate', test_file, '-q'])
            captured = capsys.readouterr()
            # Should have no output with quiet flag
            assert captured.out == ''
    
    def test_terrain_output_with_verbose(self, capsys, tmp_path):
        """Test that output is shown without quiet flag"""
        import numpy as np
        import xarray as xr
        
        with patch('era5vis.terrain.load_terrain_from_tif') as mock_load, \
             patch('era5vis.terrain.compute_terrain_aspect_dataset') as mock_aspect, \
             patch('era5vis.terrain.downsample_terrain') as mock_downsample:
            
            # Create mock return values
            mock_terrain = np.ones((50, 50), dtype=np.int32)
            mock_lats = np.linspace(47, 46, 50)
            mock_lons = np.linspace(10, 11, 50)
            mock_load.return_value = (mock_terrain, mock_lats, mock_lons, 30)
            
            # Mock downsample
            downsampled_terrain = np.ones((40, 40), dtype=np.int32)
            downsampled_lats = np.linspace(47, 46, 40)
            downsampled_lons = np.linspace(10, 11, 40)
            mock_downsample.return_value = (downsampled_terrain, downsampled_lats, downsampled_lons, 750)
            
            mock_dataset = xr.Dataset({
                'elevation': (['latitude', 'longitude'], downsampled_terrain),
                'aspect_deg': (['latitude', 'longitude'], np.ones((40, 40), dtype=np.int32) * 90),
                'slope': (['latitude', 'longitude'], np.ones((40, 40), dtype=np.int32) * 20),
                'terrain_mask': (['latitude', 'longitude'], np.ones((40, 40), dtype=bool)),
            }, coords={
                'latitude': downsampled_lats,
                'longitude': downsampled_lons,
            })
            mock_aspect.return_value = mock_dataset
            
            test_file = str(tmp_path / "test.tif")
            
            # Test without quiet mode (verbose)
            terrain(['-recreate', test_file])
            captured = capsys.readouterr()
            # Should have output
            assert 'Loading terrain from:' in captured.out
            assert 'Target resolution:' in captured.out
            assert 'Terrain dataset saved to:' in captured.out
