""" Test functions for cli """

""" 
Author: Andreas Friesinger, Lawrence Ansu Mensah
Date: 2026-01-23
"""

from unittest.mock import patch, MagicMock
import era5vis
import datetime
from era5vis.cli import terrain, analyzeH, download
import pytest


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
    def test_terrain_recreate_local_file(self, mock_aspect, mock_load, tmp_path):
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
        output_file = str(tmp_path / "terrain_dataset.nc")
        
        # Test with quiet mode and specify output path
        terrain(['-recreate', test_file, '-o', output_file, '-q'])
        
        # Verify load_terrain_from_tif was called with correct path
        mock_load.assert_called_once_with(test_file)
        mock_aspect.assert_called_once()
    
    @patch('era5vis.terrain.load_terrain_from_tif')
    @patch('era5vis.terrain.downsample_terrain')
    @patch('era5vis.terrain.compute_terrain_aspect_dataset')
    def test_terrain_recreate_with_resolution(self, mock_aspect, mock_downsample, mock_load, tmp_path):
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
        output_file = str(tmp_path / "terrain_dataset.nc")
        
        # Test with 2km resolution (should trigger downsampling)
        terrain(['-recreate', test_file, '-res', '2.0', '-o', output_file, '-q'])
        
        # Verify downsample was called
        mock_downsample.assert_called_once()
        # Check that resolution argument was passed correctly
        call_args = mock_downsample.call_args
        assert call_args[0][4] == 2000  # target_res_m = 2.0 * 1000
    
    @patch('urllib.request.urlretrieve')
    @patch('era5vis.terrain.load_terrain_from_tif')
    @patch('era5vis.terrain.compute_terrain_aspect_dataset')
    def test_terrain_download_with_force(self, mock_aspect, mock_load, mock_download):
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
    def test_terrain_download_user_declines(self, mock_download, mock_input):
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
            output_file = str(tmp_path / "terrain_dataset.nc")
            
            # Test with quiet mode
            terrain(['-recreate', test_file, '-o', output_file, '-q'])
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
            output_file = str(tmp_path / "terrain_dataset.nc")
            
            # Test without quiet mode (verbose)
            terrain(['-recreate', test_file, '-o', output_file])
            captured = capsys.readouterr()
            # Should have output
            assert 'Loading terrain from:' in captured.out
            assert 'Target resolution:' in captured.out
            assert 'Terrain dataset saved to:' in captured.out


class TestAnalyzeHCLI:
    """Tests for era5vis_analyzeH CLI function"""
    
    def test_analyzeh_version(self, capsys):
        """Test that -v and --version return version information"""
        with pytest.raises(SystemExit):
            analyzeH(['-v'])
        captured = capsys.readouterr()
        assert era5vis.__version__ in captured.out
    
    def test_analyzeh_missing_required_arguments(self, capsys):
        """Test that missing required arguments raises an error"""
        with pytest.raises(SystemExit):
            analyzeH([])
        captured = capsys.readouterr()
        assert 'required' in captured.err.lower() or 'longitude' in captured.err.lower()
    
    def test_analyzeh_missing_longitude(self, capsys):
        """Test that missing longitude argument raises an error"""
        with pytest.raises(SystemExit):
            analyzeH(['-dt', '2025-01-15-12'])
        captured = capsys.readouterr()
        assert 'required' in captured.err.lower() or 'longitude' in captured.err.lower()
    
    def test_analyzeh_missing_datetime(self, capsys):
        """Test that missing datetime argument raises an error"""
        with pytest.raises(SystemExit):
            analyzeH(['-lon', '12.5'])
        captured = capsys.readouterr()
        assert 'required' in captured.err.lower() or 'datetime' in captured.err.lower()
    
    def test_analyzeh_invalid_datetime_format(self, capsys):
        """Test that invalid datetime format raises an error"""
        with pytest.raises(SystemExit):
            analyzeH(['-lon', '12.5', '-dt', '2025-01-15'])  # Missing hour
        captured = capsys.readouterr()
        assert 'Invalid datetime format' in captured.out or 'error' in captured.out.lower()
    
    def test_analyzeh_help(self, capsys):
        """Test that --help shows help message"""
        with pytest.raises(SystemExit):
            analyzeH(['--help'])
        captured = capsys.readouterr()
        assert 'Analyze geopotential height' in captured.out
        assert 'longitude' in captured.out.lower()
        assert 'datetime' in captured.out.lower()
    
    @patch('era5vis.cli.terrain_module.load_terrain_aspect_dataset')
    @patch('era5vis.cli.era5.load_era5_data')
    @patch('era5vis.cli.visualization.create_plot')
    def test_analyzeh_invalid_longitude(self, mock_plot, mock_load_era5, mock_load_terrain, capsys):
        """Test that longitude outside terrain bounds raises an error"""
        import xarray as xr
        import numpy as np
        
        # Create mock terrain dataset with specific bounds
        mock_terrain = xr.Dataset({
            'elevation': (['latitude', 'longitude'], np.ones((10, 10))),
        }, coords={
            'latitude': np.linspace(47, 46, 10),
            'longitude': np.linspace(10, 11, 10),  # Bounds: [10, 11]
        })
        mock_load_terrain.return_value = mock_terrain
        
        # Try longitude outside bounds
        with pytest.raises(SystemExit):
            analyzeH(['-lon', '5.0', '-dt', '2025-01-15-12'])  # 5.0 is outside [10, 11]
        captured = capsys.readouterr()
        assert 'out of terrain bounds' in captured.out or 'Error' in captured.out
    
    @patch('era5vis.cli.terrain_module.load_terrain_aspect_dataset')
    @patch('era5vis.cli.era5.load_era5_data')
    @patch('era5vis.cli.visualization.create_plot')
    def test_analyzeh_date_too_far_in_future(self, mock_plot, mock_load_era5, mock_load_terrain, capsys):
        """Test that date beyond ERA5 availability raises an error"""
        import datetime
        import xarray as xr
        import numpy as np
        
        # Create mock terrain dataset
        mock_terrain = xr.Dataset({
            'elevation': (['latitude', 'longitude'], np.ones((10, 10))),
        }, coords={
            'latitude': np.linspace(47, 46, 10),
            'longitude': np.linspace(10, 11, 10),
        })
        mock_load_terrain.return_value = mock_terrain
        
        # Try future date (beyond 3 days from now)
        future_date = (datetime.datetime.now() + datetime.timedelta(days=10)).strftime('%Y-%m-%d-%H')
        with pytest.raises(SystemExit):
            analyzeH(['-lon', '10.5', '-dt', future_date])
        captured = capsys.readouterr()
        assert 'out of ERA5 data bounds' in captured.out or 'Error' in captured.out

class TestDownloadCLI:
    """Tests for era5vis_download CLI function"""

    def test_download_version(self, capsys):
        """Test that -v returns version and exits """
        with pytest.raises(SystemExit):
            download(['-v'])
        captured = capsys.readouterr()
        assert era5vis.__version__ in captured.out

    def test_download_help(self, capsys):
        """Verify that --help shows usage instructions and exits properly"""
        with pytest.raises(SystemExit):
            download(['--help'])
        captured = capsys.readouterr()
        assert 'Usage:' in captured.out
        assert '-o, --output' in captured.out
        assert '-a, --area' in captured.out
        

    def test_download_missing_mandatory_args(self, capsys):
        """Verify that the tool exits with an error if mandatory flags are missing"""
        with pytest.raises(SystemExit):
            download(['-o', 'test.nc', '-s', '2025-01-01-12'])
        captured = capsys.readouterr()
        assert 'mandatory arguments missing' in captured.out

    def test_download_invalid_date_format(self, capsys):
        """Verify that an incorrect date string format triggers an error and exits"""
        test_args = ['-o', 'test.nc', '-s', '2025.01.01.12', '-a', '47', '10', '46', '11']
        with pytest.raises(SystemExit):
            download(test_args)
        captured = capsys.readouterr()
        assert 'Error: Invalid input format' in captured.out

    @patch('era5vis.era5.load_era5_data')
    def test_download_success(self, mock_load_era5):
        """Verify that valid CLI arguments correctly trigger the ERA5 data loading process"""
        # mock the actual download function to avoid hitting the Copernicus servers
        mock_load_era5.return_value = MagicMock()
        
        # define a complete set of valid mandatory arguments
        test_args = ['-o', 'output.nc', '-s', '2025-01-15-12', '-a', '47', '10', '46', '11']
        
        # execute the download tool with these arguments
        download(test_args)

        #confirm the underlying ERA5 logic was actually called
        assert mock_load_era5.called
        
        # verify the CLI correctly converted strings into the required Python objects
        call_args = mock_load_era5.call_args[0]
        assert call_args[0] == 'output.nc'
        assert isinstance(call_args[1], datetime.datetime)
        assert call_args[2] == [47.0, 10.0, 46.0, 11.0]
