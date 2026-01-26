""" contains command line tools of ERA5vis

Authors: Andreas Friesinger, Anna Buchhauser, Lawrence Mensah
January 2026
"""

import sys
import webbrowser
import datetime
import era5vis
import argparse
import xarray as xr
import numpy as np
import tempfile
import os
from era5vis import terrain as terrain_module
from era5vis import era5
from era5vis import visualization
from era5vis import wind
from era5vis import dynamics
HELP_DOWNLOAD = """era5vis_download: Download ERA5 data for the Alpine region.

Usage:
    -o, --output [FILENAME]    : output filename, mandatory
    -s, --start [YYYY-MM-DD-HH]: start date and hour, mandatory
    -e, --end [YYYY-MM-DD-HH]  : end date and hour, optional (for time series)
    -a, --area [N W S E]       : North West South East coordinates (spaces only), mandatory
                                 (Limits: N<=48, S>=45, W>=6, E<=16)
"""

HELP_VISUALIZATION = """era5vis_visualization: Plot a vertical cross-section
                        of H from a provided dataset.

Usage:
    -i, --input [FILEPATH]     : filepath for input dataset, mandatory
    -l, --lon [LONGITUDE]      : longitude of the cross-section, mandatory
    -s, --start_lat [START_LAT]: southern border of cross-section, optional
    -e, --end_lat [END_LAT]    : northern border of cross-section, optional
"""




def download(args):
    """The actual era5vis_download command line tool.
    
    Parameters
    ----------
    args: list
        output of sys.args[1:]
    
    Author: Lawrence Mensah
    
    """

    if '--output' in args: 
        args[args.index('--output')] = '-o'
    if '--start' in args: 
        args[args.index('--start')] = '-s'
    if '--end' in args: 
        args[args.index('--end')] = '-e'
    if '--area' in args: 
        args[args.index('--area')] = '-a'

    if len(args) == 0 or args[0] in ['-h', '--help']:
        print(HELP_DOWNLOAD)
        sys.exit(0)
        
    elif args[0] in ['-v', '--version']:
        print('era5vis_download version: ' + era5vis.__version__)
        sys.exit(0)
    # Individual parameter check
    elif ('-o' in args) and ('-s' in args) and ('-a' in args):
        try:
            output = args[args.index('-o') + 1]
            start_str = args[args.index('-s') + 1]
            start_dt = datetime.datetime.strptime(start_str, "%Y-%m-%d-%H")
            
            # Handle optional end date for time series
            end_dt = None
            if '-e' in args:
                end_str = args[args.index('-e') + 1]
                end_dt = datetime.datetime.strptime(end_str, "%Y-%m-%d-%H")

            idx = args.index('-a')
            area = [float(args[idx+1]), float(args[idx+2]), 
                    float(args[idx+3]), float(args[idx+4])]
            
            era5.load_era5_data(output, start_dt, area, end_date=end_dt)
            
        except IndexError:
            print('Error: A flag was provided but no value followed it.')
        except ValueError as e:
            print(f'Error: Invalid input format. {e}')
            sys.exit(1)
        except Exception as e:
            print(f'An unexpected error occurred: {e}')
    else:
        print('era5vis_download: command not understood or mandatory arguments missing.'
              'Type "era5vis_download --help" for usage information.')
        sys.exit(1)

def terrain(args):
    """The actual era5vis_terrain command line tool.
    
    Recreate terrain datasets from GeoTIFF files with optional resolution control.
    
    Parameters
    ----------
    args: list
        output of sys.argv[1:]

    Author: Andreas Friesinger
    """
    
    parser = argparse.ArgumentParser(
        prog='era5vis_terrain',
        description='Recreate terrain datasets from GeoTIFF files for ERA5vis'
    )
    
    parser.add_argument(
        '-d', '--download',
        action='store_true',
        help='Download Alps TIF file from remote source (350MB)'
    )
    
    parser.add_argument(
        '-recreate',
        '--recreate',
        metavar='PATH',
        help='Path to GeoTIFF terrain file to process'
    )
    
    parser.add_argument(
        '-res',
        '--resolution',
        type=float,
        default=1.0,
        metavar='KM',
        help='Target resolution in kilometers (default: 1.0 km)'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress all output messages'
    )
    
    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Skip confirmation prompts and proceed directly'
    )
    
    parser.add_argument(
        '-o', '--output',
        metavar='PATH',
        help='Output path for terrain dataset (default: terrain_dataset.nc)'
    )
    
    parser.add_argument(
        '-v', '--version',
        action='version',
        version=f'era5vis_terrain {era5vis.__version__}'
    )
    
    try:
        parsed_args = parser.parse_args(args)
        
        # Check that at least one action is specified
        if not parsed_args.download and not parsed_args.recreate:
            parser.error('Either -d/--download or -recreate/--recreate must be specified')
        
        tif_path = parsed_args.recreate
        resolution_km = parsed_args.resolution
        quiet = parsed_args.quiet
        force = parsed_args.force
        output_path = parsed_args.output
        
        # Helper function to print only if not quiet
        def log(msg):
            if not quiet:
                print(msg)
        
        # Handle download
        if parsed_args.download:
            log('Alps TIF file download')
            log('File size: ~350 MB')
            
            if not force:
                response = input('Do you want to download this file? (yes/no): ').strip().lower()
                if response not in ['yes', 'y']:
                    log('Download cancelled by user')
                    return
            else:
                log('Force mode: proceeding without confirmation')
            
            # Download the file
            import urllib.request
            url = 'https://fileshare.uibk.ac.at/f/872b6e028ef74f6caf58/?dl=1'
            output_file = 'srtm_alps_30m.tif'
            
            try:
                log(f'Downloading from: {url}')
                urllib.request.urlretrieve(url, output_file)
                log(f'Downloaded to: {output_file}')
                tif_path = output_file
            except Exception as e:
                print(f'Error downloading file: {e}')
                sys.exit(1)
        
        # Process terrain if path is provided
        if tif_path:
            log(f'Loading terrain from: {tif_path}')
            log(f'Target resolution: {resolution_km} km')
            
            # Load and process terrain
            terrain_data, lats, lons, res_m = terrain_module.load_terrain_from_tif(tif_path)
            log(f'Loaded terrain with resolution: {res_m} m')
            
            # Downsample if needed
            source_res_m = res_m
            target_res_m = resolution_km * 1000
            
            if target_res_m > source_res_m:
                terrain_data, lats, lons, res_m = terrain_module.downsample_terrain(
                    terrain_data, lats, lons, source_res_m, target_res_m
                )
                log(f'Downsampled to: {res_m} m')
            else:
                log(f'Source resolution {source_res_m}m is already suitable (target: {target_res_m}m)')
            
            # Create terrain aspect dataset
            terrain_ds = terrain_module.compute_terrain_aspect_dataset(terrain_data, lats, lons)
            
            # Save terrain dataset
            import xarray as xr
            if output_path is None:
                output_path = 'terrain_dataset.nc'
            terrain_ds.to_netcdf(output_path)
            log(f'Terrain dataset saved to: {output_path}')
        
    except SystemExit:
        # argparse calls sys.exit on parse error, catch it gracefully
        raise
    except FileNotFoundError as e:
        print(f'Error: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'Error processing terrain: {e}')
        sys.exit(1)

def plot_vertical_crosssection(args):
    """The actual era5vis_visualization command line tool.

    Parameters
    ----------
    args: list
        output of sys.args[1:]
    """

    if '--input' in args: 
        args[args.index('--input')] = '-i'
    if '--lon' in args: 
        args[args.index('--lon')] = '-l'
    if '--start_lat' in args: 
        args[args.index('--start_lat')] = '-s'
    if '--end_lat' in args: 
        args[args.index('--end_lat')] = '-e'
        
    # work with command line arguements
    if len(args) == 0 or args[0] in ['-h', '--help']:
        print(HELP_VISUALIZATION)
        return

    # input data file an lon are necessary, start and end lat not
    elif ('-i' in args) and ('-l' in args):
        input_filepath = args[args.index('-i') + 1]
        lon = float(args[args.index('-l') + 1])
        start_lat = 45.5
        end_lat = 47.8
        if ('-s' in args):
            start_lat = float(args[args.index('-s') + 1])
        if ('-e' in args):
            end_lat = float(args[args.index('-e') + 1])
        if not (45 <= start_lat <= 48 and 45 <= end_lat <= 48):
            raise ValueError('Start and end latitude must be between 45 and 48')
        if start_lat >= end_lat:
            raise ValueError('Start latitude must be smaller than end latitude')

        try:
            input_ds = xr.open_dataset(input_filepath)
            visualization.create_plot(
                input_ds, lon, start_lat=start_lat, end_lat=end_lat
            )
        except Exception as e:
            print(f'Error: {e}')

    else:
        print('era5vis_visualization: command not understood or mandatory arguments missing.'
              'Type "era5vis_visualization --help" for usage information.')


## The main logic!
def analyzeH(args):
    """The actual era5vis_analyzeH command line tool.

    Parameters
    ----------
    args: list
        output of sys.argv[1:]
    
    Author: Andreas Friesinger
    """
    
    parser = argparse.ArgumentParser(
        prog='era5vis_analyzeH',
        description='Analyze geopotential height with terrain interaction and wind effects'
    )
    
    parser.add_argument(
        '-lon', '--longitude',
        type=float,
        required=True,
        metavar='LONGITUDE',
        help='Longitude of the crossection trough the alps (mandatory)'
    )
    
    parser.add_argument(
        '-dt', '--datetime',
        type=str,
        required=True,
        metavar='YYYY-MM-DD-HH',
        help='Datetime for analysis in format YYYY-MM-DD-HH (mandatory)'
    )
    
    parser.add_argument(
        '-plot_filename',
        type=str,
        metavar='FILENAME',
        help='Filename for saving the plot output (optional)'
    )
    
    parser.add_argument(
        '-ds_filename',
        type=str,
        metavar='FILENAME',
        help='Filename for saving the processed dataset (optional)'
    )
    
    parser.add_argument(
        '-v', '--version',
        action='version',
        version=f'era5vis_analyzeH {era5vis.__version__}'
    )
    
    np.seterr(all='ignore') #suppress warnings for invalid calculations
    try:
        parsed_args = parser.parse_args(args)
        
        lon = parsed_args.longitude
        datetime_str = parsed_args.datetime
        plot_filename = parsed_args.plot_filename
        ds_filename = parsed_args.ds_filename
        
        
        # Parse datetime
        try:
            date_time = datetime.datetime.strptime(datetime_str, "%Y-%m-%d-%H")
        except ValueError:
            print(f'Error: Invalid datetime format. Expected YYYY-MM-DD-HH, got {datetime_str}')
            sys.exit(1)
        
        # Load terrain dataset
        terrain_ds = terrain_module.load_terrain_aspect_dataset()
        lon_bounds = {'min': terrain_ds['longitude'].min().item(), 'max': terrain_ds['longitude'].max().item()}
        lat_bounds = {'min': terrain_ds['latitude'].min().item(), 'max': terrain_ds['latitude'].max().item()}
        
        # Validate longitude bounds
        if lon < lon_bounds['min'] or lon > lon_bounds['max']:
            raise ValueError(f"Longitude {lon} out of terrain bounds [{lon_bounds['min']}, {lon_bounds['max']}]")
        
        # Validate date bounds
        era_time_bounds = {'earliest': datetime.datetime(1940, 1, 1, 0),
                          'latest': datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(days=3)}
        if date_time < era_time_bounds['earliest'] or date_time > era_time_bounds['latest']:
            raise ValueError(f"Date {date_time} out of ERA5 data bounds [{era_time_bounds['earliest']}, {era_time_bounds['latest']}] (3 days ago)")
        
        # Define area around longitude
        area = [lat_bounds['min'], lon - 0.5, lat_bounds['max'], lon + 0.5]
        
        # Download ERA5 data
        file_dir = tempfile.NamedTemporaryFile(suffix='.nc', delete=True).name
        era_ds = era5.load_era5_data(file_dir, date_time, area)
        era_ds['gph'] = dynamics.geopotential_height(era_ds['z'])

        print('Download successful. Processing data... (This takes a few moments)')
        
        # Add terrain data to ERA dataset
        ds = terrain_module.interpolate_to_grid(era_ds, terrain_ds)
        ds = terrain_module.compute_terrain_intersection(ds, terrain_ds)
        ds['theta'] = dynamics.potential_temperature(ds['t'], ds['pressure_level']) #after interpolation
        
        # Compute wind interaction with terrain
        era5.compress_era(ds)
        ds = wind.compute_wind_terrain_interaction(ds, terrain_ds, range_km=3.0)
        
        # Compute Brunt-Väisälä frequency
        ds = dynamics.compute_N_H(ds)
        
        # Save dataset if filename provided
        if ds_filename:
            era5.safe_to_netcdf(ds,   ds_filename)
            print(f'Dataset saved to: {ds_filename}')
        
        # Visualize vertical cross-section
        visualization.create_plot(ds, lon, start_lat=lat_bounds['min'], end_lat=lat_bounds['max'], 
                                filepath=plot_filename)
        
    except SystemExit:
        raise
    except ValueError as e:
        print(f'Error: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        sys.exit(1)

   
def era5vis_download():
    """Entry point for the era5vis_download application script"""
    download(sys.argv[1:])

def era5vis_terrain():
    """Entry point for the era5vis_terrain application script"""
    terrain(sys.argv[1:])
    
def era5vis_visualization():
    """Entry point for the era5vis_visualization application script"""
    plot_vertical_crosssection(sys.argv[1:])

def era5vis_analyzeH():
    """Entry point for the era5vis_analyzeH application script"""
    analyzeH(sys.argv[1:])


