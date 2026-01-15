""" contains command line tools of ERA5vis

Manuela Lehner
November 2025
"""

import sys
import webbrowser
import datetime
import era5vis
import argparse
from era5vis import terrain as terrain_module
from era5vis import era5

HELP_DOWNLOAD = """era5vis_download: Download ERA5 data for the Alpine region.

Usage:
    -o, --output [FILENAME]    : output filename, mandatory
    -s, --start [YYYY-MM-DD-HH]: start date and hour, mandatory
    -e, --end [YYYY-MM-DD-HH]  : end date and hour, optional (for time series)
    -a, --area [N W S E]       : North West South East coordinates (spaces only), mandatory
                                 (Limits: N<=48, S>=45, W>=6, E<=16)
"""

HELP = """era5vis_modellevel: Visualization of ERA5 at a given model level.

Usage:
   -h, --help                       : print the help
   -v, --version                    : print the installed version
   -p, --parameter [PARAM]          : ERA5 variable to plot, mandatory
   -lvl, --level [LEVEL]            : pressure level to plot (hPa), mandatory
   -t, --time [TIME]                : time to plot (YYYYmmddHHMM)
   -ti, --time_index [TIME_IND]     : time index within dataset to plot (--time takes 
                                      precedence of both --time and --time_index are specified
                                      (default=0)
   --no-browser                     : the default behavior is to open a browser with the
                                      newly generated visualisation. Set to ignore
                                      and print the path to the html file instead
"""


def modellevel(args):
    """The actual era5vis_modellevel command line tool.

    Parameters
    ----------
    args: list
        output of sys.args[1:]
    """

    if '--parameter' in args:
        args[args.index('--parameter')] = '-p'
    if '--level' in args:
        args[args.index('--level')] = '-lvl'
    if '--time' in args:
        args[args.index('--time')] = '-t'
    if '--time_index' in args:
        args[args.index('--time_index')] = '-ti'

    if len(args) == 0:
        print(HELP)
    elif args[0] in ['-h', '--help']:
        print(HELP)
    elif args[0] in ['-v', '--version']:
        print('era5vis_modellevel: ' + era5vis.__version__)
        print('Licence: public domain')
        print('era5vis_modellevel is provided "as is", without warranty of any kind')
    # parameter and level must be provided, time/time_ind are optional
    elif ('-p' in args) and ('-lvl' in args):
        param = args[args.index('-p') + 1]
        level = int(args[args.index('-lvl') + 1])
        if ('-t' in args):
            time = args[args.index('-t') + 1]
            html_path = era5vis.write_html(param, level=level, time=time)
        elif ('-ti' in args):
            time = int(args[args.index('-ti') + 1])
            html_path = era5vis.write_html(param, level=level, time_ind=time)
        else:
            print('No time provided, using default (first time in the file)')
            html_path = era5vis.write_html(param, level=level, time_ind=0)
        if '--no-browser' in args:
            print('File successfully generated at: ' + str(html_path))
        else:
            webbrowser.get().open_new_tab('file://' + str(html_path))
    else:
        print('era5vis_modellevel: command not understood. '
              'Type "era5vis_modellevel --help" for usage information.')


def download(args):
    """The actual era5vis_download command line tool.
    
    Parameters
    ----------
    args: list
        output of sys.args[1:]
    
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
    elif args[0] in ['-v', '--version']:
        print('era5vis_download version: ' + era5vis.__version__)
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
        except Exception as e:
            print(f'An unexpected error occurred: {e}')
    else:
        print('era5vis_download: command not understood or mandatory arguments missing.'
              'Type "era5vis_download --help" for usage information.')

def terrain(args):
    """The actual era5vis_terrain command line tool.
    
    Recreate terrain datasets from GeoTIFF files with optional resolution control.
    
    Parameters
    ----------
    args: list
        output of sys.argv[1:]
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

## TODO ADD here verticalH()
## The main logic!



def era5vis_modellevel():
    """Entry point for the era5vis_modellevel application script"""
    modellevel(sys.argv[1:])
   
def era5vis_download():
    """Entry point for the era5vis_download application script"""
    download(sys.argv[1:])

def era5vis_terrain():
    """Entry point for the era5vis_terrain application script"""
    terrain(sys.argv[1:])


## TODO ADD HERE era5vis_verticalH 
## The Main Function to use it all together
