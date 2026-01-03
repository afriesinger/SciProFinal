""" contains command line tools of ERA5vis

Manuela Lehner
November 2025
"""

import sys
import webbrowser
import era5vis
from era5vis import era5

HELP_DOWNLOAD = """era5vis_download: Download ERA5 data for the Alpine region.

Usage:
    -o, --output [FILENAME]    : output filename, mandatory
    -y, --year [YEAR]          : year (YYYY), mandatory
    -m, --month [MONTH]        : month (MM), mandatory
    -d, --day [DAY]            : day (DD), mandatory
    -t, --time [TIME]          : time (HH:MM), mandatory
    -a, --area [N W S E]       : North West South East coordinates, mandatory
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
    if '--year' in args: 
        args[args.index('--year')] = '-y'
    if '--month' in args: 
        args[args.index('--month')] = '-m'
    if '--day' in args: 
        args[args.index('--day')] = '-d'
    if '--time' in args: 
        args[args.index('--time')] = '-t'
    if '--area' in args: 
        args[args.index('--area')] = '-a'

    if len(args) == 0 or args[0] in ['-h', '--help']:
        print(HELP_DOWNLOAD)
    elif args[0] in ['-v', '--version']:
        print('era5vis_download version: ' + era5vis.__version__)
    # Individual Parameter Checks 
    elif '-o' not in args:
        print('Error: Output filename (-o) is mandatory.')
    elif '-y' not in args:
        print('Error: Year (-y) is mandatory.')
    elif '-m' not in args:
        print('Error: Month (-m) is mandatory.')
    elif '-d' not in args:
        print('Error: Day (-d) is mandatory.')
    elif '-t' not in args:
        print('Error: Time (-t) is mandatory.')
    elif '-a' not in args:
        print('Error: Area coordinates (-a) are mandatory.')
    else:
        try:
            output = args[args.index('-o') + 1]
            year = args[args.index('-y') + 1]
            month = args[args.index('-m') + 1]
            day = args[args.index('-d') + 1]
            time = args[args.index('-t') + 1]
            idx = args.index('-a')
            area = [float(args[idx+1]), float(args[idx+2]), 
                    float(args[idx+3]), float(args[idx+4])]
            
            era5.load_era5_data(output, year, month, day, time, area)
            
        except IndexError:
            print('Error: A flag was provided but no value followed it.')
        except ValueError:
            print('Error: Area coordinates must be four numbers (N W S E).')
        except Exception as e:
            print(f'An unexpected error occurred: {e}')


def era5vis_modellevel():
    """Entry point for the era5vis_modellevel application script"""
    modellevel(sys.argv[1:])""" contains command line tools of ERA5vis

Manuela Lehner
November 2025
"""

import sys
import webbrowser
import datetime
import era5vis
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

def era5vis_modellevel():
    """Entry point for the era5vis_modellevel application script"""
    modellevel(sys.argv[1:])
   
def era5vis_download():
    """Entry point for the era5vis_download application script"""
    download(sys.argv[1:])
   
def era5vis_download():
    """Entry point for the era5vis_download application script"""
    download(sys.argv[1:])
