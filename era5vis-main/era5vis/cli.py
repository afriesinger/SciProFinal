""" contains command line tools of ERA5vis

Manuela Lehner
November 2025
"""

import sys
import webbrowser

import era5vis

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


def era5vis_modellevel():
    """Entry point for the era5vis_modellevel application script"""
    modellevel(sys.argv[1:])
