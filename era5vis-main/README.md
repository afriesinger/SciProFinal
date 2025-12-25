# A visualization package for ERA5 data

**era5vis** offers command line tools to display ERA5 data in your browser.

It was written for the University of Innsbruck's
[scientific programming](https://manuelalehner.github.io/scientific_programming)
course as a package template for the semester project and is based on the 
example packages [scispack](https://github.com/fmaussion/scispack) and
[climvis](https://github.com/fmaussion/climvis) written by
[Fabien Maussion](https://fabienmaussion.info).

## HowTo

Make sure you have all dependencies installed. These are:
- numpy
- xarray
- netcdf4
- matplotlib
- pytest

Download the package and install it in development mode. In the root directory
type:

    $ pip install -e .

## Command line interface

``setup.py`` defines an "entry point" for a script to be used as a
command line program. Currently, the only command installed is ``era5vis_modellevel``.

After installation, just type

    $ era5vis_modellevel --help

to see what the tool can do.

## Testing

I recommend to use [pytest](https://docs.pytest.org) for testing. To test
the package, run

    $ pytest .

in the package's root directory.


## License

With the exception of the ``setup.py`` file, which was adapted from the
[sampleproject](https://github.com/pypa/sampleproject) package, all the
code in this repository is dedicated to the public domain.
