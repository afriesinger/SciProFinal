# A visualization package for ERA5 data

**era5vis** offers command line tools to display ERA5 data in your browser.

It was written for the University of Innsbruck's
[scientific programming](https://manuelalehner.github.io/scientific_programming)
course as a package template for the semester project and is based on the 
example packages [scispack](https://github.com/fmaussion/scispack) and
[climvis](https://github.com/fmaussion/climvis) written by
[Fabien Maussion](https://fabienmaussion.info).

## What is it used for?

The package helps you analyze the concept of "non-dimensional-mountain-height" in real topographic scenarios using ERA5 data as input.

## HowTo

Make sure you have all dependencies installed. These are:
    'numpy',
    'xarray',
    'matplotlib',
    'netCDF4',
    'cdsapi',
    'rasterio',
    'elevation',
    'scipy',
    'pandas',

Download the package and install it in development mode. In the root directory
type:

    $ pip install -e .

## Command line interfaces

The Package provides you serveral comand line interfaces. Choose it depending on your needs. 
After installation with python setup.py install the interfaces become avialable on your comandline.

The terrain interface era5vis_terrain
The package provides a precomputed terrain file for the ALPs. If you like to go for another area or use a different gird spacing than the provided 1km you can create a terrain_dataset by your own out of an terrain TIF file. Use era5vis_terrain -help to explore your options.

The download interface: era5vis_download
With this interface you can retrieve datasets from the CDS-datastore. Make sure you have the credentials installed. Furhter information with:
era5vis_download -h

The visualisation tool: era5vis_visualization
If you already have a propper dataset with non dimensional mountain height calculated you can use this tool for getting it visualized. 

The Do-It-All-Togehter Interface era5vis_analyzeH
Just drop a longitude and a date and you get a visualization of your case. 
example: era5vis_analyzeH -lon 12 -dt 2025-01-01-12

## Testing

Run 
pytest --cov=era5vis --cov-report=term-missing
coverage html

in the package's root directory.


## License

With the exception of the ``setup.py`` file, which was adapted from the
[sampleproject](https://github.com/pypa/sampleproject) package, all the
code in this repository is dedicated to the public domain.
