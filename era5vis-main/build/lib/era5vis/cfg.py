""" Configuration module containing settings and constants. """

from pathlib import Path

#datafile = '~/scipro/data/era5_example_dataset.nc'
datafile = '/media/afriesinger/Volume/Projekte/Gleitschirmfliegen/Studium/Programming/final_project/era5vis-main/era5vis/data/era5_example_dataset.nc'

# location of data directory containing html template
pkgdir = Path(__file__).parents[0]
html_template = Path(pkgdir) / 'data' / 'template.html'
