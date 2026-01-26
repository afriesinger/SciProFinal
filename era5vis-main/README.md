# era5vis - ERA5 Data Visualization Package

> **A powerful visualization package for ERA5 climate data analysis**

[![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![License: Public Domain](https://img.shields.io/badge/License-Public%20Domain-brightgreen)](LICENSE)
[![Code style: pytest](https://img.shields.io/badge/Testing-pytest-green)]()
[![GitHub Repository](https://img.shields.io/badge/GitHub-afriesinger/SciProFinal-black?logo=github)](https://github.com/afriesinger/SciProFinal)

**era5vis** offers elegant command-line tools to visualize and analyze ERA5 atmospheric data in your browser.

## About

This package was developed for the [University of Innsbruck's Scientific Programming Course](https://manuelalehner.github.io/scientific_programming) as a semester project template, building upon the excellent foundation of [scispack](https://github.com/fmaussion/scispack) and [climvis](https://github.com/fmaussion/climvis) by [Fabien Maussion](https://fabienmaussion.info).

**GitHub Repository:** [github.com/afriesinger/SciProFinal](https://github.com/afriesinger/SciProFinal)

## What is it used for?

The package helps you analyze the concept of **non-dimensional-mountain-height** in real topographic scenarios using ERA5 data as input. Perfect for atmospheric scientists, climatologists, and mountain weather researchers!

## Quick Start

### Prerequisites

Ensure all dependencies are installed:

```
numpy
xarray
matplotlib
netCDF4
cdsapi
rasterio
elevation
scipy
pandas
```

### Installation

Clone the repository and install in development mode:

```bash
cd era5vis-main
pip install -e .
```

The command-line interfaces will be available immediately after installation.

## Command Line Interfaces

era5vis provides powerful command-line tools for different workflows:

### Terrain Interface: `era5vis_terrain`

Create custom terrain datasets from your own terrain GeoTIFF files with adjustable grid spacing. The package includes a precomputed terrain file for the Alps at 1km resolution.

```bash
era5vis_terrain -help
```

### Download Interface: `era5vis_download`

Retrieve ERA5 datasets directly from the Copernicus Data Store (CDS). Requires CDS credentials to be configured.

```bash
era5vis_download -h
```

### Visualization Tool: `era5vis_visualization`

Create stunning visualizations of pre-processed datasets with calculated non-dimensional mountain heights.

```bash
era5vis_visualization [options]
```

### All-in-One Interface: `era5vis_analyzeH`

The quickest way to analyze a location! Simply provide a longitude and date, and get a complete visualization.

```bash
era5vis_analyzeH -lon 12 -dt 2025-01-01-12
```

**Example:**
```bash
era5vis_analyzeH -lon 11.36 -dt 2026-01-20-16 -plot_filename analysis.png
```

**Tipps for usage:**
The concept of non dimensional mountain height works best in stable stratified environments. Therefor it is quite difficult to find a real case scenario that fits perfect. There is further development ongoing on handling also convective situations and give a better overview how the flow interacts with the terrain. 


## Testing

Run the comprehensive test suite with coverage reporting:

```bash
pytest --cov=era5vis --cov-report=term-missing
coverage html
```

This generates detailed coverage reports showing which parts of the code are tested.


## License

This work is dedicated to the public domain, with the exception of the `setup.py` file which was adapted from the [sampleproject](https://github.com/pypa/sampleproject) package.

---

**Made with ❤️ for atmospheric science and mountain weather research**
