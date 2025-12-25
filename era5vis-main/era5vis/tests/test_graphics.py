''' Test functions for graphics.py '''

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from era5vis import era5, graphics


def test_plot_horiz_cross_section_graphic(retrieve_param_level_from_ds):

    # create test dataset
    param, level = retrieve_param_level_from_ds
    da = era5.horiz_cross_section(param, level, 0)

    # call function to create figure object
    fig = graphics.plot_horiz_cross_section(da)

    # check that xlabel and ylabel text are found in figure and
    # the parameter name
    test = ['Longitude' in t.get_text() for t in fig.findobj(mpl.text.Text)]
    assert np.any(test)

    test = ['Latitude' in t.get_text() for t in fig.findobj(mpl.text.Text)]
    assert np.any(test)

    test = [da.long_name in t.get_text() for t in fig.findobj(mpl.text.Text)]
    assert np.any(test)

    plt.close('all')


def test_plot_horiz_cross_section_saving(tmpdir, retrieve_param_level_from_ds):

    # create test dataset
    param, level = retrieve_param_level_from_ds
    da = era5.horiz_cross_section(param, level, 0)

    # check that figure file is saved
    fpath = Path(tmpdir.join('timeseries.png'))
    graphics.plot_horiz_cross_section(da, filepath=fpath)
    assert Path.is_file(fpath)

    plt.close('all')
