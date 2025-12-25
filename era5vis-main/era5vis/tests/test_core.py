''' Test functions for core.py '''

from pathlib import Path

from era5vis import core


def test_mkdir(tmpdir):

    # check that directory is indeed created as a directory
    directory = str(tmpdir.join('html_dir'))
    core.mkdir(directory)
    assert Path.is_dir(Path(directory))


def test_write_html(retrieve_param_level_from_ds):

    # check that the html file is created and that the 
    # directory contains a png file
    param, level = retrieve_param_level_from_ds
    htmlfile = core.write_html(param, level=level, time_ind=0)
    assert Path.is_file(htmlfile)
    assert htmlfile.suffix == '.html'
    assert list(htmlfile.parent.glob('*.png'))
