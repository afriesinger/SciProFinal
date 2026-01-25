''' Test functions for core.py '''

from pathlib import Path

from era5vis import core


def test_mkdir(tmpdir):

    # check that directory is indeed created as a directory
    directory = str(tmpdir.join('html_dir'))
    core.mkdir(directory)
    assert Path.is_dir(Path(directory))
