""" Test functions for cli """

import era5vis
from era5vis.cli import modellevel


def test_help(capsys):

    # check that empty arguments return the help
    modellevel([])
    captured = capsys.readouterr()
    assert 'Usage:' in captured.out
    print(captured.out)

    # check that -h and --help return the help
    modellevel(['-h'])
    captured = capsys.readouterr()
    assert 'Usage:' in captured.out

    modellevel(['--help'])
    captured = capsys.readouterr()
    assert 'Usage:' in captured.out


def test_version(capsys):

    # check that -v and --version return version information
    modellevel(['-v'])
    captured = capsys.readouterr()
    assert era5vis.__version__ in captured.out

    modellevel(['--version'])
    captured = capsys.readouterr()
    assert era5vis.__version__ in captured.out


def test_print_html(capsys, retrieve_param_level_time_from_ds):

    param, level, time = retrieve_param_level_time_from_ds

    # check that correctly formatted calls run successfully
    modellevel(['-p', param, '-lvl', level, '--no-browser'])
    captured = capsys.readouterr()
    assert 'File successfully generated at:' in captured.out

    modellevel(['-p', param, '-lvl', level, '-ti', '0', '--no-browser'])
    captured = capsys.readouterr()
    assert 'File successfully generated at:' in captured.out

    modellevel(['-p', param, '-lvl', level, '-t', '202510010000', '--no-browser'])
    captured = capsys.readouterr()
    assert 'File successfully generated at:' in captured.out


def test_error(capsys):

    # check that incorrectly formatted calls raise an error
    modellevel(['-p', 'z'])
    captured = capsys.readouterr()
    assert 'command not understood' in captured.out
