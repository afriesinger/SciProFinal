"""Plenty of useful functions doing useful things.  """

from pathlib import Path
from tempfile import mkdtemp
import shutil


def mkdir(path, reset=False):
    """Check if directory exists and if not, create one.
        
    Parameters
    ----------
    path: str
        path to directory
    reset: bool 
        erase the content of the directory if it exists

    Returns
    -------
    path: str
        path to directory
    """
    
    if reset and Path.is_dir(path):
        shutil.rmtree(path)
    try:
        Path.mkdir(path, parents=True)
    except FileExistsError:
        pass
    return path



