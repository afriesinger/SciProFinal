"""Plenty of useful functions doing useful things.  """

from pathlib import Path
from tempfile import mkdtemp
import shutil

from era5vis import cfg, graphics, era5


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


def write_html(param, level=None, time=None, time_ind=None, directory=None):
    """ Create HTML with ERA5 plot 
    
    Returns
    -------
    param : str
        variable name to plot
    level : int
        pressure level for horizontal cross section plot
    time : str
        time for horizontal cross section plot
    time_ind : int
        time index for horizontal cross section plot (time takes 
        precedence if both time and time_ind are set)
    """

    # check that dataset actually contains the selected variable, time, ...
    era5.check_data_availability(param, level=level, time=time, time_ind=time_ind)

    # create a temporary directory for the plot
    if directory is None:
        directory = mkdtemp()
    mkdir(directory)

    print('Extracting horizontal cross-section')
    if time is not None:
        hcross = era5.horiz_cross_section(param, level, time)
    elif time_ind is not None:
        hcross = era5.horiz_cross_section(param, level, time_ind)
   
    print('Plotting data')
    png = Path(directory) / f'era5_{param}_level{level}.png'
    graphics.plot_horiz_cross_section(hcross, filepath=png)

    # create HTML from template
    outpath = Path(directory) / 'index.html'
    with open(cfg.html_template, 'r') as infile:
        lines = infile.readlines()
        out = []
        for txt in lines:
            txt = txt.replace('[PLOTTYPE]', 'Horizontal cross-section')
            txt = txt.replace('[PLOTVAR]', param)
            txt = txt.replace('[IMGTYPE]', png.name)
            out.append(txt)
        with open(outpath, 'w') as outfile:
            outfile.writelines(out)

    return outpath
