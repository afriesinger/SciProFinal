""" contains plot functions """

from datetime import datetime

import matplotlib.pyplot as plt


def plot_horiz_cross_section(da, filepath=None):
    ''' plot horizontal cross-section

    Parameters
    ----------
    da : xarray.DataArray
        horizontal cross section
    filepath : str
        plot is saved to filepath if provided
    '''

    # set up a single set of axes
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_position([0.1, 0.1, 0.7, 0.85])
    ax.set_xlabel(r'Longitude ($^{\circ}$)')
    ax.set_ylabel(r'Latitude ($^{\circ}$)')
    time = da.valid_time.to_numpy().astype('datetime64[ms]').astype(datetime)
    ax.set_title(f'{da.long_name} at {da.pressure_level.to_numpy()} '
                 + f'{da.pressure_level.units} ({time:%d %b %Y %H:%M})', fontsize=12
                )

    cf = ax.contourf(da, levels=20)
    # add colorbar in separate axes
    cax = fig.add_axes([0.83, 0.1, 0.02, 0.85])
    plt.colorbar(cf, cax=cax)
    cax.set_ylabel(f'({da.units})')

    if filepath is not None:
        fig.savefig(filepath)
        plt.close()

    return fig
