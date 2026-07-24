'''
Plot various sections based on a dataset of data from a specfic cruise
'''
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

def plot_custom_section(
        ds: xr.Dataset,
        x: str,
        y: str,
        data: str,
        fig: plt.figure,
        ax: plt.axes,
        cmap,
        cbar_label: str,
        levels = None
        ) -> tuple[plt.figure, plt.axes]:

    '''
    Plot a filled contour plot
    '''
    x_mesh, y_mesh = np.meshgrid(ds[x], ds[y])

    cs = ax.contourf(x_mesh, y_mesh, ds[data].T, cmap = cmap, levels = levels)
    fig.colorbar(cs, ax=ax, label=cbar_label)
    ax.invert_yaxis()

    return fig, ax



def plot_cast_locations(
        locations,
        fig: plt.figure,
        ax: plt.axes,
        depth: float = 0.0,
        marker: str = 'v',
        markersize: int = 5,
        color: str = 'k',
        label: str = "_nolegend_"
        ) -> tuple[plt.figure, plt.axes]:

    y_cord = np.full_like(locations, depth)
    ax.plot(locations, 
            y_cord, 
            markersize = markersize, 
            marker = marker, 
            color = color, 
            label = label, 
            linestyle = 'None',)

    return fig, ax

