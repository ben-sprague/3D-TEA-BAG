'''
Plot various 2D plots based on a dataset of data from a specfic cruise
'''
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import xarray as xr
import numpy as np

def plot_variable(
        transect: xr.Dataset,
        x: str,
        y: str,
        fig: plt.figure,
        ax: plt.axes,
        title: str,
        marker:str = 'o', 
        color:str = 'k',
        linestyle:str = '-',
        linewidth:float = 1,
        bounds = {
            'top': None,
            'bottom': None,
            'left': 0,
            'right': None
        },
        legend: bool = True,
        xlabel: bool = True,
        ylabel: bool = True,
        ) -> tuple[plt.Figure, plt.Axes]:

    ax.plot(transect[x], transect[y], marker = marker, color = color, linestyle = linestyle, linewidth = linewidth)
    
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(f"{transect[x].attrs['long_name']} [{transect[x].attrs['units']}]")
    if ylabel:
        ax.set_ylabel(f"{transect[y].attrs['long_name']} [{transect[y].attrs['units']}]")
    
    ax.set_ylim(top = bounds['top'], bottom=bounds['bottom'])
    ax.set_xlim(left = bounds['left'], right=bounds['right'])
    
        
    if legend:
        ax.legend()

    return fig, ax
