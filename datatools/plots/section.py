'''
Plot various sections based on a dataset of data from a specfic cruise
'''
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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
        levels = None
        ) -> tuple[plt.Figure, plt.Axes]:

    '''
    Plot a filled contour plot
    '''
    x_mesh, y_mesh = np.meshgrid(ds[x], ds[y])

    cbar_label = f"{ds[data].attrs['long_name']} [{ds[data].attrs['units']}]"

    levels = np.atleast_1d(levels)

    if levels.size == 1:
        #Generate symetric levels array based on a max value only
        levels = np.linspace(-levels[0], levels[0], 9)
    elif levels.size == 2:
         #Generate levels array based on a min/max value
         levels = np.linspace(levels[0], levels[1], 9)

    cs = ax.contourf(x_mesh, 
                     y_mesh, 
                     ds[data].T, 
                     cmap = cmap, 
                     levels = levels,
                     extend = 'both')

    fig.colorbar(cs, ax=ax, label=cbar_label)
    ax.invert_yaxis()

    ax.set_xlabel(f"{ds[x].attrs['long_name']} [{ds[x].attrs['units']}]")
    ax.set_ylabel(f"{ds[y].attrs['long_name']} [{ds[y].attrs['units']}]")

    return fig, ax



def plot_cast_locations(
        locations,
        fig: plt.figure,
        ax: plt.axes,
        depth: float = 0.0,
        marker: str = 'v',
        markersize: int = 5,
        color: str = 'k',
        label: str = "_nolegend_",
        vertical_markers:bool = False,
        vertical_line_style:str = '--',
        vertical_line_color:str = 'k',
        vertical_line_width:float = 1,
        vertical_line_depth:int = 4000
        ) -> tuple[plt.Figure, plt.Axes]:

    '''
    Plot the location of each CTD station
    '''

    #If desired, plot vertical lines down the plot at the location of each station
    if vertical_markers:
        bottom_y_cord = np.full_like(locations, vertical_line_depth)
        for i in range (bottom_y_cord.size):
            location = locations[i]
            x_pair = [location, location]
            y_pair = [0,bottom_y_cord[i]]
            ax.plot(x_pair,
                    y_pair,
                    marker = None, 
                    color = vertical_line_color,
                    linestyle = vertical_line_style,
                    linewidth = vertical_line_width,
                    label = '_nolegend_')

    y_cord = np.full_like(locations, depth)
    #Plot the station with markers at the top of the plot
    ax.plot(locations, 
            y_cord, 
            markersize = markersize, 
            marker = marker, 
            color = color, 
            label = label, 
            linestyle = 'None',)

    return fig, ax

def plot_seafloor(
        ax: plt.axes,
        transect: xr.Dataset,
        x_cord_variable:str,
        draw_depth:float,
        color:str = 'grey',
        edgecolor:str = None,
        ) -> plt.Axes:

    '''
    Plot the seafloor on a section based on the deepest depth of each CTD cast
    '''
    
    verticies = []
    last_depth = None

    for station_id in transect['station']:
        #Get the x coordinate of the current station
        x_cord = transect.sel(station = station_id)[x_cord_variable].item()

        #Get current max depth
        current_depth_max = transect.sel(station = station_id).dropna('depth', how = 'all')['depth'].max().item()
        #Make vertex at current x coordinate at previous depth (if avaliable) and then at the current depth
        if last_depth is None:
            #If there is not a last depth set yet, establish the first vertex at the first x coordinate with a depth of the max draw depth
            verticies.append((x_cord, draw_depth))
        else:
            verticies.append((x_cord, last_depth))

        verticies.append((x_cord, current_depth_max))

        last_depth = current_depth_max

    #Establish the last vertex at the last x coordinate with a depth of the max draw depth (so the polygon closes with a straight line at the max draw depth)
    verticies.append((x_cord, draw_depth))

    #Plot the polygon
    poly_patch = Polygon(
        verticies,
        closed=True,
        facecolor = color,
        edgecolor = edgecolor,
        alpha = 1,
        label = 'Sea Floor'
    )

    ax.add_patch(poly_patch)

    return ax

def plot_isoline(ax: plt.Axes,
                transect: xr.Dataset,
                x_cord_variable:str,
                var: str,
                value: float,
                min_depth: float = 0,
                color: str = 'black',
                linewidth: float = 1,
                linestyle: str = '--',
                inline_label: bool = True,
                ) -> plt.Axes:
    '''
    Plot a single isoline (contour line) of a given variable at a given value on a
    station/depth transect, with an inline temperature label.

    Parameters
    ----------
    ax : matplotlib Axes to plot on
    transect : xr.Dataset with dims ('station', 'depth') containing var
    x_cord_variable: variable used on the x coordinate of the plot
    value : the value to draw the isoline at
    var : name of the variable in transect
    min_depth : minimum depth to consider when contouring (shallower data excluded)
    color, linewidth, linestyle : line styling
    inline_label : whether to add an inline label on the contour line
    fmt : format string for the inline label
    '''

    transect_subset = transect.sel(depth=transect['depth'] >= min_depth)

    station_mesh, depth_mesh = np.meshgrid(transect_subset[x_cord_variable], transect_subset['depth'])

    data = transect_subset[var].T

    cs = ax.contour(station_mesh, depth_mesh, data,
                     levels=[value],
                     colors=color,
                     linewidths=linewidth,
                     linestyles=linestyle)

    units = transect[var].attrs['units']
    fmt='%1.1f '+units

    if inline_label:
        ax.clabel(cs, fmt=fmt, colors=color, fontsize=9)

    return ax


def plot_transect(
        transect: xr.Dataset,
        x: str,
        y: str,
        data: str,
        fig: plt.figure,
        ax: plt.axes,
        cmap,
        title: str,
        levels = None,
        cast_locations = True,
        isohaline = True,
        seafloor = True,
        bounds = {
            'top': 0,
            'bottom': 700,
            'left': 0,
            'right': None
        },
        ) -> tuple[plt.Figure, plt.Axes]:
    
    fig, ax = plot_custom_section(
        ds = transect,
        x = x,
        y = y,
        data = data,
        fig = fig,
        ax = ax,
        cmap = cmap,
        levels = levels
    )

    ax.set_title(title)
    ax.set_ylim(top = bounds['top'], bottom=bounds['bottom'])
    ax.set_xlim(left = bounds['left'], right=bounds['right'])


    if cast_locations:
            fig, ax = plot_cast_locations(
                locations=transect[x],
                fig = fig,
                ax = ax,
                color = 'r',
                depth=30,
                vertical_markers=True
            )
    
    if seafloor:
        ax = plot_seafloor(
                ax = ax,
                transect = transect,
                x_cord_variable = x,
                draw_depth=bounds['bottom']
            )

    if isohaline:
            ax=plot_isoline(
                ax = ax,
                transect = transect,
                x_cord_variable = x,
                var = 'psal',
                value = 34.8,
            )

    ax.legend()

    return fig, ax