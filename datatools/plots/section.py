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
        label: str = "_nolegend_",
        vertical_markers:bool = False,
        vertical_line_style:str = '--',
        vertical_line_color:str = 'k',
        vertical_line_width:float = 1,
        vertical_line_depth:int = 4000
        ) -> tuple[plt.figure, plt.axes]:

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
        ds: xr.Dataset,
        x_cord_variable:str,
        draw_depth:float,
        color:str = 'grey',
        edgecolor:str = None,
        ) -> plt.axes:

    '''
    Plot the seafloor on a section based on the deepest depth of each CTD cast
    '''

    transect = ds
    
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
