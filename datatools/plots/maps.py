'''
Tools for making various geographic plots related to CTD data
'''

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def setup_map(figure: plt.figure,
              projection: ccrs.Projection,
              extent: list,
              land: bool = True,
              coastline: bool = True,
              gridlines: bool = True
              ) -> plt.Axes:
    '''
    Draw basemap into a figure given a predefined map projection
    Returns an axes object
    '''
    #Create axes object and set extent
    ax = figure.add_subplot(1,1,1, projection=projection)

    ax.set_extent(extent, crs=ccrs.PlateCarree())

    #Draw optional features onto the basemap
    if land:
        ax.add_feature(cfeature.LAND, facecolor="lightgray")

    if coastline:
        ax.add_feature(cfeature.COASTLINE)

    if gridlines:
        ax.gridlines(draw_labels=True)


    return ax


def plot_station_locations(ax: plt.axes, 
                      ds: xr.Dataset,
                      marker: str = 'o',
                      markersize: int = 2,
                      color: str = 'k',
                      alpha: float = 1,
                      label: str = "_nolegend_"
                      ):
    '''
    Plot the location of CTD cast stations on a basemap
    '''

    for i, cruise_id in enumerate(np.atleast_1d(ds['cruise'].values.tolist())):
        #Plot all the stations in each cruise
        subset = ds.sel(cruise = cruise_id)
        ax.plot(subset['lon'], 
                subset['lat'], 
                marker = marker, 
                markersize = markersize, 
                color = color,
                linestyle = 'None',
                transform = ccrs.PlateCarree(),
                alpha = alpha,
                label = label if i == 0 else "_nolegend_")

    return ax

def plot_locations(ax: plt.axes, 
                      lat: xr.DataArray,
                      lon: xr.DataArray,
                      marker: str = 'o',
                      markersize: int = 2,
                      color: str = 'k',
                      alpha: float = 1,
                      label: str = "_nolegend_"
                      ) -> plt.Axes:
    '''
    Plot a set of lat/lon pairs on a map
    '''
    lat = np.atleast_1d(lat)
    lon = np.atleast_1d(lon)

    for i, (lon_i, lat_i) in enumerate(zip(lon, lat)):
        ax.plot(lon_i, lat_i,
                marker=marker,
                markersize=markersize,
                color=color,
                linestyle='None',
                transform=ccrs.PlateCarree(),
                alpha = alpha,
                label=label if i == 0 else '_nolegend_')

    return ax


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

def draw_bg_box(ax: plt.axes, color: str = 'k', linestyle: str = "-", linewidth: float = 1.5, label: str = 'BG Box') -> plt.axes:
    '''
    Draw the Beaufot Gyre Box
    
    Parameters:
    -----------
    ax: Axes
        The axes to draw the box o
    color: str
        The color of the box (default 'k')
    linestryle: str
        The linestyle to draw the box with (default '-')
    linewidth: float
        The linewidth to draw the box with (default 1.5)
    label: str
        The legend label for the box (default 'BG Box')

    Returns:
    --------
    bg_box_ax: Axes
        A copy of ax with the BG box drawn on top
    '''
    bg_box_ax = ax
    
    #Plot the Beaufort Gyre Box
    top_bottom_lon= np.linspace(-170, -130, 100)
    top_lat = np.full_like(top_bottom_lon, 80.5)
    bottom_lat = np.full_like(top_bottom_lon, 70.5)
    bg_box_ax.plot([-170, -170], [70.5, 80.5], color = color, linestyle = linestyle, linewidth = linewidth, transform = ccrs.PlateCarree(), label = label)
    bg_box_ax.plot([-130, -130], [70.5, 80.5], color = color, linestyle = linestyle, linewidth = linewidth, transform = ccrs.PlateCarree())
    bg_box_ax.plot(top_bottom_lon, top_lat, color = color, linestyle = linestyle, linewidth = linewidth, transform = ccrs.PlateCarree())
    bg_box_ax.plot(top_bottom_lon, bottom_lat, color = color, linestyle = linestyle, linewidth = linewidth, transform = ccrs.PlateCarree())

    return bg_box_ax