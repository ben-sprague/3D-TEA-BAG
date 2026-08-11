'''
Tools for making various geographic plots related to CTD data
'''

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
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
                label = label if i == 0 else "_nolegend_")

    return ax

def plot_locations(ax: plt.axes, 
                      lat: xr.DataArray,
                      lon: xr.DataArray,
                      marker: str = 'o',
                      markersize: int = 2,
                      color: str = 'k',
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
                label=label if i == 0 else '_nolegend_')

    return ax
