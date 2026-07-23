'''
Tools for making various geographic plots related to CTD data
'''

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
              ) -> plt.axes:
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

    for i, cruise_id in enumerate(ds['cruise'].values.tolist()):
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
