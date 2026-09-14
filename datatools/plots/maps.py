'''
Tools for making various geographic plots related to CTD data
'''

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.contour import GeoContourSet
from cmocean import cm

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

def draw_bg_box(ax: plt.Axes, color: str = 'k', linestyle: str = "-", linewidth: float = 1.5, label: str = 'BG Box') -> plt.Axes:
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

def draw_bathymetry_contours(ax: plt.Axes,
                              bathymetry: xr.DataArray = None,
                              cmap: plt.Colormap = None,
                              linestyle: str = '-',
                              linewidth: float = 1,
                              contour_levels: list = (500, 1000, 1500, 2000),
                              contours: GeoContourSet = None
                              ) -> tuple[plt.Axes, GeoContourSet, plt.Colormap]:
    '''
    Draw bathymetry contours on a basemap.

    If `contours` is provided, reuses its precomputed contour paths instead of
    recomputing them from `bathymetry` (saves the cost of re-tracing contour
    lines on large grids). In that case `bathymetry` is not needed.

    Parameters:
    -----------
    ax: Axes
        The basemap
    bathymetry: DataArray, optional
        Gridded bathymetry data. Required only if `contours` is not provided.
    cmap: Colormap
        Colormap to color contours (default is a truncated cm.deep, teal->navy)
    linestyle: str
        Linestyle for the contours (default '-')
    linewidth: float
        Linewidth for the contours (default 1)
    contour_levels: list
        Depths to contour at (default [500,1000,1500,2000]); ignored if
        `contours` is provided (levels come from the existing object)
    contours: GeoContourSet, optional
        A previously computed contour set to redraw on `ax` instead of
        recomputing from `bathymetry`.

    Returns:
    --------
    bathy_ax: Axes
        ax with the bathymetry contours drawn on it
    contours: GeoContourSet
        the contour object (either newly created, or the one passed in)
    cmap: Colormap
        the colormap used to color the contours
    '''

    bathy_ax = ax

    if cmap is None:
        cmap = truncate_colormap(cm.deep, 0.5, 1)

    if contours is None:
        if bathymetry is None:
            raise ValueError("Must provide `bathymetry` when `contours` is not given.")

        lon_mesh, lat_mesh = np.meshgrid(bathymetry['lon'], bathymetry['lat'])
        contours = bathy_ax.contour(lon_mesh, lat_mesh, bathymetry, contour_levels,
                                     cmap=cmap,
                                     linewidths=linewidth,
                                     linestyles=linestyle,
                                     transform=ccrs.PlateCarree())
    else:
        # Reuse precomputed paths instead of re-tracing contours from the grid
        levels = contours.levels
        colors = cmap(np.linspace(0, 1, len(levels)))

        for level_idx, segments in enumerate(contours.allsegs):
            for seg in segments:
                bathy_ax.plot(seg[:, 0], seg[:, 1],
                               color=colors[level_idx],
                               linestyle=linestyle,
                               linewidth=linewidth,
                               transform=ccrs.PlateCarree())

    return bathy_ax, contours, cmap