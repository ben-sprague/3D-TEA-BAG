'''
Functions for working with satelite derived dynamic ocean topography (DOT) data
from https://www.cpom.ucl.ac.uk/dynamic_topography/index.php

'''

import numpy as np
import xarray as xr

import ocean_tools as fod
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator



def get_geo_closest(
        ds: xr.Dataset,
        lat_var: str,
        lon_var: str,
        search_lat: float,
        search_lon: float
        ) -> xr.DataArray:
    '''
    Return the xarray DataArray with data from the point in a geographically gridded xarray Dataset
    closest to a given lat/lon locaiton

    Parameters
    ----------
    ds: the Dataset to search, xarray Dataset
    lat_var: the name of the variable name for latitude in the dataset, str
    lon_var: the name of the variable name for longitude in the dataset, str
    search_lat: the latitude of the location you want data for, float
    search_lon: the longitude of the location you want data for, float

    Returns
    ----------
    da: xarray DataArray with data from the point closests to the given lat/lon pair
    '''

    distances = xr.apply_ufunc(
        fod.distance_from_datum,
        ds[lat_var],
        ds[lon_var],
        kwargs={"datum": (search_lat, search_lon)},  # non-array arg passed through as-is
        input_core_dims=[["x", "y"], ["x", "y"]],   # function consumes both dims at once
        output_core_dims=[["x", "y"]],              # output has the same dims
        dask="parallelized",
        output_dtypes=[float],
        )

    cords = np.unravel_index(np.argmin(distances.values), distances.shape)
    da = ds.sel(x = cords[0], y = cords[1])

    return da


def bilinear_interp_npstere(data, x, y, target_lat, target_lon,
                             lon_0=0, lat_ts=90,
                             a=6378137.0, b=6356752.3142):
    """
    Bilinear interpolation of a field on a regular North Polar
    Stereographic grid, evaluated at arbitrary lon/lat points.

    Parameters
    ----------
    data : 2D array, shape (nx, ny)
        Gridded values, indexed as data[x, y].
    x, y : 1D arrays
        Regularly spaced, ascending grid coordinates (km) in the
        polar stereographic projection.
    target_lon, target_lat : scalar or array-like
        Points to interpolate to, in degrees.
    lon_0 : float
        Central meridian of the projection (default 0˚).
    lat_ts : float
        Latitude of true scale (default 90˚ (true polar projection)).
    a, b : float
        Ellipsoid semi-major/minor axes (default WGS84).

    Returns
    -------
    Interpolated value(s), same shape as target_lon/target_lat.
    NaN where the point falls outside the grid.
    """
    target_lon = np.atleast_1d(np.asarray(target_lon, dtype=float))
    target_lat = np.atleast_1d(np.asarray(target_lat, dtype=float))

    # Project target lon/lat into the same stereographic x/y space as the grid
    proj_str = (f"+proj=stere +lat_0=90 +lat_ts={lat_ts} +lon_0={lon_0} "
                f"+k=1 +x_0=0 +y_0=0 +a={a} +b={b} +units=km +no_defs")
    transformer = Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)
    xt, yt = transformer.transform(target_lon, target_lat)

    print((xt,yt))

    # data is indexed data[x, y], so axes tuple and query points must be (x, y)
    interpolator = RegularGridInterpolator(
        (x, y), data,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    points = np.column_stack([xt, yt])
    result = interpolator(points)

    return result.item() if result.size == 1 else result