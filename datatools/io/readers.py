'''
Load a netCDF file into a xarray dataset
'''
import numpy as np
import xarray as xr
import rioxarray as rxr

def readLSSL(path: str) -> xr.Dataset:
    '''
    Read netCDF file containing CTD data collected on cruises on the Canadian Coast Guard Ice Breaker Louis S. St-Laurent
    path: path to the netCDF file

    returns an xarray Dataset with the data
    '''

    ds = xr.load_dataset(
        path, 
        chunks = {'cruise': 1,'station': 53},
        engine="h5netcdf")

    #Convert longitude scale from 0˚-360˚ to -180˚-180˚
    ds['lon'] = (ds['lon'] + 180) % 360 - 180
    
    return ds

def readDOT(path: str, engine: str = 'h5netcdf', chunks = {'time': 50}) -> xr.Dataset:
    '''
    Read netCDF file containing satelite derived dynamic ocean topography (DOT) data
    from https://www.cpom.ucl.ac.uk/dynamic_topography/index.php

    Parameters
    ----------
    path: Path to .nc file, str
    engine: Which engine to use to open the netCDF file. h5netcdf by default, use scipy for legacy files, str

    Returns
    -------
    ds: xarray dataset with DOT data
    '''

    ds = xr.load_dataset(
        path, 
        engine=engine)
    
    return ds

def readBathymetry(path: str,
                   left: float = 140,
                   right: float = -100,
                   top: float = 90,
                   bottom: float = 68,
                   bounds: list = None) -> xr.DataArray:
    '''
    Read geoTIFF file with bathymetry data

    Parameters:
    -----------
    path: str
        Path to the geoTIFF file
    left, right, top, bottom: float
        left, right, top, and bottom lat/lon coordinates of the data
    bounds: list
        [left, right, bottom, top] lat/lon coordiante of the data. Overrides the indivudal variables when set
    
    Returns:
    --------
    da_latlon: DataArray
        DataArray of depth values in meters references by lat/lon
    '''

    if bounds is not None:
        left, right, bottom, top = bounds

    #Open the raw datafile
    da = rxr.open_rasterio(path)

    # Drop the singleton band dimension
    da = da.squeeze('band', drop=True)

    # Reproject to lat/lon if needed
    da_latlon = da.rio.reproject("EPSG:4326") #Reproject the coordinates into WGS84

    da_latlon = -da_latlon #Flip z axis so that positive is down

    #Rename y and x to lon and lat
    da_latlon = da_latlon.rename({'x': 'lon', 'y': 'lat'})

    #Mask the data to within the bounds

    lat_mask = (da_latlon['lat']>=bottom) & (da_latlon['lat']<=top)
    if left > right:
        #If the bounds cross the international date line
        lon_mask = (da_latlon['lon']>=left) | (da_latlon['lon']<=right)
    else:
        lon_mask = (da_latlon['lon']>=left) & (da_latlon['lon']<=right)

    full_lon_mask, full_lat_mask = np.meshgrid(lon_mask, lat_mask)
    mask = full_lat_mask&full_lon_mask


    da_latlon_masked = da_latlon.sel(lat = lat_mask, lon = lon_mask)
    da_latlon_masked = da_latlon_masked.sortby(da_latlon_masked['lon'])
    return da_latlon_masked
