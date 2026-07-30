'''
Load a netCDF file into a xarray dataset
'''

import xarray as xr

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
    
    return ds

def readDOT(path: str, engine: str = 'h5netcdf') -> xr.Dataset:
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
