'''
Load a netCDF file into a xarray dataset
'''

import xarray as xr

def readLSSL(path: str) -> xr.Dataset:
    '''
    Read netCDF file containin CTD data collected on cruises on the Canadian Coast Guard Ice Breaker Louis S. St-Laurent
    path: path to the netCDF file

    returns an xarray Dataset with the data
    '''

    ds = xr.load_dataset(
        path = path, 
        chunks = 'auto')
    
    