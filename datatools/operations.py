'''
A set of specialty dataset opperations for working with CTD data
'''

import xarray as xr

def var_slice(ds: xr.Dataset,
          variable: str,
          min_val: float,
          max_val: float):
    '''
    Slice a dataset along a variable (rather than a coordinate)
    '''

    mask = (ds[variable] >= min_val) & (ds[variable] <= max_val)
    return ds.where(mask)
