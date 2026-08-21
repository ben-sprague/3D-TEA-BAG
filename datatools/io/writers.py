import xarray as xr
from datetime import datetime

from ..ctd.operations import zip_by_coordinate

def save_netCDF(ds: xr.Dataset, path: str):
    ds.attrs['processed_date'] = datetime.now().isoformat()
    ds.attrs['processed_by'] = 'Ben Sprague'

    ds.to_netcdf(path, engine = "h5netcdf")


def save_netCDF_from_dict(da_dict: dict, path: str):
    ds = zip_by_coordinate(da_dict, 'cruise')
    ds.attrs['processed_date'] = datetime.now().isoformat()
    ds.attrs['processed_by'] = 'Ben Sprague'

    ds.to_netcdf(path, engine = "h5netcdf")

    