import xarray as xr
from datetime import datetime

def save_netCDF(ds: xr.Dataset, path: str):
    ds.attrs['processed_date'] = datetime.now().isoformat()
    ds.attrs['processed_by'] = 'Ben Sprague'

    ds.to_netcdf(path, engine = "h5netcdf")