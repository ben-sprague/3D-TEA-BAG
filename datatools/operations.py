'''
A set of specialty dataset opperations for working with CTD data
'''

import xarray as xr
import numpy as np

def clean_CTD_dataset(
        ds: xr.Dataset,
        min_depth: int = 1100,
        min_lat: float = 71,
        min_dist: float = 0.05,
        depth_filter: bool = True,
        dist_filter: bool = True,
        ) -> xr.Dataset:
    '''
    Clean CTD dataset in various ways and return the cleaned dataset
    '''

    clean_ds = ds

    if depth_filter:
        #First step, by default remove casts shallower than 1100m that are north of 71 degrees north 
        #Filter data north of 70.25N and south of (or on) 71N
        lat_mask = clean_ds['lat'] > min_lat
        deep_data = clean_ds.where(lat_mask, drop = True).sel(depth = clean_ds['depth'] >= min_depth)
        valid_stations = deep_data.dropna('station', how='all', subset=['TEMP','sal'])['station']
        north_stations = clean_ds.sel(station = valid_stations)

        #Add two steps and add back data south of 71N
        south_data = deep_data = clean_ds.where(clean_ds['lat'] <= min_lat, drop = True)
        clean_ds = xr.concat([south_data, north_stations], dim='station')

    if dist_filter:
        #Second Step, if two casts are taken within 0.05 (by default) degrees of latitude, discard the shallower cast

        stations_to_drop = []

        for i in range(clean_ds['station'].size - 1):
            current_lat = clean_ds['lat'][i]
            next_lat = clean_ds['lat'][i + 1]

            if np.abs(current_lat - next_lat) < min_dist:
                # Casts are close together — compare their max valid depth
                current_station = clean_ds['station'][i].item()
                next_station = clean_ds['station'][i + 1].item()

                current_depth = clean_ds.isel(station=i).dropna('depth')['depth'].max()
                next_depth = clean_ds.isel(station=i + 1).dropna('depth')['depth'].max()

                if current_depth < next_depth:
                    stations_to_drop.append(current_station)
                else:
                    stations_to_drop.append(next_station)

        clean_ds = clean_ds.drop_sel(station=stations_to_drop)
            

    #Return cleaned dataset
    return clean_ds


def var_slice(ds: xr.Dataset,
          variable: str,
          min_val: float,
          max_val: float):
    '''
    Slice a dataset along a variable (rather than a coordinate)
    '''

    mask = (ds[variable] >= min_val) & (ds[variable] <= max_val)
    return ds.where(mask, drop=True)

def split_by_coordinate(ds: xr.Dataset,
          coordinate: str
          ) -> dict:
    '''
    Split n-dimension dataset into an array of datasets of dimension n-1 along a coorinate
    '''

    split_data = {}

    for id, split_ds in ds.groupby(coordinate):
        split_data[id] = split_ds.sel(**{coordinate: id}).dropna('depth', how='all').dropna('station', how='all')

    return split_data