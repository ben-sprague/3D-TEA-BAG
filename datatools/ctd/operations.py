'''
A set of specialty dataset opperations for working with CTD data
'''

import xarray as xr
import numpy as np
import ocean_tools as fod
import gsw

def injest_CTD_transect(
        transect: xr.Dataset,
        datum: tuple,
        ) -> xr.Dataset:

    '''
    Sorts data by latitude, convert/calculate various values in a CTD transect for use in later calculations, relabel variables for uniformity, and add metadata to variables and coordinates for easy plotting. 
    By default, the following calculations/conversions are preformed:
    - Calculate coriolus parameter (fc)
    - Calculate absolute salinity (asal)
    - Calculate conservative temperature (ctmp)
    - Calculate distance along the transect based on a datum (distance)
    - Calculate the thermal wind field (m/s)

    By default, the following variables are relables:
    - Rename practical salinity (psal)
    - Rename in-situ temperature (itmp)

    By default, metadata is added to the following variables/dimentions:
    - depth
    - lat
    - lon
    - distance
    - itmp
    - ptmp
    - ctmp
    - psal
    - asal
    - pden
    - twind


    Parameters
    ----------
    transect : xarray dataset with CTD data
    datum: lat/lon coordinate of datum from which to calculate distance, Tuple (lat, lon)


    Returns
    -------
    transect : modified xarray dataset with all existing CTD data plus the calculated parameters
    '''

    #Define constants
    g = 9.81 #m/s^2

    #Sort transect by latitude
    transect = transect.sortby(transect['lat'])
    
    #Add coriolus parameter as a variable
    transect['fc'] = fod.fc(transect['lat'])

    #Calculate sea pressure from depth (but do not store in dataset)
    pressure = gsw.p_from_z(
        z = -np.asarray(transect['depth'].broadcast_like(transect['lat']).transpose('station', 'depth')),
        lat = transect['lat'].broadcast_like(transect['depth']).transpose('station', 'depth'))

    #Rename practical salinity from 'sal' to 'psal'
    transect = transect.rename_vars({'sal': 'psal'})

    #Calculate practical salinity from absolute salinity
    transect['asal'] = gsw.SA_from_SP(
        SP = transect['psal'].broadcast_like(transect['depth']).transpose('station', 'depth'),
        p = pressure,
        lat = transect['lat'].broadcast_like(transect['depth']).transpose('station', 'depth'),
        lon = transect['lon'].broadcast_like(transect['depth']).transpose('station', 'depth')
    )

    #Convert potential temperature to conservative temperature
    transect['ctmp'] = gsw.CT_from_pt(SA = transect['asal'].broadcast_like(transect['depth']).transpose('station', 'depth'),
                                        pt = transect['ptmp'].broadcast_like(transect['depth']).transpose('station', 'depth'))

    #Add distance
    transect['distance'] = (('station'), fod.distance_from_datum(transect['lat'], transect['lon'], datum)[0])

    #Rename in-situ temperature from 'TEMP' to 'itmp'
    transect = transect.rename_vars({'TEMP': 'itmp'})

    #Add metadata
    transect = set_transect_metadata(transect)

    #Clean Dataset
    transect = clean_CTD_dataset(transect)

    #Calculate the change in pden with distance
    drdx = transect.swap_dims({'station':'distance'})['pden'].differentiate('distance')/1000 #(kg/m^3)/(km) * 1km/1000m = (kg/m^3)/(m) = kg/m^4
    drdx = drdx.swap_dims({'distance':'station'}) #Swap coordinates back for contunity

    #Calculate thermal wind (du/dz, 1/s)
    rho = transect['pden']+1000 #Convert from potential density anamoly to potential density (kg/m^3)
    thermal_wind = g/transect['fc']/rho*drdx
    transect['twind'] = thermal_wind.reset_coords('distance', drop=True)

    #Update thermal wind metadata
    transect['twind'].attrs.update({
                'units': '1/s',
                'long_name': 'Thermal Wind',
                'standard_name': 'thermal_wind',
            })

    return transect

def set_transect_metadata(ds: xr.Dataset) -> xr.Dataset:
    '''
    Set standard CF-convention metadata (units, long_name, standard_name, positive)
    on common oceanographic variables in a transect/CTD dataset.
    '''

    metadata = {
        'depth': {
            'units': 'm',
            'long_name': 'Depth',
            'standard_name': 'depth',
            'positive': 'down',
        },
        'lat': {
            'units': 'degrees_north',
            'long_name': 'Latitude',
            'standard_name': 'latitude',
        },
        'lon': {
            'units': 'degrees_east',
            'long_name': 'Longitude',
            'standard_name': 'longitude',
        },
        'distance': {
            'units': 'km',
            'long_name': 'Distance Along Transect',
            'standard_name': 'distance',
        },
        'pressure': {
            'units': 'dbar',
            'long_name': 'Sea Water Pressure',
            'standard_name': 'sea_water_pressure',
            'positive': 'down',
        },
        'itmp': {
            'units': 'degC',
            'long_name': 'In-Situ Temperature',
            'standard_name': 'sea_water_temperature',
        },
        'ptmp': {
            'units': 'degC',
            'long_name': 'Potential Temperature',
            'standard_name': 'sea_water_potential_temperature',
        },
        'ctmp': {
            'units': 'degC',
            'long_name': 'Conservative Temperature',
            'standard_name': 'sea_water_conservative_temperature',
        },
        'psal': {
            'units': 'PSU',
            'long_name': 'Practical Salinity',
            'standard_name': 'sea_water_practical_salinity',
        },
        'asal': {
            'units': 'g/kg',
            'long_name': 'Absolute Salinity',
            'standard_name': 'sea_water_absolute_salinity',
        },
        'pden': {
            'units': 'kg/m^3',
            'long_name': 'Potential Density',
            'standard_name': 'sea_water_potential_density',
        },
        
    }

    for var_name, attrs in metadata.items():
        if var_name in ds.variables:
            ds[var_name].attrs.update(attrs)

    return ds


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
        valid_stations = deep_data.dropna('station', how='all', subset=['ptmp','psal'])['station']
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

def integrate_level_of_no_motion(
        twind_shear: xr.DataArray,
        level_no_motion: float,
            ) -> xr.DataArray:

    '''
    Integrate the absolute geostrophic current based on a level of no motion

    Parameters:
    -----------
    twind_shear: xr.DataArray
        Thermal wind shears (1/s) indexed by depth (same units as level of no motion)
    level_no_motion
        Level of no motion (same units as depths)
    
    Returns:
    --------
    abs_geo_vel: xr.DataArray
        Absolute geostrophic velocity indexed by depth (level of no motion units per second, i.e. m/s)
    '''

    #Integrate from surface to level of no motion
    shallow_twind_shear = twind_shear.where(twind_shear['depth']<= level_no_motion)
    shallow_agv = -shallow_twind_shear.integrate(coord='depth')

    #Integrate from level of no motion to bottom
    deep_twind_shear = twind_shear.where(twind_shear['depth'] > level_no_motion)
    deep_agv = deep_twind_shear.integrate(coord='depth')

    abs_geo_vel = xr.concat((shallow_agv, deep_agv))

    return abs_geo_vel


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
