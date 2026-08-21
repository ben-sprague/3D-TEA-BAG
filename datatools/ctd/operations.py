'''
A set of specialty dataset opperations for working with CTD data
'''

import xarray as xr
import numpy as np
from numpy.typing import ArrayLike
from numpy import linalg as LA
from pyproj import Geod
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


import ocean_tools as fod
import gsw
from ..DOT.DataServer import DataServer

def injest_CTD_transect(
        transect: xr.Dataset,
        datum: tuple,
        ) -> xr.Dataset:

    '''
    Sorts data by latitude, convert/calculate various values in a CTD transect for use in later calculations, relabel variables for uniformity, and add metadata to variables and coordinates for easy plotting. 
    By default, the following calculations/conversions are preformed:
    - Calculate coriolus parameter (fc)
    - Calculate pressure (pressure)
    - Calculate practical salinity (SP)
    - Calculate potential temperature (PT)
    - Calculate distance along the transect based on a datum (distance)
    - Calculate the thermal wind field (m/s)
    By default, metadata is added to the following variables/dimentions:
    - depth
    - pressure
    - lat
    - lon
    - distance
    - pt
    - CT
    - SP
    - SA
    - pden
    - twind


    Parameters
    ----------
    transect : xarray dataset with CTD data
    datum: tuple,
        Coordinates of datum from which to start distance calculations (lat,lon)

    Returns
    -------
    transect : modified xarray dataset with all existing CTD data plus the calculated parameters
    '''

    #Define constants
    g = 9.81 #m/s^2

    #Extract datum lat/lon
    datum_lat, datum_lon = datum

    direction = transect.attrs['direction']

    if direction == 'ns':
        #Sort transect by latitude (from south to north)
        transect = transect.sortby(transect['lat'])
    elif direction == 'ew':
        #Sort transect by latitude (from west to east)
        transect = transect.sortby(transect['lon'])
    else:
        raise ValueError(f"{direction} is not a valid input. Either pass 'ns' or 'ew'")
    
    #Add coriolus parameter as a variable
    transect['fc'] = fod.fc(transect['lat'])

    #Calculate sea pressure from depth (but do not store in dataset)
    pressure = transect['prs']

    #Calculate practical salinity from absolute salinity
    transect['SP'] = gsw.SP_from_SA(
        SA = transect['SA'].broadcast_like(transect['depth']).transpose('station', 'depth'),
        p = pressure,
        lat = transect['lat'].broadcast_like(transect['depth']).transpose('station', 'depth'),
        lon = transect['lon'].broadcast_like(transect['depth']).transpose('station', 'depth')
    )

    #Calculate potential temperature from conservative temperature
    transect['pt'] = gsw.pt_from_CT(SA = transect['SA'].broadcast_like(transect['depth']).transpose('station', 'depth'),
                                        CT = transect['CT'].broadcast_like(transect['depth']).transpose('station', 'depth'))

    #Add distance
    #Calculate distance along the transect (measured from the first point in the transect)
    distance_along_transect = np.cumsum(gsw.distance(transect['lon'], transect['lat']))/1000 #km
    distance_along_transect = np.concat((np.array([0]), distance_along_transect)) #Add 0 distance for first point

    #Add in the distance from the datum to the begining of the transect
    datum_to_transect_distance = gsw.distance([datum_lon, transect['lon'][0]], [datum_lat, transect['lat'][0]])/1000
    absolute_distance = distance_along_transect + datum_to_transect_distance

    transect['distance'] = (('station'), absolute_distance)

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
                'long_name': 'Thermal Wind Shear',
                'standard_name': 'thermal_wind_shear',
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
        'pt': {
            'units': 'degC',
            'long_name': 'Potential Temperature',
            'standard_name': 'sea_water_potential_temperature',
        },
        'CT': {
            'units': 'degC',
            'long_name': 'Conservative Temperature',
            'standard_name': 'sea_water_conservative_temperature',
        },
        'SP': {
            'units': 'PSU',
            'long_name': 'Practical Salinity',
            'standard_name': 'sea_water_practical_salinity',
        },
        'SA': {
            'units': 'g/kg',
            'long_name': 'Absolute Salinity',
            'standard_name': 'sea_water_absolute_salinity',
        },
        'pden': {
            'units': 'kg/m^3',
            'long_name': 'Potential Density Anomaly',
            'standard_name': 'sea_water_potential_density',
        },
        
    }

    for var_name, attrs in metadata.items():
        if var_name in ds.variables:
            ds[var_name].attrs.update(attrs)

    return ds


def clean_CTD_dataset(
        ds: xr.Dataset,
        distance_threshold: float = 10,
        ) -> xr.Dataset:
    '''
    Clean CTD dataset in various ways and return the cleaned dataset

    Parameters:
    -----------
    ds: Dataset
        Input Dataset with data from one cruise year
    distance_threshold: float
        How close a cast must be to be considered part of a "clump" of casts when applying the distance filter.
        Default 10km

    Returns:
    --------
    clean_ds: Dataset:
        The cleaned dataset
    '''

    clean_ds = ds
    stations_to_drop = []
    #First apply a depth minimum for all casts a certian distance offshore
    if (dir := clean_ds.attrs['direction']) == 'ns':
        #For north south, discard all casts shallower than 1100m that are more than 100km offshore
        min_distance = 100 #km
        min_depth = 600 #m
        for station_id in clean_ds['station']:
            distance = (working_station := clean_ds.sel(station = station_id))['distance']
            working_station = working_station[['SA', 'CT', 'pden']] #Only filter out casts that lack the three main measurments below 600m
            cast_depth = working_station.dropna(dim = 'depth', how = 'all')['depth'].max()
            if cast_depth < min_depth and distance > min_distance:
                stations_to_drop.append(station_id)
    elif dir == 'ew':
        #For north south, discard all casts shallower than 400m regardless of distance (because there is no point with bathymetry shallower than 400m)
        min_depth = 600 #m
        for station_id in clean_ds['station']:
            distance = (working_station := clean_ds.sel(station = station_id))['distance']
            working_station = working_station[['SA', 'CT', 'pden']] #Only filter out casts that lack the three main measurments below 600m
            cast_depth = working_station.dropna(dim = 'depth', how = 'all')['depth'].max()
            if cast_depth < min_depth:
                stations_to_drop.append(station_id)

    clean_ds = clean_ds.drop_sel(station = stations_to_drop)

    #Second, If multiple casts are taken within 5km (by default), discard all but the deepest cast
    
    stations_to_drop = np.ndarray(shape=(0,))

    #Find clumps of nearby stations
    coords = clean_ds['distance']
    n = coords.size

    #Build a KD tree with the distances along the transect
    tree = cKDTree(coords.values.reshape(-1, 1))

    #Find parts of casts that are withing the distance threshold of each other
    pairs = tree.query_pairs(r=distance_threshold)
    rows, cols = zip(*pairs) if pairs else ([], [])

    #Create an sparese row matrix of pairs and find the connected pairs
    adjacency = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_clumps, labels = connected_components(adjacency, directed=False)

    # Group into clumps
    clumps = [[] for _ in range(n_clumps)]
    for i, label in enumerate(labels):
        clumps[label].append(clean_ds['station'].values[i])
    clumps = list(c for c in clumps if len(c) > 1)

    #Discard all but the deepest cast in each clump
    for clump in clumps:
        max_depths = np.array([clean_ds.sel(station = n).dropna(dim = 'depth', how = 'all')['depth'].max().values for n in clump])
        stations_to_drop = np.concat((stations_to_drop, np.delete(clump, max_depths.argmax())))
    clean_ds = clean_ds.drop_sel(station=stations_to_drop)
            

    #Return cleaned dataset
    return clean_ds

def integrate_from_level_of_no_motion(
        transect: xr.DataArray,
        level_no_motion: float,
            ) -> xr.DataArray:

    '''
    Integrate the absolute geostrophic current based on a level of no motion

    Parameters:
    -----------
    transect: xr.Dataset
        xarray Dataset with thermal wind shears (1/s) indexed by depth (same units as level of no motion)
    level_no_motion
        Level of no motion (same units as depths)
    
    Returns:
    --------
    abs_geo_vel: xr.DataArray
        Absolute geostrophic velocity indexed by depth (level of no motion units per second, i.e. m/s)
    '''

    #First apply a depth minimum for all casts a certian distance offshore
    stations_to_drop = []
    if (dir := transect.attrs['direction']) == 'ns':
        #For north south, discard all casts shallower than the level of no motion that are more than 100km offshore
        min_distance = 100 #km
        min_depth = level_no_motion #m
        for station_id in transect['station']:
            distance = (working_station := transect.sel(station = station_id))['distance']
            working_station = working_station['twind'] #Only filter out casts that lack the three main measurments below 600m
            cast_depth = working_station.dropna(dim = 'depth', how = 'all')['depth'].max()
            if cast_depth < min_depth and distance > min_distance:
                stations_to_drop.append(station_id)
    elif dir == 'ew':
        #For north south, discard all casts shallower than the level of no motion regardless of distance (because there is no point with bathymetry shallower than 400m)
        min_depth = level_no_motion #m
        for station_id in transect['station']:
            distance = (working_station := transect.sel(station = station_id))['distance']
            working_station = working_station['twind'] #Only filter out casts that lack the three main measurments below 600m
            cast_depth = working_station.dropna(dim = 'depth', how = 'all')['depth'].max()
            if cast_depth < min_depth:
                stations_to_drop.append(station_id)
    transect = transect.drop_sel(station = stations_to_drop)
    
    twind_shear = transect['twind']

    #Split off data that doesn't reach the level of no motion, and integrate up from the bottom
    mask = twind_shear.sel(depth = level_no_motion, method = 'bfill').isnull().broadcast_like(twind_shear['depth'])
    shallow_twind_shear = twind_shear.where(mask, drop=True)
    shallow_agv = xr.full_like(shallow_twind_shear.where(shallow_twind_shear['depth'] <= level_no_motion, drop = True), np.nan)
    deep_twind_shear = twind_shear.where(~mask, drop=True)
    deep_agv = xr.full_like(deep_twind_shear, np.nan)

    #Integrate data from the bottom to the surface
    for station_id in shallow_twind_shear['station']:
        #Get valid (ie. not nan) data for that specfic station
        working_twind_shear = shallow_twind_shear.sel(station = station_id).dropna(dim = 'depth', how = 'all')

        #Flip array so integration is from bottom to surface
        working_twind_shear = working_twind_shear.isel(depth=slice(None, None, -1))

        #Integrate from bottom to surface and flip the array back so it goes from surface to bottom
        working_shallow_agv = working_twind_shear.cumulative_integrate(coord='depth').isel(depth=slice(None, None, -1))

        shallow_agv.loc[{'station': station_id, 'depth': working_shallow_agv['depth']}] = working_shallow_agv

    if ~mask.all():
        #Take all other data not masked
        deep_twind_shear = twind_shear.where(~mask, drop=True)

        #Dump off all data deeper than the level of no motion (current there assumed to be zero)
        deep_twind_shear = deep_twind_shear.where(deep_twind_shear['depth'] <= level_no_motion, drop=True)

        #Flip array so integration is from bottom to surface
        deep_twind_shear = deep_twind_shear.isel(depth=slice(None, None, -1))

        #Integrate from bottom to surface and flip the array back so it goes from surface to bottom
        deep_agv = deep_twind_shear.cumulative_integrate(coord='depth').isel(depth=slice(None, None, -1))

    abs_geo_vel = xr.concat((shallow_agv, deep_agv), dim='station', join="outer")

    #Add back nan below the level of no motion so array size matches with the rest of the dataset
    padding = xr.full_like(twind_shear.where(twind_shear['depth'] > level_no_motion, drop=True), np.nan)

    abs_geo_vel = xr.concat((abs_geo_vel, padding), dim = 'depth', join="outer")

    #Add metadata to DataArray
    depth_units = abs_geo_vel['depth'].attrs['units']
    abs_geo_vel.attrs.update({
                'units': f"{depth_units}/s",
                'long_name': 'Absolute Geostrophic Velocity',
                'standard_name': 'absolute_geostrophic_velocity',
            })

    return abs_geo_vel

def integrate_from_surface_current(
        transect: xr.Dataset,
        sat_data_server: DataServer,
        sat_data_priority: str,
        transect_direction: str,
            ) -> xr.DataArray:
    '''
    Integrate the absolute geostrophic current based on a level of no motion

    Parameters:
    -----------
    transect: xr.Dataset
        xarray Dataset with thermal wind shears (1/s) indexed by depth (m)
    sat_data_server: DataServer
        Satellite derived surface geostrophic currents
    sat_data_priority: str
        Data priority string to pass through to the satellite DataServer (either 'old' or 'new')
    transect_direction: str
        String denoting the direction of the transect (either 'ns' for north-south or 'ew' for east-west)
    
    Returns:
    --------
    abs_geo_vel: xr.DataArray
        Absolute geostrophic velocity indexed by depth (m/s)
    '''

    #Get the vector normal to the transect at each station
    dlat_dlon = np.gradient(transect['lat'], transect['lon']).reshape((-1,1))

    if transect_direction == 'ns':
        #North-south transect
        norm_vec = np.hstack((dlat_dlon, np.ones_like(dlat_dlon)))

        #For north-south transect, always have normal vector point west (where west is the negative x-direction)
        mask = norm_vec[:,0] > 0 #All rows where the x-component is greater than zero (ie. pointing east instead of west)
        norm_vec[mask,:] = -norm_vec[mask,:]
    elif transect_direction == 'ew':
        #East-west transect
        norm_vec = np.hstack((dlat_dlon, np.ones_like(dlat_dlon)))

        #For east-west transect, always have normal vector point north (where north is the positive y-direction)
        mask = norm_vec[:,1] < 0 #All rows where the y-component is less than zero (ie. pointing south instead of north)
        norm_vec[mask,:] = -norm_vec[mask,:]

    unit_norm_vec = norm_vec/(LA.norm(norm_vec, axis=1).reshape((-1,1)))

    #Get the geostrophic surface current at each station
    _, u_geo_surf_current, v_geo_surf_current = sat_data_server.get(transect['date'], transect['lat'], transect['lon'], priority=sat_data_priority)
    geo_surf_current_vector = np.column_stack((u_geo_surf_current, v_geo_surf_current)) 

    geo_surf_current_proj = np.sum(geo_surf_current_vector * unit_norm_vec, axis=1)

    #Isolate the thermal wind shear DataArray
    twind_shear = transect['twind']

    #Split off stations with a missing value at the first depth level
    mask = twind_shear.sel(depth = twind_shear['depth'][0]).isnull()
    if mask.any():
        # #If any stations have a missing value at the first depth level
        # missing_first_depth_twind_shear = twind_shear.where(mask.broadcast_like(twind_shear['depth']), drop=True)
        # missing_first_depth_surf_current = geo_surf_current_proj[mask].reshape((-1,1))

        # #Remove missing first depth level from data
        # missing_first_depth_twind_shear = missing_first_depth_twind_shear.sel(depth = twind_shear['depth'][1:])

        # #Integrate data from the surface to the bottom and add the surface current
        # missing_first_depth_agv = missing_first_depth_twind_shear.cumulative_integrate(coord='depth')+missing_first_depth_surf_current
        
        #If any stations have a missing value at the first depth level
        missing_first_depth_twind_shear = twind_shear.where(mask.broadcast_like(twind_shear['depth']), drop=True)
        missing_first_depth_surf_current = geo_surf_current_proj[mask].reshape((-1,1))
    
        missing_first_depth_agv = xr.full_like(missing_first_depth_twind_shear, np.nan)
    
        #Integrate data from the first valid value to the bottom
        for i, station_id in enumerate(missing_first_depth_twind_shear['station']):
            #Get valid (ie. not nan) data for that specfic station
            working_twind_shear = missing_first_depth_twind_shear.sel(station = station_id).dropna(dim = 'depth', how = 'all')
            working_surface_current = missing_first_depth_surf_current[i]
    
            #Integrate from bottom to surface and flip the array back so it goes from surface to bottom
            working_agv = working_twind_shear.cumulative_integrate(coord='depth').isel(depth=slice(None, None, -1))+working_surface_current
    
            missing_first_depth_agv.loc[{'station': station_id, 'depth': working_agv['depth']}] = working_agv

    if not mask.all():
        #If any stations don't have a missing value at the first depth level
        #Integrate all other stations
        complete_station_twind_shear = twind_shear.where(~mask.broadcast_like(twind_shear['depth']), drop=True)
        complete_station_surf_current = geo_surf_current_proj[~mask].reshape((-1,1))

        #Integrate data from the surface to the bottom and add the surface current
        complete_station_agv = complete_station_twind_shear.cumulative_integrate(coord='depth')+complete_station_surf_current

    if mask.all():
        #All stations did not have data at the first depth
        abs_geo_vel = missing_first_depth_agv
    elif not mask.any():
        #All stations had data at the first depth
        abs_geo_vel = complete_station_agv
    else:
        #If there were stations both with and without data at the first depth
        abs_geo_vel = xr.concat((missing_first_depth_agv, complete_station_agv), dim='station', join="outer")

    #Add metadata to DataArray
    depth_units = abs_geo_vel['depth'].attrs['units']
    abs_geo_vel.attrs.update({
                'units': f"{depth_units}/s",
                'long_name': 'Absolute Geostrophic Velocity',
                'standard_name': 'absolute_geostrophic_velocity',
            })

    return abs_geo_vel

def var_slice(ds: xr.Dataset,
          variable: str,
          min_val: float,
          max_val: float) -> xr.Dataset:
    '''
    Slice a dataset along a variable (rather than a coordinate)
    '''

    mask = (ds[variable] >= min_val) & (ds[variable] <= max_val)
    return ds.where(mask, drop=True)

def cross_track_distance(lat1, lon1, lat2, lon2, lat3, lon3):
    geod = Geod(ellps="WGS84")

    lat3 = np.atleast_1d(np.asarray(lat3, dtype=float))
    lon3 = np.atleast_1d(np.asarray(lon3, dtype=float))
    n = lat3.size

    # Distance + bearing from point 1 -> point 2 (the line)
    az12, _, dist12 = geod.inv(lon1, lat1, lon2, lat2)
    az12 = float(az12)
    dist12 = float(dist12)

    # Distance + bearing from point 1 -> each station
    az13, _, dist13 = geod.inv(
        np.full(n, lon1), np.full(n, lat1), lon3, lat3
    )
    az13 = np.asarray(az13, dtype=float).reshape(lon3.shape)
    dist13 = np.asarray(dist13, dtype=float).reshape(lon3.shape)

    R = 6371008.8  # mean Earth radius (m)
    delta13 = dist13 / R

    az12_rad = np.deg2rad(az12)
    az13_rad = np.deg2rad(az13)

    dxt = np.arcsin(np.sin(delta13) * np.sin(az13_rad - az12_rad)) * R  # shape (n,)

    return dxt, dist12, dist13

def var_slice_line(ds: xr.Dataset,
          lat1: float,
          lon1: float,
          lat2: float,
          lon2: float,
          tolerance: float = 1) -> xr.Dataset:
    '''
    Slice a dataset along a geographic line (within a certian tolerance)

    Parameters:
    -----------
    ds: Dataset
        xarray Dataset to slice
    lat1: 
        Starting latitude
    lon1: float
        Starting longitude
    lat2: float
        Ending latitude
    lon2: float
        Ending longitude
    tolerance: float
        Maximum allowable distance from the line to be included in the dataset (km). Default 1km
        
    Returns:
    --------
    sliced_ds: Dataset
        xarray Dataset with the only the data along the line
    '''


    lat3 = np.atleast_1d(ds['lat'].values)
    lon3 = np.atleast_1d(ds['lon'].values)

    dist, _, _ = cross_track_distance(lat1, lon1, lat2, lon2, lat3, lon3)

    dist_da = xr.DataArray(np.abs(dist), dims=['cruise','station'], coords={'cruise': ds['cruise'], 'station': ds['station']})

    mask = dist_da <= tolerance*1e3

    sliced_ds = ds.where(mask, drop=True)

    return sliced_ds


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

def zip_by_coordinate(da_dict: dict,
          coordinate: str
          ) -> xr.Dataset:
    '''
    Zip dictonary of dataarrays of dimension n-1 along a coorinate into a n-dimension dataset
    '''

    da_to_combine = list(da_dict[key] for key in da_dict.keys())

    comb_ds = xr.concat(da_to_combine, dim=coordinate, join = 'outer')

    return comb_ds

def to_dict_by_cruise(ds: xr.Dataset, datum: tuple = None, process:bool = True) -> dict:
    '''
    Split n-dimension dataset into an array of datasets of dimension n-1 along a coorinate and injest the data
    
    Parameters:
    -----------
    ds: Dataset
        Dataset with data from a single transect over multiple years
    datum: tuple,
            Coordinates of datum from which to start distance calculations (lat,lon)
    process: bool
        Should the CTD data be processed beyond splitting it into a dictonary by cruise year (default True)

    Returns:
    --------
    transect_dict: dict
        Dictonary with keys of cruise years and values of Datasets with the transect data from a single year
    '''

    #Split data into a dataset per cruise 
    transect_dict = split_by_coordinate(ds, 'cruise')
    if process:
        if datum is None:
            raise KeyError('Datum is required to process data')
        #Clean each dataset
        for key in transect_dict:
            #Clean and injest the transect data (add many new and useful variables)
            transect_dict[key] = injest_CTD_transect(transect_dict[key], datum=datum)
    else:
        #Just sort by distance
        for key in transect_dict:
            working_transect = transect_dict[key]
            transect_dict[key] = working_transect.sortby(working_transect['distance'])

    return transect_dict