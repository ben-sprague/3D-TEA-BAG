from datetime import datetime

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

from ..io.readers import readDOT

class DataServer:
    def __init__(self, old_data_path, new_data_path):
        '''
        A class to merge new and old satelite derived dynamic ocean topography (DOT) datasets
        from https://www.cpom.ucl.ac.uk/dynamic_topography/index.php and serve the merged dataset

        Attributes
        ----------
        sat_ds_new : xarray Dataset
            Dataset with the new DOT data
        sat_ds_old : xarray Dataset
            Dataset with the old DOT data
        start_time : datetime
            Starting time for the combined dataset
        overlap_start_time : datetime
            Time that the overlap between the two datasets starts
        overlap_end_time : datetime
            Time that the overlap between the two datasets ends
        end_time : datetime
            Time that the two datasets end

        Methods
        -------
        __merge
            Merge the overlapping portion of the two datasets
        get(time, lat, lon)
            Get a value from the merged dataset
        __closest_time
            Return the time closest to a given query
        '''

        self.sat_ds_new = readDOT(new_data_path)
        self.sat_ds_old = readDOT(old_data_path, engine = 'scipy')

        #Convert both dataset times into datatime objects

        #Old dataset
        #Old dataset averages data over a month and reports the year and month in the following format: YYYYMM
        self.sat_ds_old['date'] = pd.to_datetime(self.sat_ds_old['date'].astype(str).values, format="%Y%m")

        #Rename date to time for consistency
        self.sat_ds_old = self.sat_ds_old.rename({'date': 'time'})

        #New dataset
        #Date is in the format days since 1/1/2000

        #Setup reference date
        ref_time = datetime(year = 2000, month = 1, day = 1)

        #Create timedelta objects and add them to the reference date
        offsets = pd.to_timedelta(self.sat_ds_new['time'], unit='D')
        times = ref_time+offsets

        #Reassign time dimention to new datetime objects
        self.sat_ds_new['time'] = times

        #Set various time attributes
        self.start_time = self.sat_ds_old['time'].min()
        self.overlap_start_time = self.sat_ds_new['time'].min()
        self.overlap_end_time = self.sat_ds_old['time'].max()
        self.end_time = self.sat_ds_new['time'].max()

        #Rename DOT variable in the new dataset for consistency
        self.sat_ds_new = self.sat_ds_new.rename({'DOT_smoothed': 'DOT'}) #Select the smoothed DOT product to match the DOT product provided in the old dataset

        #Calculate the U and V components of geostrophic current with respect to lat/lon in the new dataset
        self.sat_ds_new['U'] = self.sat_ds_new['Geo_surf_current_y']*self.sat_ds_new['ang_c']+self.sat_ds_new['Geo_surf_current_x']*self.sat_ds_new['ang_s']
        self.sat_ds_new['V'] = self.sat_ds_new['Geo_surf_current_x']*self.sat_ds_new['ang_c']-self.sat_ds_new['Geo_surf_current_y']*self.sat_ds_new['ang_s']

        #Pregenerate flattened array of lat/lon points
        #Old data
        mesh_lons, mesh_lats = np.meshgrid(self.sat_ds_old['lon'], self.sat_ds_old['lat'])
        flat_lats = mesh_lats.ravel()
        flat_lons = mesh_lons.ravel()
        self.old_points = np.vstack((flat_lons, flat_lats)).T

        #New data
        flat_lats = self.sat_ds_new['lats'].values.ravel()
        flat_lons = self.sat_ds_new['lons'].values.ravel()
        self.new_points = np.vstack((flat_lons, flat_lats)).T


    def get(self, 
        time: np.datetime64, 
        lat: float,
        lon: float,
        priority: str) -> tuple[float, float, float]:

        '''
        Get the Dynamic Ocean Topgraphy (DOT), U, and V components of the geostrophic current at a specfic time

        Parameters:
        -----------
        time: np.datetime64
            Time to get data for. The nearest time will be chosen if the passed time does not match one on the temporal grid
        lat: float
            The latitude of the point to get data for. Data will be interpolared with bilinear 
            interpolation of the datapoint does no fall on the geographic grid
        lon: float
            The longitude of the point to get data for. Data will be interpolared with bilinear 
            interpolation of the datapoint does no fall on the geographic grid
        priority: str
            Whether to return 'old' or 'new' data in the overlap period (accepts 'old' or 'new')


        Returns:
        --------
        data_package: tuple[DOT: float, U: float, V: float]
            DOT: float
                Dynamic ocean topography at the specficied time/point (m)
            U, V: float
                Veloity of the geostropic current relative to latitude and longitude respectivly (m/s)
        '''

        dot = []
        u = []
        v = []

        time = np.atleast_1d(time)
        lat = np.atleast_1d(lat)
        lon = np.atleast_1d(lon)

        for i, working_time in enumerate(time):
            working_lat = lat[i]
            working_lon = lon[i]

            if working_time < self.start_time:
                #Requested time is before the start of the merged dataset
                raise KeyError(f"{working_time} is before the start of the covered timeperiod")
            elif working_time >= self.start_time and working_time < self.overlap_start_time:
                #Requested time is only covered by the old dataset
                query_time = self.__closest_time(working_time, self.sat_ds_old['time'])
                working_da = self.sat_ds_old.sel(time = query_time) #Get the dataarray of the ds at the time
                working_points = self.old_points
            elif working_time >= self.overlap_start_time and working_time < self.overlap_end_time:
                #Requested time is covered by the overlap of the two datasets

                if priority == 'old':
                    #Pass on old data
                    query_time = self.__closest_time(working_time, self.sat_ds_old['time'])
                    working_da = self.sat_ds_old.sel(time = query_time) #Get the dataarray of the ds at the time
                    working_points = self.old_points

                elif priority == 'new':
                    #Pass on new data
                    query_time = self.__closest_time(working_time, self.sat_ds_new['time'])
                    working_da = self.sat_ds_new.sel(time = query_time) #Get the dataarray of the ds at the time
                    working_points = self.new_points
                else:
                    raise AttributeError('priority accepts only "old" and "new"')

                # #For now, print both data values and just pass the new data

                # #Print old data
                # query_time = self.__closest_time(working_time, self.sat_ds_old['time'])
                # working_da = self.sat_ds_old.sel(time = query_time) #Get the dataarray of the ds at the time
                # working_points = self.old_points

                # dots = working_da['DOT'].values.ravel()
                # us = working_da['U'].values.ravel()
                # vs = working_da['V'].values.ravel()
                # data = np.vstack((dots, us, vs)).T
                # interp = LinearNDInterpolator(working_points, data)
                # working_dot, working_u, working_v = interp(working_lon, working_lat)
                # print(f"Old DOT: {working_dot}m, U: {working_u}m/s, V: {working_v}m/s")

                # #Print new data
                # query_time = self.__closest_time(working_time, self.sat_ds_new['time'])
                # working_da = self.sat_ds_new.sel(time = query_time) #Get the dataarray of the ds at the time
                # working_points = self.new_points

                # dots = working_da['DOT'].values.ravel()
                # us = working_da['U'].values.ravel()
                # vs = working_da['V'].values.ravel()
                # data = np.vstack((dots, us, vs)).T
                # interp = LinearNDInterpolator(working_points, data)
                # working_dot, working_u, working_v = interp(working_lon, working_lat)
                # print(f"New DOT: {working_dot}m, U: {working_u}m/s, V: {working_v}m/s")

                # #pass the new data
                # query_time = self.__closest_time(working_time, self.sat_ds_new['time'])
                # working_da = self.sat_ds_new.sel(time = query_time) #Get the dataarray of the ds at the time
                # working_points = self.new_points
            elif working_time >= self.overlap_end_time and working_time <= self.end_time:
                #Requested time is only covered by the new dataset
                query_time = self.__closest_time(working_time, self.sat_ds_new['time'])
                working_da = self.sat_ds_new.sel(time = query_time) #Get the dataarray of the ds at the time
                working_points = self.new_points
            else:
                #Requested time is after the end of the merged dataset
                raise KeyError(f"{working_time} is after the end of the covered timeperiod")

            #Pack DOT, U, and V values for the working time into one data array for interpolation
            dots = working_da['DOT'].values.ravel()
            us = working_da['U'].values.ravel()
            vs = working_da['V'].values.ravel()
            data = np.vstack((dots, us, vs)).T

            #Define interpolator
            interp = LinearNDInterpolator(working_points, data)
            working_dot, working_u, working_v = interp(working_lon, working_lat)
            dot.append(working_dot)
            u.append(working_u)
            v.append(working_v)

        data_package = (dot, u, v)

        return data_package

        
                 
    def __closest_time(self,
                       query: datetime,
                       time_grid: np.ndarray) -> datetime:
        '''
        Return the time closest to a given query

        Parameters:
        -----------
        query: datetime
            The time to search
        time_grid: np.ndarray
            Series of times to search from

        Returns:
        --------
        closest_time: datetime
            The closest time on the time_grid to the query
        '''

        idx = (np.abs(time_grid-query)).argmin(dim='time')

        closest_time = time_grid[idx]

        return closest_time
