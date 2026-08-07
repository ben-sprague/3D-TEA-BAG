'''
Functions to calculate various basic parameters in/related to the ocean
'''

import numpy as np
import gsw

def fc(latitude: np.ndarray) -> np.ndarray:
    '''
    Calculature the coriolis parameter (f_c) on earth

    Parameters:
    ----------
    latitude: the latitude to calculate f_c at in degrees (+ North, - South)

    Returns:
    ----------
    Coriolis parameter
    '''

    omega = 7.29e-5 #earth sidereal rotation rate (rad/s)

    lat_radians = latitude*np.pi/180 #Convert latitude from degrees to radians for np.sin

    return 2*omega*np.sin(lat_radians)

def distance_from_datum(
        latitude: np.ndarray,
        longitude: np.ndarray,
        datum: tuple
        ) -> np.ndarray:
    '''
    Calculates the distance of various points from a datum assuming a spherical earth

    Parameters
    ----------
    latitude : latitude (degrees)
    longitude : longitude (degrees)
    datum : latitude and longitude of reference datum

    Returns
    -------
    distance : distance (in km) of each lat/lon pair from datum
    '''

    datum_lat = datum[0]
    datum_lon = datum[1]

    latitude = np.atleast_2d(latitude)
    longitude = np.atleast_2d(longitude)

    distance = np.full_like(latitude, np.nan)


    for i in range (latitude.shape[0]):
        for j in range (latitude.shape[1]):
            current_lat = latitude[i, j]
            current_lon = longitude[i, j]
            distance[i, j] = gsw.distance([datum_lon, current_lon], [datum_lat, current_lat])[0]


    distance = distance/1000 #Convert from m to km

    return distance


