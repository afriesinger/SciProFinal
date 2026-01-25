"""Functions for atmospheric dynamics."""

"""
This module provides functions for calculating:
- Geopotential height
- Potential temperature        
- Brunt-Väisälä frequency     
- Non-dimensional mountain height

Author:  Andreas Friesinger 
Date: 2026-01-14
"""



import numpy as np

def geopotential_height(z:float):
    """
    Convert geopotential height to geopotential height in meters.

    Parameters:
    z : float or array-like
        Geopotential height in geopotential meters (gpm)

    Returns:
    gph : array-like
        Geopotential height in meters (m) as int32
    """
    gph = (z / 9.81)#.astype(np.uint16)
    return gph


def potential_temperature(
        T:float,        
        p:float, 
        p0=1000.0):
    """
    Calculate potential temperature (theta) in Kelvin.
    
    Parameters:
        T : float or array-like
            Temperature in Kelvin
        p : float or array-like
            Pressure at the level (hPa)
        p0 : float
            Reference pressure, default 1000 hPa
    
    Returns:
        theta : array-like
            Potential temperature in Kelvin as int32
    """
    if (T < -273).any():
        raise ValueError('T needs to be above -273 °C')
    if (p > 1015).any() or (p < 1).any():
        raise ValueError('p need to be between 1015 and 1 hPa')
    R = 287.0   # J/(kg·K)
    cp = 1004.0 # J/(kg·K)

    theta = T * (p0 / p)**(R / cp)
    theta = theta#.astype(np.float16)
    
    return theta





def brunt_vaeisaelae_freq(
        theta_up:float, 
        theta_down:float, 
        z_up:int,
        z_down:int,
        method='mean'):
    """
    Calculate the Brunt-Väisälä frequency (N) in 1/s.
    
    Parameters:
        theta_up : float
            Potential temperature at the upper level (K)
        theta_down : float
            Potential temperature at the lower level (K)
        z_up : int
            Height of the upper level (m)
        z_down : int
            Height of the lower level (m)
        method : str
            Method to calculate the Brunt-Väisälä frequency
            Valid arguments are ['mean', 'min']
    
    Returns:
        N : float
            Brunt-Väisälä frequency (1/s)
    """
    g= 9.81
    dtheta = theta_up-theta_down
    #if (np.isfinite(dtheta) == False).any():
    #    raise ValueError('theta_up and theta_down need to be finite values: theta_up: {}, theta_down: {}'.format(theta_up, theta_down)) 
    dz = z_up - z_down
    if (dz < 0).any():
        raise ValueError('z_down need to be smaller thatn z_up')
    
    if method =='mean':
        theta = (theta_up+theta_down) /2
    elif method =='min':
        theta = theta_down
    else:
        raise ValueError('Valid Arguments for method are [mean, min]')
    
    N = np.sqrt( (g/theta)*(dtheta/dz))
    N = N.astype(np.float32)

    return N


def nondim_mtn_height(
        N:float,  
        h:int,    
        U_up:float,
        U_down:float): 
    
    """
    Calculate the non-dimensional mountain height (H) which is the ratio between 
    the height of the mountain to overcome and the wind speed perpendicular 
    to the mountain.

    Parameters:
        N : float
            Brunt-Väisälä frequency (1/s)
        h : int
            height of the mountain to overcome (m)
        U : float
            perpendicular wind speed (m/s)

    Returns:
        H : float
            non-dimensional mountain height (dimensionless)
    """
    if (N < 0).any() or (N > 1).any():
        raise ValueError('N need to be positive and smaller than 1')
    if (h < 0).any():
        raise ValueError('h need to be positive')
    if (U_up < 0).any() or (U_down < 0).any():
        raise ValueError('U need to be positive')
    
    U = (U_down + U_up) / 2


    H= (N * h) / U

    return H

def compute_N_H(data):
    """
    Compute the Brunt-Väisälä frequency N and the non-dimensional mountain height H.
    
    Skip calculations where perpendicular wind speed is zero or NaN (no cross-mountain flow).
    Only compute N and H where wind speed is valid.

    Parameters:
        data : xr.Dataset
            Dataset containing 'theta', 'z', and 'perpendicular_wind_speed' variables

    Returns:
        data : xrDataset
            ERA5 dataset with added 'N' and 'H' variables
    """
    # Extract wind speeds first to check validity
    U_up = data['perpendicular_wind_speed'].shift(pressure_level=-1)
    U_down = data['perpendicular_wind_speed']
    
    # Slice to match calculation levels (remove last level which is shifted)
    U_down_sliced = U_down.isel(pressure_level=slice(0, -1))
    U_up_sliced = U_up.isel(pressure_level=slice(0, -1))
    
    # Create mask: True where wind is VALID (positive, non-zero, non-NaN)
    # Only compute where BOTH levels have valid wind
    wind_valid = (U_down_sliced > 0) & (U_up_sliced > 0) & \
                 ~np.isnan(U_down_sliced.values) & ~np.isnan(U_up_sliced.values)
    
    # Extract other variables
    theta_up = data['theta'].shift(pressure_level=-1)
    theta_down = data['theta']
    z_up = data['gph'].shift(pressure_level=-1)
    z_down = data['gph']
    h = data['downwind_terrain_height']
    
    # Only compute N and H where wind is valid
    N = brunt_vaeisaelae_freq(theta_up, theta_down, z_up, z_down)
    H = nondim_mtn_height(N, h, U_up, U_down)

    N = N.isel(pressure_level=slice(0, -1))
    H = H.isel(pressure_level=slice(0, -1))
    
    # Set to NaN where wind is invalid - no unnecessary calculations
    N = N.where(wind_valid, np.nan)
    H = H.where(wind_valid, np.nan)

    N.attrs.clear()
    H.attrs.clear()

    N.attrs.update({
        'long_name': 'Brunt-Väisälä frequency',
        'units': 's-1',
        'standard_name': 'brunt_vaisala_frequency'
    })

    H.attrs.update({
        'long_name': 'Non-dimensional mountain height',
        'units': '1',
        'standard_name': 'dimensionless_mountain_height'
    })

    data = data.assign(N=N, H=H)
    return data
