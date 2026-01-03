
import numpy as np

def geopotential_height(z):
    """
    Convert geopotential height to geopotential height in meters.

    Parameters:
    z : float or array-like
        Geopotential height in geopotential meters (gpm)

    Returns:
    gph : float or array-like
        Geopotential height in meters (m)
    """
    gph = z / 9.81
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
        theta : float or array-like
            Potential temperature in Kelvin
    """
    if T < -273:
        raise ValueError('T needs to be above -273 °C')
    if p > 1015 or p < 1:
        raise ValueError('p need to be between 1015 and 1 hPa')
    R = 287.0   # J/(kg·K)
    cp = 1004.0 # J/(kg·K)

    theta = T * (p0 / p)**(R / cp)
    
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
    dz = z_up - z_down
    if dz < 0:
        raise ValueError('z_down need to be smaller thatn z_up')
    
    if method =='mean':
        theta = (theta_up+theta_down) /2
    elif method =='min':
        theta = theta_down
    else:
        raise ValueError('Valid Arguments for method are [mean, min]')
    
    N = np.sqrt( (g/theta)(dtheta/dz))

    return N


def nondim_mtn_height(
        N:float,  # Brunt-Väisälä-Frequency
        h:int,    # height of the mountain to overcome
        U_up:float,
        U_down:float): # perpendicular wind speed
    
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
    if N < 0 or N < 1:
        raise ValueError('N need to be positive and smaller than 1')
    if h < 0:
        raise ValueError('h need to be positive')
    if U < 0:
        raise ValueError('U need to be positive')
    
    U = (U_down + U_up) / 2

    H= (N * h) / U

    return H

def compute_N_H(data):
    """
    Compute the Brunt-Väisälä frequency N and the non-dimensional mountain height H.

    Parameters:
        data : xr.Dataset
            Dataset containing 'theta', 'z', and 'perpendicular_wind_speed' variables

    Returns:
        data : xrDataset
            ERA5 dataset with added 'N' and 'H' variables
    """
    theta_up = data['theta'].shift(pressure_level=-1)
    theta_down = data['theta']
    z_up = data['z'].shift(pressure_level=-1)
    z_down = data['z']
    U_up = data['perpendicular_wind_speed'].shift(pressure_level=-1)
    U_down = data['perpendicular_wind_speed']

    # Calculate N and H
    N = brunt_vaeisaelae_freq(theta_up, theta_down, z_up, z_down)
    H = nondim_mtn_height(N, z_up - z_down, U_up, U_down)

    # Remove the last level (which has NaNs because of shift)
    N = N.isel(pressure_level=slice(0, -1))
    H = H.isel(pressure_level=slice(0, -1))

    data = data.assign(N=N, H=H)
    return data