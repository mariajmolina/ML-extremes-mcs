import xarray as xr
import numpy as np

def slice_model_grid(data_mask, data, lev, var, lat='lat', lon='lon'):
    """
    Slice the variable data on raw model grid (hybrid) to match the mask data.
    
    Args:
        mask (xarray dataset): MCS object mask.
        data (xarray dataset): The data to slice.
        lev (int): The model level to use (index).
        var (str): The variable to use.
        lat (str): Latitude dimension name. Defaults to ``lat``.
        lon (str): Longitude dimension name. Defaults to ``lon``.
        
    Returns:
        Variable data sliced to spatial extent of the mask data.
    """
    datavar = data[var]
    datavar = datavar.isel(lev=lev,
                           lat=slice(np.where(datavar[lat].values==data_mask['lat'][0].values)[0][0],
                                     np.where(datavar[lat].values==data_mask['lat'][-1].values)[0][0]+1),
                           lon=slice(np.where(datavar[lon].values==data_mask['lon'][0].values)[0][0],
                                     np.where(datavar[lon].values==data_mask['lon'][-1].values)[0][0]+1))
    return datavar

def slice_003_grid(data_mask, data, var, lat='lat', lon='lon'):
    """
    Slice the variable data on single level grid (hybrid) to match the mask data for 003 member.
    
    Args:
        mask (xarray dataset): MCS object mask.
        data (xarray dataset): The data to slice.
        var (str): The variable to use.
        lat (str): Latitude dimension name. Defaults to ``lat``.
        lon (str): Longitude dimension name. Defaults to ``lon``.
        
    Returns:
        Variable data sliced to spatial extent of the mask data.
    """
    datavar = data[var]
    datavar = datavar.isel(lat=slice(np.where(datavar[lat].values==data_mask['lat'][0].values)[0][0],
                                     np.where(datavar[lat].values==data_mask['lat'][-1].values)[0][0]+1),
                           lon=slice(np.where(datavar[lon].values==data_mask['lon'][0].values)[0][0],
                                     np.where(datavar[lon].values==data_mask['lon'][-1].values)[0][0]+1))
    return datavar

def slice_plevs_grid(data_mask, data, ncl5, var, lat='lat', lon='lon'):
    """
    Slice the variable data on pressure level grid to match the mask data.
    
    Args:
        mask (xarray dataset): MCS object mask.
        data (xarray dataset): The data to slice.
        ncl5 (int): The pressure level to use (index).
        var (str): The variable to use.
        lat (str): Latitude dimension name. Defaults to ``lat``.
        lon (str): Longitude dimension name. Defaults to ``lon``.
        
    Returns:
        Variable data sliced to spatial extent of the mask data.
    """
    datavar = data[var]
    datavar = datavar.isel(ncl5=ncl5,
                           ncl6=slice(np.where(data[lat].values==data_mask['lat'][0].values)[0][0],
                                      np.where(data[lat].values==data_mask['lat'][-1].values)[0][0]+1),
                           ncl7=slice(np.where(data[lon].values==data_mask['lon'][0].values)[0][0],
                                      np.where(data[lon].values==data_mask['lon'][-1].values)[0][0]+1))
    return datavar

def create_binary_mask(mask, mask_var='binary_tag', scaler=1.):
    """
    Create binary mask for cross entropy training.
    
    Args:
        mask (xarray dataset): MCS mask object.
        mask_var (str): The name of the mcs mask variable. Defaults to ``binary_tag`` based on previous processing.
        scaler (float): Value to scale negative class, if desired. Defaults to 1.0 (no scaling). If used, recommend value below 1.

    Returns:
        Binary mask with two categories (mcs [axis=0], no mcss [axis=1]).
    """
    binary_mask = np.ones((data_mask[mask_var].expand_dims(axis=3, channels=2).shape))
    binary_mask[:,:,:,1] = (binary_mask[:,:,:,1] - data_mask[mask_var].values) * scaler
    binary_mask[:,:,:,0] = data_mask[mask_var].values
    return binary_mask
