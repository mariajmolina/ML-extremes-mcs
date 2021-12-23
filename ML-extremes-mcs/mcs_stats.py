import numpy as np
import xarray as xr
from itertools import product


class MCSstats:
    """
    Class instantiation of MCSstats for MCS objects.
    
    Args:
        IDlist (int/float list): List of IDs created with ``id_selector``.
        mcs_path (str): Directory path to MCS masks.
        dim (int tuple): Spatial dimensions of MCS mask files.
        msk_var (str): Mask variable name in presaved file. Defaults to ``cloudtracknumber``. 
                       Options also include ``pcptracknumber`` and ``pftracknumber``.
    Todo:
        * count number of unique mcs per week
        
    """
    def __init__(self, mcs_path, IDlist=None, dim=(105, 161), msk_var='cloudtracknumber'):
        """
        Initialize.
        """
        self.IDs = IDlist
        self.mcs_path = mcs_path
        self.dim = dim
        self.msk_var = msk_var

        
    def open_masks(self, return_coords=False, lat='lat', lon='lon'):
        """
        Open masks.
        
        Args:
            return_coords (boolean): Whether to return lat/lon coordinates. Defaults to ``False``.
            lat (str): Latitude coordinate name in mask file. Defaults to ``lat``.
            lon (str): Longitude coordinate name in mask file. Defaults to ``lon``.
            
        """
        y = np.empty((len(self.IDs), *self.dim))
        
        if not return_coords:
            
            for indx, ID in enumerate(self.IDs):
                
                y[indx,:,:] = xr.open_dataset(
                    f"{self.mcs_path}/mask_{self.msk_var}_ID{ID}.nc")[self.msk_var].values
                
            return y
        
        if return_coords:
            
            for indx, ID in enumerate(self.IDs):
                
                y[indx,:,:] = xr.open_dataset(
                    f"{self.mcs_path}/mask_{self.msk_var}_ID{ID}.nc")[self.msk_var].values
                lats = xr.open_dataset(
                    f"{self.mcs_path}/mask_{self.msk_var}_ID{ID}.nc")[lat].values
                lons = xr.open_dataset(
                    f"{self.mcs_path}/mask_{self.msk_var}_ID{ID}.nc")[lon].values
                
            return y, lats, lons

        
    def slice_era5trkr(self, ds, lat='lat', lon='lon'):
        """
        Slice the ERA5 MCS masks generated with FLEXTRKR.
        To be used with the preprocess function of open_mfdataset.
        Lat/lon extents taken from original IMERG data extent.
        
        Args:
            ds (xarray data array).
            
        """
        lat0_bnd = 25
        lat1_bnd = 50
        lon0_bnd = -110
        lon1_bnd = -70
        
        data = ds.isel(lat=slice(np.where(ds[lat].values==lat0_bnd)[0][0],
                                 np.where(ds[lat].values==lat1_bnd)[0][0]+1),
                       lon=slice(np.where(ds[lon].values==lon0_bnd+360)[0][0],
                                 np.where(ds[lon].values==lon1_bnd+360)[0][0]+1))
        
        return data
    
    
    def open_masks_era5trkr(self, year, return_coords=False, lat='lat', lon='lon'):
        """
        Open ERA5 FLEXTRKR masks for intercomparison.
        Args:
            year (str): Year to open masks. 
            return_coords (boolean): Whether to return lat/lon coordinates. Defaults to ``False``.
            lat (str): Latitude coordinate name in mask file. Defaults to ``lat``.
            lon (str): Longitude coordinate name in mask file. Defaults to ``lon``.
        """
        data = xr.open_mfdataset(
            f'{self.mcs_path}/{str(year)}/mcstrack_{str(year)}*.nc', 
            combine='by_coords',
            preprocess=self.slice_era5trkr)
        
        y = data[self.msk_var].fillna(0.0).values
        
        months = data['time'].dt.month.values
        
        if return_coords:
            
            lats = data[lat].values
            lons = data[lon].values
            
            return y, months, lats, lons
        
        if not return_coords:
            
            return y, months

        
    def slice_imergtrkr(self, mask):
        """
        Slice regridded IMERG masks to match ERA5 run masks.
        """
        return mask[self.msk_var][:,:-4,:]

    
    def nontracked_total_grid(self, masks):
        """
        Grid counting all MCS frequency, without MCS tracking.
        Args:
            masks (numpy array): Mask files loaded by ``open_masks``.
        """
        return np.sum(np.where(masks > 0, 1, 0), axis=0)

    
    def tracked_total_grid(self, masks):
        """
        Grid counting all MCS frequency, with MCS tracking.
        Args:
            masks (numpy array): Mask files loaded by ``open_masks``.
        """
        y = np.zeros(masks.shape[1:])
        
        for i, j in product(range(masks.shape[1]), range(masks.shape[2])):
            
            y[i,j] = np.sum(np.where(np.unique(masks[:,i,j]) > 0, 1, 0))
            
        return y

    
    def tracked_total(self, masks):
        """
        Count total number of tracked MCSs using their unique mcs-ids (updated yearly).
        Args:
            masks (numpy array): Mask files loaded by ``open_masks``.
        """
        return len(np.unique(masks[masks > 0]))

    
    def tracked_unique_mcs_ids(self, masks):
        """
        Return IDs of tracked MCSs.
        Args:
            masks (numpy array): Mask files loaded by ``open_masks``.
        """
        return np.unique(masks[masks > 0])

    
    def total_with_mcs(self, masks):
        """
        Create temporal length binary masks of MCS or not occurrence.
        Args:
            masks (numpy array): Mask files loaded by ``open_masks``.
        """
        y = np.zeros(masks.shape[0])
        
        for i in range(masks.shape[0]):
            
            if np.any(masks[i,:,:]>0):
                
                y[i] = 1
                
        return y

    
    def percentage_with_mcs(self, array):
        """
        Percentage of hourly intervals containing an MCS.
        Args:
            masks (numpy array): Mask files loaded by ``open_masks``.
        """
        return np.sum(array)/int(array.shape[0])*100
