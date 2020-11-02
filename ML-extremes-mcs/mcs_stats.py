import numpy as np
import xarray as xr
from itertools import product

class MCSstats:
    """
    Class instantiation of MCSstats.
    Args:
        IDlist (int/float list): List of IDs created with ``id_selector``.
        mcs_path (str): Directory path to MCS masks.
        dim (int tuple): Spatial dimensions of MCS mask files.
    Todo:
        * count number of unique mcs per week??
    """
    def __init__(self, IDlist, mcs_path, dim=(105, 161)):
        """
        Initialize.
        """
        self.IDs = IDlist
        self.mcs_path = mcs_path
        self.dim = dim

    def open_masks(self, mask_var='cloudtracknumber', return_coords=False, lat='lat', lon='lon'):
        """
        Create binary mask for cross entropy training.
        Args:
            mask_var (str): Mask variable name in presaved file. Defaults to ``cloudtracknumber``.
            return_coords (boolean): Whether to return lat/lon coordinates. Defaults to ``False``.
            lat (str): Latitude coordinate name in mask file. Defaults to ``lat``.
            lon (str): Longitude coordinate name in mask file. Defaults to ``lon``.
        """
        y = np.empty((len(self.IDs), *self.dim))
        if not return_coords:
            for indx, ID in enumerate(self.IDs):
                y[indx,:,:] = xr.open_dataset(f"{self.mcs_path}/mask_ID{ID}.nc")[mask_var].values
            return y
        if return_coords:
            for indx, ID in enumerate(self.IDs):
                y[indx,:,:] = xr.open_dataset(f"{self.mcs_path}/mask_ID{ID}.nc")[mask_var].values
                lats = xr.open_dataset(f"{self.mcs_path}/mask_ID{ID}.nc")[lat].values
                lons = xr.open_dataset(f"{self.mcs_path}/mask_ID{ID}.nc")[lon].values
            return y, lats, lons

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
        Count total tracked MCSs.
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
