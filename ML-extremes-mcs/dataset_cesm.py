import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
import dl_stats

"""

Module contains pytorch dataset.

Author: Maria J. Molina, NCAR (molina@ucar.edu).

"""

class CesmDataset(Dataset):
    
    def __init__(self, cesm_array, batch_size=1, 
                 n_classes=2, shuffle=False, norm=None):
        """
        Initialization.
        
        cesm_array should be (batch, channels, lat, lon)
        """
        self.cesm_array = cesm_array
        self.samples_indx = np.arange(int(cesm_array.shape[0]))
        self.batch_size = batch_size
        self.dim = (int(cesm_array.shape[2]), int(cesm_array.shape[3]))
        self.n_channels = int(cesm_array.shape[1])
        self.n_classes = n_classes
        self.shuffle = shuffle
        
        if norm != 'zscore' and norm != 'minmax' and norm != None:
            raise Exception("Please set norm to ``zscore``, ``minmax``, or ``None``.")
            
        self.norm = norm
        if self.norm:
            self.stat_a, self.stat_b = self.compute_norm_constants()
            
        self.on_epoch_end()
        
        
    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """
        return int(np.floor(len(self.samples_indx) / self.batch_size))

    
    def __getitem__(self, index):
        """
        Generate one batch of data.
        Args:
            index (int): Index that slices the provided file indices.
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        
        # Find list of IDs
        list_IDs_temp = [self.samples_indx[k] for k in indexes]
        
        # Generate data
        X = self.__data_generation(list_IDs_temp)
        
        # convert to tensors
        X = torch.from_numpy(X).float()
        
        return {'train': X, 'minibatch_indx': indexes}

    
    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        """
        self.indexes = np.arange(len(self.samples_indx))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        return
    
    
    def compute_norm_constants(self):
        """
        Compute the nomalization or standardization constants.
        Returns: 
            a_ (numpy array): Averages or minimums for list of IDs.
            b_ (numpy array): Standard deviations or maximums for list of IDs.
        """
        a_ = np.empty((len(self.samples_indx), self.n_channels))
        b_ = np.empty((len(self.samples_indx), self.n_channels))
        
        for j in range(self.n_channels):
            
            for i in self.samples_indx:
                
                if self.norm == 'zscore':
                    
                    a_[i, j] = np.nanmean(self.cesm_array[i,j,:,:])
                    b_[i, j] = np.nanstd(self.cesm_array[i,j,:,:])
                    
                if self.norm == 'minmax':
                    
                    a_[i, j] = np.nanmin(self.cesm_array[i,j,:,:])
                    b_[i, j] = np.nanmax(self.cesm_array[i,j,:,:])
        
        return a_, b_
    
    
    def compute_stats(self, data, a, b):
        """
        Compute the standardization or normalization.
        Args:
            data (array): Data for specific variable.
            a (array): Averages or minimums for list of IDs.
            b (array): Standard deviations or maximums for list of IDs.
        Returns:
            data (array) standardized or normalized for training.
        """
        if self.norm == 'zscore':
            return dl_stats.z_score(data, a, b)
        
        if self.norm == 'minmax':
            return dl_stats.min_max_scale(data, a, b)
    
    
    def omit_nans(self, X):
        """
        Remove any ``nans`` from data.
        Args:
            X (array): Training data.
            y (array): Labels for supervised learning.
        Returns:
            Data arrays with nans removed.
        """
        maskarray = np.full(X.shape[0], True)
        masker = np.unique(np.argwhere(np.isnan(X))[:,0])
        maskarray[masker] = False
        newX = X[maskarray,:,:,:]
        return newX
    
    
    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples. # X : (n_samples, n_channels, *dim)
        Args:
            list_IDs_temp (list): List of IDs for the respective batch.
        """
        X = np.empty((self.batch_size, self.n_channels, *self.dim))

        for i, ID in enumerate(list_IDs_temp):
            
            for j in range(self.n_channels):
                
                # Store sample(s)
                X[i,j,:,:] = self.cesm_array[ID,j,:,:]
        
        if self.norm:
            for j in range(self.n_channels):
                X[:,j,:,:] = self.compute_stats(X[:,j,:,:], self.stat_a[:,j], self.stat_b[:,j])
                
        X = self.omit_nans(X)
        return X

