import numpy as np
import keras
import xarray as xr
from config import main_path

class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras training.
    """
    def __init__(self, list_IDs, labels, path_dataID, variable, height,
                 batch_size=32, dim=(106, 81), n_channels=1, n_classes=2, shuffle=True):
        """
        Initialization.
        """
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.path_dataID = path_dataID
        self.variable = variable
        self.height = height
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples. # X : (n_samples, *dim, n_channels)
        """ 
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 2), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,:,:,0] = xr.open_dataset(f"{self.path_dataID}/plev_{self.variable}_hgt{self.height}_ID{ID}.nc")[self.variable].values
            # Store class
            y = self.create_binary_mask(y, i)
        return X, y
    
    def create_binary_mask(self, y, IDindx, mask_var='binary_tag'):
        """
        Create binary mask for cross entropy training.
        """
        y[IDindx,:,:,0] = xr.open_dataset(f"{self.path_dataID}/mask_ID{IDindx}.nc")[mask_var].values
        y[IDindx,:,:,1] = np.ones(self.dim, dtype=int) - y[IDindx,:,:,0]
        return y