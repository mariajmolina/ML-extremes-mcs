import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
import xarray as xr
import dl_stats
from config import main_path_003

class DataGenerator(Sequence):
    """
    Generates data for Keras training.
    
    Args:
        list_IDs (array): List of IDs for training.
        path_dataID (str): Directory path to ID files.
        variable (str): Variable(s) string as a list.
        ens_num (str): CESM model run number (e.g., ``003``).
        h_num (str): Whether the variable is instantaneous or not. List of ``h3`` and/or ``h4``.
        height ()
        batch_size (int): Batch size for training. Defaults to ``32``.
        dim (tuple): Tuple of spatial dimensions of the variable patches. Defaults to ``(106,31)``.
        n_channels (int): Number of input features (or channels). Defaults to ``1``.
        n_classes (int): Number of output features (or channels). Defaults to ``2``.
        shuffle (boolean): Whether to shuffle for training. Defaults to ``True``.
        
    Based on tutorial/blog: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        
    """
    def __init__(self, list_IDs, path_dataID, variable, ens_num, h_num, height=None, 
                 batch_size=32, dim=(106, 81), n_channels=1, n_classes=2, shuffle=True):
        """
        Initialization.
        """
        self.list_IDs = list_IDs
        self.path_dataID = path_dataID
        self.variable = variable
        self.ens_num = ens_num
        self.h_num = h_num
        self.height = height
        self.batch_size = batch_size
        self.dim = dim
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
        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            for j, (VAR, HNUM) in enumerate(zip(self.variable, self.h_num)):
                # Store sample(s)
                if self.ens_num == '002':
                    # X[i,:,:,j] = xr.open_dataset(f"{self.path_dataID}/plev_{VAR}_hgt{self.height}_ID{ID}.nc")[VAR].values
                    raise Exception("Ensemble member 2 option not ready.")
                if self.ens_num == '003':
                    X[i,:,:,j] = xr.open_dataset(f"{self.path_dataID}/file003_{HNUM}_{VAR}_ID{ID}.nc")[VAR].values
                # Store class
            y = self.create_binary_mask(y, i, ID)
        return X, y
    
    def create_binary_mask(self, y, indx, IDindx, mask_var='binary_tag'):
        """
        Create binary mask for cross entropy training.
        
        Args:
            y (array): Labels to populate.
            indx (int): Enumerated value representing location in label array.
            IDindx (int): Mask file ID corresponding to training data.
            mask_var (str): Mask variable name in presaved file.
            
        """
        y[indx,:,:,0] = xr.open_dataset(f"{self.path_dataID}/mask_ID{IDindx}.nc")[mask_var].values
        y[indx,:,:,1] = np.ones(self.dim, dtype=int) - y[indx,:,:,0]
        return y
