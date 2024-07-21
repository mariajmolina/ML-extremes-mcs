import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
import dl_stats

"""

Module contains pytorch dataset.

Author: Maria J. Molina, NCAR (molina@ucar.edu).

"""


class CustomDataset(Dataset):

    def __init__(self, list_IDs, path_dataID, variable, batch_size=32,
                 dim=(121, 321), n_channels=1, n_classes=2, shuffle=False,
                 norm=None, mask_var='cloudtracknumber', transform=None):
        """
        Initialization.
        """
        self.list_IDs = list_IDs
        self.path_dataID = path_dataID
        self.variable = variable
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.transform = transform

        if norm != 'zscore' and norm != 'minmax' and norm is not None:
            raise Exception(
                "Please set norm to ``zscore``, ``minmax``, or ``None``."
            )

        self.norm = norm
        if self.norm:
            self.stat_a, self.stat_b = self.compute_norm_constants()

        self.on_epoch_end()
        self.msk_var = mask_var

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data.
        Args:
            index (int): Index that slices the provided file indices.
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        # convert to tensors and reshape (batch, channels, lat, lon)
        X = torch.flip(torch.from_numpy(X).permute(0, 3, 1, 2).float(), dims=(2,))
        y = torch.from_numpy(y).permute(0, 3, 1, 2).float()

        if self.transform:
            t = self.transform
            state = torch.get_rng_state()
            X = t(X)
            torch.set_rng_state(state)
            y = t(y)

        return {'train': X, 'test': y, 'minibatch_indx': indexes}

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)
        return

    def era5_vars(self, analysis_variable):
        """
        Help for grabbing variables inside ERA5 files.
        Args:
            analysis_variable (str): The filename variable for ERA5 files.
        """
        # add more variables here as work continues :)
        if analysis_variable == 'sp':
            VAR = 'SP'
        if analysis_variable == '10v':
            VAR = 'VAR_10V'
        if analysis_variable == '10u':
            VAR = 'VAR_10U'
        if analysis_variable == '2t':
            VAR = 'VAR_2T'
        if analysis_variable == '2d':
            VAR = 'VAR_2D'
        if analysis_variable == 'cape':
            VAR = 'CAPE'
        if analysis_variable == 'lsp':
            VAR = 'LSP'
        if analysis_variable == 'cp':
            VAR = 'CP'
        if analysis_variable == 'ttr':
            VAR = 'TTR'
        if analysis_variable == 'w700':
            VAR = 'W'
        if analysis_variable == 'u850' or analysis_variable == 'u500':
            VAR = 'U'
        if analysis_variable == 'v850' or analysis_variable == 'v500':
            VAR = 'V'
        if analysis_variable == 'z500':
            VAR = 'Z'
        if analysis_variable == 'q1000' or analysis_variable == 'q850':
            VAR = 'Q'
        assert (VAR), "Please enter an available variable."
        return VAR

    def compute_norm_constants(self):
        """
        Compute the nomalization or standardization constants.
        Returns: 
            a_ (numpy array): Averages or minimums for list of IDs.
            b_ (numpy array): Standard deviations or maximums for list of IDs.
        """
        a_ = np.empty((len(self.list_IDs), self.n_channels))
        b_ = np.empty((len(self.list_IDs), self.n_channels))

        for j, VAR in enumerate(self.variable):

            if self.norm == 'zscore':
                a_[:, j], b_[:, j], _, _ = dl_stats.extract_train_stats(
                    self.path_dataID+'/'+VAR+'/', VAR, self.list_IDs
                )

            if self.norm == 'minmax':
                _, _, a_[:, j], b_[:, j] = dl_stats.extract_train_stats(
                    self.path_dataID+'/'+VAR+'/', VAR, self.list_IDs
                )

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

    def create_binary_mask(self, y, indx, IDindx):
        """
        Create binary mask for cross entropy training.
        Args:
            y (array): Labels to populate.
            indx (int): Enumerated value representing location in label array.
            IDindx (int): Mask file ID corresponding to training data.
        """
        tmp_y = xr.open_dataset(
            f"{self.path_dataID}/mask/mask_{self.msk_var}_ID{IDindx}.nc"
        )[self.msk_var].values
        y[indx, :, :, 1] = np.where(tmp_y > 0, 1, 0)
        y[indx, :, :, 0] = np.ones(self.dim, dtype=int) - y[indx, :, :, 1]
        return y

    def create_singlechannel_mask(self, y, indx, IDindx):
        """
        Create single channel mask for training (e.g., for use with sigmoid output).
        Args:
            y (array): Labels to populate.
            indx (int): Enumerated value representing location in label array.
            IDindx (int): Mask file ID corresponding to training data.
        """
        tmp_y = xr.open_dataset(
            f"{self.path_dataID}/mask/mask_{self.msk_var}_ID{IDindx}.nc"
        )[self.msk_var].values
        y[indx, :, :, 0] = np.where(tmp_y > 0, 1, 0)
        return y

    def omit_nans(self, X, y):
        """
        Remove any ``nans`` from data.
        Args:
            X (array): Training data.
            y (array): Labels for supervised learning.
        Returns:
            Data arrays with nans removed.
        """
        maskarray = np.full(X.shape[0], True)
        masker = np.unique(np.argwhere(np.isnan(X))[:, 0])
        maskarray[masker] = False
        newX = X[maskarray, :, :, :]
        newy = y[maskarray]
        return newX, newy

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples.
        # X : (n_samples, *dim, n_channels)
        Args:
            list_IDs_temp (list): List of IDs for the respective batch.
        """
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        for i, ID in enumerate(list_IDs_temp):

            for j, VAR in enumerate(self.variable):

                # Store sample(s)
                X[i, :, :, j] = xr.open_dataset(
                    f"{self.path_dataID}/{VAR}/file_{VAR}_ID{ID}.nc"
                )[self.era5_vars(VAR)].values

            # Store class
            if self.n_classes > 1:
                y = self.create_binary_mask(y, i, ID)

            if self.n_classes == 1:
                y = self.create_singlechannel_mask(y, i, ID)

        if self.norm:
            for j, VAR in enumerate(self.variable):
                X[:, :, :, j] = self.compute_stats(
                    X[:, :, :, j], self.stat_a[:, j], self.stat_b[:, j]
                )

        X, y = self.omit_nans(X, y)
        return X, y