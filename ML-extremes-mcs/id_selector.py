import numpy as np
import calendar
import pickle
import pandas as pd
import xarray as xr


class IDSelector:
    """
    Class instantiation of IDSelector:
    
    Here we will be generating the list of IDs for deep learning model training.
    Note: ID generation based solely on MCS masks.
    
    Attributes:
        main_path (str): Path where mask files are located.
        start_year (int): Start year of analysis.
        end_year (int): Final year of analysis.
        month_only (int): Set to list of month integer(s) if only training with a select month. Defaults to ``None``.
        year_only (int): Set to list of year integer(s) if only training with a select year. Defaults to ``None``.
        mcs_only (boolean): Set to ``True`` if only training with masks containing MCSs. Defaults to ``False``.
        percent_train (float): Set to percentage of IDs desired for training set. Remainer will be used for test set. 
                               Defaults to ``0.7``, which is a 70/30 split, 70% for training and 30% for testing.
        percent_validate (float): Set to percentage of IDs from training data desired for validation. Defaults to ``None.``
        mask_var (str): Mask variable name in presaved file. Defaults to ``cloudtracknumber``. Options also include
                       ``pcptracknumber`` and ``pftracknumber``.
    """
    def __init__(self, main_path, start_year, end_year, month_only=None, year_only=None, mcs_only=False, 
                 percent_train=0.7, percent_validate=None, mask_var='cloudtracknumber'):
        """
        Initialization.
        """
        self.main_directory = main_path
        self.start_year = start_year
        self.end_year = end_year
        self.month_only = month_only
        self.year_only = year_only
        self.mcs_only = mcs_only
        self.percent_train = percent_train
        self.percent_validate = percent_validate
        self.msk_var = mask_var
        
        
    def open_dictionary(self, dict_freq='3H'):
        """
        Open presaved dictionary containing file respective IDs.
        Args:
            dict_freq (boolean): If ``True``, open pre-saved dictionary file of indices. Defaults to ``True``.
        """
        with open(f'{self.main_directory}/mcs_dict_{dict_freq}.pkl', 'rb') as handle:
            
            indx_array = pickle.load(handle)
            
        return indx_array

    
    def make_years(self):
        """
        Returns array of years for data generation.
        """
        return pd.date_range(start=str(self.start_year)+'-01-01', end=str(self.end_year)+'-01-01', freq='AS-JAN').year

    
    def open_mask_file(self, dict_freq=None, indx_val=None):
        """
        Open the MCS mask file.
        Args:
            dict_freq (str): Hourly frequency for training (e.g., ``3H``). Defaults to ``None``.
            indx_val (str): Defaults to ``None``.
        """
        return xr.open_dataset(f"{self.main_directory}/dl_files/{dict_freq}/mask/mask_{self.msk_var}_ID{indx_val}.nc")

    
    def generate_IDarray(self, dict_freq='3H'):
        """
        Generate an array of the respective IDs based on predefined choices.
        Note: Function based on MCS data only.
        Args:
            dict_freq (str): Files of specific hourly time spacing. Defaults to ``3H`` for 3-hourly.
        """
        print("starting ID generation...")
        
        if not self.year_only:
            yr_array = self.make_years()
        
        if self.year_only:
            yr_array = self.year_only

        # open indx dictionary
        indx_array = self.open_dictionary(dict_freq=dict_freq)
        
        # empty list for id creation
        ID_list = []
        
        for i, j in indx_array.items():
            
            if np.isin(j.year, yr_array):
                
                if not self.month_only and not self.mcs_only:
                    
                    ID_list.append(i)
                    
                if self.month_only and not self.mcs_only:
                    
                    if np.isin(j.month, self.month_only):
                        ID_list.append(i)
                        
                if self.mcs_only and not self.month_only:
                    
                    tmpmask = self.open_mask_file(dict_freq=dict_freq, indx_val=i)
                    
                    if np.any(tmpmask[self.msk_var] > 0):
                        ID_list.append(i)
                        
                if self.month_only and self.mcs_only:
                    
                    if np.isin(j.month, self.month_only):
                        tmpmask = self.open_mask_file(dict_freq=dict_freq, indx_val=i)
                        
                        if np.any(tmpmask[self.msk_var] > 0):
                            ID_list.append(i)
                            
        print("ID generation complete.")
        return np.array(ID_list)

    
    def generate_traintest_split(self, allIDs, seed=0):
        """
        Split the IDs into a train and a test set. The train set will be used to train DL model and
        the test set will be used to evaluate the DL model.
        Args:
            allIDs (numpy array): List of all IDs generated from ``generate_IDarray``.
            seed (int): Seed number for randomizing IDs. Defaults to ``0``.
        """
        np.random.seed(seed)
        permIDs = np.random.permutation(allIDs)
        
        if not self.percent_validate:
            
            trainnum = int(allIDs.shape[0] * self.percent_train)
            trainIDs = permIDs[:trainnum]
            testIDs = permIDs[trainnum:]
            
            return trainIDs, testIDs
        
        if self.percent_validate:
            
            trainnum = int(allIDs.shape[0] * (self.percent_train - self.percent_validate))
            validnum = int(allIDs.shape[0] * self.percent_validate)
            
            trainIDs = permIDs[:trainnum]
            validIDs = permIDs[trainnum:trainnum + validnum]
            testIDs = permIDs[trainnum + validnum:]
            
            return trainIDs, validIDs, testIDs
