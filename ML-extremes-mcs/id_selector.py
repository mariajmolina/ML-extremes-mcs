import xarray as xr
import numpy as np
import pandas as pd
import calendar
from config import main_path_003, start_year, end_year

class IDSelector:
    
    """Class instantiation of IDSelector:
    
    Here we will be generating the list of IDs for deep learning model training.
    
    Attributes:
        math_path (str): Path where mask files are located.
        start_year (int): Start year of analysis.
        end_year (int): Final year of analysis.
        month_only (int): Set to list of month integer(s) if only training with a select month. Defaults to ``None``.
        year_only (int): Set to list of year integer(s) if only training with a select year. Defaults to ``None``.
        mcs_only (boolean): Set to ``True`` if only training with masks containing MCSs. Defaults to ``False``.
        percent_train (float): Set to percentage of IDs desired for training set. Remainer will be used for test set. 
                               Defaults to ``0.7``, which is a 70/30 split, 70% for training and 30% for testing.
        ens_num (str): The CESM CAM ensemble number. Defaults to ``003``.
    """

    def __init__(self, main_path, start_year, end_year, month_only=None, year_only=None, mcs_only=False, 
                 percent_train=0.7, ens_num='003'):
        
        self.main_path = main_path
        self.start_year = start_year
        self.end_year = end_year
        self.month_only = month_only
        self.year_only = year_only
        self.mcs_only = mcs_only
        self.percent_train = percent_train
        self.ens_num = ens_num

    def make_month(self):
        """
        Returns all caps month string.
        """
        return calendar.month_abbr[self.month].upper()

    def make_dict(self, start_str=f'01-01-2000 03:00:00', end_str=f'01-01-2006 00:00:00'):
        """
        Create dictionary of the indices using study time range.
        These indices must be fixed values for the study time range.
        For 002 use: start_str=f'04-01-1991', end_str=f'07-31-2005 23:00:00'
        """
        alldates = pd.date_range(start=start_str, end=end_str, freq='3H')
        # cesm doesn't do leap years
        alldates = alldates[~((alldates.month == 2) & (alldates.day == 29))]
        dict_dates = {}
        for i, j in enumerate(alldates):
            dict_dates[j] = i
        return dict_dates

    def make_years(self):
        """
        Returns array of years for data generation.
        """
        years = pd.date_range(start=self.start_year+'-01-01', end=self.end_year+'-01-01', freq='AS-JAN').year
        return years

    def open_mask_file(self, year=None):
        """
        Open the MCS mask file.
        """
        if self.ens_num == '002':
            mask = xr.open_dataset(f"{self.main_path}/mask_camPD_{self.make_month()}.nc")
        if self.ens_num == '003':
            mask = xr.open_dataset(f"{self.main_path}/mask_camPD_{year}.nc")
        return mask

    def generate_IDarray(self):
        """
        Generate an array of the respective IDs based on predefined choices.
        """
        print("starting ID generation...")
        if not self.year_only:
            yr_array = self.make_years()
        if self.year_only:
            yr_array = self.year_only
        indx_array = self.make_dict()
        ID_list = []
        
        if self.ens_num == '002':
            print('Training options not yet available for member 002')
            return
        
        if self.ens_num == '003':
            for yr in yr_array:
                mask = self.open_mask_file(yr)
                for t in mask.time:
                    if not self.month_only and not self.mcs_only:
                        indx_val = indx_array[pd.to_datetime(t.astype('str').values)]
                        ID_list.append(indx_val)
                    if self.month_only and not self.mcs_only:
                        if np.isin(t.dt.month, self.month_only):
                            indx_val = indx_array[pd.to_datetime(t.astype('str').values)]
                            ID_list.append(indx_val)
                    if self.mcs_only and not self.month_only:
                        tmpmask = mask.sel(time=t)
                        if np.any(tmpmask['binary_tag']==1):
                            indx_val = indx_array[pd.to_datetime(t.astype('str').values)]
                            ID_list.append(indx_val)
                    if self.month_only and self.mcs_only:
                        if np.isin(t.dt.month, self.month_only):
                            tmpmask = mask.sel(time=t)
                            if np.any(tmpmask['binary_tag']==1):
                                indx_val = indx_array[pd.to_datetime(t.astype('str').values)]
                                ID_list.append(indx_val)
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
        trainnum = int(allIDs.shape[0]*self.percent_train)
        trainIDs = permIDs[:trainnum]
        testIDs = permIDs[trainnum:]
        return trainIDs, testIDs
