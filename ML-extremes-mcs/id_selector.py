import xarray as xr
import numpy as np
import pandas as pd
import calendar
import pickle

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
        ens_num (str): The CESM CAM ensemble number or observation/model data. Defaults to ``era5``.
    """
    def __init__(self, main_path, start_year, end_year, month_only=None, year_only=None, mcs_only=False, 
                 percent_train=0.7, ens_num='era5'):
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
        self.ens_num = ens_num

    def make_month(self):
        """
        Returns all caps month string.
        """
        return calendar.month_abbr[self.month].upper()

    def make_dict(self, start_str, end_str, frequency='3H', savepath=None):
        """
        Create dictionary of the indices using study time range.
        These indices must be fixed values for the study time range.
        Args:
            start_str and end_str (str): Start and end times for date range.
                For 002 use: start_str=f'04-01-1991', end_str=f'07-31-2005 23:00:00'
                For 003 use: start_str=f'01-01-2000 03:00:00', end_str=f'01-01-2006 00:00:00'
                For era5 use: start_str='2004-01-01 00:00:00', end_str='2016-12-31 23:00:00'
            frequency (str): Spacing for time intervals. Defaults to ``3H``.
            savepath (str): Path to save dictionary containing indices. Default to ``None``.
        """
        alldates = pd.date_range(start=start_str, end=end_str, freq=frequency)
        # cesm doesn't do leap years
        alldates = alldates[~((alldates.month == 2) & (alldates.day == 29))]
        dict_dates = {}
        
        if self.ens_num == '003' or self.ens_num == '002':
            for i, j in enumerate(alldates):
                dict_dates[j] = i
            return dict_dates
        
        if self.ens_num == 'era5':
            # file indices for dictionary -- not enumerate since some masks missing
            # this option also provides flexibility for 3H or 1H
            j = 0
            yrs = alldates.strftime('%Y')
            mos = alldates.strftime('%m')
            dys = alldates.strftime('%d')
            hrs = alldates.strftime('%H')
            yrorig = yrs[0]
            for i, yr, mo, dy, hr in zip(alldates, yrs, mos, dys, hrs):
                try:
                    self.open_mask_file(yr, mo, dy, hr)
                    dict_dates[j] = i
                    j += 1
                    if yrorig != yr:
                        print(f"{yrorig} completed")
                        yrorig = yr
                except FileNotFoundError:
                    continue
            if savepath:
                with open(f'{savepath}/mcs_dict_{frequency}.pkl', 'wb') as handle:
                    pickle.dump(dict_dates, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if not savepath:
                return dict_dates
            
    def open_dict(self, pre_dict=True, dict_freq='3H', start_str=None, end_str=None, dictsave=None):
        """
        Open dictionary. 
        Args:
            pre_dict (boolean): If ``True``, open pre-saved dictionary file of indices. Defaults to ``True``.
            dict_freq (str): Files of specific hourly time spacing. Defaults to ``3H`` for 3-hourly.
            start_str and end_str (str): Start and end times for date range. Default to ``None``.
                For 003 use: start_str=f'01-01-2000 03:00:00', end_str=f'01-01-2006 00:00:00'
                For era5 use: start_str='2004-01-01 00:00:00', end_str='2016-12-31 23:00:00'
            dictsave (str): Path to save dictionary containing indices. Default to ``None``.
        """
        print("starting mask file generation...")
        if not pre_dict:
            indx_array = self.make_dict(start_str=start_str, end_str=end_str, frequency=dict_freq, savepath=dictsave)
        if pre_dict:
            with open(f'{self.main_directory}/mcs_dict_{dict_freq}.pkl', 'rb') as handle:
                indx_array = pickle.load(handle)
        return indx_array

    def make_years(self):
        """
        Returns array of years for data generation.
        """
        years = pd.date_range(start=str(self.start_year)+'-01-01', end=str(self.end_year)+'-01-01', freq='AS-JAN').year
        return years

    def open_mask_file(self, year=None, dict_freq=None, indx_val=None):
        """
        Open the MCS mask file.
        Args:
            year (str): Year. Defaults to ``None``.
            dict_freq (str): Hourly frequency for training (e.g., ``3H``). Defaults to ``None``.
            indx_val (str): Defaults to ``None``.
        """
        if self.ens_num == '003':
            mask = xr.open_dataset(f"{self.main_directory}/mask_camPD_{year}.nc")
        if self.ens_num == 'era5':
            mask = xr.open_dataset(f"{self.main_directory}/dl_files/{dict_freq}/mask_ID{indx_val}.nc")
        return mask

    def generate_IDarray(self, pre_dict=True, dict_freq='3H', start_str=None, end_str=None, dictsave=None):
        """
        Generate an array of the respective IDs based on predefined choices.
        Args:
            pre_dict (boolean): If ``True``, open pre-saved dictionary file of indices. Defaults to ``True``.
            dict_freq (str): Files of specific hourly time spacing. Defaults to ``3H`` for 3-hourly.
            start_str and end_str (str): Start and end times for date range. Default to ``None``.
                For 003 use: start_str=f'01-01-2000 03:00:00', end_str=f'01-01-2006 00:00:00'
                For era5 use: start_str='2004-01-01 00:00:00', end_str='2016-12-31 23:00:00'
            dictsave (str): Path to save dictionary containing indices. Default to ``None``.
        """
        print("starting ID generation...")
        if not self.year_only:
            yr_array = self.make_years()
        if self.year_only:
            yr_array = self.year_only

        # open indx dictionary
        if not pre_dict:
            indx_array = self.make_dict(start_str=start_str, end_str=end_str, frequency=dict_freq, savepath=dictsave)
        if pre_dict:
            with open(f'{self.main_directory}/mcs_dict_{dict_freq}.pkl', 'rb') as handle:
                indx_array = pickle.load(handle)
        # empty list for id creation
        ID_list = []
        
        if self.ens_num == '002':
            print('Training options not yet available for member 002')
            return
        
        if self.ens_num == '003':
            for yr in yr_array:
                mask = self.open_mask_file(year=yr)
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
            print("ID generation complete.")
            return np.array(ID_list)
                                    
        if self.ens_num == 'era5':
            for i, j in indx_array.items():
                if np.isin(j.year, yr_array):
                    if not self.month_only and not self.mcs_only:
                        ID_list.append(i)
                    if self.month_only and not self.mcs_only:
                        if np.isin(j.month, self.month_only):
                            ID_list.append(i)
                    if self.mcs_only and not self.month_only:
                        tmpmask = self.open_mask_file(year=None, dict_freq=dict_freq, indx_val=i)
                        if np.any(tmpmask['cloudtracknumber'] > 0):
                            ID_list.append(i)
                    if self.month_only and self.mcs_only:
                        if np.isin(j.month, self.month_only):
                            tmpmask = self.open_mask_file(year=None, dict_freq=dict_freq, indx_val=i)
                            if np.any(tmpmask['cloudtracknumber'] > 0):
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
        trainnum = int(allIDs.shape[0]*self.percent_train)
        trainIDs = permIDs[:trainnum]
        testIDs = permIDs[trainnum:]
        return trainIDs, testIDs
