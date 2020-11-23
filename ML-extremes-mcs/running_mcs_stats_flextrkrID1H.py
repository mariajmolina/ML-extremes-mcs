######################################################################
######################################################################

from id_selector import IDSelector
import mcs_stats
import numpy as np
from config import main_path_era
import xarray as xr

######################################################################
######################################################################
##############
############## Generate MCS stats and figs for /glade/scratch/molina/cesm_mcs/cesm_era5/dl_files/1H
##############
######################################################################
######################################################################

def iterate_mcs_stats(years, dictpath, mcspath, ens_num='era5', msk_var='cloudtracknumber'):
    """
    Iterate the MCS stats.
    Args:
        years (int): List of years to iterate over. 
        dictpath (str): Path to location of dictionary.
        mcspath (str): Path to location of MCSs.
        ens_num (str): The dataset ensemble choice. Defaults to ``era5``.
        msk_var (str): Mask file variable. 
    """
    # loop thru years
    for i, yr in enumerate(years):

        # create three empty arrays for stats over 12 months with matching era5 dims
        ma_nottracked = np.zeros((12, 101, 161))
        ma_tracked = np.zeros((12, 101, 161))
        ma_trackedtotal = np.zeros((12))
        
        # loop thru months in respective year
        for month in np.arange(0,12,1):

            # instantiate id selector class
            ids_ = IDSelector(main_path = dictpath, 
                              start_year = 2004, 
                              end_year = 2016, 
                              month_only = [month+1], year_only = [yr], mcs_only = False, 
                              percent_train = 0.7, ens_num = ens_num)
            # grab ids for corresponding month and year and other settings
            IDlist = ids_.generate_IDarray(pre_dict = True, dict_freq = '1H', 
                                           start_str = None, end_str = None, dictsave = None)

            # instantiate MCS stats class object
            mcsstat = mcs_stats.MCSstats(IDlist = IDlist, mcs_path = mcspath, msk_var = msk_var)
            # return masks (and lat lon)
            masks, lat, lon = mcsstat.open_masks(return_coords=True)
            # populate the arrays with stat values
            ma_nottracked[month,:,:] = mcsstat.nontracked_total_grid(masks[:,:-4,:])
            ma_tracked[month,:,:] = mcsstat.tracked_total_grid(masks[:,:-4,:])
            ma_trackedtotal[month] = mcsstat.tracked_total(masks[:,:-4,:])

        # create dataset with all month values for the respective year
        data_assemble=xr.Dataset({
                             'nontracked_total_grid':(['month','y','x'], ma_nottracked),
                             'tracked_total_grid':(['month','y','x'], ma_tracked),
                             'tracked_total':(['month'], ma_trackedtotal),
                            },
                             coords=
                            {'year':(['year'], np.array([yr])),
                             'month':(['month'], np.arange(1,13,1)),
                             'lon':(['y','x'], lon[:-4,:]),
                             'lat':(['y','x'], lat[:-4,:]),
                            })
        # assign attrs
        data_assemble.attrs['Author'] = 'Maria J. Molina'
        data_assemble.attrs['Contact'] = 'molina@ucar.edu'
        data_assemble.attrs['Data'] = 'MCS stats'
        # save the file
        data_assemble.to_netcdf(f'{mcspath}/mcs_stats_{msk_var}_{str(yr)}.nc')
        print(f"year {str(yr)} done")
    return

def main():
    """
    Run statistics and generate associated plots for MCSs.
    """
    ############## set variables
    dictpath = main_path_era
    mcspath = f'{main_path_era}/dl_files/1H/'
    theyears = [2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]
    iterate_mcs_stats(years=theyears, dictpath=dictpath, mcspath=mcspath, 
                      ens_num='era5', msk_var='cloudtracknumber')

    ############## set variables
    dictpath = main_path_era
    mcspath = f'{main_path_era}/dl_files/1H/'
    theyears = [2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]
    iterate_mcs_stats(years=theyears, dictpath=dictpath, mcspath=mcspath, 
                      ens_num='era5', msk_var='pcptracknumber')

if __name__ == "__main__":
    main()
