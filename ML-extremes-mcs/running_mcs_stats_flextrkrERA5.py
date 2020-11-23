######################################################################
############## Author: Maria J. Molina, molina@ucar.edu
######################################################################

import mcs_stats
import numpy as np
import xarray as xr

######################################################################
######################################################################
##############
############## Generate MCS stats and figs for
############## /glade/scratch/molina/cesm_mcs/mcs_flextrkr_era5/mcstracking_3pctl or 15pctl
##############
######################################################################
######################################################################

def iterate_mcs_stats(years, mcspath, ens_num='era5', msk_var='cloudtracknumber'):
    """
    Iterate the MCS stats.
    Args:
        years (int): List of years to iterate over.
        mcspath (str): Path to location of MCSs.
        ens_num (str): The dataset ensemble choice. Defaults to ``era5``.
        msk_var (str): Mask file variable. 
    """
    # loop over years
    for i, yr in enumerate(years):
        # instantiate MCS stats class object
        mcsstat = mcs_stats.MCSstats(mcs_path=mcspath, 
                                     dim=(101, 161), 
                                     msk_var=msk_var)
        # open files
        masks, mons, lat, lon = mcsstat.open_masks_era5trkr(year=yr, return_coords=True)

        # create three empty arrays for stats over 12 months
        ma_nottracked = np.zeros((12,masks.shape[1],masks.shape[2]))
        ma_tracked = np.zeros((12,masks.shape[1],masks.shape[2]))
        ma_trackedtotal = np.zeros((12))

        #loop over 12 months and compute stats and stick values into array
        for m in np.arange(0,12,1):
            ma_nottracked[m,:,:] = mcsstat.nontracked_total_grid(masks[np.argwhere(mons==m+1)[:,0],:,:])
            ma_tracked[m,:,:] = mcsstat.tracked_total_grid(masks[np.argwhere(mons==m+1)[:,0],:,:])
            ma_trackedtotal[m] = mcsstat.tracked_total(masks[np.argwhere(mons==m+1)[:,0],:,:])

        # create dataset with all values
        data_assemble=xr.Dataset({
                         'nontracked_total_grid':(['month','y','x'], ma_nottracked),
                         'tracked_total_grid':(['month','y','x'], ma_tracked),
                         'tracked_total':(['month'], ma_trackedtotal),
                        },
                         coords=
                        {'year':(['year'], np.array([yr])),
                         'month':(['month'], np.arange(1,13,1)),
                         'lon':(['x'], lon),
                         'lat':(['y'], lat),
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
    mcspath = f'/glade/scratch/molina/cesm_mcs/mcs_flextrkr_era5/mcstracking_3pctl/'
    theyears = [2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]
    iterate_mcs_stats(years=theyears, mcspath=mcspath, ens_num='era5', msk_var='cloudtracknumber')
    
    ############## set variables
    mcspath = f'/glade/scratch/molina/cesm_mcs/mcs_flextrkr_era5/mcstracking_3pctl/'
    theyears = [2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]
    iterate_mcs_stats(years=theyears, mcspath=mcspath, ens_num='era5', msk_var='pcptracknumber')
    
    ############## set variables
    mcspath = f'/glade/scratch/molina/cesm_mcs/mcs_flextrkr_era5/mcstracking_15pctl/'
    theyears = [2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]
    iterate_mcs_stats(years=theyears, mcspath=mcspath, ens_num='era5', msk_var='cloudtracknumber')
    
    ############## set variables
    mcspath = f'/glade/scratch/molina/cesm_mcs/mcs_flextrkr_era5/mcstracking_15pctl/'
    theyears = [2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]
    iterate_mcs_stats(years=theyears, mcspath=mcspath, ens_num='era5', msk_var='pcptracknumber')

if __name__ == "__main__":
    main()
