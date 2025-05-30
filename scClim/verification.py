# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:22:35 2023

Functions for verification analysis based on data at centroids.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import maximum_filter
from climada import CONFIG

data_dir = str(CONFIG.local_data.data_dir)



def maximum_filter_nc(da,radius,spatial_dims=['chy','chx']):
    """Create a maximum filter of a xarray Dataarray, to analyse the highest
    values in a neighbourhood of a certain radius.

    Args:
        da (xr.DataArray): 3d dataaaray with spatial dimensions
        radius (int): Radius in pixels
        spatial_dims (list, optional): dimension names of spatial coords. Defaults to ['chy','chx'].

    Returns:
        da: Dataarray with maximum filter applied
    """

    #create kernel
    k_size = radius*2+1
    center_idx = radius
    kernel = np.fromfunction(lambda i,j: ((center_idx - i) ** 2 + (center_idx - j) ** 2)<=radius**2,(k_size,k_size)).astype(int)
    da_out = xr.apply_ufunc(
        maximum_filter,da.fillna(0),
        input_core_dims=[spatial_dims], output_core_dims=[spatial_dims],
        vectorize=True,kwargs={'footprint':kernel},
        dask='parallelized'#'allowed'
        )
    return da_out


def split_events_non_events(df, haz_var):
    """helper function to split events and non events

    Args:
        df (pd.DataFrame): at_centroid_data
        haz_var (str): hazard variable

    Returns:
        tuple: Dataframe with damages and without damages
    """

    at_centroid_df=df.copy()
    at_centroid_df['n_dmgs']=at_centroid_df['n_dmgs'].fillna(0)
    #at_centroid_df['dmg_val']=at_centroid_df['dmg_val'].fillna(0)
    events=at_centroid_df.loc[(at_centroid_df[haz_var]!=0) | (at_centroid_df['n_dmgs']!=0)]
    no_damages = at_centroid_df.loc[at_centroid_df['n_dmgs']==0]
    return events,no_damages

def compute_verification_stats(at_centroid_data, haz_var = 'MESHS',exposure_thresh=0):
    """
    Compute verification statistics for at centroid data

    Parameters
    ----------
    at_centroid_data : pandas.Dataframe
        Dataframe with at centroid data, required cols: n_exp, n_dmgs,date,*haz_var*
    haz_var : str, optional
        hazard variable. The default is 'MESHS'.
    exposure_thresh : int, optional
        number of exposure points above which data is considered. The default is 0.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with skill scores.
    npred : int
        number of prediction.
    """

    #set nans in exposure value to 0 before filtering with exposure thresh
    #(allows also damages at exposure == 0 to be taken into account, e.g. with exposure_thresh=-1)
    at_centroid_data['n_exp']=at_centroid_data['n_exp'].fillna(0)


    data=at_centroid_data[at_centroid_data['n_exp'] > exposure_thresh]

    #make subsets events (damage or prediction), and no_damages
    # (no damage, either false alarms or non events)
    events,no_damages=split_events_non_events(data,haz_var)

    #HITS: number of predictions with at least one observed damage (per centroid)
    A=events.groupby(haz_var).agg({'n_dmgs': lambda x: x[x != 0].count()})
    #number of prediction with no observed damage (per centroid)
    # FALSE ALARMS:
    B=events.groupby(haz_var).agg({'n_dmgs': lambda x: x[x == 0].count()})
    # non
    D=no_damages.groupby(haz_var)['n_dmgs'].count()

    npred=np.int64(np.sum(A[1::]+B[1::]).values[0])
    Asum=np.ones(len(A.index[1::]))*np.nan
    Bsum=np.ones(len(A.index[1::]))*np.nan
    Csum=np.ones(len(A.index[1::]))*np.nan
    Dsum=np.ones(len(A.index[1::]))*np.nan

    index=range(len(A.index[1::]))
    for i in index:
        #indexing of original dataframe (to exclude values at haz intensity 0)
        l=i+1
        # count number hits above threshold
        Asum[i]=A['n_dmgs'].values[l::].sum()
        # count number false alarms above threshold
        Bsum[i]=B['n_dmgs'].values[l::].sum()
        # count number of misses
        Csum[i]=A['n_dmgs'].values[0:l].sum()
        # count number of non_events
        Dsum[i]=D.values[0:l].sum()

    #check here: https://www.cawcr.gov.au/projects/verification/#Types_of_forecasts_and_verifications
    FAR=Bsum/(Asum+Bsum)
    POD=Asum/(Asum+Csum)
    CSI=Asum/(Asum+Bsum+Csum)
    NER=Dsum/(Bsum+Dsum)
    POFD=Bsum/(Bsum+Dsum)
    PC=(Asum+Dsum)/(Asum+Bsum+Csum+Dsum) #proportion correct
    fyes=(Asum+Csum)/(Bsum+Dsum) #fraction of yes observations on all observations
    # SEDI=(np.log(POFD)-np.log(POD)+np.log(1-POD)-np.log(1-POFD))/(np.log(POFD)+np.log(POD)+np.log(1-POD)+np.log(1-POFD))
    HSS=2*(Asum*Dsum-Bsum*Csum)/((Asum + Csum)*(Csum + Dsum) + (Asum + Bsum)*(Bsum + Dsum))
    base_rate=(Asum+Csum)/(Asum+Bsum+Csum+Dsum)
    #Hansen Kuyper Discriminant
    HK = POD - POFD
    #Bias
    B=(Asum+Bsum)/(Asum+Csum)
    #total number of centroids with hail predictions
    Pred=Asum+Bsum
    #total number of centroids with hail damage
    Obs=Asum+Csum

    df=pd.DataFrame({'FAR': FAR,
                     '1-FAR': 1-FAR,
                     'POD': POD,
                     'NER': NER,
                     'CSI': CSI,
                     'HSS': HSS,
                    #  'SEDI': SEDI,
                     'PC': PC,
                     'POFD': POFD,
                     'HK': HK,
                     'B':B,
                     'PRED': Pred,
                     'Fyes': fyes,
                     'OBS': Obs,
                     's': base_rate},
                    index = A.index[1::])
    return df, npred



def get_at_gridpoints_from_xr(ds,haz_var,min_count=50,haz_range=None):
    """Creat at_gridpoints dataframe from xarray dataset

    Args:
        ds (xr.DataSet): Dataset with n_exp, n_dmgs and haz_var as variables
        haz_var (str): hazard variable
        min_count (int, optional): Minimum exposure points per gridcell. Defaults to 50.

    Returns:
        at_gridpoint_df: dataframe with one row per unique gridpoind-date combination
    """

    #rename variables in case ds_KGV is used as input directly
    if 'n_count_exposure' in ds and 'n_count' in ds:
        ds = ds.rename({'n_count_exposure':'n_exp','n_count':'n_dmgs'})

    assert('n_exp' in ds and 'n_dmgs' in ds and haz_var in ds)
    #stack dimensions
    ds_stacked = ds[['n_exp',haz_var,'n_dmgs']].stack({'stacked':['chx','chy']})
    #filter out gridpoints with too few exposure points
    ds_stacked = ds_stacked.loc[dict(stacked=ds_stacked.n_exp>min_count)]

    #create dataframe
    at_gridpoint_df = ds_stacked.to_dataframe()

    #select relevant variables and fill nan values of n_dmg and n_exp with 0
    at_gridpoint_df = at_gridpoint_df[['n_exp','n_dmgs',haz_var]].fillna({'n_exp':0,'n_dmgs':0,haz_var:np.nan})

    if haz_var == 'MESHS': #fill nan MESHS values with 0
        at_gridpoint_df[haz_var] = at_gridpoint_df[haz_var].fillna(0)

    #select only relevant hazard range
    if haz_range is not None:
        at_gridpoint_df = at_gridpoint_df.loc[(at_gridpoint_df[haz_var]>=haz_range[0]) &
                                              (at_gridpoint_df[haz_var]<=haz_range[1]),:]

    return at_gridpoint_df