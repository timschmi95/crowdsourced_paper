"""
Utility function for subproject E (Timo)
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from scipy.ndimage import affine_transform



def stretch_array_any_direction2(arr, angle, primary_factor=1.5, secondary_factor=1/1.5):
    """
    Stretch a 2D numpy array along a given direction (angle) with two defined factors.

    Parameters:
    - arr: Input n x n 2D numpy array
    - angle: Direction to stretch in degrees (0 = horizontal, 90 = vertical, etc.)
    - primary_factor: Stretching factor along the specified angle.
    - secondary_factor: Stretching factor perpendicular to the specified angle.

    Returns:
    - Stretched 2D numpy array
    """
    # Convert angle to radians
    theta = np.deg2rad(angle)

    # Create the affine transformation matrix for rotation (clockwise to align x-axis with the angle)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    # Rotation matrix for counterclockwise rotation
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])

    # Stretching matrix (applies scaling along x-axis for primary_factor and y-axis for secondary_factor)
    stretch_matrix = np.array([[1/secondary_factor, 0],
                               [0, 1/primary_factor]])

    # Inverse of the rotation matrix to rotate back after stretching
    inv_rotation_matrix = np.linalg.inv(rotation_matrix)

    # The combined affine transformation matrix (rotate -> stretch -> rotate back)
    combined_matrix = inv_rotation_matrix @ stretch_matrix @ rotation_matrix

    # Calculate the center of the image
    # Assert that the array has an uneven number of rows and columns (and is square)
    assert arr.shape[0] % 2 == 1 and arr.shape[0] == arr.shape[1]
    center = np.array(arr.shape) // 2

    # Apply affine transformation with the computed matrix and center
    transformed_array = affine_transform(arr, combined_matrix, offset=center - center @ combined_matrix)

    return transformed_array



def create_imp_df(imp_now,imp_obs_now,xmin):
    """create a dataframe with modelled and observed impacts

    Args:
        imp_now (climada.engine.Impact): impact modelled
        imp_obs_now (climada.engine.Impact): impact observed
        xmin (float): minimum total impact to consider

    Returns:
        imp_df: Dataframe with modelled and observed impacts
    """
    ord_dates_nonZero = np.sort(np.unique(np.concatenate((imp_now.date[imp_now.at_event>xmin],
                                                        imp_obs_now.date[imp_obs_now.at_event>xmin]))))
    imp_df = pd.DataFrame(index=ord_dates_nonZero,data={
        'date':[dt.datetime.fromordinal(d) for d in ord_dates_nonZero],
        "prePost2012":[dt.datetime.fromordinal(d).year>2012 for d in ord_dates_nonZero]})

    imp_dfMod = pd.DataFrame(data={"imp_modelled":imp_now.at_event},index = imp_now.date)
    imp_dfObs = pd.DataFrame(data={"imp_obs":imp_obs_now.at_event},index = imp_obs_now.date)

    imp_df= imp_df.join(imp_dfMod,how='left') #join on index
    imp_df= imp_df.join(imp_dfObs,how='left') #join on index
    imp_df = imp_df.fillna(0) #fill NaN
    return imp_df


def calc_skill_scores(df_now,dmg_thresh,calc_additional_scores=False,imp_modelled="imp_modelled",imp_obs="imp_obs"):

        if imp_modelled is not None:
            df_now

        #get number of points
        considered_events = sum((df_now[imp_modelled]>dmg_thresh) | (df_now[imp_obs]>dmg_thresh))

        neither_is_zero = (df_now[imp_modelled]!=0) & (df_now[imp_obs]!=0)
        rmse = np.sqrt(np.mean((df_now[imp_modelled][neither_is_zero]-df_now[imp_obs][neither_is_zero])**2))
        rmsf = np.exp(np.sqrt(np.mean(np.log(np.divide(
            df_now[imp_modelled][neither_is_zero],df_now[imp_obs][neither_is_zero]))**2)))
        rmsf_weighted = np.exp(np.sqrt(np.mean(np.log(np.divide(
            df_now[imp_modelled][neither_is_zero],df_now[imp_obs][neither_is_zero]))**2
            *df_now[imp_obs][neither_is_zero])/np.sum(df_now[imp_obs][neither_is_zero])))

        above_either_thresh = (df_now[imp_obs]>dmg_thresh) | (df_now[imp_modelled]>dmg_thresh)

        FAR = (sum((df_now[imp_modelled]>dmg_thresh) & (df_now[imp_modelled]>df_now[imp_obs]*10)))/sum((df_now[imp_modelled]>dmg_thresh))
        POD = sum((df_now[imp_obs]>dmg_thresh) & (df_now[imp_obs]>df_now[imp_modelled]/10) &
                  (df_now[imp_obs]<df_now[imp_modelled]*10))/sum((df_now[imp_obs]>dmg_thresh))
        p_within_OOM = sum(above_either_thresh & (df_now[imp_obs]>df_now[imp_modelled]/10) &
                           (df_now[imp_obs]<df_now[imp_modelled]*10))/sum(above_either_thresh)

        if calc_additional_scores:
            #calculate RMSF only for events above the damage threshold
            rmsf_above_thresh = np.exp(np.sqrt(np.mean(np.log(np.divide(
            df_now[imp_modelled][above_either_thresh & neither_is_zero],df_now[imp_obs][above_either_thresh & neither_is_zero]))**2)))

            return rmse,rmsf,rmsf_weighted,FAR,POD,p_within_OOM,considered_events,rmsf_above_thresh

        return rmse,rmsf,rmsf_weighted,FAR,POD,p_within_OOM,considered_events

def get_mod_obs_ratio(imp_df,PAA_thresh,MDR_thresh,imp_obs='imp_observed',
                      imp_mod='imp_MESHS',imp_PAA_obs='n_count',
                      imp_PAA_mod='n_buildings_MESHS',q=0.75,plot=True):
    """Calculate the model-to-observed ratio for for monetary damages and number of affected buildings
    The model-to-observed ratio is calculated for events above a certain threshold (PAA_thresh, MDR_thresh)

    Args:
        imp_df (pd.DataFrame): Dataframe with modelled and observed impacts, with columns
        as indicated in the function arguments
        PAA_thresh (float): Threshold for minimum number of affected buildings
        MDR_thresh (float): Threshold for minimum monetary damages (modelled or observed)
        imp_obs (str, optional): column name of observed damages. Defaults to 'imp_observed'.
        imp_mod (str, optional): Column name of modelled damages. Defaults to 'imp_MESHS'.
        imp_PAA_obs (str, optional): Columns name of observed number of damaged assets. Defaults to 'n_count'.
        imp_PAA_mod (str, optional): Columns name of modelled number of damages assets. Defaults to 'n_buildings_MESHS'.
        q (float, optional): Quantile for the secondary evaluatgion metric. Defaults to 0.75.

    Returns:
        tuple: all PAA ratios, all MDR ratios
    """


    imp_sel = imp_df.loc[(imp_df[imp_obs]>=MDR_thresh) | (imp_df[imp_mod]>=MDR_thresh)]
    imp_sel['model-obs-ratio'] = imp_sel[imp_mod]/imp_sel[imp_obs]

    imp_selPAA = imp_df.loc[(imp_df[imp_PAA_obs]>=PAA_thresh) | (imp_df[imp_PAA_mod]>=PAA_thresh)]
    imp_selPAA['model-obs-ratio'] = imp_selPAA[imp_PAA_mod]/imp_selPAA[imp_PAA_obs]

    #invert negative ratios
    with np.errstate(divide='ignore'): # Suppress warnings for divide-by-zero. They are correctly handled as np.exp(np.log(0).abs()) = np.inf
        MDR_ratio_factor = np.exp(np.log(imp_sel['model-obs-ratio']).abs())
        PAA_ratio_factor = np.exp(np.log(imp_selPAA['model-obs-ratio']).abs())

    if plot:
        plot_log = False
        if plot_log:
            bins = np.logspace(0,2,20)
            xscale = 'log'
        else:
            bins = np.arange(1,11,0.5)
            xscale = 'linear'

        fig,ax = plt.subplots()
        ax.hist(MDR_ratio_factor,bins=bins,label='MDR') # note 10**(np.log10()) leads to equal result
        ax.hist(PAA_ratio_factor,bins=bins,
                label='PAA',color='orange',alpha=0.5) # note 10**(np.log10()) leads to equal result

        ymax = ax.get_ylim()[1]
        #add quantiles
        for q in [0.5,q]:
            q_mdr = MDR_ratio_factor.quantile(q)
            if q_mdr <= bins.max():
                ax.axvline(q_mdr,color='blue',linestyle='--')#,label=f'Q{q} MDR')
                ax.text(q_mdr,ymax*0.9,f'Q {q}',color = 'blue')
            q_paa = PAA_ratio_factor.quantile(q)
            if q_paa <= bins.max():
                ax.axvline(q_paa,color='orange',linestyle='--')#,label=f'Q{q} PAA')
                ax.text(q_paa,ymax*0.9,f'Q {q}',color = 'darkorange')
            # ax.axvline(MDR_ratio_factor.quantile(0.5),color='blue',linestyle='--')

        ax.set(xscale=xscale,xlabel = 'Modelled-to-observed factor(+/-)',ylabel='No. of events',
                xlim=[bins.min(),bins.max()])
        ax.legend(loc='center right')

    #print key numbersprint(f"PAA mean factor: {PAA_ratio_factor.mean():.2f} | MDR mean factor: {MDR_ratio_factor.mean():.2f}")
    print(f"PAA median factor: {PAA_ratio_factor.median():.1f} | MDR median factor: {MDR_ratio_factor.median():.1f}")
    print(f"PAA {q} quantile: {PAA_ratio_factor.quantile(q):.1f} | MDR {q} quantile: {MDR_ratio_factor.quantile(q):.1f}")

    return PAA_ratio_factor,MDR_ratio_factor


def ds_to_df(ds_all,window_size=30,vars=None):
    """Short helper function to transform an xr.Dataset containing per-grid cell values
    of crowdsourced hail size and observed damages to a pandas DataFrame

    Args:
        ds_all (xr.Dataset): dataset containing per-grid cell values
        window_size (int, optional): windowsize (Number of points, not mm) for
            rolling output. Defaults to 30.
        vars (list, optional): list of variables. Defaults to None.

    Returns:
        pd.DataFrame: pandas dataframe with same data
    """
    if vars is None:
        vars = ['h_smooth','PAA','n_count_exposure']
    df = ds_all[vars].to_dataframe()
    df = df.dropna(how='any')
    # df.plot.scatter(x='h_smooth',y='PAA',c='n_count_exposure',cmap='Blues',vmax=100)


    df['h_smooth_r'] = df['h_smooth'].round(0)
    df = df.sort_values('h_smooth_r')
    if window_size is not None:
        df['PAA_rolling'] = df['PAA'].rolling(window_size,center=True,min_periods=1).mean()
        if 'MDR' in vars:
            df['MDR_rolling'] = df['MDR'].rolling(window_size,center=True,min_periods=1).mean()
    return df


def plot_dmg_bin(df,ax,bin_size=5,ymin=0,color='green',alpha=0.3,new_axis=True,
                 pl_var = 'count_dmg',ymax=None,relative=False,**kwargs):

    #if 0 in df.index:
    bins = pd.cut(df.index,right=False,
                  bins = np.arange(min(df.index),max(df.index)+bin_size,bin_size))
    bin_mids = bins.categories.mid.values
    bin_right = bins.categories.right.values
    bin_l = bins.categories.left.values

    if df.index[0]==0 and df.index[1]>bin_size: #first index is 0 and only contains 0 values (no rolling!)
        bin_mids[0] = 0 #change to zero
        bin_l[0] -= bin_size/2 #center first bin at 0
    df_bins=df.groupby(bins).sum()
    if relative:
        df_bins[pl_var] = df_bins[pl_var]/df_bins[pl_var].sum()
    if new_axis:
        ax2 = ax.twinx()
    else:
        ax2 = ax

    #ax2.step(bin_mids,df_bins[pl_var],color = 'green',label = '#damages',alpha=0.2)
    ax2.bar(bin_l,df_bins[pl_var],width=bin_size,color = color,label = '#damages',
            alpha=alpha,align='edge',**kwargs)
    # ax2.set(ylabel="Number of damage reports",ylim=[ymin,max(df_bins[pl_var])*1.1])

    if pl_var == 'count_all': #create axis break
        ylabel = "Number of exposed assets"
        ymax = max(df_bins[pl_var][1:])*1.3 if ymax is None else ymax
        # ax2.set_ylim([ymin,ymax])
        ax2.annotate(f'Out of axis: {(df_bins[pl_var][0]):.1e}',xy=(bin_mids[0],ymax),
                     xytext=(bin_right[0]*1.1,ymax*0.9),
                     arrowprops=dict(facecolor=color,shrink=0.05),color=color)

    elif pl_var == 'count_dmg':
        ylabel = "Fraction of damage reports" if relative else "Number of damage reports"
        ymax=max(df_bins[pl_var])*1.1 if ymax is None else ymax
    elif pl_var == 'count_cells':
        ylabel = "Number of grid cells"
        ymax=max(df_bins[pl_var])*1.1 if ymax is None else ymax

    ax2.set(ylabel=ylabel,ylim=[ymin,ymax])
    return ax2