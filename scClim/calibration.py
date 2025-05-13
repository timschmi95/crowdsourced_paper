# -*- coding: utf-8 -*-
"""
Empirical calibration functions for spatially explicit vulnerability function calibration


"""
import xarray as xr

# other
import numpy as np
import pandas as pd
import sys, os  # argparse
from scipy import sparse
import climada
from climada.hazard import Hazard
from climada import CONFIG

sys.path.append(str(CONFIG.local_data.func_dir))
import scClim as sc
from scClim.constants import CUT_OFF_DICT, INT_RANGE_DICT
import datetime as dt
import copy
import json
import matplotlib
import matplotlib.pyplot as plt


# %% Helper functions
def log_func(x, a, b, c, d):
    return a / (1.0 + np.exp(-c * (x - d))) + b


def fit_log_curve(xData, yData, p0):
    from scipy.optimize import curve_fit

    fittedParameters, pcov = curve_fit(log_func, xData, yData, p0)
    print("Fitted parameters:", fittedParameters)
    print()

    xrange = np.arange(20, 121, 1)
    modelPredictions = log_func(xData, *fittedParameters)

    absError = modelPredictions - yData

    SE = np.square(absError)  # squared errors
    MSE = np.mean(SE)  # mean squared errors
    RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(absError) / np.var(yData))

    modelPredictions_out = log_func(xrange, *fittedParameters)

    print()
    print("RMSE:", RMSE)
    print("R-squared:", Rsquared)

    print()

    return xrange, modelPredictions_out, Rsquared, fittedParameters


def filter_exp(exp, yearMin, yearMax):
    """filter exposure to remove buildings that are not built yet

    Args:
        exp (climada.entity.Exposures): exposure to be filtered
        yearMin (int): minimum year to be kept (usually 0, except if only subset
                                                of buildings is to be kept)
        yearMax (max): maximum year to be kept (usually the year of the hazard event)

    Returns:
        exp_out: filtered exposure
    """

    exp_out = exp.copy()
    sel = np.logical_and(exp.gdf.Baujahr >= yearMin, exp.gdf.Baujahr <= yearMax)
    exp_out.gdf = exp.gdf.loc[sel, :]
    if "centr_HL" in exp_out.gdf.columns:
        exp_out.gdf.centr_HL = exp_out.gdf.centr_HL.astype(int)
    return exp_out
    # def filter_exp_fast(exp, yearMin, yearMax): #only slightly faster!
    # sel = np.logical_and(exp.gdf.Baujahr >= yearMin,
    # exp.gdf.Baujahr <= yearMax)
    # out_gdf = exp.gdf.where(sel).dropna()
    # if 'centr_HL' in out_gdf.columns:
    # out_gdf.centr_HL = out_gdf.centr_HL.astype(int)
    # return out_gdf


def get_exposed_assets(exposure, variable, exposure_type, variable_2=None):
    """

    Get number and total value of exposed assets

    Parameters
    ----------
    exposure : climada.entity.exposures.base.Exposures
        exposure for which the damage function is to be calibrated
    variable : str
        hazard variable name
    exposure_type : str
        type of exposure provided. "GVZ" or "agriculture"
    variable_2:

    Returns
    -------
    all_count : pandas.Dataframe
        Dataframe with hazard intensities as index and number of exposed assets as column
    all_value : pandas.Dataframe
        Dataframe with hazard intensities as index and the total value of the
        exposed assets as column
    at_centroid : pandas.Dataframae
        Dataframe with centroids as index and colums with the hazard intensity,
        the number of exposed assets (counts), and the total value

    """

    # count number of exposed assets per MESHS
    if exposure_type == "GVZ":
        all_count = (
            exposure.gdf.groupby(variable).VersicherungsID.count().rename("count_all")
        )
        count_centr = (
            exposure.gdf.groupby("centr_HL").VersicherungsID.count().rename("counts")
        )
    elif exposure_type == "agriculture":
        all_count = exposure.gdf.groupby(variable).n_fields.sum().rename("count_all")
        count_centr = exposure.gdf.groupby("centr_HL").n_fields.sum().rename("counts")

    # get intensity at centroid
    intens_centr = exposure.gdf.groupby("centr_HL")[variable].mean().rename("intensity")

    # compute total value exposed to hail hazard per value hazard variable
    all_value = exposure.gdf.groupby(variable).value.sum().rename("value_all")

    # get total value at centroids
    value_centr = exposure.gdf.groupby("centr_HL").value.sum().rename("value")

    if variable_2 is not None and variable_2 in exposure.gdf.keys():
        intens_2_centr = (
            exposure.gdf.groupby("centr_HL")[variable_2].max().rename("intensity_2")
        )
        # concat intensity values, counts, and values at centroid
        at_centroid = pd.concat(
            [intens_centr, count_centr, value_centr, intens_2_centr], axis=1
        )
    else:
        at_centroid = pd.concat([intens_centr, count_centr, value_centr], axis=1)

    return all_count, all_value, at_centroid


def add_field_area_to_damages(exp_dmg, exposure, field_area_mean):
    """
    for agriculture, assign average field area from exposure to each
    reported damage claim if damage is reported in a region without exposure,
    use Swiss average field area
    """

    # add field area column
    exp_dmg.gdf["field_area"] = np.nan

    # loop over all damages and add field area
    for index in exp_dmg.gdf.index:

        # get centroids
        centroid = exp_dmg.gdf.loc[index, "centr_HL"]

        # read field area from exposure
        field_area = exposure.gdf.loc[
            exposure.gdf[exposure.gdf["centr_HL"] == centroid].index, "field_area"
        ].values

        if field_area.size == 1:
            exp_dmg.gdf.loc[index, "field_area"] = field_area[0]
        elif field_area.size == 0:
            # issue warning if damage is recorded where no exposure is present
            exp_dmg.gdf.loc[index, "field_area"] = field_area_mean
            # warnings.warn('Damage reported in region without exposure. Swiss average field size used.')
        else:
            raise ValueError("more than one field area found.")

    return exp_dmg


def get_damaged_assets(
    damage,
    variable,
    date_now,
    exposure_type,
    get_PVA=False,
    variable_2=None,
    fraction_insured=1,
):
    """get variables specific to damage claims as a function of hazard variable

    Parameters
    ----------
    damage: climada impact
            damage claims of an event as climada impact object
    variable: str
            hazard variable name
    exposure_type: str
        type of exposure ('agriculture' or 'GVZ')

    Returns
    -------
    dmg_count: pandas.Dataframe
            number of damage claims grouped by hazard intensity
    dmg_value: pandas.Dataframe
            sum of damage value grouped by hazard intensity
    affected_value: pandas.Dataframe
            total value affected grouped by hazard intensity
            (i.e. Versicherungssumme or total field area of damaged fields)
    at_centroid: pandas.Dataframe
           counts, value, and value affected by centroid
    """

    # date selection
    date_idx = damage.date == date_now.toordinal()
    dmg_1 = damage.imp_mat[date_idx, :]
    # exposure (centroid) selection
    obj_idx = dmg_1.nonzero()[1]
    # get impact at current date
    imp_now = damage.imp_mat[date_idx, obj_idx].getA1()
    # get hazard intensity at current date
    haz_intensity_now = damage.haz_intensity[date_idx, obj_idx].getA1()
    # get second hazard intensity at current date
    if hasattr(damage, "haz_intensity_2"):
        haz_intensity_2_now = damage.haz_intensity_2[date_idx, obj_idx].getA1()

    # get value affected (needs attribute aff_mat for damage impact object)
    if get_PVA == True:
        val_affected_now = damage.aff_mat[date_idx, obj_idx].getA1()

    # create Dataframe with columns damages, hazard intensity, and centroid
    if hasattr(damage, "haz_intensity_2"):
        dmg_df = pd.DataFrame(
            {
                "damage": imp_now,
                variable: haz_intensity_now,
                variable_2: haz_intensity_2_now,
                "centr_HL": damage.centr_HL[obj_idx],
            }
        )
    else:
        dmg_df = pd.DataFrame(
            {
                "damage": imp_now,
                variable: haz_intensity_now,
                "centr_HL": damage.centr_HL[obj_idx],
            }
        )

    # if requested, compute total value affected per intensity value and per centroid
    if get_PVA == True:
        dmg_df["value_affected"] = val_affected_now
        affected_value = (
            dmg_df.groupby(variable).value_affected.sum().rename("value_affected")
            / fraction_insured
        )
        aff_val_centr = (
            dmg_df.groupby("centr_HL").value_affected.sum().rename("value_affected")
            / fraction_insured
        )

    else:
        affected_value = None

    # get number of damage claims and damage value per intensity value and per centroid
    dmg_count = (
        dmg_df.groupby(variable).damage.count().rename("count_dmg") / fraction_insured
    )
    dmg_count = dmg_count.round(decimals=0)  # only allow integers for count
    dmg_value = (
        dmg_df.groupby(variable).damage.sum().rename("value_dmg") / fraction_insured
    )
    dmg_count_centr = (
        dmg_df.groupby("centr_HL").damage.count().rename("counts") / fraction_insured
    )
    dmg_count_centr = dmg_count_centr.round(decimals=0)  # only allow integers for count
    dmg_value_centr = (
        dmg_df.groupby("centr_HL").damage.sum().rename("value") / fraction_insured
    )
    intens_centr = dmg_df.groupby("centr_HL")[variable].max().rename("intensity")

    # get value of second hazard variable at centroids
    if hasattr(damage, "haz_intensity_2"):
        intens_2_centr = (
            dmg_df.groupby("centr_HL")[variable_2].max().rename("intensity_2")
        )
        at_centroid_list = [
            dmg_count_centr,
            dmg_value_centr,
            intens_centr,
            intens_2_centr,
        ]
    else:
        at_centroid_list = [dmg_count_centr, dmg_value_centr, intens_centr]

    # concatenate per centroid data
    # concat counts, damage value, and value affected
    if get_PVA:
        at_centroid_list.append(aff_val_centr)

    at_centroid = pd.concat(at_centroid_list, axis=1)

    return dmg_count, dmg_value, affected_value, at_centroid


def smooth_data_running_mean(ds, variable, intensity_range, cut_off, window_size):
    """

    Parameters
    ----------
    ds : xr.dataset
        containing calibration data.
    variable : str
        Hazard variable.
    intensity_range : np.array
        Intensity range of hazard variable.
    cut_off : int
        cut off value for the smooth-empirical fit.
    window_size : int
        windowsize for rolling average.

    Returns
    -------
    ds_roll : xr.dataset
        same as ds, but with rolling average values.
    ds_roll_cut : xr.dataset
        same as ds_roll, but with values above cut_off as constant defined by
        the weighted average.

    """

    if (
        intensity_range[0] == 0
    ):  # all values except 0, because it should not be used for rolling
        ds_roll = ds.sel({variable: slice(intensity_range[1], intensity_range[-1])})
        ds_roll = ds_roll.rolling(
            dim={variable: window_size},
            min_periods=max(
                1,
                int(  # add max(1,x) in case windowsize=1 (no rolling)
                    np.floor(window_size / 2)
                ),
            ),
            center=True,
        ).sum()
        ds_roll = xr.concat([ds.sel({variable: 0}), ds_roll], dim=variable)
    else:  # include all values in rolling average
        ds_roll = ds.sel({variable: slice(intensity_range[0], intensity_range[-1])})
        ds_roll = ds_roll.rolling(
            dim={variable: window_size},
            min_periods=max(
                1,
                int(  # add max(1,x) in case windowsize=1 (no rolling)
                    np.floor(window_size / 2)
                ),
            ),
            center=True,
        ).sum()

    # set values <1e-8 to zero. Bc of a bug (rounding errors) in xarray.rolling()
    ds_roll = ds_roll.where(ds_roll > 1e-8, other=0)

    # create rolling average cut off at a threshold
    ds_roll_cut = ds_roll.copy(deep=True)
    ds_roll_cut.loc[{variable: slice(cut_off, intensity_range.max())}] = (
        ds_roll_cut.loc[{variable: slice(cut_off, intensity_range.max())}]
        .sum(dim=variable)
        .expand_dims({variable: intensity_range[intensity_range >= cut_off]}, axis=1)
    )

    return ds_roll, ds_roll_cut


def get_unique_dates_from_impacts(damages, dates_modeled_imp=None):
    """
    Get unique dates from a measured impact (observed dates) and modeled impacts (modeled dates)'

    Parameters
    ----------
    damages : climada.engine.impact.Impact
        Impact object containing observed impacts
    dates_modeled_imp : list
        list of dates (dt.datetime) based on modeled data (e.g. a modeled impact)
        The default is None.

    Returns
    -------
    dt_dates : list
        unique list of dates (dt.datetime) that comining observed and modeled dates with impact

    """
    import datetime as dt

    # str_dates = [dt.datetime.fromordinal(d).strftime('%d%m%Y') for d in damages.date]
    dt_dates = [dt.datetime.fromordinal(d) for d in damages.date]

    if dates_modeled_imp is not None:
        dt_dates = np.unique(np.append(dt_dates, dates_modeled_imp))

    # dates must be sorted for correct exposure filtering (with 'Baujahr' column')
    return sorted(dt_dates)


def assign_intensity_to_damages(damages, haz, variable, haz_2=None, variable_2=None):
    """


    Parameters
    ----------
    damages : climada.Impact
        damages impact object
    haz : climada.Hazard
        hazard object
    variable : str
        variable name

    Returns
    -------
    damages : climada.Impact
        damages with new attribute haz_intensity

    """
    haz_date_idx = np.array([np.where(haz.date == date)[0][0] for date in damages.date])
    haz_loc_idx = damages.centr_HL
    damages.haz_intensity = haz.intensity[haz_date_idx, :][:, haz_loc_idx]

    if haz_2 is None:
        pass
    else:
        damages.haz_intensity_2 = haz_2.intensity[haz_date_idx, :][:, haz_loc_idx]

    return damages


def at_centroid_from_exp_and_dmg(
    exp_at_centroid,
    variable,
    date_now,
    dmg_at_centroid=None,
    variable_2=None,
    get_PVA=False,
):
    """
    Parameters
    ----------
    exp_at_centroid : pd.DataFrame
        Dataframe containing exposure data at centroids
    variable : str
        hazard variable
    date_now : datetime
        current date
    dmg_at_centroid : pd.DataFrame, optional
        Dataframe containing damage data at centroids. The default is None.
    variable_2 : str, optional
        secondary hazard variable to be traced along analysis. The default is None.
    get_PVA : boolean, optional
        get PVA and MDD in output, requires a column 'value_affected' in
        dmg_at_centroids. The default is False.

    Returns
    -------
    at_centroid_dict : Dictionary
        Dictionary containing the important values at all centroids
        that have at least a damage or an overlap between
        exposure and hazard (i.e., a damage prediction)

    """

    at_centroid_dict = {}

    if dmg_at_centroid is not None:
        # reindex damages at centroid to exposure indices
        new_df = pd.DataFrame(
            index=np.unique(
                np.concatenate([dmg_at_centroid.index, exp_at_centroid.index])
            )
        )
        dmg_at_centroid = dmg_at_centroid.reindex(new_df.index)
        exp_at_centroid = exp_at_centroid.reindex(new_df.index)

        # compute PAA and MDR at centroid
        at_centroid_dict["PAA"] = dmg_at_centroid["counts"] / exp_at_centroid["counts"]
        at_centroid_dict["MDR"] = dmg_at_centroid["value"] / exp_at_centroid["value"]
        # get intensity at centroid
        at_centroid_dict[variable] = np.nanmax(
            (
                exp_at_centroid["intensity"].replace("nan", np.nan),
                dmg_at_centroid["intensity"].replace("nan", np.nan),
            ),
            axis=0,
        )
        at_centroid_dict["n_exp"] = exp_at_centroid["counts"]
        at_centroid_dict["n_dmgs"] = dmg_at_centroid["counts"]
        at_centroid_dict["exp_val"] = exp_at_centroid["value"]
        at_centroid_dict["dmg_val"] = dmg_at_centroid["value"]
        at_centroid_dict["date"] = date_now

        if variable_2 is not None:
            at_centroid_dict[variable_2] = np.nanmax(
                (
                    exp_at_centroid["intensity_2"].replace("nan", np.nan),
                    dmg_at_centroid["intensity_2"].replace("nan", np.nan),
                ),
                axis=0,
            )

        if get_PVA == True:
            at_centroid_dict["PVA"] = (
                dmg_at_centroid["value_affected"] / exp_at_centroid["value"]
            )
            at_centroid_dict["MDD"] = (
                dmg_at_centroid["value"] / dmg_at_centroid["value_affected"]
            )
        else:
            at_centroid_dict["PVA"] = None
            at_centroid_dict["MDD"] = None
    else:
        at_centroid_dict[variable] = exp_at_centroid["intensity"].replace("nan", np.nan)
        at_centroid_dict["n_exp"] = exp_at_centroid["counts"]
        at_centroid_dict["date"] = date_now
        at_centroid_dict["n_dmgs"] = np.nan
        at_centroid_dict["dmg_val"] = np.nan

        if variable_2 is not None:
            at_centroid_dict[variable_2] = exp_at_centroid["intensity_2"].replace(
                "nan", np.nan
            )

    # make dataframe
    at_centroid_df = pd.DataFrame(
        {k: v for k, v in at_centroid_dict.items() if v is not None}
    )
    at_centroid_df["centr_HL"] = at_centroid_df.index

    # only include data points with variable > 0 or damages > 0
    at_centroid_df["n_dmgs"] = at_centroid_df["n_dmgs"].fillna(0)
    at_centroid_df["dmg_val"] = at_centroid_df["dmg_val"].fillna(0)
    # at_centroid_df=at_centroid_df.loc[(at_centroid_df[variable]!=0) | (at_centroid_df['n_dmgs']!=0)]

    return at_centroid_df


# %% Main calibration functions
def empirical_calibration_per_exposure(
    hazard_object,
    exposure_object,
    damages,
    exposure_type="agriculture",
    variable="MESHS",
    hazard_object_2=None,
    variable_2=None,
    get_PVA=False,
    filter_year=None,
    dates_modeled_imp=None,
    roll_window_size=10,
    fraction_insured=1,
    cut_off_overwrite=None,
):
    """
    main function for calibration - computes a range of variables as
    function of values of the hazard variable

    Parameters
    ----------
    hazard_object: climada.hazard
            hazard object used for calibraiton
    exposure_object: climada.entity.exposures.base.Exposures
            exposure object used for calibration
    damages: dict or climada.engine.impact.Impact
            either dictionary of exposure objects, dictionary of impact objects
            or a single impact objects hat contain damage claims
    exposure_type: str
            type of exposure ('agriculture' or 'GVZ')
    variable: str
            hazard variable name
    hazard_object_2: climada.hazard
            hazard object with a secondary hazard variable that is traced trough the analysis
    variable_2: str
            secondary hazard variable name
    filter_year : tuple
            (yearMin,yearMax): to filter exposure data by year when an exposure was built
    dates_modeled_imp : np.array
        array of dt.datetime() with all dates where modelled impact is not zero
        (i.e. there is a hazard_intensity>0 somewhere)
    roll_window_size : int
        Size of rolling window. default to 10 for 10mm moving window in MESHS
    cut_off_overwrite : int
        Overwrite the cutoff value for the rolling average. Default is None.

    Returns
    -------

    ds: xarray.Dataset
        Dataset with additive calibration-relevant variables as function of hazard variable for each event
            count_cell: number of grid cells with non-zero exposure
            count_all: number of exposed assets
            value_all: total value of exposed assets
            count_dmg: number of affected assets
            value_dmg: total damage
            value_affected: total value affected
    df_all: pandas.Dataframe
        Dataframe which corresponds to the sum of the data in ds over all events
    ds_roll: xarray.Dataset
        ds with rolling average applied
    ds_roll_cut: xarray.Dataset
        ds_roll with cutoff for high values of hazard variable
    values_at_centroid_all: pandas.Dataframe
        A Dataframe with information on damages per gridpoint (PAA, MDD, MDR, n_dmgs),
        and exposure (n_exp), as well as hazard intensity (variable and variable_2)
    intensity_range: numpy.array
        Intensity array for hazard variable used in the calibration
    """
    ### DECLARE VARIABLES ##
    df_all = pd.DataFrame()  # output dataframe
    haz = copy.deepcopy(hazard_object)  # create copy not to modify original hazard
    haz_2 = copy.deepcopy(hazard_object_2)  # create copy not to modify original hazard

    exp_sel = (
        exposure_object.copy()
    )  # create a copy of exposure not to modify original exposure

    # CHECKS
    if haz_2 is None:
        if variable_2 is None:
            pass
        else:
            print("Variable_2 set to None as no secondary hazard has been passed.")
            variable_2 = None
    else:
        if variable_2 is None:
            raise ValueError("Need to pass variable_2 to consider secondary hazard.")

    # assert <500 intensity steps in hazard data. For this method to wokr, the
    # hazard must be discretized as data is grouped according to hazard
    # intensity values. Recommended ~<100 steps
    if len(np.unique(haz.intensity.data)) > 500:
        raise ValueError(
            "Too many unique hazard intensity values. Recommended <100 steps."
        )

    # assert no duplicate dates in hazard
    assert len(haz.date) == len(np.unique(haz.date))

    # ASSIGN CENTROIDS
    exp_sel.assign_centroids(haz)
    sc.assign_centroids_imp(damages, haz)

    # GET DATELISTS
    # from observed damages and modeled damages
    dt_dates = get_unique_dates_from_impacts(
        damages, dates_modeled_imp=dates_modeled_imp
    )
    # from damages only
    dt_dates_dmgs = [dt.datetime.fromordinal(date) for date in damages.date]

    # DEFINE INTENSITY RANGE
    try:
        intensity_range = INT_RANGE_DICT[variable]
    except KeyError:
        raise ValueError(
            f"Calibration not implemented for variable {variable}."
            " Add value to INT_RANGE_DICT"
        )

    # ASSIGN INTENSITY VALUES TO DAMAGES
    if variable == "HKE":  # round hazard intensity to steps of e.g. 50J/m2
        values, counts = np.unique(np.diff(intensity_range), return_counts=True)
        step = values[np.argmax(counts)]  # get most common step size
        haz.intensity.data = np.round(haz.intensity.data / step) * step

    # assign intensities
    if hazard_object_2 is None:
        damages = assign_intensity_to_damages(damages, haz, variable)
    else:
        damages = assign_intensity_to_damages(
            damages, haz, variable, haz_2=haz_2, variable_2=variable_2
        )
    year = -np.inf  # initialize year for year filtering

    # loop over dates
    for date_now in dt_dates:

        # filter exposure and recreate exp_sel once for each year
        if filter_year and date_now.year > year:
            year = date_now.year
            exp_sel = exposure_object.copy()
            exp_sel.assign_centroids(haz)
            exp_sel = filter_exp(exp_sel, filter_year[0], min(year, filter_year[1]))

        # add intensity values to exposure gdf of current date
        intensity = (
            haz.intensity[haz.date == date_now.toordinal(), :].toarray().squeeze()
        )
        exp_sel.gdf[variable] = intensity[exp_sel.gdf.centr_HL]
        if hazard_object_2 is not None:
            intensity_2 = (
                haz_2.intensity[haz_2.date == date_now.toordinal(), :]
                .toarray()
                .squeeze()
            )
            exp_sel.gdf[variable_2] = intensity_2[exp_sel.gdf.centr_HL]

        # for agriculture exposure compute an average field area per grid cell
        if exposure_type == "agriculture":
            exp_sel.gdf["field_area"] = exp_sel.gdf["value"] / exp_sel.gdf["n_fields"]
            # compute Swiss average field area to use if no field area can be defined
            # field_area_mean=exp_sel.gdf['value'].sum()/exp_sel.gdf['n_fields'].sum()

        # get counts, values ordered by intensity and centroid based information
        all_count, all_value, exp_at_centroid = get_exposed_assets(
            exp_sel, variable, exposure_type, variable_2=variable_2
        )

        ### GET DAMAGED ASSETS

        # if damages have been reported at the current date, get them
        if date_now in list(dt_dates_dmgs):

            dmg_count, dmg_value, affected_value, dmg_at_centroid = get_damaged_assets(
                damages,
                variable,
                date_now,
                exposure_type,
                get_PVA,
                variable_2=variable_2,
                fraction_insured=fraction_insured,
            )

            # get at centroid data
            at_centroid_df = at_centroid_from_exp_and_dmg(
                exp_at_centroid,
                variable,
                date_now,
                dmg_at_centroid=dmg_at_centroid,
                variable_2=variable_2,
                get_PVA=get_PVA,
            )

        else:  # if no damages are recorded add empty series for damage value/count

            dmg_count = pd.Series(dtype=int, name="count_dmg")
            dmg_value = pd.Series(dtype=int, name="value_dmg")
            affected_value = None

            # get at centroid data
            at_centroid_df = at_centroid_from_exp_and_dmg(
                exp_at_centroid,
                variable,
                date_now,
                dmg_at_centroid=None,
                variable_2=variable_2,
                get_PVA=get_PVA,
            )

            if get_PVA == True:
                affected_value = pd.Series(dtype=int, name="value_affected")
                # raise NotImplementedError('get_PVA with additional impact dates is not yet implemented. \
                #     All info is there->could be easily implemented')

        # concatenate at_centroid dataframe
        if "values_at_centroid_all" not in locals():
            values_at_centroid_all = at_centroid_df
        else:
            values_at_centroid_all = pd.concat(
                [values_at_centroid_all, at_centroid_df], ignore_index=True
            )

        # get number of grid cells with non-zero exposure per value of hazard variable
        vals, counts = np.unique(
            intensity[np.unique(exp_sel.gdf.centr_HL)], return_counts=True
        )
        cell_count = pd.Series(counts, index=vals).rename("count_cell")

        # create a pandas.DataFrame with all relevant data as function of hazard intensity
        df = pd.concat(
            [cell_count, all_count, all_value, dmg_count, dmg_value, affected_value],
            axis=1,
        )

        # add a label for reported damages per day, and store full info in xarray
        if df_all.empty:
            df_all = df.fillna(0).copy()
            ds = (
                df.reindex(intensity_range)
                .fillna(0)
                .to_xarray()
                .expand_dims(dim={"date": [date_now]})
            )
        else:
            df_all = df_all.add(df, fill_value=0)
            ds_temp = (
                df.reindex(intensity_range)
                .fillna(0)
                .to_xarray()
                .expand_dims(dim={"date": [date_now]})
            )
            ds = xr.concat([ds, ds_temp], dim="date")

    # rename the index of the output dataset
    ds = ds.rename({"index": variable})

    # SMOOTH DATA USING A RUNNING MEAN
    if cut_off_overwrite is None:
        cut_off = CUT_OFF_DICT[variable]
    else:
        cut_off = cut_off_overwrite

    ds_roll, ds_roll_cut = smooth_data_running_mean(
        ds, variable, intensity_range, cut_off, roll_window_size
    )

    return ds, df_all, ds_roll, ds_roll_cut, values_at_centroid_all, intensity_range


def compute_empirical_damage_functions(
    ds, ds_roll, ds_roll_cut, get_monotonic_fit=True
):
    """
    Parameters
    ----------
    ds: xarray.Dataset
        Dataset with relevant values grouped per MESHS value
    ds_roll: xarray.Dataset
        moving average of ds
    ds_roll_cut: xarray.Dataset
        moving average of ds but with a cutoff after MESHS=60

    Returns
    --------
    df: pandas.Dataframe
        Dataframe with values computed over all events
    df_roll: pandas.Dataframe
        moving average of df
    df_roll_cut: pandas.Dataframe
        moving average of ds but with acutoff
    n_dmgs: pandas.Dataframe
        number of damage claims per intensity value
    """

    if "value_affected" in ds.data_vars:
        use_PVA = True
    else:
        use_PVA = False

    # create dataframe including all events
    df = ds.sum(dim="date").to_dataframe()
    df_roll = ds_roll.sum(dim="date").to_dataframe()
    df_roll_cut = ds_roll_cut.sum(dim="date").to_dataframe()

    # compute measures relevant for impact function
    for dframe in [df_roll, df_roll_cut, df, ds, ds_roll, ds_roll_cut]:
        dframe["MDR"] = dframe.value_dmg / dframe.value_all
        dframe["PAA"] = (
            dframe.count_dmg / dframe.count_all
        )  # PAA: Percent Assets Affected
        if use_PVA:
            dframe["MDD"] = dframe.value_dmg / dframe.value_affected
            dframe["PVA"] = (
                dframe.value_affected / dframe.value_all
            )  # PVA: percent value affected
    n_dmgs = ds.sum(dim="date").count_dmg

    if get_monotonic_fit:
        df_roll_cut_fit = fit_monotonic(df_roll_cut)
        return df, df_roll, df_roll_cut, df_roll_cut_fit, n_dmgs
    else:
        return df, df_roll, df_roll_cut, n_dmgs


# %% Bootstrapping
def var_tuple(n_intensity, n_samples, variable):
    """get tuple for mapping used to initialized xarray.Dataset
    var name: (tuple of dimension names, array-like)

    Parameters
    ---------
    n_intensity: int
        number of intensity values / length of dimension "variable"
    n_samples: int
        number of bootstrap samples / length of dimension 'b_sample'

    Returns
    --------
    tuple of dimension names and array-like
    """

    nan_arr = np.full((n_intensity, n_samples), np.nan)

    return ([variable, "b_sample"], nan_arr)


def bootstrapping(
    ds,
    ds_roll,
    variable,
    n_samples,
    intensity_range,
    log_fit=False,
    cut_off=60,
    keep_raw_values=False,
    do_cross_validation=False,
    by_year=False,
):
    """
    Parameters
    ds: xarray.Dataset with date for each sample date
    ds_roll: ds but for rolling averages
    variable: hazard variable
    n_samples: number of bootstrap samples. If do_cross_validation=True,
                n_samples is the number of splits for cross validation
    cut_off: intensity value where function is cut off (default to 60 for MESHS=60mm)
    log_fit: boolean
        if True, fit a logistic function to the data
    keep_raw_values: boolean
        if True, also add raw values to output dataset
    do_cross_validation: boolean
        if True, do cross validation INSTEAD of bootstrapping
    n_splits: int
        number of splits for cross validation
    by_year: boolean
        if True, select cross validation splits by year

    Returns
    -------
    """
    # initialize bootstrap xarrays
    # https://elizavetalebedeva.com/bootstrapping-confidence-intervals-the-basics/

    # initialize bootstrap array with

    # get number of intensity values
    n_intensity = len(ds[variable])

    # initialize data variables
    data_vars = {
        "PAA": var_tuple(n_intensity, n_samples, variable),
        "MDR": var_tuple(n_intensity, n_samples, variable),
    }

    # determine if PVA should be calculated
    if "value_affected" in ds.data_vars:  # use PVA
        data_var_list = ["PAA", "MDD", "MDR", "PVA"]
        data_vars.update(
            {
                "MDD": var_tuple(n_intensity, n_samples, variable),
                "PVA": var_tuple(n_intensity, n_samples, variable),
            }
        )

    else:  # don't use PVA,MDD
        data_var_list = ["PAA", "MDR"]

    if keep_raw_values:
        # also add raw values to output dataset (e.g. count_dmg, value_all)
        data_var_list = data_var_list + [
            "count_cell",
            "count_all",
            "count_dmg",
            "value_all",
            "value_dmg",
            "value_affected",
        ]
        data_vars.update(
            {
                "count_cell": var_tuple(n_intensity, n_samples, variable),
                "count_all": var_tuple(n_intensity, n_samples, variable),
                "count_dmg": var_tuple(n_intensity, n_samples, variable),
                "value_all": var_tuple(n_intensity, n_samples, variable),
                "value_dmg": var_tuple(n_intensity, n_samples, variable),
                "value_affected": var_tuple(n_intensity, n_samples, variable),
            }
        )

    ds_boot = xr.Dataset(
        data_vars=data_vars,
        coords={variable: ds[variable], "b_sample": np.arange(n_samples)},
    )

    # bootstrap array for rolling means
    ds_boot_roll = ds_boot.copy(deep=True)
    # bootstrap array for  roll cut
    ds_boot_roll_cut = ds_boot.copy(deep=True)

    # logistic fit data for each sample
    fit_data = {
        var: pd.DataFrame(index=np.arange(n_samples), columns=["R2", "params"])
        for var in data_var_list
    }

    if do_cross_validation:
        assert n_samples <= 10  # cross validation only for n_samples<=10

        # Create n_samples equally sized CV_splits
        if by_year:
            unique_years = np.sort(np.unique(ds.date.dt.year))
            n_years = unique_years.size
            split_size = np.floor(n_years / n_samples)
            CV_split_year = np.repeat(np.arange(n_samples), split_size)

            # pad the last CV_split to use all datapoints
            pad_length = n_years - len(CV_split_year)
            # assert that the last split is not more than 30% larger than others
            assert pad_length / split_size < 0.3
            CV_split_year = np.pad(
                CV_split_year,
                (0, pad_length),
                mode="constant",
                constant_values=max(CV_split_year),
            )
            CV_split_dict = dict(zip(unique_years, CV_split_year))

            CV_split = np.vectorize(CV_split_dict.get)(ds.date.dt.year)
            dates_CV = pd.Series(data=CV_split, index=ds.date, name="CV_split")
        else:
            n_dates = len(ds.date)
            split_size = np.floor(n_dates / n_samples)
            CV_split = np.repeat(np.arange(n_samples), split_size)

            # pad the last CV_split to use all datapoints
            pad_length = n_dates - len(CV_split)
            CV_split = np.pad(
                CV_split,
                (0, pad_length),
                mode="constant",
                constant_values=max(CV_split),
            )
            dates_CV = pd.Series(data=CV_split, index=ds.date, name="CV_split")

    # bootstrapping n samples with replacement and with the size of the original dataset
    for num in np.arange(n_samples):

        # get random sample dates
        if do_cross_validation:

            # #get random sample dates
            # sel_samples = ds.date[num*split_size:(num+1)*split_size]
            # #get all dates except 1 split
            # sel_samples = np.setdiff1d(ds.date, sel_samples)
            sel_samples = dates_CV.index[dates_CV != num].values

        else:
            sel_samples = np.random.choice(ds.date, size=len(ds.date), replace=True)

        # compute sum values for sample
        ds_now = ds.sel(date=sel_samples).sum(dim="date")
        ds_roll_now = ds_roll.sel(date=sel_samples).sum(dim="date")
        ds_roll_now_cut = ds_roll_now.copy(deep=True)
        ds_roll_now_cut.loc[{variable: slice(cut_off, intensity_range.max())}] = (
            ds_roll_now_cut.loc[{variable: slice(cut_off, intensity_range.max())}].sum(
                dim=variable
            )
        )

        # Compute relevant metrics for this sample
        for dset in [ds_now, ds_roll_now, ds_roll_now_cut]:
            dset["MDR"] = dset.value_dmg / dset.value_all
            dset["PAA"] = dset.count_dmg / dset.count_all
            if "PVA" in data_var_list:
                dset["MDD"] = dset.value_dmg / dset.value_affected
                dset["PVA"] = dset.value_affected / dset.value_all

        # add sample metrics to output arrays
        ds_boot.loc[{"b_sample": num}] = ds_now[data_var_list]
        ds_boot_roll.loc[{"b_sample": num}] = ds_roll_now[data_var_list]
        ds_boot_roll_cut.loc[{"b_sample": num}] = ds_roll_now_cut[data_var_list]

        if log_fit == True:

            for var in data_var_list:
                xData = intensity_range[1:41]
                yData = ds_roll_now[var][1:41].values

                # initial parameter estimation
                p0 = [0.2, 0.075, 0.2, 48]

                try:
                    xrange, modelPredictions, Rsquared, parameters = fit_log_curve(
                        xData, yData, p0
                    )
                except:
                    Rsquared = None
                    parameters = None

                fit_data[var].loc[num, "R2"] = Rsquared
                fit_data[var].loc[num, "params"] = parameters
        else:
            fit_data = None

    # Rename datasets in case of cross validation
    if do_cross_validation:
        ds_boot = ds_boot.rename({"b_sample": "CV_split"})
        ds_boot_roll = ds_boot_roll.rename({"b_sample": "CV_split"})
        ds_boot_roll_cut = ds_boot_roll_cut.rename({"b_sample": "CV_split"})
        if by_year:
            return (
                ds_boot,
                ds_boot_roll,
                ds_boot_roll_cut,
                fit_data,
                dates_CV,
                pd.Series(CV_split_dict, name="CV_split"),
            )
        else:
            return ds_boot, ds_boot_roll, ds_boot_roll_cut, fit_data, dates_CV
    else:
        return ds_boot, ds_boot_roll, ds_boot_roll_cut, fit_data


def fit_monotonic(df):
    """fit monotonic function to variables in df (MDR, PAA, MDD, PVA) and
    return a new dataframe with fitted values


    Parameters
    ----------
    df : pandas.Dataframe
        dataframe with values of MDR, PAA, MDD, PVA that a monotonic fit needs
        to be applied to
    Returns
    -------
    df_monotonic : pandas.Dataframe
        dataframe with monotonic fot of values in df
    """

    var_list = ["MDR", "PAA", "MDD", "PVA"]
    data_out = {}
    # logistic fit data for each sample
    for var in var_list:

        if var in df.keys():
            if df.index[0] == 0:  # skip_zero
                assert np.where(df.index == 0) == np.array([0])
                x = df.index[1:]
                y = df[var][1:]
            else:
                x = df.index
                y = df[var]
            # avoid nan values
            no_nan = ~np.isnan(x) & ~np.isnan(y)
            x = x[no_nan]
            y = y[no_nan]

            monotone_fit = sc.both.smooth_monotonic(x, y, plot=False)

            data_out[var] = monotone_fit

    df_monotonic = pd.DataFrame(data_out, index=x)

    return df_monotonic


def extend_haz(haz, dates_to_extent):
    """extend hazard object with empty events for calibration

    Args:
        haz (climada.hazard): hazard object to extend
        dates_to_extent (array-like): dates to extend hazard object with
    """
    # check that none of the new dates already exists
    assert np.all(np.isin(dates_to_extent, haz.date) == False)
    add_intensity = sparse.csr_matrix(
        (len(dates_to_extent), haz.intensity.shape[1]), dtype=float
    )
    stacked_int = sparse.vstack([haz.intensity, add_intensity])

    dates_combined = np.concatenate([haz.date, dates_to_extent])
    haz2 = Hazard(
        haz_type=haz.haz_type,
        units=haz.units,
        centroids=haz.centroids,
        event_id=np.arange(1, len(dates_combined) + 1),
        frequency=np.full_like(dates_combined, haz.frequency[0], dtype=float),
        event_name=[
            dt.datetime.fromordinal(d).strftime("ev_%Y-%m-%d") for d in dates_combined
        ],
        date=dates_combined,
        orig=np.full_like(dates_combined, haz.orig[0], dtype=bool),
        intensity=stacked_int,
    )
    haz2.check()
    return haz2


# %% Plot functions


def plot_calibration_results(
    df,
    df_roll,
    ax,
    variable,
    df_monotonic=None,
    ds_boot=None,
    fit_data=None,
    n_dmgs=None,
    modelPredictions=None,
    xrange=None,
    Rsquared=None,
    ylim=None,
    measure="PAA",
    try_log_fit=True,
):

    matplotlib.rcParams.update({"font.size": 15})

    if measure == "PVA":
        label = "fraction of area affected"
    elif measure == "PAA":
        label = "fraction of fields affected"
    elif measure == "MDD":
        label = "mean harvest loss ratio at individual field"
    elif measure == "MDR":
        label = "mean harvest loss ratio at grid point"
    # plot means
    # df[measure].plot(color='red',label=label+' (mean)',alpha=0.5)

    # plot rolling average
    df_roll[measure].plot(color="green", linewidth=2, label=label + " (running mean)")

    if df_monotonic is None:
        pass
    else:
        # plot monotonic fit average
        df_monotonic[measure].plot(
            color="red",
            linewidth=2,
            linestyle="dashed",
            label=label + " (monotonic fit)",
        )

    if Rsquared:
        Rsquared = np.round(Rsquared, decimals=3)
        ax.plot(
            xrange,
            modelPredictions,
            color="orange",
            label="logistic fit (R$^2$ =" + str(Rsquared) + ")",
        )

    if fit_data is not None:
        for index in fit_data[measure].index:
            parameters = fit_data[measure].loc[index, "params"]
            if parameters is not None:
                y = log_func(xrange, *parameters)
                ax.plot(
                    xrange,
                    y,
                    color="dimgrey",
                    linestyle="dashed",
                    linewidth=0.4,
                    alpha=0.2,
                    zorder=0,
                )

    if ds_boot is not None:
        percentile05 = ds_boot[measure].quantile(q=0.05, dim="b_sample")
        percentile95 = ds_boot[measure].quantile(q=0.95, dim="b_sample")

        names = ["5th", "95th"]
        percentile05.plot(ax=ax, color="k", linewidth=1, label=names[0] + " percentile")
        percentile95.plot(ax=ax, color="k", linewidth=1, label=names[1] + " percentile")
        parameters_percentile = []
        # fit curve to percentiles
        p0 = [0.2, 0.075, 0.2, 48]

        for n, percentile in enumerate([percentile05, percentile95]):

            if try_log_fit == True:
                try:
                    # print(df.index.values[0:41],percentile.values[1:41])
                    xrange, modelPredictions, Rsquared, parameters = fit_log_curve(
                        df.index.values[1:41], percentile.values[1:41], p0
                    )
                except:
                    Rsquared = None

                # print(Rsquared)
                if Rsquared:
                    Rsquared = np.round(Rsquared, decimals=3)
                    ax.plot(
                        xrange,
                        modelPredictions,
                        color="k",
                        label=names[n]
                        + " perc. log. fit (R$^2$ ="
                        + str(Rsquared)
                        + ")",
                    )
                    parameters_percentile.append(parameters)

    if n_dmgs is not None:
        ax2 = ax.twinx()
        n_dmgs.plot(
            ax=ax2, label="number of damage claims", color="blue", zorder=0, alpha=0.5
        )
        ax2.set_ylim([0, 250])
        ax2.legend(loc="lower left")
        ax2.set_ylabel("number of damage claims")
        ax2.yaxis.label.set_color("blue")
        ax2.tick_params(axis="y", colors="blue")
        ax2.spines["right"].set_color("blue")

    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim([0, 0.25])

    if variable == "MESHS":
        ax.set_xlim([20, 120])
    elif variable == "HKE":
        ax.set_xlim([0, 3000])
    ax.set_ylabel(measure)
    ax.legend(loc="upper left")

    # shade part of plot not used for calib
    # ax.axvspan(61,81,0,1,color='dimgrey',alpha=0.2)

    return


###############################################################################
# Quantile match calibration
###############################################################################


def quantile_match_calib(
    haz,
    exp_in,
    imp_obs_rel,
    start_date,
    end_date,
    thresh_exp=0,
    metric="paa",
    date_mode="all_summers",
    n_members=1,
    bs_samples=None,
    method="sort",
    plot_level=1,
    save_path=None,
    bias_weight=0,
    verbose=True,
):
    """Calibrate an impact function by maching quantiles of hazard and
       RELATIVE impact (i.e. PAA or MDR)

    Args:
        haz (climada.Hazard or callable): hazard data or callable to get hazard file paths per date
        exp_in (climada.Exposure): exposure data
        imp_obs_rel (climada.engine.Impact): observed (relative) impact
            (i.e. PAA or MDR). Must have same spatial coords as exp_in
        start_date (str): start date in format 'YYYY-MM-DD'
        end_date (str): end date in format 'YYYY-MM-DD'
        thresh_exp (float, optional): Threshold for minimum exposure value to
            be considered. Defaults to 0.
        metric (str, optional): relative damage metric. Defaults to 'paa'.
        date_mode (str, optional): Date selection. Defaults to 'all_summers'.
        n_members (int, optional): number of ensemble members. Defaults to 1.
        bs_samples (int, optional): number of bootstrap samples. Defaults to None.
        method (str, optional): calculation method. Defaults to 'sort'.
        plot_level (int, optional): level of plotting. Defaults to 1.
        save_path (str, optional): path to save results. Defaults to None.
        bias_weight (int, optional): weight of bias in calibration (compared to MSE). Defaults to 0.
        verbose (bool, optional): print outputs. Defaults to True.

    Returns:
        tuple: tuple of impact function parameters, flexible impact function
            parameters, figure, impact
    """

    exp = copy.deepcopy(exp_in)

    if not method == "sort":
        raise NotImplementedError(
            f"method={method} not implemented. Use 'sort', which only works"
            f"with the same spatial and temporal extent in hazard and imp_obs."
            f"See subproj_A\impfct_hailcast\match_quantiles_hail_dmg_buildings_season_centroids.ipynb"
            f"for details on the 'quant' method"
        )

    # get a list of all dates
    if not date_mode == "all_summers":
        raise NotImplementedError(f"date_mode={date_mode} not implemented")

    date_list = pd.date_range(start_date, end_date, freq="D")
    date_list = date_list[(date_list.month >= 4) & (date_list.month <= 9)]
    n_days = len(date_list)

    # Select only impact between start and end date
    if dt.datetime.fromisoformat(start_date) > dt.datetime.fromordinal(
        imp_obs_rel.date[0]
    ) or dt.datetime.fromisoformat(end_date) < dt.datetime.fromordinal(
        imp_obs_rel.date[-1]
    ):
        print(
            "Not all impact dates are within the selected date range. Selecting subset of impact dates."
        )
        imp_obs_rel = imp_obs_rel.select(
            dates=(
                dt.datetime.fromisoformat(start_date).toordinal(),
                dt.datetime.fromisoformat(end_date).toordinal(),
            )
        )

    months = np.array([dt.datetime.fromordinal(d).month for d in imp_obs_rel.date])
    if any(months < 4) or any(months > 9):
        raise ValueError("Some impact dates are not within the summer season.")

    # assert consistent coordinates for exposure and impact
    np.testing.assert_array_almost_equal(
        np.array([[coord.y, coord.x] for coord in exp.gdf.geometry.values]), imp_obs_rel.coord_exp
    )

    # select subset of centroids, which fullfill the min_exposure criterion
    if verbose: print(f"Centroids before subsetting: {exp.gdf.shape[0]}")
    condition = exp.gdf.value > thresh_exp
    exp.set_gdf(exp.gdf[condition])
    if verbose: print(f"Centroids after subsetting:   {exp.gdf.shape[0]}")

    # select a subset of centroids in impact
    imp_sel = imp_obs_rel.imp_mat[:, condition.values].toarray()
    if verbose: print(f"Impact after subsetting  (events, centroids):  {imp_sel.shape}")

    # get dates of events with reported damages
    date_events = [dt.date.fromordinal(date) for date in imp_obs_rel.date]
    n_events = len(date_events)
    if verbose: print(f"Number of events with reported impacts: {n_events}")

    # create imp_filled with dimensions (dates_in_period, selected_centroids) filled with zeros
    imp_filled = np.zeros((n_days, imp_sel.shape[1]))
    date_idx = [date_list.to_list().index(pd.Timestamp(date)) for date in date_events]
    # alternative with same results : [np.where(date_list==pd.Timestamp(date))[0][0] for date in date_events]
    imp_filled[date_idx, :] = imp_sel
    imp_sel = imp_filled  # overwrite imp_sel with imp_filled


    n_events = len(date_events)
    if verbose:
        print(f"\nImpact after filling dates with no damages: {imp_sel.shape}")
        print(f"{n_events} events in the period: {date_events[0]} - {date_events[-1]}")
        print(f"{n_days - n_events} of {n_days} days without damage")

    # get hazard intensity for all dates
    if callable(haz):
        hazard, dates_missing = get_hazard_from_files(
            haz, date_list, plot_level=plot_level
        )
        # Remove impact dates not in hazard data
        if len(dates_missing) > 0:
            print(
                f"\nRemove {len(dates_missing)} dates from imp_sel which are missing in hazard dataset."
            )
            idx_missing = [date_list.to_list().index(date) for date in dates_missing]
            print(f"Shape of imp_sel with all dates:             {imp_sel.shape}")
            imp_sel = np.delete(imp_sel, idx_missing, axis=0)
            print(f"Shape of imp_sel with missing dates removed: {imp_sel.shape}")

    else:
        hazard = haz

    # get corresponding hazard centroids
    exp.assign_centroids(hazard)
    haz_sel = hazard.intensity[:, exp.gdf.centr_HL].toarray()
    haz_sel_coord = hazard.centroids.coord[exp.gdf.centr_HL, :]

    # assert consistent coordinates for exposure and hazard (closest match)
    np.testing.assert_array_almost_equal(
        np.array([[coord.y, coord.x] for coord in exp.gdf.geometry.values]), haz_sel_coord, decimal=1
    )

    if plot_level >= 2:
        # plot selected hazard data
        fig, ax = plt.subplots()
        c1 = ax.scatter(
            x=haz_sel_coord[:, 1],
            y=haz_sel_coord[:, 0],
            c=np.max(haz_sel, axis=0),
            vmin=0,
            vmax=45,
            s=10,
        )
        plt.colorbar(c1)
        ax.set(title="Selected hazard data (max value per centroid)")

    if n_members > 1:

        haz_sel_orig = copy.deepcopy(haz_sel)

        haz_sel = haz_sel.reshape((imp_sel.shape[0], n_members, imp_sel.shape[1]))
        # re-order dimensions
        haz_sel = np.moveaxis(haz_sel, 1, 2)
        # Version1:  haz_sel = haz_sel.reshape((imp_sel.shape[0],imp_sel.shape[1],n_members))

        # check if hazard and impact have the same number of timesteps and centroids
        assert haz_sel.shape[:2] == imp_sel.shape

        # DEBUG: Check that the data is re-shaped into the correct dimensions
        n_time = 0
        for i in range(len(hazard.event_id)):
            if np.mod(i, n_members) == 0 and i > 0:
                n_time += 1
            ens_member = i - n_members * n_time

            # print(f"time: {n_time},ens member: {ens_member}")
            # print(hazard.event_name[i])
            # if not haz_test[n_time,:,ens_member].max() == 0:
            #     print(haz_sel_orig[i,:].max())

            np.testing.assert_array_equal(
                haz_sel_orig[i, :], haz_sel[n_time, :, ens_member]
            )

    else:
        # check if hazard and impact have the same number of timesteps and centroids
        assert haz_sel.shape == imp_sel.shape

    # flatten and sort the two arrays
    haz_vals_sorted, imp_vals_sorted = get_sorted_data(haz_sel, imp_sel, n_members)

    # get impact functions from quantile matching
    params, params_flex, impf_em, impf_em_flex, df_impf, x_max_obs = get_impfs(
        haz_vals_sorted, imp_vals_sorted, metric, min_v_thresh="min_haz",
        verbose=verbose, bias_weight=bias_weight,
    )

    # get impact function with flexible lower threshold
    _, params_flex_thresh, _, impf_em_flex_thresh, _, _ = get_impfs(
        haz_vals_sorted, imp_vals_sorted, metric, min_v_thresh="none",
        verbose=verbose, bias_weight=bias_weight,
    )

    if save_path:
        for fp, p in zip(
            [
                f"{save_path}.json",
                f"{save_path}_flex.json",
                f"{save_path}_flex_thresh.json",
            ],
            [params, params_flex, params_flex_thresh],
        ):
            save_params(fp, p)

    # Plot the resulting impact functions
    if plot_level >= 1:
        fig = plot_quant_match_impf(
            haz_vals_sorted,
            imp_vals_sorted,
            impf_em,
            impf_em_flex,
            df_impf,
            params_flex,
            thresh_exp,
            date_mode,
            start_date,
            end_date,
            x_max_obs,
            metric,
        )
        ax = fig.get_axes()[0]
    else:
        fig = None

    # get bootstrapped hazard and impact values
    if bs_samples is not None:

        bootstrapped_haz_sorted, bootstrapped_imp_sorted, haz_counts = (
            get_bootstrapped_haz_imp_vals(
                haz_sel,
                imp_sel,
                bs_samples,
                n_members,
                return_full_array=False,
                haz_decimal=1,
            )
        )
        if save_path:
            np.savez(
                f"{save_path}_bs.npz",
                bs_haz=bootstrapped_haz_sorted,
                bs_imp=bootstrapped_imp_sorted,
            )


        # get quantile of bootstrap sample
        q_arrs = []
        for quantile, scale_bounds in zip(
            [0.05, 0.95], [[0, params_flex["scale"]], [params_flex["scale"], 1]]
        ):
            # Note: scale_bound are ignored for now

            q_arr = np.nanquantile(
                sc.both.numpy_ffill(bootstrapped_imp_sorted), quantile, axis=0
            )
            (
                paramsBS,
                params_flexBS,
                impf_emBS,
                impf_em_flexBS,
                df_impfBS,
                x_max_obsBS,
            ) = get_impfs(
                bootstrapped_haz_sorted[0, :],
                q_arr,
                metric,
                min_v_thresh="min_haz",
                value_counts=haz_counts,
                verbose=False,
                scale_bounds=scale_bounds,#None,
                bias_weight=bias_weight,
            )
            print(f"Q {quantile*100:.0f}%: {params_flexBS}")
            # ax.plot(bootstrapped_haz_sorted[0,:] ,q_arr,color='black',alpha=0.9)
            if plot_level >= 1:
                ax.plot(
                    impf_em_flexBS.intensity,
                    impf_em_flexBS.mdd*100,
                    color="lightblue",
                    label=f"Emanuel-type Q{quantile}  power={params_flexBS['power']:.1f} (opt)",
                )
            q_arrs.append(q_arr)
            if save_path:
                save_params(f"{save_path}_Q{quantile:.2f}_flex.json", params_flexBS)

        if plot_level >= 1:
            ax.fill_between(
                bootstrapped_haz_sorted[0, :],
                q_arrs[0]*100,
                q_arrs[1]*100,
                color="grey",
                alpha=0.2,
                label="5-95% quantile",
            )

            for i in range(bs_samples):
                not_nan = ~np.isnan(bootstrapped_imp_sorted[i, :])
                ax.plot(
                    bootstrapped_haz_sorted[i, :][not_nan],
                    bootstrapped_imp_sorted[i, :][not_nan]*100,
                    linewidth=1,
                    linestyle="-",
                    color="grey",
                    alpha=0.1,
                )

            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

            # Debug: Check the spatial distribution of bootstrapped samples to make sure they are correct
            if plot_level >= 4:
                plot_spatial_distribution_of_bs_samples(haz_sel,imp_sel,bs_samples,n_members,haz_sel_coord,ax)

        return (
            params,
            params_flex,
            fig,
            haz_vals_sorted,
            bootstrapped_haz_sorted,
            bootstrapped_imp_sorted,
            haz_counts,
        )



    return params, params_flex, fig, haz_vals_sorted


def plot_spatial_distribution_of_bs_samples(haz_sel,imp_sel,bs_samples,n_members,
                                            haz_sel_coord,ax):
    """Function for debugging only. Plot the spatial distribution of
    bootstrapped samples to make sure they are correct.
    Only works for forecast data with n_members > 1"""
    bootstrapped_haz, bootstrapped_imp = get_bootstrapped_haz_imp_vals(
        haz_sel, imp_sel, bs_samples, n_members, return_full_array=True
    )
    for i in range(bs_samples):
        haz_vals_bs, imp_vals_bs = get_sorted_data(
            bootstrapped_haz[i], bootstrapped_imp[i], n_members
        )
        ax.plot(
            haz_vals_bs,
            imp_vals_bs*100,
            linewidth=1,
            linestyle="-",
            color="salmon",
            alpha=0.1,
        )

    i, j = np.where(
        bootstrapped_haz[0, :].sum(axis=1)
        > np.quantile(bootstrapped_haz[0, :].sum(axis=1), 0.9)
    )
    for count in range(min(20, len(i))):
        i_now = i[count]
        j_now = j[count]
        haz_now = bootstrapped_haz[0, i_now, :, j_now]
        # plot selected hazard data
        # c1 = plt.tricontourf(haz_sel_coord[:,1], haz_sel_coord[:,0], np.max(haz_sel,axis=0), levels=50)
        fig, ax = plt.subplots()
        c1 = ax.scatter(
            x=haz_sel_coord[:, 1],
            y=haz_sel_coord[:, 0],
            c=haz_now,
            vmin=0,
            vmax=45,
            s=5,
        )
        plt.colorbar(c1)
        ax.set(title="Selected hazard data (max value per centroid)")

############# Helper functions for quantile match calibration ################
def save_params(path, params):
    with open(path, "w") as file:
        json.dump(params, file)


def get_sorted_data(
    haz_sel,
    imp_sel,
    n_members=1,
    subsampling=False,
    subsample_decimals=1,
    haz_vals_subsampled=None,
):
    """Get sorted hazard and impact data. If n_members > 1, subsample
    hazard data to quantile match centroids correctly

    Args:
        haz_sel (np.array): selected hazard values
        imp_sel (np.array): selected impact values
        n_members (int, optional): number of ensemble members. Defaults to 1.
        subsampling (bool, optional): subsample data to hazard values with given
          precision (reduces memory usage). Defaults to False.
        subsample_decimals (int, optional): number of decimals for subsampling.
            Defaults to 1.
        haz_vals_subsampled (np.array, optional): subsampled hazard values

    Returns:
        tuple: tuple of sorted hazard and impact data
    """

    # flatten and sort the two arrays
    imp_vals_sorted = np.sort(imp_sel.flatten())
    haz_vals_sorted = np.sort(haz_sel.flatten())

    if n_members > 1:
        # subsample hazard data to quantile match centroids correctly
        # select every n-th value of haz_vals_sorted
        # start at n_members/2 to get the middle value of each "n_memeber group"
        haz_vals_sorted = haz_vals_sorted[int(n_members / 2) :: n_members]

    if subsampling:
        haz_vals_sorted = haz_vals_sorted.round(decimals=subsample_decimals)
        if haz_vals_subsampled is None:
            # Subsample data so that only one value per 0.1 of hazard intensity is used
            haz_vals_subsampled = np.unique(haz_vals_sorted)
        else:
            # assert that the precision of the given subsampled values is correct, and that they are sorted
            assert np.all(
                np.round(np.sort(haz_vals_subsampled), decimals=subsample_decimals)
                == haz_vals_subsampled
            )

        imp_vals_subsampled = np.array(
            [
                np.mean(imp_vals_sorted[haz_vals_sorted == haz])
                for haz in haz_vals_subsampled
            ]
        )

        assert len(haz_vals_subsampled) == len(imp_vals_subsampled)
        return haz_vals_subsampled, imp_vals_subsampled

    else:
        assert len(haz_vals_sorted) == len(imp_vals_sorted)
        return haz_vals_sorted, imp_vals_sorted


def get_bootstrapped_haz_imp_vals(
    haz_sel, imp_sel, n_samples, n_members=1, return_full_array=False, haz_decimal=1
):
    """Get bootstrapped hazard and impact values

    Args:
        haz_sel (np.array): selected hazard values
        imp_sel (np.array): selected impact values
        n_samples (int): number of bootstrap samples
        return_full_array (bool, optional): return full array of bootstrapped
            values (needs more memory; for debugging mainly). Defaults to False.
        haz_decimal (int, optional): number of decimals for unique hazard values. Defaults to 1.

    Returns:
        tuple: tuple of bootstrapped hazard and impact values
    """

    # Set random seed for reproducibility
    rg = np.random.default_rng(seed=123)

    if return_full_array:
        # get bootstrapped hazard and impact values (without ensemble memebrs)
        if n_members == 1:  # and haz_sel.ndim == 2 and imp_sel.ndim == 2:
            bootstrapped_haz = np.zeros((n_samples, haz_sel.shape[0], haz_sel.shape[1]))
            bootstrapped_imp = np.zeros((n_samples, imp_sel.shape[0], imp_sel.shape[1]))

            for i in range(n_samples):
                if i == 0:
                    #select original data
                    sel_sample_dates = np.arange(haz_sel.shape[0])
                else:
                    sel_sample_dates = rg.choice(
                        haz_sel.shape[0], size=haz_sel.shape[0], replace=True
                    )
                bootstrapped_haz[i, :, :] = haz_sel[sel_sample_dates, :]

                bootstrapped_imp[i, :, :] = imp_sel[sel_sample_dates, :]

        # get bootstrapped hazard and impact values (with ensemble members)
        elif haz_sel.ndim == 3 and imp_sel.ndim == 2:
            # hazard dims: (n_events, n_centroids, n_members)
            # impact dims: (n_events, n_centroids)
            bootstrapped_haz = np.zeros(
                (n_samples, haz_sel.shape[0], haz_sel.shape[1], haz_sel.shape[2])
            )
            bootstrapped_imp = np.zeros((n_samples, imp_sel.shape[0], imp_sel.shape[1]))

            for i in range(n_samples):
                if i == 0:
                    #select original data
                    sel_sample_dates = np.arange(haz_sel.shape[0])
                else:
                    sel_sample_dates = rg.choice(
                        haz_sel.shape[0], size=haz_sel.shape[0], replace=True
                    )
                bootstrapped_haz[i, :, :, :] = haz_sel[sel_sample_dates, :, :]

                bootstrapped_imp[i, :, :] = imp_sel[sel_sample_dates, :]

        return bootstrapped_haz, bootstrapped_imp

    else:
        # unique hazard values with a precision of 0.1 (1 decimal)
        haz_vals = np.round(haz_sel, decimals=haz_decimal)
        unique_haz_vals, haz_counts = np.unique(haz_vals, return_counts=True)

        # count occurences of each unique hazard value

        # get bootstrapped hazard and impact values (without ensemble memebrs)
        bootstrapped_haz = np.zeros((n_samples, len(unique_haz_vals)))
        bootstrapped_imp = np.zeros((n_samples, len(unique_haz_vals)))

        for i in range(n_samples):
            if i == 0:
                sel_sample_dates = np.arange(haz_sel.shape[0])
            else:
                sel_sample_dates = rg.choice(
                    haz_sel.shape[0], size=haz_sel.shape[0], replace=True
                )
            haz_vals_sorted, imp_vals_sorted = get_sorted_data(
                haz_sel[sel_sample_dates, :],
                imp_sel[sel_sample_dates, :],
                n_members=n_members,
                subsampling=True,
                subsample_decimals=haz_decimal,
                haz_vals_subsampled=unique_haz_vals,
            )
            bootstrapped_haz[i, :] = haz_vals_sorted
            bootstrapped_imp[i, :] = imp_vals_sorted

        return bootstrapped_haz, bootstrapped_imp, haz_counts


def get_impfs(
    haz_vals_sorted,
    imp_vals_sorted,
    metric,
    min_v_thresh="none",
    value_counts=None,
    scale_bounds=None,
    verbose=False,
    bias_weight=0,
):
    """Get impact functions from quantile matching

    Args:
        haz_vals_sorted (np.array): sorted hazard values
        imp_vals_sorted (np.array): sorted RELATIVE impact values (PAA or MDR)
        metric (str): relative damage metric
        min_v_thresh (str, optional): minimum value for v_thresh. Defaults to 'none'.
            if 'min_haz' use minimum hazard value with nonzero impact
        value_counts (np.array, optional): counts of unique hazard values. Defaults to None.
        scale_bounds (tuple, optional): bounds for scale parameter. Defaults to None.
        verbose (bool, optional): print verbose output. Defaults to False.
        bias_weight (float, optional): weight for bias in comparision to MSE,
                                        when optimizing the impact function. Defaults to 0.

    Returns:
        tuple: tuple of impact function parameters, flexible impact function
            parameters, figure, hazard values
    """

    # create dataframe for fit_emanuel_to_impf function
    both_zero = (imp_vals_sorted == 0) & (haz_vals_sorted == 0)
    df_impf = pd.DataFrame(
        {"intensity": haz_vals_sorted, metric.upper(): imp_vals_sorted}
    ).set_index("intensity")

    if value_counts is None:
        df_impf["count_cell"] = 1  # give equal weight (=1) to each row
    else:
        df_impf["count_cell"] = value_counts

    opt_var = metric.upper()  # PAA or MDR
    df_impf = df_impf[~both_zero]  # remove double zeros for faster calculation

    # edit dataframe to have one value per FULL INTEGER value of hazard intensity
    column_mapping = {col: "mean" for col in df_impf.columns}
    column_mapping["count_cell"] = "sum"
    df_impf["haz_rounded"] = df_impf.index.values.round().astype(int)
    df_1steps = df_impf.groupby("haz_rounded").agg(column_mapping)

    # Define parameter bounds for Emanuel-type impact function
    x_max_obs = int(df_1steps.index.max())
    if min_v_thresh == "min_haz":
        v_thresh = haz_vals_sorted[imp_vals_sorted > 0].min()
    elif min_v_thresh == "none":
        v_thresh = 0
    y_bounds = [
        v_thresh * 1.01,
        x_max_obs,
    ]  # add 1% to avoid v_tresh==v_half in starting array
    v_tresh_bounds = [v_thresh, x_max_obs]

    if scale_bounds is None:
        scale_bounds = [imp_vals_sorted.max() / 10, min(1, imp_vals_sorted.max() * 10)]

    pbounds = {
        "v_thresh": v_tresh_bounds,
        "v_half": y_bounds,
        "scale": scale_bounds,
        "power": (3, 3),
    }

    param_start = {
        "v_thresh": v_thresh,
        "v_half": (x_max_obs - v_thresh) / 2 + v_thresh,
        "scale": min(1, imp_vals_sorted.max()),
        "power": 3,
    }

    # fit Emanuel-type sigmoid function to impact function
    params, res, impf_emanuel = sc.calib_opt.fit_emanuel_impf_to_emp_data(
        df_1steps, #alternative: df_impf
        pbounds,
        opt_var,
        plot=False,
        max_iter=1000,
        verbose=verbose,
        param_start=list(param_start.values()),
        bias_weight=bias_weight,
    )

    impf_em = sc.calib_opt.get_emanuel_impf(
        **params, impf_id=1, intensity=np.arange(0, x_max_obs, 1)
    )

    pbounds["power"] = (1, 20)  # flexible power law in Emanuel-type function
    param_start["power"] = 5
    params_flex, res2, impf_emanuel_flex = sc.calib_opt.fit_emanuel_impf_to_emp_data(
        df_1steps, #alternative: df_impf
        pbounds,
        opt_var,
        plot=False,
        max_iter=1000,
        verbose=verbose,
        param_start=list(param_start.values()),
        bias_weight=bias_weight,
    )

    impf_em_flex = sc.calib_opt.get_emanuel_impf(
        **params_flex, impf_id=1, intensity=np.arange(0, x_max_obs, 1)
    )

    return params, params_flex, impf_em, impf_em_flex, df_impf, x_max_obs


def get_hazard_from_files(haz, date_list, plot_level=1):
    """Get hazard intensity for all dates in date_list

    Args:
        haz (callable): callable to get hazard file paths per date
        date_list (pandas.DatetimeIndex): list of dates
        plot_level (int, optional): level of plotting. Defaults to 1.

    Returns:
        climada.hazard: hazard object
    """
    f_haz = []
    dates_missing = []
    for date in date_list:
        file_path = haz(date)
        # check if file exists
        if os.path.exists(file_path):
            # if the file exists, write the path to the f_haz list
            f_haz.append(file_path)
        else:
            # if the file does not exist, write the date to the f_haz_missing list
            dates_missing.append(date)
    print(f"Hazard files available: {len(f_haz)}")
    print(f"Hazard files missing:   {len(dates_missing)}")

    # load data
    hazard = sc.hazard_from_radar(
        f_haz,
        varname="DHAIL_MX",
        time_dim="time",
        ensemble_dim="epsd_1",
        spatial_dims=["x_1", "x_2"],
    )
    print(
        f"Hazard dimensions (n_time*n_members, n_centroids): {hazard.intensity.shape}"
    )
    if plot_level >= 2:
        hazard.plot_intensity(0, vmin=0, vmax=45)

    return hazard, dates_missing


############# Plotting functions for quantile match calibration ###############
def plot_quant_match_impf(
    haz_vals_sorted,
    imp_vals_sorted,
    impf_em,
    impf_em_flex,
    df_impf,
    p2,
    thresh_exp,
    date_mode,
    start_date,
    end_date,
    x_max_obs,
    metric="paa",
):
    """Plot impact function from quantile matching

    Args:
        haz_vals_sorted (np.array): sorted hazard values
        imp_vals_sorted (np.array): sorted RELATIVE impact values (PAA or MDR)
        impf_em (climada.entity.Impf): Optimized impact function with fixed power=3
        impf_em_flex (climada.entity.Impf): Optimized impact function with flexible power (1 to 20)
        df_impf (pd.DataFrame): calibration data
        p2 (dict): parameters of flexible Emanuel-type impact function
        thresh_exp (float): Min. considered exposure value
        date_mode (str): Date selection mode. Defaults to 'all_summers'
        start_date (str): start date
        end_date (str): End date
        x_max_obs (float): Maximum observed relative impact (PAA or MDR)
        metric (str, optional): relative damage metric. Defaults to 'paa'.

    Returns:
        plt.Figure: Plot of impact function
    """

    if metric == "paa":
        ylabel = "Percent of Assets Affected [%]"
    elif metric == "mdr":
        ylabel = "Mean Damage Ratio [%]"

    fig, ax = plt.subplots()

    ax.plot(
        haz_vals_sorted,
        imp_vals_sorted*100,
        linewidth=2,
        linestyle="-",
        color="k",
        label="member mean",
    )
    ax.plot(
        impf_em.intensity, impf_em.mdd*100, color="lightgreen", label="Emanuel-type (opt)"
    )
    ax.plot(
        impf_em_flex.intensity,
        impf_em_flex.mdd*100,
        color="tab:blue",
        label=f"Emanuel-type  power={p2['power']:.1f} (opt)",
    )
    text_size = f"No. of non-zero cells = {int(imp_vals_sorted.shape[0])}"

    ax.set(xlim=(0, x_max_obs * 1.05), xlabel="HAILCAST DHAIL$_\mathrm{max}$ [mm]",
           ylabel=ylabel)
    text = (
        f"centroids with exposure > {thresh_exp}\n{date_mode} "
        f"from {start_date} to {end_date}\n\n"
    )

    ax.text(0.02, 0.98, text + text_size, va="top", ha="left", transform=ax.transAxes)

    # add legend below the figure
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    ax2 = ax.twinx()
    _ = ax2.hist(df_impf.index, bins=40, alpha=0.5, color="gray", log=True)
    ax2.set(ylabel="Number of cells")
    return fig
