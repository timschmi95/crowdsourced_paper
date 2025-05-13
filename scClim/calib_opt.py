"""
Impact function calibration functionalities:
    Optimization and manual calibration

based in climada.engine.calib_opt.py and extended by Timo to include
bayesian optimization
"""

import copy
import numpy as np
import pandas as pd
import warnings
import sys
from scipy import interpolate
from scipy.optimize import minimize
from itertools import combinations
import matplotlib.pyplot as plt

from climada.entity import ImpfTropCyclone, ImpactFunc
from climada import CONFIG
sys.path.append(str(CONFIG.local_data.func_dir))
import scClim as sc

try:
    from bayes_opt import BayesianOptimization
except:
    pass


def init_impf(impf_name_or_instance, param_dict,intensity_range, df_out=pd.DataFrame(index=[0])):
    """create an ImpactFunc based on the parameters in param_dict using the
    method specified in impf_parameterisation_name and document it in df_out.

    Parameters
    ----------
    impf_name_or_instance : str or ImpactFunc
        method of impact function parameterisation e.g. 'emanuel' or an
        instance of ImpactFunc
    param_dict : dict, optional
        dict of parameter_names and values
        e.g. {'v_thresh': 25.7, 'v_half': 70, 'scale': 1}
        or {'mdd_shift': 1.05, 'mdd_scale': 0.8, 'paa_shift': 1, paa_scale': 1}
    intensity_range : array
        tuple of 3 intensity numbers along np.arange(min, max, step)
    Returns
    -------
    imp_fun : ImpactFunc
        The Impact function based on the parameterisation
    df_out : DataFrame
        Output DataFrame with headers of columns defined and with first row
        (index=0) defined with values. The impact function parameters from
        param_dict are represented here.
    """
    impact_func_final = None
    if isinstance(impf_name_or_instance, str):
        if impf_name_or_instance == 'emanuel':
            impact_func_final = ImpfTropCyclone.from_emanuel_usa(**param_dict)
            impact_func_final.haz_type = 'TC'
            impact_func_final.id = 1
            df_out['impact_function'] = impf_name_or_instance
        if impf_name_or_instance == 'emanuel_HL':
            impact_func_final = get_emanuel_impf(
                **param_dict,intensity=intensity_range,haz_type='HL')
            df_out['impact_function'] = impf_name_or_instance
        elif impf_name_or_instance == 'sigmoid_HL':
            assert('L' in param_dict.keys() and 'k' in param_dict.keys() and 'x0' in param_dict.keys())
            impact_func_final = ImpactFunc.from_sigmoid_impf(
                **param_dict,intensity=intensity_range)#,haz_type='HL')
            impact_func_final.haz_type = 'HL'
            if intensity_range[0]==0 and not impact_func_final.mdd[0]==0:
                warnings.warn('sigmoid impact function has non-zero impact at intensity 0. Setting impact to 0.')
                impact_func_final.mdd[0]=0
            df_out['impact_function'] = impf_name_or_instance

    elif isinstance(impf_name_or_instance, ImpactFunc):
        impact_func_final = change_impf(impf_name_or_instance, param_dict)
        df_out['impact_function'] = ('given_' +
                                     impact_func_final.haz_type +
                                     str(impact_func_final.id))
    for key, val in param_dict.items():
        df_out[key] = val
    return impact_func_final, df_out

def change_impf(impf_instance, param_dict):
    """apply a shifting or a scaling defined in param_dict to the impact
    function in impf_istance and return it as a new ImpactFunc object.

    Parameters
    ----------
    impf_instance : ImpactFunc
        an instance of ImpactFunc
    param_dict : dict
        dict of parameter_names and values (interpreted as
        factors, 1 = neutral)
        e.g. {'mdd_shift': 1.05, 'mdd_scale': 0.8,
        'paa_shift': 1, paa_scale': 1}

    Returns
    -------
    ImpactFunc : The Impact function based on the parameterisation
    """
    ImpactFunc_new = copy.deepcopy(impf_instance)
    # create higher resolution impact functions (intensity, mdd ,paa)
    paa_func = interpolate.interp1d(ImpactFunc_new.intensity,
                                    ImpactFunc_new.paa,
                                    fill_value='extrapolate')
    mdd_func = interpolate.interp1d(ImpactFunc_new.intensity,
                                    ImpactFunc_new.mdd,
                                    fill_value='extrapolate')
    temp_dict = dict()
    temp_dict['paa_intensity_ext'] = np.linspace(ImpactFunc_new.intensity.min(),
                                                 ImpactFunc_new.intensity.max(),
                                                 (ImpactFunc_new.intensity.shape[0] + 1) * 10 + 1)
    temp_dict['mdd_intensity_ext'] = np.linspace(ImpactFunc_new.intensity.min(),
                                                 ImpactFunc_new.intensity.max(),
                                                 (ImpactFunc_new.intensity.shape[0] + 1) * 10 + 1)
    temp_dict['paa_ext'] = paa_func(temp_dict['paa_intensity_ext'])
    temp_dict['mdd_ext'] = mdd_func(temp_dict['mdd_intensity_ext'])
    # apply changes given in param_dict
    for key, val in param_dict.items():
        field_key, action = key.split('_')
        if action == 'shift':
            shift_absolut = (
                ImpactFunc_new.intensity[np.nonzero(getattr(ImpactFunc_new, field_key))[0][0]]
                * (val - 1))
            temp_dict[field_key + '_intensity_ext'] = \
                temp_dict[field_key + '_intensity_ext'] + shift_absolut
        elif action == 'scale':
            temp_dict[field_key + '_ext'] = \
                    np.clip(temp_dict[field_key + '_ext'] * val,
                            a_min=0,
                            a_max=1)
        else:
            raise AttributeError('keys in param_dict not recognized. Use only:'
                                 'paa_shift, paa_scale, mdd_shift, mdd_scale')

    # map changed, high resolution impact functions back to initial resolution
    ImpactFunc_new.intensity = np.linspace(ImpactFunc_new.intensity.min(),
                                           ImpactFunc_new.intensity.max(),
                                           (ImpactFunc_new.intensity.shape[0] + 1) * 10 + 1)
    paa_func_new = interpolate.interp1d(temp_dict['paa_intensity_ext'],
                                        temp_dict['paa_ext'],
                                        fill_value='extrapolate')
    mdd_func_new = interpolate.interp1d(temp_dict['mdd_intensity_ext'],
                                        temp_dict['mdd_ext'],
                                        fill_value='extrapolate')
    ImpactFunc_new.paa = paa_func_new(ImpactFunc_new.intensity)
    ImpactFunc_new.mdd = mdd_func_new(ImpactFunc_new.intensity)
    return ImpactFunc_new


########################################################################################
#non calibration functions
def plot_impf(param_dict_result,optimizer,impf_name_or_instance,intensity_range,haz=None,err_name=''):
    """plot the best impact function and the target function (over time)"""

    target = np.array([i['target'] for i in optimizer.res])
    error = 1/target

    #plot target function (over time)
    fig,ax = plt.subplots()
    ax.plot(error,'o',label=f'error: {err_name}')
    ax.set(yscale = 'log',xlabel='iteration',ylabel=f'error: {err_name}')
    ax.scatter(np.argmin(error),min(error),color='tab:red',zorder=4,label=f'best fit')
    ax.legend()
    #plot best impact function
    fig,ax = plt.subplots()
    title = [f'{key}: {param_dict_result[key]:.2f}' if param_dict_result[key]>0.1 else
             f'{key}: {param_dict_result[key]:.2e}' for key in param_dict_result.keys()]
    impf = init_impf(impf_name_or_instance,param_dict_result,intensity_range)[0]
    ax.plot(impf.intensity,impf.mdd*impf.paa*100,zorder=3,color='tab:blue',label=f'MDR best fit')#\nerr={min(error):.1e}')
    max_y = max(impf.mdd*impf.paa)*100
    ax.set(ylim=(0,max_y),title=title,xlabel=f'Intensity [{impf.intensity_unit}]',ylabel = 'MDR [%]')

    #plot all impact functions within the given quantile (top 10% by default)
    # q90=np.quantile(target,plot_quant)
    # params_over90 = [i['params'] for i in np.array(a)[target>q90]]
    # error_over90 = [1/i['target'] for i in np.array(a)[target>q90]]

    #Plot all impact functions with an Error less than 10% larger than the best
    params_over90 = [i['params'] for i in np.array(optimizer.res)[error<=min(error)*1.1]]
    error_over90 = [1/i['target'] for i in np.array(optimizer.res)[error<=min(error)*1.1]]

    for params,error_now in zip(params_over90,error_over90):
        impf = init_impf(impf_name_or_instance,params,intensity_range)[0]
        label = 'within 10 percent of best fit' if error_now==error_over90[0] else None
        ax.plot(impf.intensity,impf.mdd*impf.paa*100,color='grey',alpha=0.3,label=label)
        max_y = max(max_y,max(impf.mdd*impf.paa)*100)
    ax.set(ylim=(0,max_y))
    if haz:
        ax2 = ax.twinx()
        ax2.hist(haz.intensity[haz.intensity.nonzero()].getA1(),bins=40,color='tab:orange',
                 alpha=0.3,label='Haz intensity')
        ax2.set(ylabel='Intensity distribution in hazard data')
        ax2.legend(loc='upper right')
    ax.legend()
    return ax


from matplotlib import colors
def plot_param_space(param_dict_result,optimizer,err_name):
    """plot the parameter space with the best result highlighted"""

    #get all parameter combinations for 2d plots
    var_combs = list(combinations(param_dict_result.keys(),2))

    #set up figure
    rows = np.ceil(len(param_dict_result.keys())/3).astype(int)
    cols = min(len(param_dict_result.keys()),3)
    fig,axes = plt.subplots(rows,cols,figsize=(4.5*cols,3.5*rows),gridspec_kw={'wspace':0.3})

    #scatter plot each parameter combination
    for i,var_comb in enumerate(var_combs):
        ax = axes.flatten()[i]
        idx1=list(param_dict_result.keys()).index(var_comb[0])
        idx2=list(param_dict_result.keys()).index(var_comb[1])
        # fig,ax = plt.subplots()
        err_finite = (1/optimizer.space.target)[np.isfinite(1/optimizer.space.target)]
        min_c=min(err_finite)
        max_c=max(err_finite)
        norm = colors.LogNorm(vmin=min_c, vmax=max_c)
        scat=ax.scatter(optimizer.space.params[:,idx1],optimizer.space.params[:,idx2],
                        c=1/optimizer.space.target,norm=norm,cmap='viridis_r')
        # scat.cmap.set_over('grey') #not working atm
        ax.scatter(optimizer.max['params'][var_comb[0]],optimizer.max['params'][var_comb[1]],marker='x',color='red')
        if 'v_thresh' in var_comb and 'v_half' in var_comb:
            ax.plot([0,100],[0,100],color='grey')
        ax.set(xlabel=var_comb[0],ylabel=var_comb[1],title=var_comb)

    #set colorbar
    cbar_ax = fig.add_axes([0.25, -0.1, 0.5, 0.06]) #left,bot,width,height
    fig.colorbar(scat, cax=cbar_ax,orientation='horizontal').set_label(f'Error measure: {err_name}')


def get_emanuel_vals(intensity,v_thresh=20, v_half=60, scale=1e-3,power=3):
    """Get the Emanuel-type impact function values for a given intensity array"""

    #Check whether the input parameters are valid
    if v_half <= v_thresh:
        raise ValueError('Shape parameters out of range: v_half <= v_thresh.')
    if v_thresh < 0 or v_half < 0:
        raise ValueError('Negative shape parameter.')
    if scale > 1 or scale <= 0:
        raise ValueError('Scale parameter out of range.')

    #Calculate the impact function values
    v_temp = (intensity - v_thresh) / (v_half - v_thresh)
    v_temp[v_temp < 0] = 0
    v_temp = v_temp**power / (1 + v_temp**power)
    v_temp *= scale
    return v_temp


def get_emanuel_impf(v_thresh=20, v_half=60, scale=1e-3,power=3,
                    impf_id=1, intensity=np.arange(0, 110, 1),
                    intensity_unit='mm',haz_type='HL'):
    """
    Init TC impact function using the formula of Kerry Emanuel, 2011:
    https://doi.org/10.1175/WCAS-D-11-00007.1

    Parameters
    ----------
    impf_id : int, optional
        impact function id. Default: 1
    intensity : np.array, optional
        intensity array in m/s. Default:
        5 m/s step array from 0 to 120m/s
    v_thresh : float, optional
        first shape parameter, wind speed in
        m/s below which there is no damage. Default: 25.7(Emanuel 2011)
    v_half : float, optional
        second shape parameter, wind speed in m/s
        at which 50% of max. damage is expected. Default:
        v_threshold + 49 m/s (mean value of Sealy & Strobl 2017)
    scale : float, optional
        scale parameter, linear scaling of MDD.
        0<=scale<=1. Default: 1.0
    power : int, optional
        Exponential dependence. Default to 3 (as in Emanuel (2011))

    Raises
    ------
    ValueError

    Returns
    -------
    impf : ImpfTropCyclone
        TC impact function instance based on formula by Emanuel (2011)
    """

    #Get the function values. Note that invalid input parameters are checked
    # within get_emanuel_vals(). (e.g. V_half <= V_thresh)
    v_temp = get_emanuel_vals(intensity,v_thresh,v_half,scale,power)

    impf = ImpactFunc(haz_type=haz_type, id=impf_id,intensity=intensity,
                        intensity_unit=intensity_unit,name='Emanuel-type')
    impf.paa = np.ones(intensity.shape)
    impf.mdd = v_temp
    return impf


def fit_emanuel_impf_to_emp_data(emp_df,pbounds,opt_var='MDR',options=None,
                                 optimizer='Nelder-Mead',plot=True,param_start=None,
                                 verbose=True,max_iter=500,bias_weight=0):
    """Fit emanuel-type impact function to empirical data

    Args:
        emp_df (pd.DataFrame): DF with empirical data, including 'MDR', 'PAA', and 'count_cell'
        pbounds (dict): dictionary of parameter bounds
        opt_var (str, optional): Variable from emp_df to fit data to. Defaults to 'MDR'.
        options (dict, optional): Additional options for the Bayesian optimizer. Defaults to None.
        optimizer (str, optional): Choice of optimizer. Defaults to 'Nelder-Mead'.
        plot (bool, optional): Whether or not to plot the data. Defaults to True.
        param_start (dict, optional): Initial parameter values. Defaults to None.
        verbose (bool, optional): Whether or not to print the results. Defaults to True.
        max_iter (int, optional): Maximum number of iterations for the optimizer. Defaults to 500.
        bias_weight (int, optional): Weight of the bias in the optimization. Defaults to 0.
                                    if >0: weight MSE with |impact bias [%]|. i.e. if the impact bias is 5%
                                    and the weight is 1, the MSE is multiplied by 1.05

    Raises:
        ValueError: if get_emanuel_impf returns an error except for v_half <= v_thresh

    Returns:
        tuple: Parameters, optimizer object, and impact function
    """
    if not emp_df.index.name:
        raise ValueError('Careful, emp_df.index has no name. Double check if the \
                            index corresponds to the intensity unit!')

    def weighted_MSE(**param_dict):
        try:
            impf = get_emanuel_impf(**param_dict,intensity=emp_df.index.values)
        except ValueError as e:
            if 'v_half <= v_thresh' in str(e):
                #if invalid input to Emanuel_impf (v_half <= v_thresh), return zero
                return np.inf
            else:
                raise ValueError(f'Unknown Error in init_impf:{e}. Check inputs!')

        imp_mod = impf.mdd*impf.paa
        SE = np.square(imp_mod-emp_df[opt_var])

        SE_weigh = SE*emp_df.count_cell
        SE_weigh_noZero = SE_weigh[emp_df.index.values!=0]
        MSE = np.mean(SE_weigh_noZero) #mean over all intensity values (equivalent result with sum)


        if bias_weight>0:
            # #select only values with nonzero hazard and not zero impact for both empirical and model
            #NOTE: less intuitive than equivalent impact
            # nonzero = (emp_df.index.values!=0) & ~((emp_df[opt_var]==0) & (imp_mod==0) )
            # bias = imp_mod - emp_df[opt_var]
            # weighted_bias = (bias[nonzero]*emp_df.count_cell[nonzero]).sum()/emp_df.count_cell[nonzero].sum()
            # print(f'weighted bias: {weighted_bias:.2e}')

            #calculate equivalent impacts
            imp_mod_sum = (imp_mod*emp_df.count_cell).sum()
            imp_emp_sum = (emp_df[opt_var]*emp_df.count_cell).sum()
            imp_bias = (imp_mod_sum-imp_emp_sum)/imp_emp_sum
            # print(f'impact difference: {imp_bias*100:.2e}%')

            #add bias to MSE
            MSE = MSE*(1+abs(imp_bias)*bias_weight)

        return MSE

    def weighted_inverse_MSE(**param_dict):
        MSE = weighted_MSE(**param_dict)
        return 1/MSE

    if optimizer == 'Bayesian':
        if options is None: options = {'init_points' : 20, 'n_iter' : 80}
        optimizer = BayesianOptimization(f = weighted_inverse_MSE,
                                        pbounds = pbounds,
                                        verbose = 2,
                                        random_state = 4)

        optimizer.maximize(**options)#(init_points = 10, n_iter = 30)#
        print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))

        param_dict_result = optimizer.max["params"]
        # assert(param_dict_result.keys()==param_dict.keys())
        plot_param_space(param_dict_result,optimizer)
        impf = init_impf('emanuel_HL',param_dict_result,emp_df.index.values)[0]
        ax=plot_impf(param_dict_result,optimizer,'emanuel_HL',emp_df.index.values)

    elif optimizer == 'Nelder-Mead' or 'trust-constr':
        if param_start is None:
            #use mean of bounds as starting point
            param_means = [(v[0]+v[1])/2 for v in pbounds.values()]
        elif type(param_start)==dict:
            param_means = param_start.values()
        else:
            param_means = param_start
        param_dict = dict(zip(pbounds.keys(),param_means))
        bounds = [(v[0],v[1]) for v in pbounds.values()]
        x0 = list(param_dict.values())

        #define function that returns the MSE, with an array x as input
        def mse_x(x):
            param_dict_temp = dict(zip(param_dict.keys(), x))
            # return -weighted_inverse_MSE(**param_dict_temp)
            return weighted_MSE(**param_dict_temp)

        if optimizer == 'trust-constr':
            print(param_dict,bounds)
            np.testing.assert_array_equal((list(pbounds.keys())[:2]),['v_thresh', 'v_half'])
            cons = ({'type': 'ineq', 'fun': lambda x:  x[1] - x[0]}, #v_half-v_tresh is to be non-negative
                    {'type':'ineq', 'fun': lambda x: x[2]}) #scale is to be non-negative
            options= None

        elif optimizer == 'Nelder-Mead':
            cons = None
            options={'disp': verbose, 'maxiter': max_iter}

        elif optimizer == 'DIRECT': #seems to also work
            from scipy.optimize import direct
            raise NotImplementedError('DIRECT optimizer not yet implemented')

        res = minimize(mse_x, x0,
                        bounds=bounds,
                        constraints=cons,
                        # jac=gradient_respecting_bounds(bounds, mse_x) if optimizer == 'trust-constr' else None,
                        # method='SLSQP',
                        method = optimizer,
                        options=options)

        optimizer = res
        param_dict_result = dict(zip(param_dict.keys(), res.x))
        impf = init_impf('emanuel_HL',param_dict_result,emp_df.index.values)[0]

        if verbose:
            print(param_dict_result)
            # if bias_weight>0:
            imp_mod = impf.mdd*impf.paa
            imp_mod_sum = (imp_mod*emp_df.count_cell).sum()
            imp_emp_sum = (emp_df[opt_var]*emp_df.count_cell).sum()
            imp_bias = (imp_mod_sum-imp_emp_sum)/imp_emp_sum
            print(f'Equiv. impact bias: {imp_bias*100:.3f}%')

        if plot:
            ax=impf.plot(zorder=3)
            title = [f'{key}: {param_dict_result[key]:.2f}' if param_dict_result[key]>0.1 else f'{key}: {param_dict_result[key]:.2e}' for key in param_dict_result.keys()]
            ax.set(ylim=(0,max(impf.mdd*100)),title=title)
            #add empirical function to plot
            ax.plot(emp_df.index,emp_df[opt_var]*100,label=f'Empirical {opt_var}')
            plt.legend()

    return param_dict_result, optimizer, impf

def fit_emanuel_to_bootstrap(ds_boot,impf_best_fit,pbounds,opt_var='MDR',b_sample_name='b_sample',plot=None,bias_weight=0,param_start=None):
    """fit bootstrap sample to each subsample of bootstrapped dataset

    Args:
        ds_boot (sr.Dataset): dataset with dimensions (b_sample, intensity)
        impf_best_fit (climada.ImpactFunc): best fit impact function
        pbounds (dict): dictionary of parameter bounds
        opt_var (str, optional): Variable to optimize (MDR,PAA or MDD).
            Defaults to 'MDR'.
        b_sample_name (str, optional): Name of the dimension with the subsamples
        plot (str, optional): 'Quant' to plot 10-90% quatile, 'all' to plot each
            subsample. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with impact function values for each subsample
    """
    n_samples = ds_boot[b_sample_name].shape[0]

    df_boot_emanuel = pd.DataFrame(index = impf_best_fit.intensity, dtype=float,
                                   columns = [f'b_{i}' for i in range(n_samples)])
    df_boot_params = pd.DataFrame(index = range(n_samples), columns = pbounds.keys())
    for i in range(n_samples):
        df_now = ds_boot.isel({b_sample_name:i}).to_dataframe()
        pNow,resNow,impf_emanuelNow = fit_emanuel_impf_to_emp_data(
            df_now,pbounds,opt_var=opt_var,plot=False,verbose=False,bias_weight=bias_weight,param_start=param_start)
        df_boot_emanuel.loc[impf_best_fit.intensity,f'b_{i}']=impf_emanuelNow.mdd

        df_boot_params.loc[i] = pNow

    if plot is not None:
        fig,ax=plt.subplots()
        ax.plot(impf_best_fit.intensity,impf_best_fit.mdd*100)
        if plot == 'quant':
            sc.plot_funcs.fill_df_quantile(df_boot_emanuel,0.1,0.9,ax)
            ax.legend()
        elif plot == 'all':
            (df_boot_emanuel*100).plot(ax=ax,legend=False,alpha=0.2,color='grey')
        else:
            raise ValueError('plot must be "quant" or "all"')

    return df_boot_emanuel,df_boot_params