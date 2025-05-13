# -*- coding: utf-8 -*-
"""
Ploting functions for climada and non-climada objects used in scClim

"""

#Import packages
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap, LogNorm, ListedColormap
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
from matplotlib.ticker import FuncFormatter
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cf
import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union

from climada.engine import Impact
from climada import CONFIG
from climada.util.constants import CMAP_IMPACT
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scClim as sc

cmap_imp = CMAP_IMPACT
cmap_imp.set_under('white',alpha=0)
data_dir = str(CONFIG.local_data.data_dir)

# %% General plotting functions
def plot_canton(ax,canton='Zürich',edgecolor = 'black',facecolor = "none",
                lakes=True,ch_border=None,zorder=0.5,lakeEdgeColor='none',
                lakeFaceColor='lightblue',union=False,**kwargs):
    """Plots cantonal borders on a catropy axis.

    Args:
        ax (axes): cartopy axis
        canton (str, optional): canton(s) name. Defaults to 'Zürich'.
        edgecolor (str, optional): Edgecolor. Defaults to 'black'.
        facecolor (str, optional): Facecolor. Defaults to "none".
        lakes (bool, optional): Whether to plot lakes. Defaults to True.
        ch_border (str, optional): color of Swiss border. Defaults to None.
        zorder (float, optional): Zorder. Defaults to 0.5.
        lakeEdgeColor (str, optional): Edgecolors of lakes. Defaults to 'none'.
        lakeFaceColor (str, optional): Facecolor of lakes. Defaults to 'lightblue'.
        union (bool, optional): Whether to unionze polygons of multiple cantons.
            Defaults to False.
    """
    ch_shp_path = str(CONFIG.ch_shp_path)
    reader = shpreader.Reader("%s/swissTLMRegio_KANTONSGEBIET_LV95.shp"%ch_shp_path)

    if canton=='all':
        sel_cantons = [place for place in reader.records() if
                       place.attributes["OBJEKTART"]=='Kanton' and
                       place.attributes["ICC"]=='CH']
        for sel_canton in sel_cantons:
            shape_feature = cf.ShapelyFeature([sel_canton.geometry],
                                               ccrs.epsg(2056), edgecolor=edgecolor,
                                               facecolor = facecolor,**kwargs)
            ax.add_feature(shape_feature,zorder=zorder)
    elif type(canton)==list:
        sel_cantons = [place for place in reader.records() if
                       place.attributes["NAME"] in canton]
        if union:
            union_geom = unary_union([sel_canton.geometry for sel_canton in sel_cantons])
            shape_feature = cf.ShapelyFeature([union_geom], ccrs.epsg(2056),
                                              edgecolor=edgecolor, facecolor=facecolor,
                                               **kwargs)
            ax.add_feature(shape_feature, zorder=zorder)
        else:
            for sel_canton in sel_cantons:
                shape_feature = cf.ShapelyFeature([sel_canton.geometry],
                                                ccrs.epsg(2056), edgecolor=edgecolor,
                                                facecolor = facecolor,**kwargs)
                ax.add_feature(shape_feature,zorder=zorder)
    elif type(canton)==str:
        sel_canton = [place for place in reader.records() if
                      place.attributes["NAME"]==canton][0]
        shape_feature = cf.ShapelyFeature([sel_canton.geometry],
                                           ccrs.epsg(2056), edgecolor=edgecolor,
                                           facecolor = facecolor,**kwargs)
        ax.add_feature(shape_feature)

    # add lake
    if lakes:
        add_lakes(ax,facecolor=lakeFaceColor,edgecolor=lakeEdgeColor,zorder=zorder)

    if ch_border:

        reader3 = shpreader.Reader("%s/swissTLMRegio_LANDESGEBIET_LV95.shp"%ch_shp_path)
        sel_ctry = [place for place in reader3.records() if
                    place.attributes["ICC"]=='CH'][0]
        shape_feature3 = cf.ShapelyFeature([sel_ctry.geometry],
                                           ccrs.epsg(2056), edgecolor=ch_border,
                                           facecolor = "none")
        ax.add_feature(shape_feature3,zorder = zorder-1)
        shape_feature4 = cf.ShapelyFeature([sel_ctry.geometry],
                                           ccrs.epsg(2056), edgecolor='None',
                                           facecolor = 'white')
        ax.add_feature(shape_feature4,zorder=0)

def add_lakes(ax,facecolor='lightblue',edgecolor='none',zorder=0.5,
              crs=ccrs.epsg(2056),min_size=1e7,**kwargs):
    reader2 = shpreader.Reader("%s/Hydrography/swissTLMRegio_Lake.shp"%str(CONFIG.ch_shp_path))
    geometry = reader2.geometries()
    geometry = np.array([g for g in geometry])
    lakesize = np.array([a.area for a in reader2.geometries()])
    geometry = geometry[lakesize>min_size]
    shape_feature2 = cf.ShapelyFeature(geometry,crs, edgecolor=edgecolor,
                                        facecolor = facecolor,**kwargs)
    ax.add_feature(shape_feature2,zorder=zorder)#0.5,2


def plot_nc(nc,ax=None,fillna='none',extent = None,title='',discrete='none',vmax=None,
            vmin=None,borders=True,cbar_lbl='',pl_type='',crs='default',canton=None,
            logScale=False,cbar_horizontal=False,extend=None,cbar=True,return_plot=False,
            border_color='black',**kwargs):
    """
     Parameters
    ----------
    nc : xr.DataArray
        with 1 timestep only
    """


    if crs == 'default': #PlateCaree()
        transform = ccrs.PlateCarree()
        x_dim = 'lon'
        y_dim = 'lat'
        if 'clat' in nc.coords:
            nc = nc.rename({'clat':'lat','clon':'lon'})
    elif crs=='EPSG:2056':
        transform = ccrs.epsg(2056)
        x_dim = 'chx'
        y_dim = 'chy'
    elif crs=='EPSG:21781':
        transform = ccrs.epsg(21781)
        x_dim = 'chx'
        y_dim = 'chy'
    elif type(crs) == ccrs.AlbersEqualArea: #crs given by rxr object
        transform = crs #nc.rio.crs
        x_dim = 'x'
        y_dim = 'y'

    if not fillna=='none':
        nc = nc.fillna(fillna)
    if not discrete=='none':
        nc = nc>discrete
    if not ax:
        fig,ax = plt.subplots(1,1,subplot_kw={'projection':transform})
    if extent: #
        ax.set_extent(extent,crs=transform)
        # lon_min, lon_max, lat_min, lat_max = extent
        # lon_cond = np.logical_and(nc.lon >= lon_min, nc.lon <= lon_max)
        # lat_cond = np.logical_and(nc.lat >= lat_min, nc.lat <= lat_max)
        # nc = nc.where(np.logical_and(lat_cond,lon_cond),drop=True)

    if logScale:
        vmin = vmin if vmin else nc.min().values
        vmax = vmax if vmax else nc.max().values
        vmin = max(1,vmin) #avoid log(0)
        vmax = max(1,vmax) #avoid empty norm
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        kwargs['norm'] = norm
        vmin = None; vmax = None
    else:
        norm = None

    if pl_type=='contour':
        plot = ax.contour(nc[x_dim],nc[y_dim],nc,transform=transform,
                          extend=extend,**kwargs)
    elif pl_type=='contourf':
        if nc.ndim == 2:
            plot = ax.contourf(nc[x_dim],nc[y_dim],nc,transform=transform,
                            extend=extend,**kwargs)
        elif nc.ndim == 1:
            plot = ax.tricontourf(nc[x_dim],nc[y_dim],nc,transform=transform,
                            extend=extend,**kwargs)
    elif pl_type=='bool_field':
        #create bool colormap
        if 'cmap' in kwargs:
            cmap = ListedColormap(['white',kwargs['cmap']])
            # cmap = plt.cm.get_cmap(kwargs['cmap']).copy()
            cmap.set_under('white',alpha=0)
            kwargs['cmap'] = cmap
        if not nc.max()==0:
            plot = ax.pcolormesh(nc[x_dim],nc[y_dim],nc,vmax=vmax,shading='auto',
                                 transform=transform,vmin=0.5,**kwargs)
    else:
        plot = ax.pcolormesh(nc[x_dim],nc[y_dim],nc,vmax=vmax,vmin=vmin,
                             transform=transform,shading='auto',**kwargs)
    if cbar:
        add_cbar(ax,plot,cbar_lbl,extend=extend,horizontal=cbar_horizontal)

    if borders:
        ax.add_feature(cf.BORDERS,edgecolor=border_color)
    if title:
        try:
            ax.set_title(title) #+nc.name+nc.time)
        except:
            ax.set_title(title+nc.name)
    if canton:
        plot_canton(ax,canton=canton)


    # axis.set_title("\n".join(wrap(tit)))
    # if fontsize:
        # cbar.ax.tick_params(labelsize=fontsize)
        # cbar.ax.yaxis.get_offset_text().set_fontsize(fontsize)
        # for item in [axis.title, cbar.ax.xaxis.label, cbar.ax.yaxis.label]:
            # item.set_fontsize(fontsize)
    # else:
        # plt.colorbar(plot,ax=ax,shrink = cbar_shrink)
    if return_plot:
        return ax, plot
    else:
        return ax

def add_cbar(ax,plot,cbar_lbl='',extend=None,horizontal=False,pad=0.1,**kwgs):
        if horizontal:
            cbax = make_axes_locatable(ax).append_axes(
                'bottom', size="6.5%", pad=pad, axes_class=plt.Axes)
            cbar = plt.colorbar(plot, cax=cbax, orientation='horizontal',
                                extend=extend,**kwgs)
        else:
            cbax = make_axes_locatable(ax).append_axes(
                'right', size="6.5%", pad=pad, axes_class=plt.Axes)
            cbar = plt.colorbar(plot, cax=cbax, orientation='vertical',
                                extend=extend,**kwgs)
        cbar.set_label(cbar_lbl)

def add_cbar_v2(fig,ax,c_mappable,x0=0.3,y0=0,dx=0.7,dy=0.08,label="",ticks=None,
                labelsize=8,labelpad=None,orientation='horizontal',**kwargs):
    """add new colorbar to figure in a specific position"""

    #Define position of colorbar
    pos = ax.get_position()
    x0_ = pos.x0 + x0*pos.width
    y0_ = pos.y0 + y0*pos.height
    width = dx*pos.width
    height = dy*pos.height

    #Add new axis
    cbax2 = fig.add_axes((x0_,y0_,width,height))

    #Add colorbar
    cbar2 = plt.colorbar(c_mappable, cax=cbax2, orientation=orientation,**kwargs)#, extend=extend)
    cbar2.set_label(label, fontsize=labelsize,labelpad=labelpad)
    if ticks is not None:
        cbar2.set_ticks(ticks)
    cbar2.ax.tick_params(labelsize=labelsize)

def cut_cmap(cmap, minval=0.0, maxval=1.0, n=-1,return_weak=False):

    if n == -1:
        n = cmap.N
    new_cmap = LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n))**1.1)
    new_cmap_light = LinearSegmentedColormap.from_list(
         'truncLight({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n))**0.3)
    if return_weak:
        return new_cmap,new_cmap_light
    else:
        return new_cmap

# %% Specific plotting functions

########################### Subproject E #####################################
def plot_crowdsourced_overview(crowd_sel,res_km = 2,norm = LogNorm(vmin=1,vmax=1e3),figsize=(6,4)):

    #Create plot
    prj = ccrs.epsg(2056) #ccrs.AlbersEqualArea(8.222665776, 46.800663464)
    fig,ax = plt.subplots(1,1,figsize=figsize,subplot_kw={'projection':prj},dpi=200)

    #Plot crowdsourced data (2d histogram)
    ext = sc.constants.CH_EXTENT_EPSG2056
    step = res_km*1e3
    bins = [np.arange(ext[0],ext[1],step=step),np.arange(ext[2],ext[3],step=step)]
    cmap = cut_cmap(plt.cm.get_cmap('Purples'),0.1,1)
    cmap.set_under('white',alpha=0)
    pl = ax.hist2d(crowd_sel['chx'],crowd_sel['chy'],bins=bins,cmap=cmap,norm=norm,transform=ccrs.epsg(2056))
    sc.plot_funcs.add_cbar(ax,pl[3],cbar_lbl='Number of reports')

    #add cantons
    col='brown'
    sc.plot_canton(ax,canton=['Zürich','Aargau','Luzern','Bern'],
                facecolor='none',edgecolor=col,zorder=2,ch_border='black',lakeEdgeColor='steelblue')
    #Add canton names
    canton_names = ['Zurich','Aargau','Lucerne','Berne']
    canton_locs = [(8.4,47.72),(7.85,47.63),(8.19,46.9),(7.1,47.35)]

    transform = ccrs.PlateCarree()#._as_mpl_transform(ax)
    for i in range(len(canton_names)):
        ax.annotate(canton_names[i], xy = canton_locs[i], xytext = canton_locs[i],
                    xycoords=transform,fontsize=10,zorder=3,color=col,
                    path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')])

    #add scalebar
    y= 1090e3
    x = 2733e3
    scaleBar = Line2D((x,x+100e3),(y,y),transform = ccrs.epsg(2056),color='black',linewidth=2.5,zorder=6)
    ax.add_line(scaleBar)
    ax.text(x+50e3, y+5e3, '100 km', transform=ccrs.epsg(2056),fontsize=9,ha='center',va='bottom',weight='bold')

    # Add line from (6, 46.1) to (10, 47.5)
    pts_1 = [(6.5,45.7),(6.85,46.15),(8.1,46.75),(9.8,47.2),(11,47.3)]
    line = Line2D([x[0] for x in pts_1],[x[1] for x in pts_1],transform=ccrs.PlateCarree(),color='grey',linewidth=2)
    pts_2 = [(8,45.8),(8.35,46.3),(9.3,46.33),(10.5,45.9)]

    line2 = Line2D([x[0] for x in pts_2],[x[1] for x in pts_2],transform=ccrs.PlateCarree(),color='grey',linewidth=2)
    ax.add_line(line)
    ax.add_line(line2)

    ax.annotate('Alps', xy = (8.7,46.5), xytext = (8.7,46.5),xycoords=transform,fontsize=14,zorder=3,color='grey', rotation=10,
                path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')])


    # #set figure options and save the figure
    gl=ax.gridlines(draw_labels=["bottom","left"], x_inline=False, y_inline=False,linestyle='--',
                    linewidth=0.5,color='darkgrey',xlabel_style={'color':'grey','size':8,'rotation':'horizontal'},
                    ylabel_style={'color':'grey','size':8})
    ax.set_extent(sc.constants.CH_EXTENT_EPSG2056,crs = ccrs.epsg(2056))
    return fig,ax

def plot_imp_comp_V3(date, ds_KGV, ds_haz, canton_list=None):
    """
    Plot comparison of observed, crowdsourced and radar-based impacts

    Args:
        date (str): Date of event (format: 'YYYY-MM-DD')
        ds_KGV (xr.Dataset): Building insurance dataset (KGV)
        ds_haz (xr.Dataset): Hazard data (gridded crowdsourced reports)
        var (str, optional): PAA or MDR. Defaults to 'PAA'.
        canton_list (list, optional): List of cantons to plot. Defaults to None.
    """
    text_kw = dict(ha='right', va='center', fontsize=12, rotation='vertical')
    if canton_list is None:
        canton_list = ['Zürich', 'Aargau', 'Luzern', 'Bern']

    # Set up figure
    fig, axes = plt.subplots(5, 3, figsize=(15, 25), subplot_kw={'projection': ccrs.epsg(2056)},
                             dpi=80, gridspec_kw={'hspace': -0.35, 'wspace': 0.05, 'height_ratios': [1, 1, 1, 1, 1]})


    # ----------- Plot Exposure -----------
    ax, plot = sc.plot_nc(ds_KGV['n_count_exposure'], ax=axes[0, 0], title=f"Observations",
                          borders=False, cmap='Greys', vmin=1, return_plot=True, cbar=False)
    sc.plot_funcs.add_cbar_v2(fig, axes[0, 0], plot, label="No. of buildings", y0=0.09,labelpad=0) #dy=0.07)
    rect = Rectangle((0.3, 0), 0.75, 0.16,  # (x, y, width, height) in axes coordinates
                     facecolor='white', edgecolor='grey', alpha=1, zorder=10,
                        transform=axes[0, 0].transAxes)
    axes[0, 0].add_patch(rect)

    axes[0, 0].text(-0.02, 0.5, "Exposure & hazard", transform=axes[0, 0].transAxes, **text_kw)

    # ----------- Plot hazard -----------
    # Convert date to datetime object
    date_dt = dt.datetime.strptime(date, '%Y-%m-%d').date()

    # Plot gridded crowdsourced data
    _, plot = sc.plot_nc(ds_haz.sel(time=date).h_smooth.where(ds_haz.sel(time=date).h_smooth > 0),
                         ax=axes[0, 1], borders=False, cbar_lbl='Hail size [mm]',
                           cbar=False, return_plot=True,title='Crowdsourced')

    sc.plot_funcs.add_cbar_v2(fig, axes[0, 1], plot, label="Hail size [mm]",
                              extend='neither', y0=-0.04,labelpad=0)  # ticks=[1,20,40,60],
    rect = Rectangle((0.3, 0), 0.75, 0.16,  # (x, y, width, height) in axes coordinates
                     facecolor='white', edgecolor='grey', alpha=1, zorder=10,
                     transform=axes[0,1].transAxes)
    axes[0,1].add_patch(rect)

    meshs = xr.open_dataset(f'{data_dir}/V5/MZC/MZC_X1d66_{date_dt.year}.nc').sel(time=date).MZC.clip(0, 100)
    _, plot = sc.plot_nc(meshs.where(meshs > 0), ax=axes[0, 2], cmap='viridis',
                         borders=False, cbar_lbl='MESHS [mm]', cbar=False,
                         return_plot=True,title='Radar-based')
    sc.plot_funcs.add_cbar_v2(fig, axes[0, 2], plot, label="MESHS [mm]", y0=-0.04,labelpad=0)  # ticks=[20,50,75,100]
    rect = Rectangle((0.3, 0), 0.75, 0.16,  # (x, y, width, height) in axes coordinates
                     facecolor='white', edgecolor='grey', alpha=1, zorder=10,
                        transform=axes[0,2].transAxes)
    axes[0,2].add_patch(rect)

    # Define variables for percentage of assets affected (PAA) or mean damage ratio (MDR)
    for i, var in enumerate(['PAA', 'MDR']):
        if var == 'PAA':
            var_name = "Percentage of assets affected"
            abs_str = 'n_count'
            abs_name = "No. of claims"
            ds_KGV['n_count_MESHS'] = ds_KGV['n_buildings_MESHS'].copy()
            vmin = 0.01
            cmaps = ["Reds", "Purples"]

        elif var == 'MDR':
            var_name = "Mean damage ratio"
            abs_str = "imp"
            abs_name = "Monetary damages [CHF]"
            ds_KGV['imp'] = ds_KGV['imp_observed'].copy()
            vmin = 0.0001
            cmaps = ["Oranges","Greens"]

        # ----------- Plot absolute impacts -----------
        # Select the 95th percentile of the reported damage as max. value for colorbar
        q95_noNan = ds_KGV[abs_str].sel(date=date).where(ds_KGV[abs_str].sel(date=date) > 0).quantile(0.95)
        cmap = sc.cut_cmap(get_cmap(cmaps[0]).copy(), minval=0.02)
        cmap.set_under('white')
        kwargs = {"cmap": cmap, "vmin": 1, "vmax": q95_noNan, 'borders': False}

        # Observed
        sc.plot_nc(ds_KGV[abs_str].sel(date=date), ax=axes[2 + 2 * i, 0], **kwargs, cbar=False)
        axes[2+2*i,0].text(-0.02, 0.5, abs_name, transform=axes[2+2*i,0].transAxes, **text_kw)
        # Crowdsourced
        sc.plot_nc(ds_KGV[f"{abs_str}_crowd"].sel(date=date), ax=axes[2 + 2 * i, 1], **kwargs, cbar=False)

        # ----------- Plot relative impacts -----------
        # Select 95th percentile of the relative impact as max. value for colorbar
        q95_noNan = ds_KGV[var].sel(date=date).where(ds_KGV[var].sel(date=date) > 0).quantile(0.95)
        cmap2 = sc.cut_cmap(get_cmap(cmaps[1]).copy(), minval=0.05)
        cmap2.set_under('white')
        kwargs2 = {"cmap": cmap2, "vmax": q95_noNan, "vmin": vmin, 'borders': False}

        # Define formatter for colorbar
        if q95_noNan > 0.05:
            fmt = lambda x, pos: '{:.0%}'.format(x)
        elif q95_noNan > 0.005:
            fmt = lambda x, pos: '{:.1%}'.format(x)
        else:
            fmt = lambda x, pos: '{:.2%}'.format(x)
        fmter = FuncFormatter(fmt)

        # Observed
        sc.plot_nc(ds_KGV[var].sel(date=date), ax=axes[1+2*i,0], **kwargs2, cbar=False)
        axes[1+2*i,0].text(-0.02, 0.5, var_name, transform=axes[1+2*i,0].transAxes, **text_kw)
        # Crowdsourced
        sc.plot_nc(ds_KGV[f'{var}_crowd'].sel(date=date), ax=axes[1 + 2 * i, 1], **kwargs2, cbar=False)

        # ----------- Plot second hazard data (radar) -----------
        cbar_kw = dict(x0=1.02, y0=-0.02, dx=0.07, dy=1.04, labelsize=10,orientation='vertical')
        # Absolute impacts
        _, plot = sc.plot_nc(ds_KGV[f"{abs_str}_MESHS"].sel(date=date), ax=axes[2 + 2 * i, 2],
                             cbar=False, return_plot=True, **kwargs)
        sc.plot_funcs.add_cbar_v2(fig, axes[2 + 2 * i, 2], plot, label=f"{abs_name}",
                                  **cbar_kw
                                  )

        # Relative impacts
        _, plot = sc.plot_nc(ds_KGV[f"{var}_MESHS"].sel(date=date), ax=axes[1 + 2 * i, 2],
                             cbar=False, return_plot=True, **kwargs2)
        sc.plot_funcs.add_cbar_v2(fig, axes[1 + 2 * i, 2], plot, label=f"{var}",
                                  format=fmter,**cbar_kw)

    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)',
              '(k)', '(l)', '(m)', '(n)', '(o)']
    for ax, lbl in zip(axes.flatten(), labels):
        ax.set_extent(sc.constants.SUB_CH_EXTENT_2056_TIGHT, crs=ccrs.epsg(2056))
        sc.plot_canton(ax, canton=canton_list, edgecolor='grey', facecolor='none',
                       zorder=6)
        txt = ax.text(0.02, 0.97, f"{lbl}", transform=ax.transAxes, ha='left',
                      va='top', fontsize=14, weight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

    return fig

def plot_filtering_V2(date,crowd_data,crowd_raw,ds_haz,fs=None,return_fig=False,
                   title=True):
    """
    Plot the filtering process of crowd data: 1) MESHS/POH based filter, 2) DBSCAN

    """
    crowd_plot = crowd_data[(crowd_data.hailday==date) & (crowd_data.cluster!=-1)]
    crowd_excl = crowd_data[(crowd_data.hailday==date) & (crowd_data.cluster==-1)]

    #Plot sizes on only clustered points
    fig,axes = plt.subplots(2,3,figsize=(18,10),subplot_kw={'projection': ccrs.epsg(2056)},
                            dpi=80,gridspec_kw={'hspace':0.1,'wspace':-0.02})

    #Raw data
    _ = sc.crowd_process.plot_crowd(crowd_raw[(crowd_raw.hailday==date)],fig_ax=(fig,axes[0,0]),gridlines=False,legend=False)

    #MESHS/POH filtered data
    _ = sc.crowd_process.plot_crowd(crowd_data[(crowd_data.hailday==date) ],fig_ax=(fig,axes[0,1]),gridlines=False,legend=False)

    #Clustering
    axes[1,1].scatter(crowd_plot.chx,crowd_plot.chy,c=crowd_plot.cluster,cmap='tab20c',
                    s=10,transform=ccrs.epsg(2056),zorder=30)
    axes[1,1].scatter(crowd_excl.chx,crowd_excl.chy,c='black',s=10,
                    transform=ccrs.epsg(2056),zorder=30)
    cols= ['tab:blue','tab:orange','tab:green']
    patchList = [None,None]
    patchList[0] = [Line2D([], [], color=c, marker='o', linestyle='None',markersize=5) for c in cols]
    patchList[1] = Line2D([], [], color='black', marker='o', linestyle='None',markersize=None)
    axes[1,1].legend(handles=patchList, labels=['Clustered reports','Outliers'],
            handler_map = {list: HandlerTuple(None)},loc='lower right',fontsize=fs)


    #Plot sizes on only clustered points
    _ = sc.crowd_process.plot_crowd(crowd_plot,fig_ax=(fig,axes[0,2]),
                                    gridlines=False,loc='lower right',
                                    bbox_to_anchor=(1.02, -0.02),all_lables=True)


    #Plot POH and MESHS
    year = date.year
    meshs=xr.open_dataset(f'{data_dir}/V5/MZC/MZC_X1d66_{year}.nc').sel(time=date.strftime("%Y-%m-%d")).MZC
    poh=xr.open_dataset(f'{data_dir}/V5/BZC/BZC_X1d66_{year}.nc').sel(time=date.strftime("%Y-%m-%d")).BZC
    sc.plot_nc(poh,ax=axes[1,0],cmap='lightblue',discrete=0,pl_type='bool_field',cbar=False,borders=False)
    ax,plot = sc.plot_nc(meshs.where(meshs>20),ax=axes[1,0],cmap='viridis',cbar=False,borders=False,return_plot=True,vmin=20,vmax=80)
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0+0.05*pos.width, pos.y0 - 0.25 * pos.height, pos.width*0.9, 0.1 * pos.height])
    cb = plt.colorbar(plot,cax=cax,orientation='horizontal',label='MESHS [mm]')
    cb.set_label('MESHS [mm]',fontsize=fs)
    cb.ax.tick_params(labelsize=fs)

    #Plot poh legend
    patch = Patch(color="lightblue",alpha=1, label=f'Probability of Hail > 0% ')
    legend=ax.legend(handles=[patch],loc='lower right',fontsize=fs)
    legend.get_frame().set_alpha(1)
    legend.set_zorder(8)

    # #Plot contours within first plot
    # sc.plot_nc((sc.verification.maximum_filter_nc(poh,radius=4)>0),ax = axes[0,0],pl_type='bool_field',zorder=0.5,cmap='lightgreen',cbar=False)
    # sc.plot_nc((sc.verification.maximum_filter_nc(meshs,radius=4)>20),ax = axes[0,0],pl_type='bool_field',zorder=0.5,cmap='green',cbar=False)

    #Plot gridded hazard data in last subplot
    ax,plot = sc.plot_nc(ds_haz['h_smooth'].where(ds_haz['h_smooth']>0).sel(time=date.strftime("%Y-%m-%d")),
                         ax=axes[1,2],cbar=False,borders=False,cmap='viridis',
                         return_plot=True,zorder=0.5)
    pos = ax.get_position()



    cax = fig.add_axes([pos.x0+0.05*pos.width, pos.y0 - 0.25 * pos.height, pos.width*0.9, 0.1 * pos.height])
    cb2=plt.colorbar(plot,cax=cax,orientation='horizontal')
    cb2.set_label('Gridded hail size [mm]',fontsize=fs)
    cb2.ax.tick_params(labelsize=fs)

    labels = ['(a)','(b)','(c)','(d)','(e)','(f)']
    for ax,lbl in zip(axes.flatten(),labels):
        ax.set_extent(sc.constants.SUB_CH_EXTENT_2056, crs=ccrs.epsg(2056))
        sc.plot_canton(ax,canton=['Bern','Zürich','Luzern','Aargau'],edgecolor='grey',lakeEdgeColor='royalblue',lakeFaceColor='lightblue')
        txt = ax.text(0.02,0.97,f"{lbl}", transform=ax.transAxes,ha='left',
                    va='top',fontsize=14,weight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

    if title:
        fig.suptitle(f"{date.strftime('%Y-%m-%d')}",y=0.95,weight='bold',fontsize=fs)

    axes[0,0].text(0.8,1.02,f"Step 1: Radar-based filter",transform=axes[0,0].transAxes,fontsize=fs)#,color='tab:blue')
    axes[0,1].text(0.8,1.02,f"Step 2: 4D-DBSCAN clustering",transform=axes[0,1].transAxes,fontsize=fs)
    axes[0,2].text(1.02, 0, f"Step 3: Gridding of filtered reports",
                  transform=axes[0,2].transAxes, fontsize=fs,
                  rotation=-90, ha='left', va='center')

    # Make frame of axes[1,0] blue
    # axes[1,0].spines['geo'].set_edgecolor('tab:blue')
    # axes[1,0].spines['geo'].set_linewidth(2)

    # axes[0,0].set_title('Radar-based filter',fontsize=fs,)
    # axes[0,1].set_title('4D-DBSCAN clustering',fontsize=fs)
    # axes[0,2].set_title('Gridding of filtered reports',fontsize=fs)

    axes[1,2].spines['geo'].set_linewidth(3)
    if return_fig:
        return fig,axes



#impact function plots (from empirical calibration)
def fill_df_quantile(df,q1,q2,ax,label=None,**kwargs):
    q1_arr=df.quantile(q=q1,axis=1,numeric_only=True)*100
    q2_arr=df.quantile(q=q2,axis=1,numeric_only=True)*100
    if label is None:
        label = 'Q %.2f-%.2f' % (q1, q2)
    color = kwargs.pop('color','black')
    alpha = kwargs.pop('alpha',0.2)
    ax.fill_between(df.index,q1_arr,q2_arr,color=color,alpha=alpha,
                    label = label,**kwargs)


def get_emp_quantile(ds_roll_gc, q):
    ds_roll_gc['PAA'] = ds_roll_gc.n_dmgs/ds_roll_gc.n_exp
    ds_roll_gc['MDR'] = ds_roll_gc.dmg_val/ds_roll_gc.exp_val
    ds = ds_roll_gc[['PAA', 'MDR']]
    quant = ds.quantile(q, dim='index')
    return quant
