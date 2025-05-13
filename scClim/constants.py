# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:16:50 2022
by Timo Schmid

Constants to be used for hail calculations
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy



CH_EXTENT = [5.8, 10.6, 45.7, 47.9]#x0 x1 y0 y1
CH_EXTENT_EPSG2056 = (2485014.011782823, 2837016.892254034, 1075214.1686203221, 1299782.7670088513)
ZRH_EXTENT = [8.35, 9, 47.15, 47.7]
ZRH_EXTENT_EPSG2056 = (2668000, 2721000, 1222000, 1284000)
SUB_CH_EXTENT_2056 = (2.55e6, 2.75e6, 1.125e6,1.29e6)
SUB_CH_EXTENT_2056_TIGHT = (2.56e6, 2.71e6, 1.15e6,1.28e6)
SUB_CH_EXTENT = (6.5, 9, 46.2, 47.8)
CMAP_VIR = copy.copy(plt.cm.get_cmap('viridis'))
CMAP_VIR.set_under('white',alpha=0)

# Variable_dictionaries
INT_RANGE_DICT = {
    'MESHS':    np.concatenate(([0],np.arange(20,100))),
    'dBZ':      np.arange(45, 75, 0.5),
    'crowd':    np.arange(0, 70, 1),
    'crowdFiltered': np.arange(0, 70, 1),
    'POH':      np.arange(0, 100, 1),
    'E_kinCC':  np.arange(0,2000,50),
}

INT_LABEL_DICT = {
    'MESHS_4km': 'Intensity: MESHS [mm]',
    'MESHS': 'Intensity: MESHS [mm]',
    'dBZ': 'Intensity: Reflectivity [dBZ]',
    'crowdFiltered': 'Intensity: Crowd-sourced hail size [mm]',
    'crowd': 'Intensity: Crowd-sourced hail size [mm]',
    'POH': 'Intensity: Probability of hail [%]',
    'E_kinCC': 'Intensity: E$_{kin}$ [J m$^{-2}$]',
}

CUT_OFF_DICT = {
    'MESHS':    60,
    'dBZ':      65,
    'crowd':    45,
    'crowdFiltered': 45,
    'POH':      100,
    'E_kinCC':  800,
}

ID_COL_DICT = {
    'GVL':          'Kantonale Versicherungsnummer',
    'AGV':          'VertragsNr',
    'GVB':          'Vertragsnummer',
    'GVZ':          'VersicherungsID',
    'MFZrandom_':   'POLNR',
    'KGV':          'id_col'
}


#Dictionary of windowsize for rolling window (in #steps)
W_SIZE_DICT = {
    'MESHS':    11,
    'MESHS_4km':11,
    'MESHS_opt_corr':11,
    'MESHS_20to23':11,
    'MESH':     9,
    'HKE':      15,
    'dBZ':      3,
    'crowd':    10,#5,
    'crowdFiltered': 5,
    'POH':      7,
    'E_kin':    3,
    'E_kinCC':  3,
    'MESHSdBZ': 5,
    'MESHSdBZ_p3':5,
    'VIL':      7,
}

DMG_BIN_DICT = {
    'MESHS': 5,
    'MESHS_4km':5,
    'MESHS_opt_corr':5,
    'MESHS_20to23':5,
    'MESH': 5,
    'HKE': 10,
    'dBZ': 2,
    'crowd': 5,
    'crowdFiltered': 2,
    'POH': 2,
    'E_kin': 50,
    'E_kinCC': 50,
    'MESHSdBZ': 5,
    'MESHSdBZ_p3':5,
    'VIL': 5,
}

BAUINDEX = pd.DataFrame(
    index=np.arange(2000,2023+1),
    data = {
    #See PDF in ../data/AGV
    'AGV':[402,417,436,436,422,422,436,436,464,482,482,482,498,498,498,498,498,486,486,486,486,486,486,486],
    #https://gvb.ch/de/versicherungen/baukostenindex.html
    'GVB': [120,125,127,123,124,127,130,134,140,139,138,141,141,141,141,141,141,140,141,143,144,147,147,147],
    #source: https://www.gvl.ch/versicherung/versicherungswert/
    'GVL':[810,810,815,808,810,820,846,880,917,902,912,926,927,921,918,916,912,906,911,907,901,928,928,928],
    #GVZ source: GVZ hail loss data (.csv)
    'GVZ':[840,  900,  900,  900,  900,  900,  900,  900,  970, 1025, 1025,
       1025, 1025, 1025, 1025, 1025, 1025, 1025, 1025, 1025, 1025, 1025,1025,1025],
    #KGV damages are already indexed! see scClim/grid_cantonal_data.py
    'KGV':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    }
)



BAUJAHR_DICT = {
    '' : (0,2022),
    'before1960' : (0,1959),
    '1960-2002' : (1960,2002),
    'after2002' : (2003,2022)
}
