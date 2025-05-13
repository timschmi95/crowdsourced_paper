# Improved real-time hail damage estimates leveraging dense crowdsourced observations
The scripts here reproduce the main results of the paper:
Schmid T., Gebhart, V., Bresch D. N. (2025) Improved real-time hail damage estimates leveraging dense crowdsourced observations. Meteorological Applications, DOI to be added

Publication status: accepted
<!-- Publication status: [accepted](https://doi.org/) -->

Contact: [Timo Schmid](timo.schmid@usys.ethz.ch)

## Content

### ./notebooks/

Jupyter notebooks to reproduce figures that appear in the paper.

### ./scClim/

Contains functions which are called in other scripts for data pre-processing, calibration, visualizing, as well as utility functions and constants.
These function represent a subset of all utility functions used in the damage modelling part of the [scClim project](https://scclim.ethz.ch/)

## Requirements
Requires:
* Python 3.9+ environment (best to use conda for CLIMADA repository)
* _CLIMADA_ repository version 5.0:
        https://wcr.ethz.ch/research/climada.html
        https://github.com/CLIMADA-project/climada_python
* Exposure and damage data for the calibration. The hail damage data used in the paper are not public and only available within the [scClim](https://scclim.ethz.ch/) project.