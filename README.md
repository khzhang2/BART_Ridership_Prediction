# BART Ridership Prediction
Author: HaTT2018
## Introduction
Models (Linear Rregression, Convolutional Neural Network, Graph Convolutional Network) are for ridership prediction of Bay Area Rapid Transit (BART). This repository (excluding "Model - Deep Gravity.ipynb") is for UC Berkeley course [CE 259](https://classes.berkeley.edu/content/2022-spring-civeng-259-001-lec-001) term project.

## Python Environment
Should install 
  > numpy, pandas, geopandas, matplotlib, torch, sklearn, skmob

## Table of Contents
- Data Processing of [Smart Location Data](./Data%20-%20SLD.ipynb) and [BART Ridership Data](./Data%20-%20bart_data.py)
- [Deep Gravity Model](./Model%20-%20Deep%20Gravity.ipynb):
  - Model input: socio-economics data (see table below, [source](https://www.epa.gov/sites/default/files/2021-06/documents/epa_sld_3.0_technicaldocumentationuserguide_may2021.pdf)) of location 1 and location 2, as well as their geographical distance.
  - Model output: predicted trip counts between location 1 and location 2.
  - Model layout: 6 layers with dimension 256, and 9 layers with dimension 128.
    <img src="./imgs/features.png" width="400">
- [Ridership Prediction using Linear Regression Model](./Model%20-%20Linear.ipynb): Use previous *5 days* ridership to predict the next following *4 days* ridership. (Prediction horizon may subject to change, see "[config.csv](./config.csv)" for details)
- [Spatio-temporal Prediction of BART Ridership Using Convolutional Neural Network](./Model%20-%20Conv.ipynb): Use previous *5 days* ridership to predict the next following *4 days* ridership. (Prediction horizon may subject to change, see "[config.csv](./config.csv)" for details)
- [Spatio-temporal Prediction of BART Ridership Using Graph Convolutional Neural Network](./Model%20-%20ConvGraph%202.ipynb): Use previous *5 days* ridership to predict the next following *4 days* ridership. (Prediction horizon may subject to change, see "[config.csv](./config.csv)" for details)
- A detailed comparison report of the Linear Regression, CNN and GCN can be found [here](./reports/Report.pdf).
