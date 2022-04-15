# Deep Gravity
## Introduction
"Deep Gravity - Model.ipynb" is an Python implementation of deep gravty model (Simini *et al.*, 2021). Other two models are spatial-temporal prediction of Bay Area Rapid Transit (BART) ridership.

## Reference
Simini, F., Barlacchi, G., Luca, M., & Pappalardo, L. (2021). A Deep Gravity model for mobility flows generation. *Nature Communications*, 12(1), 6576. https://doi.org/10.1038/s41467-021-26752-4

## Python Environment
> pip install requirement.txt

### Table of Contents
- Data Processing of [Smart Location Data](https://github.com/HaTT2018/Deep_Gravity/blob/main/Deep%20Gravity%20-%20Data%20Processing.ipynb) and [BART Ridership Data](https://github.com/HaTT2018/Deep_Gravity/blob/main/bart_data.py)
- [Deep Gravity Model](https://github.com/HaTT2018/Deep_Gravity/blob/main/Deep%20Gravity%20-%20Model.ipynb):
  - Model input: socio-economics data (see table below, [source](https://www.epa.gov/sites/default/files/2021-06/documents/epa_sld_3.0_technicaldocumentationuserguide_may2021.pdf)) of location 1 and location 2, as well as their geographical distance.
  - Model output: predicted trip counts between location 1 and location 2.
  - Model layout: 6 layers with dimension 256, and 9 layers with dimension 128.
    <img src="./imgs/features.png" width="400">
- [Spatio-temporal Prediction of BART Ridership Using Convolutional Neural Network](https://github.com/HaTT2018/Deep_Gravity/blob/main/Deep%20Gravity%20-%20Model%20-%20Conv.ipynb): Use previous *5 days* ridership to predict the next following *4 days* ridership. (Prediction horizon may subject to change)
- [Spatio-temporal Prediction of BART Ridership Using Graph Convolutional Neural Network](https://github.com/HaTT2018/Deep_Gravity/blob/main/Deep%20Gravity%20-%20Model%20-%20ConvGraph.ipynb): Use previous *5 days* ridership to predict the next following *4 days* ridership. (Prediction horizon may subject to change)
