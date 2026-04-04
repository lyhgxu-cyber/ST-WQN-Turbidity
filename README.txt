ST-WQN: Spatiotemporal Water Quality Network for Remote Sensing Turbidity Inversion
Anonymous Repository for Peer Review


This repository includes the full processing pipeline for constructing a matched
satellite-in-situ turbidity dataset and the deep learning model (ST-WQN) for
water turbidity remote sensing inversion.


Environment Requirements
- Google Earth Engine (GEE) Code Editor (for JavaScript satellite matching)
- Python 3.8 - 3.11
- PyTorch >= 1.10
- Python packages: pandas, numpy, scikit-learn, matplotlib, seaborn


File Structure
01_GEE_Landsat_Matchup.js      Satellite-in-situ matching for Landsat 8/9
02_GEE_Sentinel2_Matchup.js    Satellite-in-situ matching for Sentinel-2
03_Data_Cleaning_and_QC.py     Four-step data quality control and filtering
04_ST_WQN_Train_and_Inference.py  Model training, validation, and visualization
wqn.pth                Trained model weights
README.txt                     This document


Pipeline Execution Order (Run step by step)

Step 1:
Run 01_GEE_Landsat_Matchup.js and 02_GEE_Sentinel2_Matchup.js on the GEE Code Editor
to obtain matched satellite-in-situ CSV files.

Step 2:
Run 03_Data_Cleaning_and_QC.py locally to perform data cleaning.
Steps include physical filtering, CV spatial homogeneity check,
Isolation Forest anomaly detection, and Huber regression residual removal.

Step 3:
Run 04_ST_WQN_Train_and_Inference.py to train the ST-WQN model, conduct validation,
and generate evaluation metrics and prediction plots.


Model Overview
- Input: Spectral bands, spectral indices, DOY encoding, geographic encoding, sensor flag
- Architecture: Spectral attention module + spatiotemporal feature fusion
- Loss: Asymmetric weighted Huber loss (alpha=2.0 for under-prediction penalty)
- Train/test split: Spatially independent split by monitoring stations


Output Files
- LS_1day_clean.csv (cleaned matched dataset)
- wqn.pth (trained model weights)
- wqn.png (prediction scatter plot)
- wqn.txt (evaluation metrics: R2, RMSE, MAE, MAPE)


Note
This repository is for peer review purpose only.
All rights reserved.