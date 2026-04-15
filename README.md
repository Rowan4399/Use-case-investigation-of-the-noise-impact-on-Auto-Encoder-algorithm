# ADS-B Noise Impact on Auto-encoder Models
This repository provides the implementation for the study: **“Investigation of the Noise Impact on Autoencoder Algorithms for ADS-B Trajectory Data”**

ADS-B (Automatic Dependent Surveillance–Broadcast) data are widely used in aviation research, but they often contain noise such as missing data, drift, and anomalies.

This project investigates how different types of noise affect the performance of autoencoder-based models for trajectory reconstruction.

We evaluate three architectures:

* Fully Connected Autoencoder (**FC-AE**)
* Long Short-Term Memory Autoencoder (**LSTM-AE**)
* Gated Recurrent Unit Autoencoder (**GRU-AE**)

##  Experiment Description

The workflow includes:

1. **Data preprocessing**

   * Filtering trajectories near airports
   * Removing anomalies and incomplete data
   * Resampling trajectories to fixed length

2. **Baseline selection**

   * Selecting high-quality trajectories using reconstruction error

3. **Noise injection**

   * Gaussian noise (random perturbation)
   * Drift noise (systematic offset)
   * Spike noise (random outliers)
   * Missing data (point removal + interpolation)

4. **Model evaluation**

   * Reconstruction performance measured using RMSE

##  Key Findings

* **Spike noise** has the least impact on model performance
* **Drift noise** causes the most consistent degradation
* **Missing data** shows a nonlinear impact (critical threshold effect)
* Noise sensitivity depends more on **dataset characteristics** than model architecture

##  Dataset

Experiments are conducted on four airport datasets:

* Zurich (LSZH)
* Harbin (ZYHB)
* Hangzhou (ZSHC)
* Guangzhou (ZGGG)

Each dataset contains aircraft trajectories with:

* Latitude & longitude
* Altitude
* Speed & heading
* Timestamp

The datasets for Harbin, Hangzhou, and Guangzhou are available at: https://drive.google.com/drive/folders/14-9x59wCM8_CSwlmTDpLF2uKGtoEHzKN?usp=drive_link
##  Output

The code produces:

* Reconstructed trajectories
* RMSE evaluation results
* Noise sensitivity analysis plots

