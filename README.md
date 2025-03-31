# EEG Project: Frontal Electrode Analysis

## Overview

This project aims to analyze the effect of frontal electrodes on EEG data. While the original study was implemented in MATLAB, we have reproduced the workflow using Python. The original study is titled Visuo-haptic prediction errors: a multimodal dataset (EEG, motion) in BIDS format indexing mismatches in haptic interaction.

## Pipeline
![image](https://github.com/user-attachments/assets/7c2a7b4b-40d9-411e-8e2d-0dd63f2a8cf0)

## Workflow

To reproduce our results, follow these steps:

## Prerequisites

* Tested on Python 3.12

## Installation

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
## Dataset and Configuration

### Dataset

Download the latest dataset from OpenNeuro:

* Dataset Link: [https://openneuro.org/datasets/ds003846/versions/2.0.2/download]

### Configuration

Update the dataset directory paths as needed in your configuration files (e.g., in your python notebooks).

## Jupyter notebooks

## EEG
The eeg.ipynb is our main python notebook which plots the ERP of different sessions of all the subjects.

## SNR
The snr.ipynb does the SNR for ERP data. It is one of the parts of our sanity checks.

## TFR
the tfr.ipynb removes the 1/f noise from the signal making it suitable to do the analysis.

## License

This project is licensed under [CC0]

## References

* [Original Paper Reference]
* Lukas Gehrke and Sezen Akman and Albert Chen and Pedro Lopes and Klaus Gramann (2024). Prediction Error. OpenNeuro.
* [Dataset] doi: doi:10.18112/openneuro.ds003846.v2.0.2
