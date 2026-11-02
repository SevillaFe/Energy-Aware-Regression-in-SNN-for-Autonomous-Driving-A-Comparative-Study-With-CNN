# Scripts_Paper_3

Repository with code used for experiments and analysis in the published article:

**Energy-Aware Regression in Spiking Neural Networks for Autonomous Driving: A Comparative Study With Convolutional Networks**

## Overview

This project contains scripts for:

- ETL and encoding of driving images into spike-based representations (rate, latency, delta)
- Training and evaluation of SNN and CNN models for steering-angle regression
- Hyperparameter optimization with Optuna
- Emissions and energy tracking with CodeCarbon
- Plotting and result analysis utilities

The repository has been cleaned up to keep an active, unified workflow while preserving older scripts in excluded folders.

## Main Structure

- `unified_scripts/`
  - Unified scripts used as the main entry points for current experiments
  - `00_ETL_Encode_SNN_delta.py`: data encoding pipeline (rate/latency/delta)
  - `00_CNN_AllNets_MSE_MAPE_EER.py`: unified CNN training (PilotNet/LaksNet/MiniNet)
  - `00_h5_SNN_AllNets_MSE_MAPE_EER.py`: unified H5-based SNN training (PilotNet/LaksNet/MiniNet)
  - `00_Optuna_Leaky_SNN_PilotNet.py`: Optuna optimization workflow

- `utils/`
  - Helper scripts for plotting and auxiliary analysis

- `archived_scripts/`, `deprecated_scripts/`
  - Legacy code kept for reference
  - Ignored by Git for clean commits

## Environment

Recommended Python version:

- Python 3.10+ (tested workflows may vary by machine)

Core dependencies used across scripts:

- torch
- torchvision
- pytorch-lightning
- snntorch
- optuna
- h5py
- pandas
- pillow
- tqdm
- torchmetrics
- codecarbon

Install example:

```bash
pip install torch torchvision pytorch-lightning snntorch optuna h5py pandas pillow tqdm torchmetrics codecarbon
```

## Quick Start

### 1) Encode dataset (SNN inputs)

```bash
python unified_scripts/00_ETL_Encode_SNN_delta.py --encoding all
```

Examples:

```bash
python unified_scripts/00_ETL_Encode_SNN_delta.py --encoding delta --threshold 0.25,0.5
python unified_scripts/00_ETL_Encode_SNN_delta.py --encoding latency --num-steps 25,50 --tau 2.5,5.0
```

### 2) Train unified CNN model

```bash
python unified_scripts/00_CNN_AllNets_MSE_MAPE_EER.py --model pilotnet --epochs 20
```

### 3) Train unified SNN model from H5

```bash
python unified_scripts/00_h5_SNN_AllNets_MSE_MAPE_EER.py --model laksnet --encoding latency --num-steps 25 --tau 5.0
```

### 4) Run Optuna optimization

```bash
python unified_scripts/00_Optuna_Leaky_SNN_PilotNet.py
```

## Metrics and Logging

The training workflows are focused on regression quality and efficiency:

- MSE
- MAPE
- EER (energy-efficiency related metric defined in scripts)
- CodeCarbon emissions tracking
- TensorBoard logs

## Notes

- Absolute dataset paths are used in several scripts; adjust CLI arguments to your local machine.
- Legacy scripts are intentionally excluded from versioned changes via `.gitignore`.
- Unified scripts are the recommended baseline for reproducible runs.

## Citation

If you use this code in academic work, please cite the article:

**Energy-Aware Regression in Spiking Neural Networks for Autonomous Driving: A Comparative Study With Convolutional Networks**

(Include DOI / conference / journal metadata as needed.)
