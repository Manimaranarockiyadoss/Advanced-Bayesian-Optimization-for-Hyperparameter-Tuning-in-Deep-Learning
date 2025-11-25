# Advanced Bayesian Optimization for Hyperparameter Tuning in Deep Learning

This repository implements **Bayesian Optimization using Gaussian Processes** to tune hyperparameters of a deep neural network (DNN),
and compares its performance against **Random Search**.

## Project Contents

- `bayesian_opt.py` – main Python script with:
  - data generation
  - model definition (Keras MLP)
  - Bayesian Optimization using scikit-optimize (GP + EI)
  - Random Search baseline
  - final evaluation and metrics
- `README.md` – project overview and usage
- `report.md` – detailed report-style explanation
- `requirements.txt` – required Python packages
- `LICENSE` – MIT license
- `results/` – directory for saving metrics/plots
- `notebooks/` – place for Jupyter experiments (optional)
- `plots/` – place to store generated figures

## How to Run

```bash
pip install -r requirements.txt
python bayesian_opt.py
```

## Summary

The goal is to show that **Bayesian Optimization** is more sample-efficient than **Random Search**
for deep learning hyperparameter tuning, achieving better accuracy with fewer trials.
