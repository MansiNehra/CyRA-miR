# CyRA-miR

Machine learning framework for predicting cyclophosphamide response in breast cancer using miRNA expression profiles.

## Project Overview

This repository contains the complete machine learning workflow used for drug response prediction.

## Repository Structure

```
data/
src/
requirements.txt
README.md
.gitignore
```

## Machine Learning Models

The following machine learning models are implemented and evaluated:

- Logistic Regression (LR)
- Elastic Net Logistic Regression (EN-LR)
- Balanced Random Forest (BRF)
- LightGBM (LGBM)
- CatBoost (CAT)
- Easy Ensemble Classifier (EEC)


## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

## Outputs

The pipeline generates:

- Trained models
- Performance metrics
- ROC and PR curves
- Stability analysis heatmaps
- SHAP explanations plots
- Other Figures

## Data

The repository contains the processed dataset used for model development.
The original miRNA expression and clinical data were obtained from The Cancer Genome Atlas (TCGA) and processed prior to model training.
