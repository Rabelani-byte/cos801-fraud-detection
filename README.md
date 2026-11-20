# COS801: Deep Learning on Tabular Data for Fraud Detection under Extreme Class Imbalance 

## Project Overview

This project investigates fraud detection on imbalanced tabular data using the IEEE-CIS Fraud Detection dataset.

This study it compares the  LightGBM, TabNet and FT-Transformer using the chronological data splitting. It also examines the class imbalance strategies, probability calibration,
performance on later transactions, and the operational costs of classification errors.

## Dataset

The IEEE-CIS Fraud Detection dataset is available from Kaggle:

https://www.kaggle.com/competitions/ieee-fraud-detection/data

The original dataset is not included in this repository.

## Models

- LightGBM
- TabNet
- FT-Transformer

The Class imbalance experiments it includes class weighting, random oversampling and focal loss.

## Main Results

| Model | Test PR-AUC | Test ROC-AUC | Test F1 |
|---|---:|---:|---:|
| LightGBM | 0.5480 | 0.9023 | 0.4975 |
| TabNet | 0.4082 | 0.8473 | 0.4267 |
| FT-Transformer | 0.4414 | 0.8554 | 0.4565 |

LightGBM achieved the highest test PR-AUC among the three evaluated models.

## Project Structure

## Project Structure

- `notebooks/`: Data exploration, preprocessing, baseline models,
  deep learning models, and final evaluation.
- `src/`: Python scripts for preprocessing and reproducing results.
- `Results/`: Saved experimental results.
- `Data/`: Dataset access instructions.
- `README.md`: Project overview and instructions.
- `requirements.txt`:The Required Python packages.
- `.gitignore`: Files excluded from Git.

## How to Run

1. Download the IEEE-CIS dataset from Kaggle.
2. Install the dependencies using `pip install -r requirements.txt`.
3. Open the notebooks in Google Colab.
4. Run the notebooks in numerical order, adjusting dataset paths as required.

The `src/reproduce_results.py` script recalculates evaluation metrics from a combined CSV containing the actual test labels and saved prediction probabilities for LightGBM, TabNet and FT-Transformer.

Example:

```bash
python src/reproduce_results.py --predictions results/test_predictions.csv
```

The combined prediction CSV is not currently included, so this command cannot yet reproduce the reported scores directly from the repository. The script does not retrain the models.

## Limitations

The experiments use one dataset. SAINT was not evaluated, and performance changes were assessed using validation and later test periods rather than multiple rolling time windows. 
Repeated random-seed variability was not established for all headline models.

## Reproducibility

The notebooks contain the experimental implementation and saved results. Some experiments require substantial computation and access to the original dataset.

## Academic Use

Developed for the COS801 Deep Learning project.

## Project Report and Presentation 
- Presentation is submitted through google form 
- And Final Report it is submitted Through Google Form with a link of Git repo. 


