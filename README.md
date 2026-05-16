# Cervical Cancer Risk Prediction — Machine Learning Thesis

> Predicting biopsy outcomes using supervised machine learning in R.  
> \*\*Authors:\*\* Viet Tien Trinh · Zerui Wang

\---

## Table of Contents

* [Background](#background)
* [Dataset](#dataset)
* [Project Structure](#project-structure)
* [Workflow](#workflow)
* [Models](#models)
* [Class Imbalance Strategies](#class-imbalance-strategies)
* [Evaluation Metrics](#evaluation-metrics)
* [How to Run](#how-to-run)
* [Dependencies](#dependencies)

\---

## Background

Cervical cancer is one of the most preventable cancers when detected early. This thesis investigates whether machine learning models can accurately predict a patient's **biopsy outcome** (positive/negative) based on demographic, lifestyle, and medical history data. After identifying the best-performing model, the most influential risk factors are extracted to support clinical understanding.

\---

## Dataset

|Property|Details|
|-|-|
|**Source**|Hospital Universitario de Caracas, Venezuela|
|**Observations**|858 patients|
|**Features**|36 variables — demographics, habits, STD history, contraceptive use|
|**Target variable**|`Biopsy`: `1` = cancer positive, `0` = negative|
|**Missing values**|Present — encoded as `?` in the raw file; some patients declined to answer|
|**UCI Repository**|[Cervical Cancer Risk Factors](https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors)|


### Preprocessing decisions

* **Removed — zero variance:** `STDs:AIDS`, `STDs:cervical condylomatosis`
* **Removed — data leakage:** `Dx`, `Dx:Cancer`, `Dx:CIN`, `Dx:HPV`, `Hinselmann`, `Schiller`, `Citology` (these are other test outcomes from the same exam as Biopsy)
* **Removed — >90% missing:** `STDs: Time since first diagnosis`, `STDs: Time since last diagnosis`
* **Imputation:** median (numeric) and mode (categorical) — performed **inside the recipe** per fold to prevent data leakage from test data

\---

## Project Structure

```
├── Dataset/
│   └── risk\_factors\_cervical\_cancer.csv   # Place dataset here (not tracked)
│
├── Thesis\_hantera\_NA.R                    # Step 1: Preprocessing \& EDA
├── Thesis\_ML\_no\_SMOTE.R                   # Step 2a: Scenario 1 — Baseline
├── Thesis\_ML\_SMOTE.R                      # Step 2b: Scenario 2 — SMOTE
├── Thesis\_ML\_class\_weights.R              # Step 2c: Scenario 3 — Class Weights
│
└── README.md
```

\---

## Workflow

Scripts must be run **in order**. The preprocessing script creates the `df` object that all ML scripts depend on.

```
Thesis\_hantera\_NA.R
        │
        ▼  (creates df object)
        ├── Thesis\_ML\_no\_SMOTE.R
        ├── Thesis\_ML\_SMOTE.R
        └── Thesis\_ML\_class\_weights.R
```

\---

## Models

Four classification models are trained and compared in each scenario:

|Model|Details|
|-|-|
|**Logistic Regression**|Linear baseline, fully interpretable|
|**LASSO Logistic Regression**|L1-regularized; automatically performs feature selection (zero-out coefficients)|
|**Random Forest**|Ensemble of decision trees via `ranger`; variable importance extracted|
|**XGBoost**|Gradient boosting via `xgboost`; importance extracted with `xgb.importance`|

All models use **stratified 80/20 train/test split** (`set.seed(123)`) and **cross-validation** for hyperparameter tuning. Classification thresholds are optimized per model using **Youden's J statistic** (maximizes sensitivity + specificity) rather than a fixed cutoff.

\---

## Class Imbalance Strategies

The `Yes` (cancer positive) class is a minority in the dataset. Three scenarios are tested:

|Scenario|Script|Description|
|-|-|-|
|**Baseline**|`Thesis\_ML\_no\_SMOTE.R`|Models trained directly on cleaned, imbalanced data|
|**SMOTE**|`Thesis\_ML\_SMOTE.R`|Synthetic Minority Over-sampling applied **only to training data** via `themis::step\_smote`|
|**Class Weights**|`Thesis\_ML\_class\_weights.R`|Minority class weighted by `n\_total / (2 × n\_class)` — applied inside model specs|

This gives **12 model–scenario combinations** in total.

\---

## Evaluation Metrics

Given the clinical nature of the task — where missing a cancer case (false negative) is costlier than a false alarm — a comprehensive set of metrics is used:

|Metric|Why it matters|
|-|-|
|**Confusion Matrix**|Full breakdown of TP, FP, TN, FN|
|**Sensitivity (Recall)**|How well the model catches true cancer cases|
|**Specificity**|How well the model rules out non-cancer|
|**Precision**|Of all predicted positives, how many are real|
|**PR AUC**|Performance on the minority class across all thresholds — primary selection metric|
|**ROC AUC**|Overall discriminability|
|**MCC**|Matthews Correlation Coefficient — robust metric for imbalanced data|
|**Brier Score**|Calibration of predicted probabilities|

> \*\*Model selection\*\* is based primarily on \*\*PR AUC\*\*, since it is most informative under class imbalance.

\---

## How to Run

### 1\. Install R dependencies

```r
install.packages(c(
  "tidymodels", "tidyverse", "themis",
  "ranger", "xgboost", "glmnet",
  "naniar", "corrplot", "janitor",
  "pROC", "hardhat"
))
```

### 2\. Place the dataset

Download from [UCI ML Repository](https://archive.uci.edu/dataset/383/cervical+cancer+risk+factors) and save as:

```
Dataset/risk\_factors\_cervical\_cancer.csv
```

### 3\. Run scripts in order

```r
# In R or RStudio — run in sequence within the same session:
source("Thesis\_hantera\_NA.R")          # Preprocessing + EDA (creates df)
source("Thesis\_ML\_no\_SMOTE.R")         # Scenario 1: Baseline
source("Thesis\_ML\_SMOTE.R")            # Scenario 2: SMOTE
source("Thesis\_ML\_class\_weights.R")    # Scenario 3: Class Weights
```

> All scripts must run in the \*\*same R session\*\* since they share the `df` object created by `Thesis\_hantera\_NA.R`.

\---

## Dependencies

|Package|Purpose|
|-|-|
|`tidymodels`|Modeling framework (recipes, workflows, tuning)|
|`tidyverse`|Data wrangling and visualization|
|`themis`|SMOTE and other resampling methods|
|`ranger`|Random Forest implementation|
|`xgboost`|XGBoost implementation|
|`glmnet`|LASSO logistic regression|
|`pROC`|ROC analysis and Youden's J threshold optimization|
|`naniar`|Missing data visualization|
|`corrplot`|Spearman correlation matrix|
|`janitor`|Data cleaning utilities|
|`hardhat`|Case weights support for class weighting|



